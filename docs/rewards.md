# The reward function

PersuaRL's Selector never receives a supervised label. Everything it learns about
which experts to consult comes from this scalar:

```
R(y_t, x_t) = Σ_k β_k · R_k(y_t, x_t)  −  (complexity + repetition + load-balance)  +  load-bonus
```

Five terms score the **generated response**; three penalties score the **routing
decision**. This document covers what each one measures, why it exists, and how
to change it.

Code: [`src/persuarl/rewards/`](../src/persuarl/rewards/). Configuration:
[`configs/rewards/insuredial.yaml`](../configs/rewards/insuredial.yaml).

---

## Why a composite reward

A single metric cannot capture persuasive effectiveness. A response can be
strategically well-framed but miss the user's intent; it can address the intent
precisely while sounding mechanical; it can be fluent and on-topic while
repeating the previous turn verbatim. Prior work evaluates these dimensions
*after* training with human judgments. PersuaRL turns them into training signal,
which is what lets the Selector learn coordination patterns that balance
competing objectives rather than optimising one of them.

Every `R_k` is normalised to `[0, 1]` before weighting. This is not cosmetic: the
`β` weights only mean what they look like they mean if the terms share a range,
and GRPO's group-relative advantages are computed on the raw scalar.

---

## R1 — Engagement Strategy Consistency

**Weight β₁ = 0.15.** Code: `rewards/consistency.py`.

Does the response use a persuasion strategy appropriate to the user's turn?

A BERT classifier fine-tuned on InsureDial's six strategy labels (82.14% accuracy,
Table 9) gives `P_p(u_T)` — the distribution over strategies the *user's*
utterance implies. Separately, each strategy `p` has a **prototype** embedding:
the mean encoder embedding of every training utterance labelled `p`. The
generated response is embedded in the same space and compared to each prototype
by cosine similarity, `S_p(r_T)`.

```
R1 = Σ_p P_p(u_T) · S_p(r_T)  +  λ · max_p S_p(r_T)
```

The first term rewards matching the strategy distribution the user's turn calls
for. The `λ` term is a confidence bonus: being *strongly* aligned with some
strategy beats being weakly aligned with all of them. The paper uses
`0.3 < λ < 1`; the shipped config uses `0.1`, which weights distribution-matching
more heavily.

Cosine lives in `[-1, 1]`, so the raw score is rescaled by `(x + 1) / 2` and
clamped — the `λ` term can push it past 1.

> **On the annotation mismatch.** Strategy labels are annotated on *agent* turns,
> but R1 conditions on the *user* turn. This is deliberate (Appendix FAQ 1): it
> makes the Selector anticipate the appropriate response strategy from what the
> user expressed, and avoids needing gold strategy labels at inference time. It
> is an approximation, and the paper says so.

## R2 — Intent Consistency

**Weight β₂ = 0.15.** Same machinery as R1, over the six user-intent labels, with
its own classifier (84.91% accuracy) and its own prototypes.

```
R2 = Σ_i P_i(u_T) · S_i(r_T)  +  λ · max_i S_i(r_T)
```

Both classifiers and both prototype tensors come from one command:

```bash
bash scripts/rewards/train_reward_models.sh
```

Prototypes are computed from the **training split only** — computing them over
all data would leak test utterances into the reward signal. The tensor is saved
next to the classifier it was built from, and `PrototypeScorer` refuses to load a
prototype tensor whose row count disagrees with the classifier's label count,
because a silent mismatch there produces plausible-looking garbage.

## R3 — Contextual Appropriateness

**Weight β₃ = 0.20.** Code: `rewards/contextual.py`.

Is the response anchored to the conversation, and especially to what the user
just said?

```
R3 = min( ( BERTScore_F1(x_i, y_i) + 2 · BERTScore_F1(u_i, y_i) ) / 3 , 1 )
```

where `x_i` is the full dialogue context, `u_i` the current user utterance, and
`y_i` the generated response. The latest turn is weighted double because
relevance to the user's last message dominates perceived appropriateness.
Dividing by 3 renormalises; `min(·, 1)` clips outliers that would otherwise
dominate a GRPO group's advantage.

The single-model baselines set `use_reference: true`, which scores against the
ground-truth reply instead — there is no Generator in that setting, and that is
the formulation the original baseline used.

## R4 — Non-Repetitiveness

**Weight β₄ = 0.15.** Lexical, no model, effectively free.

```
R4 = 1 − |r_{T−1} ∩ r_T| / |r_{T−1} ∪ r_T|
```

Jaccard distance between the current response and the previous agent turn. This
is the counterweight to R3: a response that maximises similarity to the context
by *copying* it scores well on R3 and badly here. Empirically the pair improves
Distinct-2 alongside relevance, which is the evidence that the model is not just
echoing input (Appendix FAQ 3).

The first turn of a conversation scores 0, not 1. An opening line cannot
demonstrate non-repetitiveness, and giving it free full credit would bias the
policy toward whatever route it happened to pick on turn one.

## R5 — LLM-as-a-Judge

**Weight β₅ = 0.35 — the largest.** Code: `rewards/judge.py`.

Prometheus-7B-v2.0 scores each response 1–5 against a persuasion rubric
(`data/prompts.py`), mapped to `[0, 1]` by `(score − 1) / 4`. This is the only
term that evaluates persuasive quality above the surface level — whether the
response reframes the product around the user's actual needs, anticipates
objections, and moves toward a next step.

Three implementation choices worth knowing:

- **Greedy decoding** (`do_sample=False`). A stochastic judge injects variance
  straight into the advantage estimate.
- **Explicit anti-clustering instruction.** Without instruction 3 in the prompt,
  Prometheus assigns 3/5 to nearly everything, which flattens the signal GRPO
  needs.
- **Defensive parsing.** An unparseable verdict falls back to 3 rather than
  aborting a multi-hour run. The failure rate is tracked and reported in
  `training_summary.json`; anything above a few percent means the prompt and the
  checkpoint have drifted apart.

R5 is also the expensive term — one 7B generation per rollout, i.e.
`batch_size × num_generations` per step. To develop without it:

```bash
bash scripts/rl/train_persuarl.sh --no-judge   # R5 becomes a constant 0.5
```

---

## The penalties

The five `R_k` score the response. These three score the **decision**, and they
are what stop the Selector from degenerating.

### Complexity — `α · N`, α = 0.025

Linear in the number of activated experts. Without it, "select everything" is
weakly dominant: more context rarely hurts a single response, so the reward alone
never pays for restraint. **This penalty is what makes PersuaRL a selector rather
than AllExpert.**

### Route repetition — `min(β · max(0, F − 1), P_max)`, β = 0.2, P_max = 0.15

`F` is a route's usage divided by its uniform-usage share. This is exploration
pressure with a specific mechanism behind it: GRPO computes advantages *within* a
group of rollouts, so once the policy collapses onto one route every rollout in a
group scores identically and the gradient vanishes. The penalty keeps the group
diverse enough to produce signal.

Inactive for the first `repetition_warmup: 16` rollouts — early on, every route
looks overused relative to a tiny history, and penalising that is penalising
noise.

### Load balance — `γ (R_k − 1)²` when `R_k > 1`, γ = 0.4, capped at 0.15

Per-expert version of the same idea. `R_k` is expert *k*'s usage divided by the
mean usage of the *other* experts. Overuse is penalised; underuse earns a
mirrored bonus (`γ = 0.05`, capped at 0.08). Both are averaged over the route's
experts so a 4-expert route is not penalised four times for one imbalanced
expert.

Counters live in `RoutingStatistics`, backed by `multiprocessing.Manager` proxies
so dataloader workers share one view of usage history.

---

## Weights

The shipped weights are the tuned setting from Table 11, selected on a 15%
held-out split by perplexity:

| β₁ | β₂ | β₃ | β₄ | β₅ | PPL |
|---|---|---|---|---|---|
| 0.10 | 0.15 | 0.35 | 0.30 | 0.10 | 4.0165 |
| 0.20 | 0.30 | 0.10 | 0.20 | 0.20 | 3.9173 |
| 0.30 | 0.10 | 0.30 | 0.10 | 0.20 | 3.9196 |
| **0.15** | **0.15** | **0.20** | **0.15** | **0.35** | **3.9172** |

Change them from the command line:

```bash
bash scripts/rl/train_persuarl.sh --set rewards.weights.judge=0.25
```

`RewardWeights` warns if the weights do not sum to 1 — the composite is then no
longer in `[0, 1]` and is not comparable to the paper's numbers.

### Ablating a component

```bash
bash scripts/rl/train_persuarl.sh --ablate engagement    # zero β₁
```

Remaining weights are **not** renormalised, matching the paper: the ablation asks
what each term contributes, not how the model does with a rebalanced objective.

Results from Table 12 (Qwen 2.5 3B):

| configuration | B-2 | MT | BF1 | D-2 | R1 |
|---|---|---|---|---|---|
| PersuaRL (full) | **0.375** | **0.250** | **0.760** | **0.991** | **0.609** |
| − R1 engagement | 0.341 | 0.214 | 0.694 | 0.965 | 0.521 |
| − R2 intent | 0.349 | 0.226 | 0.706 | 0.969 | 0.536 |
| − R3 contextual | 0.351 | 0.229 | 0.721 | 0.985 | 0.563 |
| − R4 repetition | 0.357 | 0.234 | 0.729 | 0.989 | 0.571 |
| − R5 judge | 0.342 | 0.219 | 0.701 | 0.966 | 0.529 |
| − (R1 + R2) | 0.335 | 0.211 | 0.691 | 0.961 | 0.511 |
| no rewards | 0.323 | 0.205 | 0.684 | 0.948 | 0.501 |

R1, R2 and R5 are the load-bearing terms; R3 and R4 are complementary.

---

## Reward circularity

A reasonable objection: if the model is trained against reward models, does it
learn to satisfy the reward models rather than to persuade?

PersuaRL's answer is architectural (Appendix FAQ 6–7). RL updates **only the
expert-selection policy**. The Generator is frozen during reward computation, and
the reward models are frozen throughout — so no gradient ever reaches the thing
being evaluated or the thing doing the evaluating. The Selector influences
generation only indirectly, by choosing which expert signals the Generator sees.
That separation is why the human-evaluation gains in Table 4 (which the reward
models cannot see) track the automatic ones.

The one place the Generator does update is the co-adaptation SFT step — and that
step optimises ordinary NLL against the **ground-truth reply**, not the reward.

---

## Adding a reward

1. Write a scorer returning a `[0, 1]` array of length `batch_size`, in a new
   module under `rewards/`.
2. Add its name to `REWARD_NAMES` and a field to `RewardWeights` in
   `rewards/composite.py`.
3. Wire it into `PersuasiveRewardModel.score` and `build_reward_model`.
4. Add its weight to `configs/rewards/insuredial.yaml`, rebalancing so the
   weights still sum to 1.

Both training regimes pick it up automatically — they share one reward
implementation, which is the point.
