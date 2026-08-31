# The pipeline, stage by stage

Seven stages. Each is a script, each is independently runnable, and each writes
into `$MODELS_ROOT` where the next one picks it up.

Stages run in order 0 to 6: data preparation, expert modules, reward
classifiers, generator SFT, selector GRPO, inference, evaluation. Three of them
also feed sideways into the GRPO stage rather than only forward — Stage 1 caches
the expert answers it scores against, Stage 2 trains the R1/R2 classifiers it
rewards with, and Stage 3 produces the generator it warm-starts from.

Everything below assumes `$MODELS_ROOT` (default `./outputs`) and `$DATA_ROOT`
(default `./data`), both settable in `.env`.

---

## Stage 0 — Data preparation

```bash
bash scripts/prepare_data.sh
```

The shipped expert CSVs store each answer as prose ("The persuasion strategy is
Logical appeal and the reason is …"). Expert SFT and the reward classifiers need
that split back into `(utterance, label, reason)`. This stage does the split
using the per-expert patterns in `experts/registry.py`.

**Writes** `data/processed/experts/*.csv` and `data/processed/classifiers/*.csv`.
Takes under a minute, no GPU.

Rows whose answer does not parse are counted and dropped, never guessed at.

---

## Stage 1 — Expert modules

```bash
bash scripts/experts/train_all_experts.sh
bash scripts/experts/train_all_experts.sh Qwen/Qwen2.5-3B-Instruct   # any backbone
bash scripts/experts/train_expert.sh intent                          # just one
```

Four decoder-only LMs, LoRA fine-tuned on `(context, user turn) → (label,
reason)` with an NLL objective. They differ **only** by system prompt and output
template — both declared in `experts/registry.py`, both consumed by one trainer.

**Writes** `$MODELS_ROOT/experts/<expert>/`. Roughly 40 min each on an A100.

**Optional.** The repo ships precomputed answers for all 13,383 turns, so you can
skip straight to Stage 2 and use those. Train the experts if you want to change
the backbone, the label space, or annotate a new corpus.

To regenerate the cached answers from your own checkpoints:

```bash
bash scripts/experts/run_expert_inference.sh
```

Experts are loaded and freed one at a time — four 3B models at once will not fit
alongside anything else on a single 80GB card.

---

## Stage 2 — Reward classifiers

```bash
bash scripts/rewards/train_reward_models.sh
BASE_MODEL=answerdotai/ModernBERT-base bash scripts/rewards/train_reward_models.sh
```

Two BERT sequence classifiers (engagement strategy, user intent) plus, for each,
a `(num_classes, hidden_dim)` tensor of **class prototypes** — the mean encoder
embedding of every training utterance carrying that label. R1 and R2 need both.

**Writes** `$MODELS_ROOT/reward_models/<dimension>_classifier/` containing the
model and its `prototypes.pt`. About 15 min on one A100.

Reproducing Table 9:

| classifier | BERT-large | | DistilBERT | | ModernBERT | |
|---|---|---|---|---|---|---|
| | Acc | F1 | Acc | F1 | Acc | F1 |
| Engagement (ESCR) | 82.14 | 75.53 | 81.49 | 71.80 | 81.64 | 72.39 |
| Intent (ICR) | 84.91 | 74.49 | 81.52 | 72.34 | 80.78 | 70.26 |

Prototypes are built in the same command, from the same checkpoint, on the
training split only. Building them separately is how you end up with a prototype
tensor that does not match the classifier it is scored against.

---

## Stage 3 — Generator SFT

```bash
bash scripts/sft/train_generator_sft.sh
bash scripts/sft/train_generator_sft.sh microsoft/Phi-3-mini-128k-instruct
```

Fine-tunes the Generator on `(history, user turn, <expert>…</expert> analysis) →
reference reply`, with the prompt masked out of the loss.

**Writes** `$MODELS_ROOT/generator_sft/`. About 3–4 h on an A100.

**Do not skip this.** An un-adapted Generator cannot exploit the `<expert>`
blocks, so every route produces a comparably mediocre response, every rollout in
a GRPO group scores alike, and the Selector has no signal to learn from. That is
Table 15 in the paper, and it is the single most common way to get a PersuaRL run
that trains without improving.

`data.route_source` controls which experts appear in each training example:

| value | behaviour | use for |
|---|---|---|
| `all` | every expert, every turn | the default warm start; also the AllExpert ablation |
| `random` | a random route per turn | robustness to *any* subset before RL explores |
| `logged` | replay `best_routes.csv` from a finished RL run | rebuilding a matched Generator offline |

---

## Stage 4 — PersuaRL selector GRPO

```bash
bash scripts/rl/train_persuarl.sh
bash scripts/rl/train_persuarl.sh --selector meta-llama/Llama-3.2-3B-Instruct
bash scripts/rl/train_persuarl.sh --freeze-generator     # ablation D.3.4
bash scripts/rl/train_persuarl.sh --ablate intent        # ablation Table 12
```

The method. One training step:

1. The Selector sees `(history, user turn)` and samples **G = 8** route letters
   under a single-token constraint. Each letter is a binary expert mask.
2. For each rollout, the selected experts' answers are packed into the Generator
   prompt and the **frozen** Generator produces a response.
3. The composite reward scores each response; the routing penalties adjust it.
4. GRPO computes group-relative advantages over the 8 rollouts and updates
   **only** the Selector.
5. The highest-reward rollout becomes one NLL step for the Generator — the
   co-adaptation that lets `A_φ` track the Selector's evolving policy.

**Writes** `$MODELS_ROOT/persuarl/selector/`, `$MODELS_ROOT/persuarl/generator/`,
plus `best_routes.csv` and `training_summary.json`. Roughly 25–28 h on an A100
80GB with a 3B Selector, 3B Generator and 7B judge resident.

### Constrained decoding

The action space is 15 routes, labelled `A`–`O`. On the first generated token the
logits processor masks everything except those 15 ids; afterwards it forces EOS.
One token per action, no JSON parsing, no malformed-route retries.

`route_token_ids` verifies each letter is genuinely single-token in the Selector's
vocabulary and fails loudly if not — a multi-token label would quietly break the
constraint. With a fifth expert (31 routes) the single-letter scheme still fits,
but past 26 you need `AllowedSequencesProcessor` instead.

Because TRL owns the `generate` call, the constraint is installed by wrapping the
bound method (`patch_generate_with_constraint`). A **fresh** processor is built
per call — the processor caches the prompt boundary, and reusing one across
batches of different prompt lengths is a real bug.

### What to watch

| signal | healthy | trouble |
|---|---|---|
| `reward/total` | climbing, plateauing near 0.65–0.67 | flat from step 0 → check the Generator warm start |
| route distribution | several routes recurring | one route → raise `temperature` or `repetition_beta` |
| `invalid_routes` | ~0 | non-zero → the constraint is not installed |
| `judge_failure_rate` | < 2% | higher → judge prompt/checkpoint mismatch |
| expert usage | roughly balanced | one expert dominating → raise `load_balance_gamma` |

Reward curves are logged to TensorBoard: `tensorboard --logdir outputs/persuarl/selector`.

---

## Stage 5 — Inference

```bash
bash scripts/inference/run_pipeline.sh
bash scripts/inference/run_pipeline.sh --mode all         # AllExpert baseline
bash scripts/inference/run_pipeline.sh --mode prompting   # prompted, no RL
bash scripts/inference/run_pipeline.sh --live-experts     # call expert LMs
bash scripts/inference/run_pipeline.sh --limit 20         # smoke run
```

Selector → Experts → Generator over the held-out test split, one conversation at
a time.

**One difference from training that matters:** history is built from the model's
own replies, not the reference transcript. Errors compound across a dialogue,
which is what you want to measure — a system that only looks good under teacher
forcing is not a system.

**Writes** the results CSV plus a `_routes.json` route histogram.

| flag | effect |
|---|---|
| `--mode persuarl` | trained Selector, constrained decoding (default) |
| `--mode all` | every expert every turn |
| `--mode prompting` | Selector backbone prompted to choose, unconstrained, no RL |
| `--live-experts` | call the expert LMs instead of the cache (slower, honest latency) |

---

## Stage 6 — Evaluation

```bash
bash scripts/eval/compute_metrics.sh results/persuarl.csv
bash scripts/eval/compute_metrics.sh results/persuarl.csv --with-judge
bash scripts/eval/compute_metrics.sh results/persuarl.csv --with-ppl meta-llama/Llama-3.2-3B-Instruct
```

BLEU-2, METEOR, BERTScore-F1, Distinct-2 and ROUGE-1 always; perplexity and
LLM-as-a-judge opt in, because each loads an extra model.

**Writes** a per-turn CSV and a `.summary.json` with the corpus means — the row
you paste into a results table.

BERTScore is batched over the whole file. The LLM-J column reuses the *training*
judge, so the reported number and the R5 signal come from the same rubric.

---

## Baselines

Everything in Table 2 comes from the same code:

```bash
# single-shot: no training at all
bash scripts/inference/run_pipeline.sh --mode all --set generator.adapter_path=null

# SFT
bash scripts/sft/train_baseline_sft.sh meta-llama/Llama-3.2-3B-Instruct

# GRPO from the instruct checkpoint (Table 13)
bash scripts/rl/train_grpo_single.sh meta-llama/Llama-3.2-3B-Instruct

# Single → SFT → RL (warm start)
bash scripts/rl/train_grpo_warmstart.sh
```

The warm start **merges** the SFT adapter into the base weights before attaching
a fresh LoRA. Stacking a second adapter on a live one means the RL gradient works
against the SFT adapter as well as the base model, and the effective learning
rate on the composition is not the one you configured.

---

## Ablation map

| paper section | command |
|---|---|
| D.3.1 rewards (Table 12) | `train_persuarl.sh --ablate <component>` |
| D.3.2 tool selection (Fig. 2/4/5) | `run_pipeline.sh --mode all` / `--mode prompting` |
| D.3.3 base model + GRPO (Table 13) | `train_grpo_single.sh` |
| D.3.4 frozen generator (Table 14) | `train_persuarl.sh --freeze-generator` |
| D.3.5 AllExpert + untrained gen (Table 15) | `run_pipeline.sh --mode all --set generator.adapter_path=null` |
| D.3.6 AllExpert + trained gen (Table 16) | `run_pipeline.sh --mode all` |
| D.3.7 untrained small models (Table 17) | `run_pipeline.sh --set generator.adapter_path=null --set selector.adapter_path=null` |
| Experts (Table 3) | `train_persuarl.sh` with an expert removed from `EXPERT_KEYS` |

---

## Hardware

The paper ran everything on a single **A100 80GB**, ~25–28 h per model.

Approximate resident memory during Stage 4, all bf16:

| component | 3B | 24B |
|---|---|---|
| Selector + LoRA | ~7 GB | — |
| Generator + LoRA | ~7 GB | ~50 GB (4-bit) |
| Prometheus-7B judge | ~14 GB | ~14 GB |
| BERT classifiers + BERTScore | ~2 GB | ~2 GB |

Fitting on less than 80 GB:

```bash
# 4-bit the policy
--set generator.quantization.bits=4

# smaller GRPO groups (weaker advantage estimates, less memory)
--set train.num_generations=4

# drop the judge entirely during development
--no-judge
```
