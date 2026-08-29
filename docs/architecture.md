# Code architecture

Where things live and why they live there. Read this before adding a backbone, an
expert, or a reward.

```
src/persuarl/
├── constants.py        expert keys, column names, label sets
├── routes.py           the Selector's 15-route action space
├── config.py           YAML + defaults chains + --set overrides
│
├── data/               InsureDial -> model inputs
│   ├── dataset.py        loading, key-joins, turn iteration with history
│   ├── splits.py         conversation-level splitting
│   ├── prompts.py        every system prompt, incl. the judge rubric
│   └── formatting.py     prompt construction + prompt-masked tokenisation
│
├── models/             backbone-agnostic plumbing
│   ├── loader.py         quantisation, dtype, LoRA target auto-detection
│   └── decoding.py       constrained decoding for the Selector
│
├── experts/            the four expert modules (T_i)
│   ├── registry.py       ExpertSpec x4: prompts, templates, parsers
│   ├── training.py       one trainer for all four
│   └── inference.py      live ExpertRunner + CachedExpertPool
│
├── rewards/            R = sum_k beta_k R_k, minus penalties
│   ├── consistency.py    R1, R2 (prototype similarity)
│   ├── contextual.py     R3 (BERTScore), R4 (Jaccard)
│   ├── judge.py          R5 (Prometheus)
│   ├── penalties.py      complexity, route repetition, load balance
│   ├── classifiers.py    training the R1/R2 classifiers + prototypes
│   └── composite.py      assembly; the class both trainers share
│
├── training/
│   ├── sft.py            Generator SFT and the SFT baselines
│   ├── grpo_selector.py  PersuaRL (the method)
│   └── grpo_single.py    single-model GRPO baselines
│
├── inference/pipeline.py Selector -> Experts -> Generator
├── evaluation/           BLEU-2, METEOR, BERTScore, Distinct-2, ROUGE-1, PPL, LLM-J
└── cli/                  thin argparse wrappers; all logic lives above
```

---

## Four ideas the layout is built around

### 1. The action space is data, not code

`routes.py` enumerates every non-empty subset of `EXPERT_KEYS` and assigns each a
single letter. Nothing else in the codebase hard-codes "15" or "four experts" —
the Selector prompt, the constrained decoder, the penalty normalisation and the
route log all derive from that one table.

Adding a fifth expert widens it automatically (31 routes, labels `A`–`Z` plus 5
more; the assertion in `routes.py` catches the point where single-letter labels
run out).

### 2. The experts are a registry, not four programs

The original repo had four fine-tuning scripts that were byte-identical apart
from a system prompt, an output template and a filename. `experts/registry.py`
declares those differences as four `ExpertSpec` objects; `experts/training.py` is
the one program that consumes them.

Each spec also carries an `answer_pattern` that parses a rendered answer back
into `(label, reason)` — which is what lets `prepare_data.py` rebuild training
data from the shipped output CSVs.

### 3. One reward implementation, two training regimes

`PersuasiveRewardModel` scores a batch of candidate responses. It does not know
or care whether those responses came from a routed Generator (PersuaRL) or
straight from the policy (single-model GRPO). Both `SelectorRewardFunction` and
`SingleModelRewardFunction` wrap the same object.

This matters for correctness, not just tidiness: in the original scripts the
method and its baseline each had their own copy of the reward and they had
already drifted — different R1/R2 formulations, different judge prompts — which
makes a comparison between them not quite a comparison.

### 4. The backbone is a config value

`models/loader.py` handles everything that varies across model families:

- **LoRA targets** are auto-detected by inspecting the loaded module names.
  Llama/Qwen/Mistral expose `q_proj`/`k_proj`/`v_proj`; Phi-3 fuses them into
  `qkv_proj` and gate/up into `gate_up_proj`; GPT-NeoX uses `query_key_value`.
  An explicit override is validated against the model, because a typo there
  trains nothing and reports success.
- **Quantisation** is a `QuantizationSpec` built from config, with
  `prepare_model_for_kbit_training` applied automatically when 4-/8-bit is on.
- **Padding side** is explicit at every call: `right` for SFT (labels must align
  with inputs), `left` for anything that generates (right-padding pushes the
  generation cue away from the sequence end).

So `--set model.id=<anything>` works for SFT, and `--set selector.id=` /
`--set generator.id=` work for GRPO.

---

## Design decisions worth knowing

**Conversation-level splitting, always.** `data/splits.py` is the only place a
split happens, and it splits on `conversation_id`. It uses a local
`np.random.default_rng`, not `np.random.seed`, so it never disturbs a caller's
RNG state.

**Key-joins, not positional concatenation.** `merge_expert_outputs` joins on
`(conversation_id, turn_no)`. The original code used `pd.concat(axis=1)`, which
is correct only while every file has identical row order and silently misaligns
otherwise.

**Prompt masking in exactly one function.** `tokenize_with_prompt_mask` is used
by expert SFT, Generator SFT, baseline SFT and the co-adaptation step. It clamps
the mask boundary to the truncated length, so a long history eats into the
completion rather than masking past the end of the sequence.

**Fresh logits processors per `generate` call.** The processors cache the prompt
boundary on first call. Reusing one across batches of different prompt lengths
misidentifies where generation started — a real bug in the original inference
script.

**Shared routing counters.** `RoutingStatistics` can be backed by
`multiprocessing.Manager` proxies so TRL's dataloader workers observe one usage
history. Proxy containers do not support in-place mutation of nested values,
which is why every update reassigns.

**Graceful degradation over hard failure.** A missing reward classifier scores 0
and logs a warning; an unparseable judge verdict falls back to neutral; NLTK
falls back to whitespace tokenisation. A multi-hour RL run should not die at hour
three because one component is unavailable — but it should say so loudly at
startup, and every one of these paths does.

---

## Extending

### A new backbone

Nothing to write. `--set model.id=<hf-id>` (or `selector.id` / `generator.id`).
Add a file under `configs/models/` if you want to pin quantisation or dtype for
it.

### A fifth expert

1. Add its key to `EXPERT_KEYS` in `constants.py` (order is load-bearing).
2. Add an `ExpertSpec` to `experts/registry.py` — system prompt, completion
   template, answer pattern, label set.
3. Add a config under `configs/experts/`.

The route table, Selector prompt, Generator analysis block, penalty
normalisation and load balancing all widen automatically. The assertion at the
bottom of `registry.py` fails at import time if the registry and `EXPERT_KEYS`
disagree, rather than three hours into a run.

### A new reward

See [rewards.md](rewards.md#adding-a-reward). Both trainers pick it up, because
they share one reward object.

### A new training regime

Add a module under `training/`, a config under `configs/`, a CLI wrapper under
`cli/` and a script under `scripts/`. Reuse `build_reward_model`,
`split_by_conversation` and `attach_lora` rather than reimplementing them — that
is what keeps two regimes comparable.

---

## Tests

```bash
make test          # or: python -m pytest tests/ -v
```

95 tests, no GPU, no model downloads. They cover the invariants that fail
silently otherwise: the route table, the reward weights and penalty caps,
conversation-level split disjointness, key-based expert merging, prompt masking,
and config resolution.

`tests/conftest.py` puts `src/` on the path, so the suite runs from a fresh clone
without `pip install -e .`.
