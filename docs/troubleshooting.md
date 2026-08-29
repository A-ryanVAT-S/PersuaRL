# Troubleshooting

Failures we hit, in rough order of how often they come up.

---

## Training runs but the reward never improves

The most common failure, and almost always the same cause: **the Generator was
not fine-tuned first.**

An un-adapted Generator cannot exploit the `<expert>…</expert>` blocks. Every
route then produces a comparably mediocre response, every rollout in a GRPO group
scores about the same, the group-relative advantages are ~0, and the Selector
receives no gradient. `reward/total` sits flat from step 0.

```bash
bash scripts/sft/train_generator_sft.sh
# then point the RL config at it (this is already the default)
bash scripts/rl/train_persuarl.sh --set generator.adapter_path=outputs/generator_sft
```

`train_persuarl.sh` warns if the adapter directory is missing. This is Table 15
in the paper.

Second most common: **the reward classifiers are missing**, so R1 and R2 score 0
and 30% of the reward signal is dead. The startup log says so explicitly:

```
WARNING reward components unavailable (scored as 0): R1 engagement, R2 intent
```

Fix with `bash scripts/rewards/train_reward_models.sh`.

---

## The Selector collapses onto one route

Check the route histogram in `training_summary.json` or the `_routes.json` an
inference run writes. If one label accounts for nearly everything:

```bash
# more exploration pressure
--set rewards.penalties.repetition_beta=0.35
--set rewards.penalties.repetition_max=0.25

# hotter sampling (the default 1.2 is already high for a reason)
--set train.temperature=1.5

# larger groups -> better advantage estimates
--set train.num_generations=12
```

If it collapses specifically onto the all-experts route `O`, the complexity
penalty is too weak relative to how much the reward rewards more context:

```bash
--set rewards.penalties.complexity_alpha=0.05
```

---

## `invalid_routes` is non-zero

The constrained decoder is not installed. `patch_generate_with_constraint`
replaces `trainer.model.generate` *after* `GRPOTrainer` builds the model — if you
call it before, or reorder that block, the policy free-runs and emits
whatever it likes.

Also check `route_token_ids` did not silently pass on a tokenizer where the
letters are not single tokens; it raises a clear error, so if you see one, your
Selector backbone needs `AllowedSequencesProcessor` instead.

---

## CUDA out of memory

Stage 4 holds a Selector, a Generator, a 7B judge, two BERT classifiers and
BERTScore at once. In descending order of effect:

```bash
# 1. drop the judge during development (frees ~14 GB)
bash scripts/rl/train_persuarl.sh --no-judge

# 2. 4-bit the Generator
--set generator.quantization.bits=4

# 3. smaller GRPO groups
--set train.num_generations=4

# 4. shorter rollouts
--set generator.max_new_tokens=96
--set train.max_prompt_length=384

# 5. gradient checkpointing (SFT stages)
--set train.gradient_checkpointing=True
```

If OOM hits specifically inside the reward function, it is usually BERTScore
batching against a long context:

```bash
--set rewards.contextual.batch_size=8
--set rewards.judge.batch_size=4
```

---

## `bitsandbytes` will not import / 4-bit fails

`bitsandbytes` is Linux-first. On Windows use WSL2, or run without quantisation:

```bash
--set model.quantization.bits=null
```

On a cluster, a CUDA-version mismatch between torch and bitsandbytes is the usual
cause. Reinstall torch for your actual CUDA version first, then bitsandbytes:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install --force-reinstall bitsandbytes
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

---

## Gated model downloads fail (Llama, Mistral)

Accept the licence on the model's HuggingFace page, then:

```bash
huggingface-cli login
# or put HF_TOKEN=hf_... in .env
```

On an offline cluster, pre-download to shared storage and point the config at the
local path:

```bash
--set model.id=/scratch/models/llama-3.2-3b-instruct
```

---

## `ValueError: LoRA target module(s) [...] do not exist in this backbone`

You set `lora.target_modules` explicitly for a backbone that names its
projections differently. Phi-3 fuses QKV into `qkv_proj` and gate/up into
`gate_up_proj`; GPT-NeoX-style models use `query_key_value`.

The fix is almost always to delete the override and let auto-detection do it:

```yaml
lora:
  target_modules: null
```

The error message lists the candidates that *are* present, if you do need to be
explicit.

---

## Expert merge produces zero rows

```
ValueError: expert merge produced zero rows -- the CSVs do not share
(conversation_id, turn_no) keys.
```

The expert CSVs were generated from a different dialogue file than the one
`data.dialogues_path` points at. Regenerate them against the current corpus:

```bash
bash scripts/experts/run_expert_inference.sh
```

The join is on keys, not row order, on purpose — positional concatenation would
"succeed" here and silently pair every turn with the wrong analysis.

---

## `prepare_data.sh` reports thousands of unparseable intent rows

Expected with the shipped files: `intent.csv` currently duplicates
`engagement.csv`, so its answers are engagement prose and do not match the intent
pattern. See [the data README](../data/README.md#known-issues-in-the-shipped-files)
for what this affects and how to fix it. Training still runs; R2 contributes 0.

---

## NLTK `punkt` download fails

R4 falls back to whitespace tokenisation automatically — the reward is slightly
coarser but the run continues. To fix properly:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Newer NLTK versions renamed the resource, which is why `setup_env.sh` tries both.

---

## Judge parse-failure rate is high

Check `judge_failure_rate` in `training_summary.json`. Above a few percent means
the judge prompt and the judge checkpoint disagree. Either:

- you swapped in a non-Prometheus judge that does not emit `[RESULT] <n>` — adapt
  `parse_judge_score` in `rewards/judge.py`, or
- `max_new_tokens` is too small and the verdict is being cut off:
  `--set rewards.judge.max_new_tokens=300`.

Failures silently score 3/5, so a high rate makes R5 an expensive constant.

---

## Results do not match the paper

Check these in order:

1. **The split.** RL and inference must use the same ratios (`0.85 / 0.0`) and
   the same seed. A different split is a different test set.
2. **Reward weights.** Defaults are the tuned Table 11 setting. If
   `RewardWeights` warned that they do not sum to 1, the composite is no longer
   in `[0, 1]`.
3. **Decoding.** Appendix D.2 uses `temperature=0.8`, `top_p=0.95`, `top_k=40`,
   `max_tokens=512` at inference. The defaults here match.
4. **The Generator warm start.** Without it you are measuring Table 15, not
   Table 2.
5. **Run-to-run variance.** GRPO rollouts are sampled, and the paper reports
   single runs. Small differences are expected; a gap of several BLEU points is
   a configuration problem, not noise.

---

## Multiprocessing errors on Windows

`RoutingStatistics.shared` uses a `multiprocessing.Manager`, which needs the
spawn-safe entry-point guard. The CLIs all have `if __name__ == "__main__":`, so
run them as modules (`python -m persuarl.cli.train_selector`) rather than
importing and calling `train_selector` from a notebook on Windows.

Simpler: run training under WSL2 or Linux. `bitsandbytes`, `flash-attn` and
`vllm` are all happier there.

---

## Still stuck

Turn on debug logging and capture the run:

```bash
python -m persuarl.cli.train_selector --config configs/rl/persuarl.yaml \
    --log-level DEBUG --log-file logs/debug.log
```

The first ~40 lines record the resolved config, every override, the visible GPUs,
which reward components loaded, and the constrained action token ids. That header
answers most questions on its own.
