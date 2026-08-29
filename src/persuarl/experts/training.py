"""One trainer for all four experts.

Each expert is a decoder-only LM fine-tuned with LoRA on
``(dialogue context, user utterance) -> (label, reason)``, optimising the NLL of
the target string (Appendix C.2). The only per-expert differences -- the system
prompt and the target template -- come from :mod:`persuarl.experts.registry`.

The context handling is worth understanding: within a conversation the prompt
accumulates the *ground-truth* transcript, but the training target for turn *t*
is only turn *t*'s analysis. So the expert learns to analyse the latest user
turn in context, never to continue the dialogue.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from ..constants import (
    COL_AGENT_REPLY,
    COL_CONVERSATION_ID,
    COL_TURN_NO,
    DEFAULT_SEED,
)
from ..data.formatting import has_trainable_tokens, tokenize_with_prompt_mask
from ..data.splits import split_by_conversation
from ..models.loader import attach_lora, load_causal_lm, load_tokenizer, quantization_from_config
from ..utils.logging import get_logger
from .registry import ExpertSpec, get_expert

LOGGER = get_logger(__name__)


def build_expert_dataset(
    frame: pd.DataFrame,
    spec: ExpertSpec,
    tokenizer,
    *,
    text_column: str = "utterance",
    label_column: str = "label",
    reason_column: str = "reason",
    max_length: int = 1024,
) -> Dataset:
    """Turn an annotated CSV into prompt-masked training examples.

    Expected columns: ``conversation_id``, ``turn_no``, the utterance column,
    ``label`` and ``reason``. Use ``persuarl.cli.prepare_data`` to derive this
    shape from the shipped expert-output CSVs.
    """
    required = {COL_CONVERSATION_ID, COL_TURN_NO, text_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{spec.key}: dataset is missing column(s) {sorted(missing)}")

    frame = frame.sort_values([COL_CONVERSATION_ID, COL_TURN_NO])
    examples: list[dict[str, Any]] = []
    skipped = 0

    for _, group in frame.groupby(COL_CONVERSATION_ID, sort=True):
        # Conversation-scoped message history, reset per conversation.
        messages: list[dict[str, str]] = [{"role": "system", "content": spec.system_prompt}]

        for _, row in group.iterrows():
            utterance = str(row[text_column])
            messages.append({"role": "user", "content": utterance})

            completion = spec.render_completion(
                label=row.get(label_column, ""),
                reason=row.get(reason_column, ""),
            )
            encoded = tokenize_with_prompt_mask(
                tokenizer, messages, completion, max_length=max_length
            )
            if has_trainable_tokens(encoded):
                examples.append(encoded)
            else:
                skipped += 1

            # The *agent's* reply, not the expert's answer, continues the
            # dialogue -- the expert observes conversations, it does not drive them.
            messages.append({"role": "assistant", "content": str(row.get(COL_AGENT_REPLY, ""))})

    if skipped:
        LOGGER.warning("%s: skipped %d fully-truncated examples (raise max_length?)", spec.key, skipped)
    LOGGER.info("%s: built %d training examples", spec.key, len(examples))
    return Dataset.from_list(examples)


def train_expert(config, expert_key: str) -> dict[str, float]:
    """Fine-tune one expert end to end. Returns held-out loss and perplexity."""
    spec = get_expert(expert_key)
    LOGGER.info("=== training %s (%s) ===", spec.display_name, spec.key)

    model_config = config.section("model")
    train_config = config.section("train")
    data_config = config.section("data")

    output_dir = Path(config.get("output_dir")) / spec.key
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(
        model_config.get("id"),
        padding_side="right",  # SFT: labels must stay aligned with inputs
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )

    quantization = quantization_from_config(model_config.section("quantization"))
    model = load_causal_lm(
        model_config.get("id"),
        dtype=model_config.get("dtype", "auto"),
        device_map=model_config.get("device_map", "auto"),
        quantization=quantization,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = config.section("lora")
    model = attach_lora(
        model,
        r=int(lora_config.get("r", 16)),
        alpha=int(lora_config.get("alpha", 32)),
        dropout=float(lora_config.get("dropout", 0.05)),
        target_modules=lora_config.get("target_modules", None),
        attention_only=bool(lora_config.get("attention_only", False)),
        prepare_for_kbit=quantization is not None,
    )

    frame = pd.read_csv(data_config.get("path"))
    split = split_by_conversation(
        frame,
        train_ratio=float(data_config.get("train_ratio", 0.80)),
        validation_ratio=float(data_config.get("validation_ratio", 0.05)),
        seed=int(config.get("seed", DEFAULT_SEED)),
    )
    LOGGER.info("%s split: %s", spec.key, split.describe())

    build = lambda subset: build_expert_dataset(  # noqa: E731 - three uses, one line
        subset,
        spec,
        tokenizer,
        text_column=data_config.get("text_column", "utterance"),
        label_column=data_config.get("label_column", "label"),
        reason_column=data_config.get("reason_column", "reason"),
        max_length=int(data_config.get("max_length", 1024)),
    )

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_config.get("epochs", 3)),
        per_device_train_batch_size=int(train_config.get("batch_size", 2)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(train_config.get("learning_rate", 2e-4)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.03)),
        eval_strategy=train_config.get("eval_strategy", "steps"),
        eval_steps=int(train_config.get("eval_steps", 500)),
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=int(train_config.get("save_steps", 500)),
        save_total_limit=int(train_config.get("save_total_limit", 2)),
        logging_steps=int(train_config.get("logging_steps", 50)),
        load_best_model_at_end=bool(train_config.get("load_best_model_at_end", True)),
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", False)),
        seed=int(config.get("seed", DEFAULT_SEED)),
        report_to=train_config.get("report_to", ["tensorboard"]),
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=build(split.train),
        eval_dataset=build(split.validation),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    LOGGER.info("%s adapter saved to %s", spec.key, output_dir)

    # Held-out perplexity is the number to compare across expert backbones.
    metrics = trainer.evaluate(eval_dataset=build(split.test))
    loss = float(metrics.get("eval_loss", float("nan")))
    perplexity = math.exp(loss) if loss == loss else float("nan")
    LOGGER.info("%s test loss=%.4f perplexity=%.4f", spec.key, loss, perplexity)

    return {"eval_loss": loss, "perplexity": perplexity}


__all__ = ["build_expert_dataset", "train_expert"]
