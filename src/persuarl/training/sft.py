"""Supervised fine-tuning for the Generator and for the SFT baselines.

Two prompt styles, one trainer, selected by ``data.style`` in the config:

``generator``
    Conditions on ``(history, user utterance, <expert>...</expert> analysis)``.
    This produces the ``A_phi`` that PersuaRL's Selector routes into. Which
    experts appear in the analysis block is controlled by ``data.route_source``:

    ``all``     every expert on every turn -- the AllExpert ablation (D.3.6),
                and the right warm start before selector RL.
    ``logged``  replay the best routes GRPO discovered, from
                ``best_routes.csv``. Use this to rebuild a Generator offline
                that matches an RL run, without re-running RL.
    ``random``  sample a route per turn; a cheap way to make the Generator
                robust to *any* expert subset before the Selector starts
                exploring.

``baseline``
    Plain ``(history, user utterance) -> reply``. This is the "SFT" column of
    Table 2 and the warm start for the single-model GRPO baselines.

Any causal LM works: the backbone, quantisation and LoRA targets all come from
the config, and target modules are auto-detected when unset.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from ..constants import DEFAULT_SEED, EXPERT_KEYS
from ..data.dataset import DialogueTurn, iter_dialogue_turns, load_dialogues, merge_expert_outputs
from ..data.formatting import (
    build_baseline_messages,
    build_generator_messages,
    has_trainable_tokens,
    tokenize_with_prompt_mask,
)
from ..data.splits import split_by_conversation
from ..models.loader import (
    attach_lora,
    describe_devices,
    load_causal_lm,
    load_tokenizer,
    load_with_adapter,
    quantization_from_config,
)
from ..routes import ROUTES, Route, route_from_mask
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything

LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------
# Route selection for generator-style SFT
# --------------------------------------------------------------------------


def _load_logged_routes(path: str | Path) -> dict[tuple[str, int], Route]:
    """Read ``best_routes.csv`` (written during selector RL) into a lookup."""
    frame = pd.read_csv(path)
    lookup: dict[tuple[str, int], Route] = {}
    for _, row in frame.iterrows():
        route = route_from_mask({key: int(row.get(key, 0)) for key in EXPERT_KEYS})
        if route is not None:
            lookup[(str(row["conversation_id"]), int(row["turn_no"]))] = route
    LOGGER.info("loaded %d logged routes from %s", len(lookup), path)
    return lookup


def _route_for_turn(
    turn: DialogueTurn,
    style: str,
    logged: dict[tuple[str, int], Route] | None,
    rng: random.Random,
) -> Sequence[str]:
    """Decide which experts appear in this training example's analysis block."""
    if style == "all":
        return EXPERT_KEYS
    if style == "random":
        return rng.choice(ROUTES).experts
    if style == "logged":
        if logged is None:
            raise ValueError("route_source='logged' requires data.routes_path")
        route = logged.get((str(turn.conversation_id), int(turn.turn_no)))
        # Turns the RL run never reached fall back to all experts rather than
        # being dropped -- dropping them would bias the Generator toward
        # whatever subset of the corpus GRPO happened to visit.
        return route.experts if route else EXPERT_KEYS
    raise ValueError(f"unknown route_source {style!r}; expected all|random|logged")


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------


def build_sft_dataset(
    frame: pd.DataFrame,
    tokenizer,
    *,
    style: str = "generator",
    route_source: str = "all",
    logged_routes: dict[tuple[str, int], Route] | None = None,
    max_length: int = 1536,
    include_history: bool = True,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    """Build prompt-masked SFT examples, one per agent turn."""
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    skipped_empty = 0
    skipped_truncated = 0

    for turn in iter_dialogue_turns(frame, include_history=include_history):
        if not turn.agent_reply.strip():
            skipped_empty += 1
            continue

        if style == "generator":
            experts = _route_for_turn(turn, route_source, logged_routes, rng)
            messages = build_generator_messages(turn, experts)
        elif style == "baseline":
            messages = build_baseline_messages(turn, include_history=include_history)
        else:
            raise ValueError(f"unknown SFT style {style!r}; expected generator|baseline")

        encoded = tokenize_with_prompt_mask(tokenizer, messages, turn.agent_reply, max_length=max_length)
        if has_trainable_tokens(encoded):
            examples.append(encoded)
        else:
            skipped_truncated += 1

    if skipped_empty:
        LOGGER.info("skipped %d turns with an empty reference reply", skipped_empty)
    if skipped_truncated:
        LOGGER.warning(
            "skipped %d examples whose completion was truncated away "
            "(increase data.max_length)", skipped_truncated,
        )
    LOGGER.info("built %d %s-style SFT examples", len(examples), style)
    return Dataset.from_list(examples)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def train_sft(config) -> dict[str, float]:
    """Run supervised fine-tuning as described by ``config``."""
    seed = int(config.get("seed", DEFAULT_SEED))
    seed_everything(seed)
    LOGGER.info("devices: %s", describe_devices())

    model_config = config.section("model")
    data_config = config.section("data")
    train_config = config.section("train")
    style = data_config.get("style", "generator")

    output_dir = Path(config.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(
        model_config.get("id"),
        padding_side="right",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )

    quantization = quantization_from_config(model_config.section("quantization"))
    warm_start = model_config.get("adapter_path", None)
    if warm_start:
        # Continuing from an existing adapter: merge it so the new LoRA starts
        # from the warm-started weights rather than stacking on a live adapter.
        LOGGER.info("warm-starting from adapter %s", warm_start)
        model = load_with_adapter(
            model_config.get("id"),
            warm_start,
            merge=True,
            dtype=model_config.get("dtype", "auto"),
            device_map=model_config.get("device_map", "auto"),
            quantization=quantization,
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )
    else:
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

    # -- data ---------------------------------------------------------------
    dialogues = load_dialogues(data_config.get("dialogues_path"))
    if style == "generator":
        expert_paths = data_config.section("expert_outputs").as_dict()
        if not expert_paths:
            raise ValueError("generator-style SFT needs data.expert_outputs paths")
        dialogues = merge_expert_outputs(dialogues, expert_paths)

    split = split_by_conversation(
        dialogues,
        train_ratio=float(data_config.get("train_ratio", 0.80)),
        validation_ratio=float(data_config.get("validation_ratio", 0.05)),
        seed=seed,
    )
    LOGGER.info("split: %s", split.describe())

    routes_path = data_config.get("routes_path", None)
    logged_routes = _load_logged_routes(routes_path) if routes_path else None

    build = lambda subset: build_sft_dataset(  # noqa: E731
        subset,
        tokenizer,
        style=style,
        route_source=data_config.get("route_source", "all"),
        logged_routes=logged_routes,
        max_length=int(data_config.get("max_length", 1536)),
        include_history=bool(data_config.get("include_history", True)),
        seed=seed,
    )

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_config.get("epochs", 1)),
        per_device_train_batch_size=int(train_config.get("batch_size", 2)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(train_config.get("learning_rate", 2e-4)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "linear"),
        eval_strategy=train_config.get("eval_strategy", "steps"),
        eval_steps=int(train_config.get("eval_steps", 500)),
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=int(train_config.get("save_steps", 500)),
        save_total_limit=int(train_config.get("save_total_limit", 2)),
        logging_steps=int(train_config.get("logging_steps", 50)),
        load_best_model_at_end=bool(train_config.get("load_best_model_at_end", True)),
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", False)),
        bf16=bool(train_config.get("bf16", False)),
        fp16=bool(train_config.get("fp16", False)),
        seed=seed,
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
    LOGGER.info("adapter saved to %s", output_dir)

    metrics = trainer.evaluate(eval_dataset=build(split.test))
    loss = float(metrics.get("eval_loss", float("nan")))
    perplexity = math.exp(loss) if loss == loss else float("nan")
    LOGGER.info("held-out loss=%.4f perplexity=%.4f", loss, perplexity)
    return {"eval_loss": loss, "perplexity": perplexity}


__all__ = ["build_sft_dataset", "train_sft"]
