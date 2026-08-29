"""Single-model GRPO baselines (Table 13, D.3.3).

No Selector, no experts: one policy generates the agent reply directly and is
optimised against the *same* composite reward PersuaRL uses. Two variants,
distinguished only by ``model.adapter_path``:

``adapter_path: null``   -- **GRPO from the instruct checkpoint.** Isolates what
                            the reward buys you without any domain SFT.
``adapter_path: <sft>``  -- **Warm-start GRPO.** The SFT adapter is merged into
                            the base weights first, then a fresh LoRA is trained
                            with GRPO on top. This is the ``Single -> SFT ->
                            RL`` progression of Table 2.

Merging (rather than stacking a second adapter on a live one) matters: with two
active adapters the RL gradient has to work against the SFT adapter's output as
well as the base model's, and the effective learning rate on the composition is
not the one you configured.

The routing penalties do not apply here -- there is no route to penalise -- so
this trains against the raw ``sum_k beta_k R_k``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from ..constants import DEFAULT_SEED
from ..data.dataset import iter_dialogue_turns, load_dialogues
from ..data.formatting import build_baseline_messages
from ..data.splits import split_by_conversation
from ..models.loader import (
    attach_lora,
    describe_devices,
    load_causal_lm,
    load_tokenizer,
    load_with_adapter,
    quantization_from_config,
)
from ..rewards.composite import PersuasiveRewardModel, build_reward_model
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything

LOGGER = get_logger(__name__)


def build_single_model_dataset(frame, tokenizer, *, include_history: bool = True) -> Dataset:
    """One row per turn, with the prompt already chat-templated.

    Unlike the Selector dataset, the prompt goes through the chat template here:
    the policy is producing free-form dialogue, and skipping the template is how
    you get a model that never emits EOS and runs to ``max_completion_length``
    every time.
    """
    rows = []
    for turn in iter_dialogue_turns(frame, experts=(), include_history=include_history):
        messages = build_baseline_messages(turn, include_history=include_history)
        rows.append(
            {
                "prompt": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ),
                "conversation_id": turn.conversation_id,
                "turn_no": str(turn.turn_no),
                "user_utterance": turn.user_utterance,
                "dialogue_context": turn.history,
                "previous_agent_reply": turn.previous_agent_reply,
                "reference_reply": turn.agent_reply,
            }
        )
    LOGGER.info("single-model dataset: %d turns", len(rows))
    return Dataset.from_list(rows)


class SingleModelRewardFunction:
    """Scores the policy's own completions with the shared reward stack."""

    def __init__(self, reward_model: PersuasiveRewardModel, *, log_every: int = 1) -> None:
        self.reward_model = reward_model
        self.log_every = log_every
        self._batches = 0

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        user_utterance: list[str],
        dialogue_context: list[str],
        previous_agent_reply: list[str],
        reference_reply: list[str],
        **kwargs,
    ) -> list[float]:
        self._batches += 1
        responses = [str(completion).strip() for completion in completions]

        breakdown = self.reward_model.score(
            responses,
            user_utterances=user_utterance,
            contexts=dialogue_context,
            previous_replies=previous_agent_reply,
            references=reference_reply,
        )

        if self._batches % self.log_every == 0:
            for index in range(len(responses)):
                LOGGER.info("  [%d/%d] %s | %s",
                            index + 1, len(responses),
                            breakdown.format_row(index),
                            responses[index][:110].replace("\n", " "))

        return [float(value) for value in np.clip(breakdown.total, 0.0, 1.0)]


def train_single_model_grpo(config) -> dict[str, float]:
    """Train a single policy with GRPO, optionally warm-started from SFT."""
    seed = int(config.get("seed", DEFAULT_SEED))
    seed_everything(seed)
    LOGGER.info("devices: %s", describe_devices())

    model_config = config.section("model")
    data_config = config.section("data")
    train_config = config.section("train")

    output_dir = Path(config.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(
        model_config.get("id"),
        padding_side="left",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )

    quantization = quantization_from_config(model_config.section("quantization"))
    adapter_path = model_config.get("adapter_path", None)

    if adapter_path:
        LOGGER.info("warm start: merging SFT adapter %s into the base weights", adapter_path)
        model = load_with_adapter(
            model_config.get("id"),
            adapter_path,
            merge=True,
            dtype=model_config.get("dtype", "float16"),
            device_map=model_config.get("device_map", "auto"),
            quantization=quantization,
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )
    else:
        LOGGER.info("cold start: GRPO directly on the instruct checkpoint")
        model = load_causal_lm(
            model_config.get("id"),
            dtype=model_config.get("dtype", "float16"),
            device_map=model_config.get("device_map", "auto"),
            quantization=quantization,
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )

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

    reward_device = config.get("reward_device", "cuda" if torch.cuda.is_available() else "cpu")
    reward_model = build_reward_model(config.section("rewards"), device=reward_device)
    reward_function = SingleModelRewardFunction(
        reward_model, log_every=int(train_config.get("log_every", 1))
    )

    dialogues = load_dialogues(data_config.get("dialogues_path"))
    split = split_by_conversation(
        dialogues,
        train_ratio=float(data_config.get("train_ratio", 0.85)),
        validation_ratio=float(data_config.get("validation_ratio", 0.0)),
        seed=seed,
    )
    LOGGER.info("split: %s", split.describe())
    train_dataset = build_single_model_dataset(
        split.train, tokenizer, include_history=bool(data_config.get("include_history", True))
    )

    arguments = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_config.get("batch_size", 8)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 2)),
        learning_rate=float(train_config.get("learning_rate", 2e-5)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.03)),
        num_train_epochs=float(train_config.get("epochs", 1)),
        num_generations=int(train_config.get("num_generations", 4)),
        max_prompt_length=int(train_config.get("max_prompt_length", 1024)),
        max_completion_length=int(train_config.get("max_completion_length", 256)),
        temperature=float(train_config.get("temperature", 0.9)),
        top_p=float(train_config.get("top_p", 1.0)),
        beta=float(train_config.get("kl_beta", 0.04)),
        epsilon=float(train_config.get("clip_epsilon", 0.2)),
        loss_type=train_config.get("loss_type", "bnpo"),
        save_steps=int(train_config.get("save_steps", 100)),
        logging_steps=int(train_config.get("logging_steps", 1)),
        report_to=train_config.get("report_to", ["tensorboard"]),
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        train_dataset=train_dataset,
        args=arguments,
        processing_class=tokenizer,
    )

    LOGGER.info("starting single-model GRPO on %d turns", len(train_dataset))
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    LOGGER.info("policy saved to %s", output_dir)

    summary = {"batches": float(reward_function._batches)}
    if hasattr(reward_model.judge, "stats"):
        summary.update(reward_model.judge.stats())
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "SingleModelRewardFunction",
    "build_single_model_dataset",
    "train_single_model_grpo",
]
