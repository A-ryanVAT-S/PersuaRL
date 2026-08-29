"""PersuaRL: GRPO over the expert-selection policy, with a co-adapting Generator.

This is the method of Section 4. One training step:

1. The **Selector** ``pi_theta`` sees ``(history, user utterance)`` and samples
   ``G`` route letters under the single-token constraint. Each letter is a
   binary expert mask ``o_t``.
2. For each rollout, the selected experts' cached answers are packed into the
   Generator prompt ``U(x_t, o_t)`` and the frozen **Generator** ``A_phi``
   produces a response ``y_t``.
3. The composite reward scores ``y_t`` on R1-R5; the routing penalties adjust it
   for complexity, route repetition and expert load imbalance.
4. GRPO computes group-relative advantages over the ``G`` rollouts and updates
   only ``pi_theta``.
5. **Co-adaptation** (optional, on by default): the highest-reward rollout in the
   group becomes one SFT example for the Generator, so ``A_phi`` gradually gets
   better at using the routes the Selector actually picks.

Why the Generator is frozen during the reward computation and updated only on
the winning rollout: if the reward's gradient reached the Generator, the two
models would be optimising the same objective jointly and the Generator could
learn to satisfy the reward models directly (Appendix FAQ 7, reward
circularity). Updating it *only* on the argmax rollout, and only via ordinary
NLL against the ground-truth reply, keeps the supervision grounded in the
reference data while still letting it track the Selector's evolving policy.

Set ``train.freeze_generator: true`` for the D.3.4 ablation.
"""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from ..constants import DEFAULT_SEED, EXPERT_ANSWER_COLUMN, EXPERT_KEYS
from ..data.dataset import DialogueTurn, iter_dialogue_turns, load_dialogues, merge_expert_outputs
from ..data.formatting import (
    build_generator_messages,
    build_selector_prompt,
    tokenize_with_prompt_mask,
)
from ..data.splits import split_by_conversation
from ..models.decoding import patch_generate_with_constraint, route_token_ids
from ..models.loader import (
    attach_lora,
    describe_devices,
    load_tokenizer,
    load_with_adapter,
    quantization_from_config,
)
from ..rewards.composite import PersuasiveRewardModel, build_reward_model
from ..rewards.penalties import PenaltyConfig, RoutingStatistics, compute_penalties, penalty_config_from
from ..routes import Route, route_from_label
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything

LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


def build_selector_dataset(frame, *, include_history: bool = True) -> Dataset:
    """One row per dialogue turn: the Selector prompt plus everything the
    reward function needs.

    TRL passes any extra dataset columns through to the reward function as
    keyword arguments, which is how the expert answers and the reference reply
    reach :class:`SelectorRewardFunction` without global state. This requires
    ``remove_unused_columns=False`` in :class:`GRPOConfig`.
    """
    rows: list[dict[str, str]] = []
    for turn in iter_dialogue_turns(frame, include_history=include_history):
        row = {
            "prompt": build_selector_prompt(turn),
            "conversation_id": turn.conversation_id,
            "turn_no": str(turn.turn_no),
            "user_utterance": turn.user_utterance,
            "dialogue_context": turn.history,
            "previous_agent_reply": turn.previous_agent_reply,
            "reference_reply": turn.agent_reply,
        }
        row.update(
            {EXPERT_ANSWER_COLUMN.format(expert=key): turn.expert_answer(key) for key in EXPERT_KEYS}
        )
        rows.append(row)

    LOGGER.info("selector dataset: %d turns", len(rows))
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------
# Reward function
# --------------------------------------------------------------------------


class SelectorRewardFunction:
    """TRL-compatible reward: route letter -> generated response -> scalar.

    Callable with the signature ``GRPOTrainer`` expects. Kept as a class rather
    than a closure so its state -- the Generator, the optimiser, the routing
    counters, the best-route log -- is inspectable and testable.
    """

    def __init__(
        self,
        reward_model: PersuasiveRewardModel,
        generator,
        generator_tokenizer,
        *,
        penalties: PenaltyConfig,
        statistics: RoutingStatistics,
        generator_optimizer: optim.Optimizer | None = None,
        best_routes_path: Path | None = None,
        csv_lock=None,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        max_prompt_tokens: int = 1024,
        sft_max_length: int = 1536,
        log_every: int = 1,
    ) -> None:
        self.reward_model = reward_model
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.penalties = penalties
        self.statistics = statistics
        self.generator_optimizer = generator_optimizer
        self.best_routes_path = best_routes_path
        self.csv_lock = csv_lock
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_prompt_tokens = max_prompt_tokens
        self.sft_max_length = sft_max_length
        self.log_every = log_every

        self._batches = 0
        self._invalid_routes = 0
        self.generator.eval()

        if self.best_routes_path is not None:
            self._init_route_log()

    # -- generator rollouts ------------------------------------------------

    @torch.no_grad()
    def _generate_responses(self, turns: Sequence[DialogueTurn], routes: Sequence[Route | None]) -> list[str]:
        """Run the frozen Generator once per valid route.

        Invalid routes are skipped entirely rather than being sent through with
        an empty prompt: a wasted forward pass per malformed rollout is real
        cost at ``num_generations=8``.
        """
        prompts: list[str] = []
        positions: list[int] = []

        for index, (turn, route) in enumerate(zip(turns, routes)):
            if route is None:
                continue
            messages = build_generator_messages(turn, route.experts)
            prompts.append(
                self.generator_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            positions.append(index)

        responses = [""] * len(turns)
        if not prompts:
            return responses

        inputs = self.generator_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens,
        ).to(self.generator.device)

        generated = self.generator.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            pad_token_id=self.generator_tokenizer.eos_token_id,
        )
        decoded = self.generator_tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for position, text in zip(positions, decoded):
            responses[position] = text.strip()
        return responses

    # -- generator co-adaptation -------------------------------------------

    def _generator_sft_step(self, turn: DialogueTurn, route: Route) -> float | None:
        """One NLL step on the Generator using the group's best route.

        Returns the loss, or ``None`` if the step was skipped. Skips are normal:
        no reference reply, or a NaN loss from an unlucky truncation.
        """
        if self.generator_optimizer is None or not turn.agent_reply.strip():
            return None

        messages = build_generator_messages(turn, route.experts)
        encoded = tokenize_with_prompt_mask(
            self.generator_tokenizer, messages, turn.agent_reply, max_length=self.sft_max_length
        )

        device = self.generator.device
        batch = {
            key: torch.tensor([value], device=device)
            for key, value in encoded.items()
        }

        self.generator.train()
        try:
            with torch.enable_grad():  # the caller's no_grad context is still active
                self.generator_optimizer.zero_grad(set_to_none=True)
                loss = self.generator(**batch).loss
                if not torch.isfinite(loss):
                    LOGGER.warning("generator SFT loss was not finite; skipping the step")
                    return None
                loss.backward()
                self.generator_optimizer.step()
                return float(loss.item())
        finally:
            # Always restore eval mode -- an exception here would otherwise leave
            # dropout active for every subsequent reward rollout.
            self.generator.eval()

    # -- route logging ------------------------------------------------------

    def _init_route_log(self) -> None:
        self.best_routes_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.best_routes_path.exists():
            header = ",".join(["conversation_id", "turn_no", "reward", *EXPERT_KEYS])
            self.best_routes_path.write_text(header + "\n", encoding="utf-8")

    def _log_best_route(self, turn: DialogueTurn, route: Route, reward: float) -> None:
        """Append the winning route. Used later by ``route_source: logged`` SFT."""
        if self.best_routes_path is None:
            return
        mask = route.as_dict()
        line = ",".join(
            [str(turn.conversation_id), str(turn.turn_no), f"{reward:.6f}",
             *(str(mask[key]) for key in EXPERT_KEYS)]
        )
        if self.csv_lock is not None:
            with self.csv_lock:
                with self.best_routes_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        else:
            with self.best_routes_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    # -- the callable TRL invokes ------------------------------------------

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        conversation_id: list[str],
        turn_no: list[str],
        user_utterance: list[str],
        dialogue_context: list[str],
        previous_agent_reply: list[str],
        reference_reply: list[str],
        **expert_columns,
    ) -> list[float]:
        batch_size = len(completions)
        self._batches += 1

        # Rebuild the turn objects the reward path needs. TRL flattens groups,
        # so the same turn appears num_generations times, once per rollout.
        turns = [
            DialogueTurn(
                conversation_id=conversation_id[i],
                turn_no=int(turn_no[i]),
                user_utterance=user_utterance[i],
                agent_reply=reference_reply[i],
                history=dialogue_context[i],
                previous_agent_reply=previous_agent_reply[i],
                expert_answers={
                    key: expert_columns.get(EXPERT_ANSWER_COLUMN.format(expert=key), [""] * batch_size)[i]
                    for key in EXPERT_KEYS
                },
            )
            for i in range(batch_size)
        ]

        routes = [route_from_label(completion) for completion in completions]
        invalid = sum(route is None for route in routes)
        self._invalid_routes += invalid
        if invalid == batch_size:
            LOGGER.warning("every route in this batch was malformed; returning zero rewards")
            return [0.0] * batch_size

        responses = self._generate_responses(turns, routes)

        breakdown = self.reward_model.score(
            responses,
            user_utterances=[turn.user_utterance for turn in turns],
            contexts=[turn.history for turn in turns],
            previous_replies=[turn.previous_agent_reply for turn in turns],
            references=[turn.agent_reply for turn in turns],
        )

        # Penalties read a single snapshot of the shared counters, then update a
        # local copy as the batch proceeds so rollouts within a batch still see
        # each other's usage.
        route_counts, expert_counts = self.statistics.snapshot()
        final_rewards: list[float] = []

        for index, route in enumerate(routes):
            penalty = compute_penalties(route, route_counts, expert_counts, self.penalties)
            reward = float(np.clip(breakdown.total[index] - penalty.total, 0.0, 1.0))
            final_rewards.append(reward)

            if route is not None:
                route_counts[route.as_json()] += 1
                for expert in route.experts:
                    expert_counts[expert] += 1

            if self._batches % self.log_every == 0:
                LOGGER.info(
                    "  [%d/%d] route=%-3s %s | pen C/R/L=%.3f/%.3f/%.3f bonus=%.3f -> %.4f",
                    index + 1, batch_size,
                    completions[index].strip() or "??",
                    breakdown.format_row(index),
                    penalty.complexity, penalty.repetition, penalty.load_balance,
                    penalty.load_bonus, reward,
                )

        self.statistics.commit([route for route in routes if route is not None])

        # -- co-adaptation: SFT the Generator on the group's best rollout ----
        best = int(np.argmax(final_rewards))
        if routes[best] is not None:
            loss = self._generator_sft_step(turns[best], routes[best])
            if loss is not None:
                LOGGER.info("  generator SFT step: loss=%.4f on route %s (reward %.4f)",
                            loss, routes[best], final_rewards[best])
            self._log_best_route(turns[best], routes[best], final_rewards[best])

        return final_rewards

    def summary(self) -> dict[str, float]:
        """End-of-run diagnostics."""
        stats = {
            "batches": float(self._batches),
            "invalid_routes": float(self._invalid_routes),
            **{f"expert_usage/{k}": float(v) for k, v in self.statistics.usage_summary().items()},
        }
        if hasattr(self.reward_model.judge, "stats"):
            stats.update(self.reward_model.judge.stats())
        return stats


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def train_selector(config) -> dict[str, float]:
    """Train the Selector with GRPO, co-adapting the Generator alongside it."""
    seed = int(config.get("seed", DEFAULT_SEED))
    seed_everything(seed)
    LOGGER.info("devices: %s", describe_devices())

    output_dir = Path(config.get("output_dir"))
    generator_output_dir = Path(config.get("generator_output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_output_dir.mkdir(parents=True, exist_ok=True)

    selector_config = config.section("selector")
    generator_config = config.section("generator")
    data_config = config.section("data")
    train_config = config.section("train")

    # -- Selector tokenizer (the policy itself is constructed by GRPOTrainer) --
    selector_tokenizer = load_tokenizer(
        selector_config.get("id"),
        padding_side="left",  # the Selector generates
        trust_remote_code=bool(selector_config.get("trust_remote_code", False)),
    )
    allowed_tokens = route_token_ids(selector_tokenizer)
    LOGGER.info("constrained action tokens: %s", allowed_tokens)

    # -- Generator: frozen for rewards, LoRA-adapted for co-training ---------
    generator_tokenizer = load_tokenizer(
        generator_config.get("id"),
        padding_side="left",
        trust_remote_code=bool(generator_config.get("trust_remote_code", False)),
    )
    generator = load_with_adapter(
        generator_config.get("id"),
        generator_config.get("adapter_path", None),
        merge=bool(generator_config.get("merge_adapter", True)),
        dtype=generator_config.get("dtype", "auto"),
        device_map=generator_config.get("device_map", "auto"),
        quantization=quantization_from_config(generator_config.section("quantization")),
        trust_remote_code=bool(generator_config.get("trust_remote_code", False)),
    )
    generator.config.pad_token_id = generator_tokenizer.pad_token_id

    freeze_generator = bool(train_config.get("freeze_generator", False))
    generator_optimizer = None
    if freeze_generator:
        LOGGER.info("generator frozen (ablation D.3.4): only the Selector will be updated")
        for parameter in generator.parameters():
            parameter.requires_grad_(False)
    else:
        generator_lora = generator_config.section("lora")
        generator = attach_lora(
            generator,
            r=int(generator_lora.get("r", 16)),
            alpha=int(generator_lora.get("alpha", 32)),
            dropout=float(generator_lora.get("dropout", 0.05)),
            target_modules=generator_lora.get("target_modules", None),
            attention_only=bool(generator_lora.get("attention_only", False)),
        )
        generator_optimizer = optim.AdamW(
            [p for p in generator.parameters() if p.requires_grad],
            lr=float(train_config.get("generator_learning_rate", 1e-5)),
        )
    generator.eval()

    # -- reward stack --------------------------------------------------------
    reward_device = config.get("reward_device", "cuda" if torch.cuda.is_available() else "cpu")
    reward_model = build_reward_model(config.section("rewards"), device=reward_device)
    penalties = penalty_config_from(config.section("rewards.penalties"))

    # Manager-backed counters so TRL dataloader workers share one view of usage.
    manager = multiprocessing.Manager()
    statistics = RoutingStatistics.shared(manager, history_size=penalties.history_size)

    # -- data ----------------------------------------------------------------
    dialogues = load_dialogues(data_config.get("dialogues_path"))
    dialogues = merge_expert_outputs(dialogues, data_config.section("expert_outputs").as_dict())
    split = split_by_conversation(
        dialogues,
        train_ratio=float(data_config.get("train_ratio", 0.85)),
        validation_ratio=float(data_config.get("validation_ratio", 0.0)),
        seed=seed,
    )
    LOGGER.info("split: %s", split.describe())
    train_dataset = build_selector_dataset(split.train)

    reward_function = SelectorRewardFunction(
        reward_model,
        generator,
        generator_tokenizer,
        penalties=penalties,
        statistics=statistics,
        generator_optimizer=generator_optimizer,
        best_routes_path=output_dir / "best_routes.csv",
        csv_lock=manager.Lock(),
        max_new_tokens=int(generator_config.get("max_new_tokens", 128)),
        temperature=float(generator_config.get("temperature", 0.8)),
        top_p=float(generator_config.get("top_p", 0.95)),
        top_k=int(generator_config.get("top_k", 40)),
        max_prompt_tokens=int(generator_config.get("max_prompt_tokens", 1024)),
        log_every=int(train_config.get("log_every", 1)),
    )

    # -- GRPO ----------------------------------------------------------------
    grpo_arguments = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_config.get("batch_size", 2)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(train_config.get("learning_rate", 2e-5)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.03)),
        num_train_epochs=float(train_config.get("epochs", 1)),
        num_generations=int(train_config.get("num_generations", 8)),
        max_prompt_length=int(train_config.get("max_prompt_length", 512)),
        # The action is a single letter; 4 tokens leaves room for the EOS and
        # any leading-space token the backbone insists on emitting.
        max_completion_length=int(train_config.get("max_completion_length", 4)),
        temperature=float(train_config.get("temperature", 1.2)),
        top_p=float(train_config.get("top_p", 1.0)),
        beta=float(train_config.get("kl_beta", 0.04)),
        epsilon=float(train_config.get("clip_epsilon", 0.2)),
        loss_type=train_config.get("loss_type", "bnpo"),
        save_steps=int(train_config.get("save_steps", 200)),
        logging_steps=int(train_config.get("logging_steps", 1)),
        report_to=train_config.get("report_to", ["tensorboard"]),
        # Required: the reward function reads the expert-answer columns.
        remove_unused_columns=False,
        seed=seed,
        model_init_kwargs={
            "device_map": selector_config.get("device_map", "auto"),
            "dtype": selector_config.get("dtype", "auto"),
            "trust_remote_code": bool(selector_config.get("trust_remote_code", False)),
        },
    )

    selector_lora = selector_config.section("lora")
    from peft import LoraConfig

    peft_config = LoraConfig(
        r=int(selector_lora.get("r", 16)),
        lora_alpha=int(selector_lora.get("alpha", 32)),
        lora_dropout=float(selector_lora.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=selector_lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    trainer = GRPOTrainer(
        model=selector_config.get("id"),
        reward_funcs=reward_function,
        train_dataset=train_dataset,
        args=grpo_arguments,
        peft_config=peft_config,
        processing_class=selector_tokenizer,
    )

    # Constrain the policy's rollouts to legal route tokens. This has to happen
    # after the trainer builds the model, and it wraps the bound method because
    # TRL owns the generate call.
    patch_generate_with_constraint(trainer.model, selector_tokenizer, allowed_tokens)

    LOGGER.info("starting GRPO: %d turns, %d rollouts each",
                len(train_dataset), grpo_arguments.num_generations)
    trainer.train()

    trainer.save_model(str(output_dir))
    selector_tokenizer.save_pretrained(str(output_dir))
    LOGGER.info("selector saved to %s", output_dir)

    if not freeze_generator:
        generator.save_pretrained(str(generator_output_dir))
        generator_tokenizer.save_pretrained(str(generator_output_dir))
        LOGGER.info("co-adapted generator saved to %s", generator_output_dir)

    summary = reward_function.summary()
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("run summary: %s", summary)
    return summary


__all__ = ["SelectorRewardFunction", "build_selector_dataset", "train_selector"]
