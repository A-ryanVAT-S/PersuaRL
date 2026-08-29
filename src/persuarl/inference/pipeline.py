"""End-to-end inference: Selector -> Experts -> Generator.

Deployment-time behaviour, which differs from training in one important way:
**history is built from the model's own replies, not the reference transcript.**
Errors therefore compound across a dialogue, which is exactly what you want to
measure -- a system that only looks good under teacher forcing is not a system.

Three routing modes, matching the ablations in D.3.2:

``persuarl``   the trained Selector picks the route (the method)
``all``        every expert every turn (AllExpert baseline)
``prompting``  the Selector backbone is *prompted* to choose, with no RL
               (Prompting Tools baseline)

Expert answers come either from the shipped cached CSVs (``--expert-source
cached``, fast, what the paper's tables use) or from live expert LMs
(``--expert-source live``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from ..constants import COL_CONVERSATION_ID, COL_TURN_NO, EXPERT_KEYS
from ..data.dataset import DialogueTurn, load_dialogues, merge_expert_outputs
from ..data.formatting import build_generator_messages, build_selector_prompt
from ..data.splits import split_by_conversation
from ..experts.inference import CachedExpertPool, ExpertRunner
from ..models.decoding import build_selector_logits_processor, route_token_ids
from ..models.loader import load_tokenizer, load_with_adapter
from ..routes import ROUTES, Route, route_from_label, route_from_mask
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TurnResult:
    """One row of the inference output CSV."""

    conversation_id: str
    turn_no: int
    user_utterance: str
    reference_reply: str
    selected_route: str
    selected_experts: str
    model_reply: str
    expert_outputs: dict[str, str] = field(default_factory=dict)

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            key: value for key, value in asdict(self).items() if key != "expert_outputs"
        }
        row.update({f"expert_{key}": value for key, value in self.expert_outputs.items()})
        return row


class PersuaRLPipeline:
    """Selector + experts + Generator, wired together for inference."""

    def __init__(
        self,
        *,
        selector_model=None,
        selector_tokenizer=None,
        generator_model,
        generator_tokenizer,
        expert_pool: CachedExpertPool | None = None,
        expert_runners: dict[str, ExpertRunner] | None = None,
        routing_mode: str = "persuarl",
        selector_temperature: float = 0.8,
        generator_max_new_tokens: int = 512,
        generator_temperature: float = 0.8,
        generator_top_p: float = 0.95,
        generator_top_k: int = 40,
    ) -> None:
        if routing_mode not in {"persuarl", "all", "prompting"}:
            raise ValueError(f"unknown routing_mode {routing_mode!r}")
        if routing_mode in {"persuarl", "prompting"} and selector_model is None:
            raise ValueError(f"routing_mode={routing_mode!r} requires a selector model")
        if expert_pool is None and not expert_runners:
            raise ValueError("provide either a cached expert pool or live expert runners")

        self.selector_model = selector_model
        self.selector_tokenizer = selector_tokenizer
        self.generator_model = generator_model
        self.generator_tokenizer = generator_tokenizer
        self.expert_pool = expert_pool
        self.expert_runners = expert_runners or {}
        self.routing_mode = routing_mode
        self.selector_temperature = selector_temperature
        self.generator_max_new_tokens = generator_max_new_tokens
        self.generator_temperature = generator_temperature
        self.generator_top_p = generator_top_p
        self.generator_top_k = generator_top_k

        self._allowed_tokens = (
            route_token_ids(selector_tokenizer) if selector_model is not None else []
        )
        self._route_counts: dict[str, int] = {}

        if selector_model is not None:
            selector_model.eval()
        generator_model.eval()

    # -- step 1: routing ----------------------------------------------------

    @torch.no_grad()
    def select_route(self, turn: DialogueTurn) -> Route:
        """Choose the expert subset for this turn."""
        if self.routing_mode == "all":
            # route_from_mask over an all-ones mask is the last route ("O").
            return route_from_mask({key: 1 for key in EXPERT_KEYS}) or ROUTES[-1]

        prompt = build_selector_prompt(turn)
        inputs = self.selector_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.selector_model.device)

        # 'prompting' mode deliberately skips the constrained decoder: the point
        # of that baseline is that an unconstrained prompted model has to be
        # *asked* to produce a valid route, and sometimes will not.
        logits_processor = (
            build_selector_logits_processor(self.selector_tokenizer, self._allowed_tokens)
            if self.routing_mode == "persuarl" else None
        )

        generated = self.selector_model.generate(
            **inputs,
            max_new_tokens=4 if self.routing_mode == "persuarl" else 16,
            do_sample=True,
            temperature=self.selector_temperature,
            logits_processor=logits_processor,
            pad_token_id=self.selector_tokenizer.eos_token_id,
        )
        decoded = self.selector_tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        route = route_from_label(decoded)
        if route is None:
            # Falling back to all experts (rather than none) keeps a malformed
            # route from silently producing an unconditioned reply, which would
            # flatter the baseline it is meant to measure.
            LOGGER.debug("unparseable route %r; falling back to all experts", decoded.strip())
            route = route_from_mask({key: 1 for key in EXPERT_KEYS}) or ROUTES[-1]

        self._route_counts[route.label] = self._route_counts.get(route.label, 0) + 1
        return route

    # -- step 2: expert outputs ---------------------------------------------

    def gather_expert_outputs(self, turn: DialogueTurn, route: Route) -> dict[str, str]:
        """Fetch the selected experts' answers, cached or live."""
        outputs: dict[str, str] = {}
        for expert in route.experts:
            runner = self.expert_runners.get(expert)
            if runner is not None:
                outputs[expert] = runner.annotate([turn], batch_size=1)[0]
            elif self.expert_pool is not None:
                outputs[expert] = self.expert_pool.lookup(expert, turn.conversation_id, turn.turn_no)
            else:
                outputs[expert] = ""
        return outputs

    # -- step 3: generation --------------------------------------------------

    @torch.no_grad()
    def generate_reply(self, turn: DialogueTurn, route: Route) -> str:
        messages = build_generator_messages(turn, route.experts)
        prompt = self.generator_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.generator_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.generator_model.device)

        generated = self.generator_model.generate(
            **inputs,
            max_new_tokens=self.generator_max_new_tokens,
            do_sample=True,
            temperature=self.generator_temperature,
            top_p=self.generator_top_p,
            top_k=self.generator_top_k,
            pad_token_id=self.generator_tokenizer.eos_token_id,
        )
        return self.generator_tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    # -- full conversation ---------------------------------------------------

    def run_conversation(self, turns: Sequence[dict], *, verbose: bool = False) -> list[TurnResult]:
        """Replay one conversation, feeding generated replies back as history."""
        history_parts: list[str] = []
        previous_reply = ""
        results: list[TurnResult] = []

        for row in turns:
            turn = DialogueTurn(
                conversation_id=str(row[COL_CONVERSATION_ID]),
                turn_no=int(row[COL_TURN_NO]),
                user_utterance=str(row["user_utterance"]),
                agent_reply=str(row.get("new_agent_reply", "")),
                history="\n".join(history_parts),
                previous_agent_reply=previous_reply,
            )

            route = self.select_route(turn)
            turn.expert_answers = self.gather_expert_outputs(turn, route)
            reply = self.generate_reply(turn, route)

            results.append(
                TurnResult(
                    conversation_id=turn.conversation_id,
                    turn_no=turn.turn_no,
                    user_utterance=turn.user_utterance,
                    reference_reply=turn.agent_reply,
                    selected_route=route.label,
                    selected_experts="+".join(route.experts),
                    model_reply=reply,
                    expert_outputs=dict(turn.expert_answers),
                )
            )

            if verbose:
                LOGGER.info("conv %s turn %s | route %s", turn.conversation_id, turn.turn_no, route)
                LOGGER.info("  user : %s", turn.user_utterance)
                LOGGER.info("  agent: %s", reply)

            # The generated reply, not the reference, becomes the next context.
            history_parts.append(f"User: {turn.user_utterance}")
            history_parts.append(f"Agent: {reply}")
            previous_reply = reply

        return results

    def route_distribution(self) -> dict[str, int]:
        """How often each route fired -- the diagnostic for a collapsed policy."""
        return dict(sorted(self._route_counts.items()))


# --------------------------------------------------------------------------
# Config-driven construction
# --------------------------------------------------------------------------


def build_pipeline(config) -> PersuaRLPipeline:
    """Instantiate a pipeline from an inference config."""
    selector_config = config.section("selector")
    generator_config = config.section("generator")
    routing_mode = config.get("routing_mode", "persuarl")

    selector_model = selector_tokenizer = None
    if routing_mode != "all":
        selector_tokenizer = load_tokenizer(
            selector_config.get("id"), padding_side="left",
            trust_remote_code=bool(selector_config.get("trust_remote_code", False)),
        )
        selector_model = load_with_adapter(
            selector_config.get("id"),
            selector_config.get("adapter_path", None),
            merge=True,
            dtype=selector_config.get("dtype", "bfloat16"),
            device_map=selector_config.get("device_map", "auto"),
            trust_remote_code=bool(selector_config.get("trust_remote_code", False)),
        )

    generator_tokenizer = load_tokenizer(
        generator_config.get("id"), padding_side="left",
        trust_remote_code=bool(generator_config.get("trust_remote_code", False)),
    )
    generator_model = load_with_adapter(
        generator_config.get("id"),
        generator_config.get("adapter_path", None),
        merge=False,
        dtype=generator_config.get("dtype", "bfloat16"),
        device_map=generator_config.get("device_map", "auto"),
        trust_remote_code=bool(generator_config.get("trust_remote_code", False)),
    )

    expert_source = config.get("expert_source", "cached")
    expert_pool = None
    expert_runners: dict[str, ExpertRunner] = {}

    if expert_source == "cached":
        expert_pool = CachedExpertPool.from_paths(config.section("data.expert_outputs").as_dict())
    elif expert_source == "live":
        experts_config = config.section("experts")
        for key in EXPERT_KEYS:
            spec = experts_config.section(key)
            expert_runners[key] = ExpertRunner.from_pretrained(
                key,
                spec.get("id", experts_config.get("id")),
                spec.get("adapter_path", None),
                dtype=spec.get("dtype", "bfloat16"),
                device_map=spec.get("device_map", "auto"),
            )
    else:
        raise ValueError(f"unknown expert_source {expert_source!r}; expected cached|live")

    return PersuaRLPipeline(
        selector_model=selector_model,
        selector_tokenizer=selector_tokenizer,
        generator_model=generator_model,
        generator_tokenizer=generator_tokenizer,
        expert_pool=expert_pool,
        expert_runners=expert_runners,
        routing_mode=routing_mode,
        selector_temperature=float(selector_config.get("temperature", 0.8)),
        generator_max_new_tokens=int(generator_config.get("max_new_tokens", 512)),
        generator_temperature=float(generator_config.get("temperature", 0.8)),
        generator_top_p=float(generator_config.get("top_p", 0.95)),
        generator_top_k=int(generator_config.get("top_k", 40)),
    )


def run_inference(config) -> pd.DataFrame:
    """Run the pipeline over the held-out test split and write a results CSV."""
    data_config = config.section("data")
    output_path = Path(config.get("output_path"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dialogues = load_dialogues(data_config.get("dialogues_path"))
    if config.get("expert_source", "cached") == "cached":
        # Inner-join first so a turn is never routed to an expert that has no
        # cached answer for it.
        dialogues = merge_expert_outputs(dialogues, data_config.section("expert_outputs").as_dict())

    split = split_by_conversation(
        dialogues,
        train_ratio=float(data_config.get("train_ratio", 0.85)),
        validation_ratio=float(data_config.get("validation_ratio", 0.0)),
        seed=int(config.get("seed", 42)),
    )
    test_frame = split.test
    LOGGER.info("evaluating on %d turns / %d conversations",
                len(test_frame), test_frame[COL_CONVERSATION_ID].nunique())

    limit = config.get("max_conversations", None)
    pipeline = build_pipeline(config)

    grouped = list(test_frame.groupby(COL_CONVERSATION_ID, sort=True))
    if limit:
        grouped = grouped[: int(limit)]

    rows: list[dict[str, object]] = []
    for _, group in tqdm(grouped, desc="conversations"):
        turns = group.sort_values(COL_TURN_NO).to_dict("records")
        for result in pipeline.run_conversation(turns, verbose=bool(config.get("verbose", False))):
            rows.append(result.to_row())

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    LOGGER.info("wrote %d turns -> %s", len(frame), output_path)

    distribution = pipeline.route_distribution()
    (output_path.parent / f"{output_path.stem}_routes.json").write_text(
        json.dumps(distribution, indent=2), encoding="utf-8"
    )
    LOGGER.info("route distribution: %s", distribution)
    return frame


__all__ = ["PersuaRLPipeline", "TurnResult", "build_pipeline", "run_inference"]
