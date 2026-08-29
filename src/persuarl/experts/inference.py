"""Running a trained expert over a dialogue file.

Two ways to get expert outputs into the pipeline:

**Cached** (what the paper's experiments use, and what ships in
``data/insuredial/expert_outputs/``) -- run each expert once over the whole
corpus, write a CSV, and have training and inference look answers up by
``(conversation_id, turn_no)``. Expert outputs do not depend on the Selector or
the Generator, so recomputing them inside the RL loop would burn GPU hours on
identical results.

**Live** -- load the expert adapters and call them per turn. Slower, but it is
the honest setting for a deployment estimate and the only option on a corpus
you have not pre-annotated. ``persuarl.cli.run_pipeline --expert-source live``
takes this path.

:class:`ExpertRunner` implements both by exposing the same ``annotate`` call.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from ..constants import COL_CONVERSATION_ID, COL_TURN_NO, EXPERT_ANSWER_COLUMN
from ..data.dataset import DialogueTurn, iter_dialogue_turns
from ..models.loader import load_tokenizer, load_with_adapter
from ..utils.logging import get_logger
from .registry import ExpertSpec, get_expert

LOGGER = get_logger(__name__)


@dataclass
class GenerationSettings:
    """Decoding parameters for expert inference.

    Defaults follow Appendix D.2 (``temperature=0.8``, ``top_p=0.95``,
    ``top_k=40``) but with a short ``max_new_tokens``: an expert answer is a
    label plus one or two lines, and letting it run to 512 tokens just invites
    the model to start writing dialogue.
    """

    max_new_tokens: int = 96
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    do_sample: bool = True


class ExpertRunner:
    """A loaded expert LM that annotates dialogue turns."""

    def __init__(
        self,
        spec: ExpertSpec,
        model,
        tokenizer,
        settings: GenerationSettings | None = None,
    ) -> None:
        self.spec = spec
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings or GenerationSettings()
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        expert_key: str,
        base_model_id: str,
        adapter_path: str | Path | None,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        settings: GenerationSettings | None = None,
        trust_remote_code: bool = False,
    ) -> ExpertRunner:
        spec = get_expert(expert_key)
        LOGGER.info("loading %s from %s (+%s)", spec.display_name, base_model_id, adapter_path)
        tokenizer = load_tokenizer(base_model_id, padding_side="left", trust_remote_code=trust_remote_code)
        model = load_with_adapter(
            base_model_id,
            str(adapter_path) if adapter_path else None,
            merge=False,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        return cls(spec, model, tokenizer, settings)

    def _prompt(self, turn: DialogueTurn) -> str:
        messages = [
            {"role": "system", "content": self.spec.system_prompt},
            {"role": "user", "content": turn.user_utterance},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def annotate(self, turns: Sequence[DialogueTurn], *, batch_size: int = 8) -> list[str]:
        """Generate one answer per turn, in input order."""
        answers: list[str] = []

        for start in range(0, len(turns), batch_size):
            batch = turns[start:start + batch_size]
            inputs = self.tokenizer(
                [self._prompt(turn) for turn in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.model.device)

            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                top_k=self.settings.top_k,
                do_sample=self.settings.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            answers.extend(
                text.strip()
                for text in self.tokenizer.batch_decode(
                    generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
            )

        return answers


def annotate_corpus(
    runner: ExpertRunner,
    frame: pd.DataFrame,
    *,
    batch_size: int = 8,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Annotate every turn in ``frame`` and return a merge-ready expert CSV.

    Output columns: ``conversation_id``, ``turn_no``, ``utterance``,
    ``<expert>_answer`` -- exactly the schema
    :func:`persuarl.data.dataset.load_expert_outputs` expects.
    """
    turns = list(iter_dialogue_turns(frame, experts=()))
    answer_column = EXPERT_ANSWER_COLUMN.format(expert=runner.spec.key)

    answers: list[str] = []
    progress = tqdm(range(0, len(turns), batch_size), desc=f"{runner.spec.key} annotation")
    for start in progress:
        answers.extend(runner.annotate(turns[start:start + batch_size], batch_size=batch_size))

    result = pd.DataFrame(
        {
            COL_CONVERSATION_ID: [turn.conversation_id for turn in turns],
            COL_TURN_NO: [turn.turn_no for turn in turns],
            "utterance": [turn.user_utterance for turn in turns],
            answer_column: answers,
        }
    )

    unparsed = sum(1 for answer in answers if runner.spec.parse_answer(answer) is None)
    if unparsed:
        LOGGER.warning(
            "%s: %d/%d answers did not match the expected output format "
            "(the expert may be under-trained, or max_new_tokens too small)",
            runner.spec.key, unparsed, len(answers),
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        LOGGER.info("wrote %d %s annotations -> %s", len(result), runner.spec.key, output_path)

    return result


class CachedExpertPool:
    """Serves precomputed expert answers keyed by ``(conversation_id, turn_no)``.

    Drop-in alternative to holding four expert LMs in memory. Missing keys
    return an empty string, and the miss count is logged once at teardown --
    silent misses would show up much later as a mysteriously weak reward.
    """

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self._lookup: dict[str, dict[tuple[int, int], str]] = {}
        self._misses = 0

        for expert, frame in tables.items():
            column = EXPERT_ANSWER_COLUMN.format(expert=expert)
            self._lookup[expert] = {
                (int(row[COL_CONVERSATION_ID]), int(row[COL_TURN_NO])): str(row[column])
                for _, row in frame.iterrows()
            }
            LOGGER.info("cached %d %s answers", len(self._lookup[expert]), expert)

    @classmethod
    def from_paths(cls, paths: dict[str, str | Path]) -> CachedExpertPool:
        from ..data.dataset import load_expert_outputs

        return cls({expert: load_expert_outputs(expert, path) for expert, path in paths.items()})

    def lookup(self, expert: str, conversation_id: int | str, turn_no: int) -> str:
        table = self._lookup.get(expert)
        if table is None:
            return ""
        answer = table.get((int(conversation_id), int(turn_no)))
        if answer is None:
            self._misses += 1
            return ""
        return answer

    def annotate_turn(self, turn: DialogueTurn, experts: Iterable[str]) -> dict[str, str]:
        return {
            expert: self.lookup(expert, turn.conversation_id, turn.turn_no)
            for expert in experts
        }

    @property
    def misses(self) -> int:
        return self._misses


__all__ = ["CachedExpertPool", "ExpertRunner", "GenerationSettings", "annotate_corpus"]
