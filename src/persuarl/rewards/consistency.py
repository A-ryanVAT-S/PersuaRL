"""R1 (engagement strategy) and R2 (intent) -- the two prototype rewards.

Both have the same shape (Eqs. 3 and 4 in the paper). Given a user utterance
``u_T`` and a generated response ``r_T``:

1. A fine-tuned BERT classifier gives ``P_c(u_T)``, the probability that the
   *user's* turn belongs to class ``c``.
2. ``Embed(r_T)`` is mean-pooled from the same classifier's encoder, and
   ``S_c(r_T) = cos(Embed(r_T), Proto_c)`` scores the *response* against each
   class prototype.
3. The reward mixes the two::

       R = sum_c P_c(u_T) * S_c(r_T)  +  lambda * max_c S_c(r_T)

The first term asks "does the response match the class distribution the user's
turn implies"; the ``lambda`` term (0.3 < lambda < 1 in the paper, 0.1 in the
released configs) adds a confidence bonus for being *strongly* aligned with
some class rather than weakly aligned with all of them.

Cosine similarity lives in [-1, 1], so the raw score is rescaled to [0, 1]
before it is mixed into the composite reward -- every ``R_k`` has to share a
range or the ``beta`` weights stop meaning what they look like they mean.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def mean_pooled_embedding(
    texts: Sequence[str],
    model,
    tokenizer,
    *,
    max_length: int = 128,
    batch_size: int = 32,
) -> torch.Tensor:
    """Mean-pool the encoder's last hidden states over non-special tokens.

    We deliberately pool the *encoder*, not the classification head: prototypes
    have to live in the same space as the embeddings they are compared against,
    and the head's 5-or-6-dim logits are far too coarse for a cosine similarity.
    """
    encoder = _encoder_of(model)
    device = next(model.parameters()).device
    chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [str(text) for text in texts[start:start + batch_size]]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            outputs = encoder(**inputs, output_hidden_states=True)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs.hidden_states[-1]

            # Mask-weighted mean so padding does not drag the vector toward zero.
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(pooled)

    return torch.cat(chunks, dim=0)


def _encoder_of(model):
    """Return the transformer trunk of a ``*ForSequenceClassification`` model.

    ``model.base_model`` covers BERT/RoBERTa/ModernBERT; the ``model_type``
    fallback catches wrappers that expose the trunk under its family name.
    """
    encoder = getattr(model, "base_model", None)
    if encoder is None:
        family = getattr(model.config, "model_type", "bert")
        encoder = getattr(model, family, None)
    if encoder is None:
        LOGGER.warning("could not locate encoder trunk; pooling the full classifier instead")
        encoder = model
    return encoder


def class_probabilities(
    texts: Sequence[str],
    model,
    tokenizer,
    *,
    max_length: int = 128,
) -> torch.Tensor:
    """Softmax over the classifier head -- the ``P_c(u_T)`` term."""
    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = tokenizer(
            [str(text) for text in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        return torch.softmax(model(**inputs).logits, dim=-1)


@dataclass
class PrototypeScorer:
    """A classifier plus its class prototypes: everything R1 or R2 needs.

    ``lambda_confidence`` is the ``lambda`` of Eqs. 3-4. Raising it makes the
    reward care more about peak alignment and less about matching the user's
    full class distribution.
    """

    model: torch.nn.Module
    tokenizer: object
    prototypes: torch.Tensor
    lambda_confidence: float = 0.1
    name: str = "consistency"

    def __post_init__(self) -> None:
        self.model.eval()
        num_classes = int(self.model.config.num_labels)
        if self.prototypes.shape[0] != num_classes:
            raise ValueError(
                f"{self.name}: prototype tensor has {self.prototypes.shape[0]} rows but the "
                f"classifier has {num_classes} labels -- regenerate the prototypes "
                f"(persuarl.cli.train_reward_models) against this checkpoint"
            )

    @classmethod
    def from_pretrained(
        cls,
        classifier_path: str | Path,
        prototypes_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        lambda_confidence: float = 0.1,
        name: str = "consistency",
    ) -> PrototypeScorer:
        from ..models.loader import load_sequence_classifier

        model, tokenizer = load_sequence_classifier(str(classifier_path), device=device)
        prototypes = torch.load(str(prototypes_path), map_location="cpu")
        LOGGER.info(
            "%s scorer ready: %s (%d classes, prototypes %s)",
            name, classifier_path, model.config.num_labels, tuple(prototypes.shape),
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            prototypes=prototypes,
            lambda_confidence=lambda_confidence,
            name=name,
        )

    def score(self, user_utterances: Sequence[str], responses: Sequence[str]) -> torch.Tensor:
        """Compute the reward for a batch, normalised to [0, 1]."""
        if len(user_utterances) != len(responses):
            raise ValueError("user_utterances and responses must be the same length")

        probabilities = class_probabilities(user_utterances, self.model, self.tokenizer)
        embeddings = mean_pooled_embedding(responses, self.model, self.tokenizer)
        prototypes = self.prototypes.to(embeddings.device, dtype=embeddings.dtype)

        # (batch, 1, dim) vs (1, classes, dim) -> (batch, classes)
        similarity = F.cosine_similarity(
            embeddings.unsqueeze(1), prototypes.unsqueeze(0), dim=-1
        )

        aligned = (probabilities.to(similarity.device) * similarity).sum(dim=-1)
        confident = similarity.max(dim=-1).values
        raw = aligned + self.lambda_confidence * confident

        # cos in [-1,1] -> [0,1]; clamp because the lambda term can push past 1.
        return torch.clamp((raw + 1.0) / 2.0, 0.0, 1.0).detach().cpu()


__all__ = [
    "PrototypeScorer",
    "class_probabilities",
    "mean_pooled_embedding",
]
