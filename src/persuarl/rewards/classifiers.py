"""Training the reward classifiers and building their class prototypes.

R1 and R2 need two artefacts per dimension:

1. a sequence classifier over persuasion strategies / user intents, and
2. a ``(num_classes, hidden_dim)`` tensor of **class prototypes** -- the mean
   encoder embedding of every training utterance carrying that label.

This module produces both in one pass, replacing the four notebooks
(``EngangementBert.ipynb``, ``EngamentModernBert.ipynb``, ``Intetntbert.ipynb``,
``IntentModernBert.ipynb``) that differed only in ``BASE_MODEL_ID`` and the
label list. Swap the backbone from the config: ``bert-base-uncased`` reproduces
the paper's numbers (ESCR 82.14 acc, ICR 84.91 acc, Table 9);
``answerdotai/ModernBERT-base`` and ``distilbert-base-uncased`` are the
alternatives that table compares against.

Prototypes are computed from the **training split only** -- computing them over
all data would leak test utterances into the reward signal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..constants import DEFAULT_SEED
from ..utils.logging import get_logger
from .consistency import mean_pooled_embedding

LOGGER = get_logger(__name__)


@dataclass
class ClassifierSpec:
    """Everything that differs between the engagement and intent classifiers."""

    name: str
    labels: Sequence[str]
    dataset_path: str
    text_column: str = "utterance"
    label_column: str = "label"
    normalize_labels: bool = False
    """Lower-case and strip a trailing ' appeal' before matching.

    The engagement annotations are free-text ("Logical appeal", "logical"), the
    intent annotations are already canonical identifiers.
    """


def _prepare_frame(spec: ClassifierSpec) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    """Load, clean and label-encode the classifier's training CSV."""
    frame = pd.read_csv(spec.dataset_path, usecols=[spec.text_column, spec.label_column])
    frame = frame.dropna()

    labels = frame[spec.label_column].astype(str).str.strip()
    if spec.normalize_labels:
        labels = labels.str.lower().str.replace(" appeal", "", regex=False)
    frame[spec.label_column] = labels

    known = set(spec.labels)
    before = len(frame)
    frame = frame[frame[spec.label_column].isin(known)]
    dropped = before - len(frame)
    if dropped:
        # Multi-label annotations ("logical and emotional") land here. They are
        # ~0.1% of InsureDial and there is no principled single-label mapping.
        LOGGER.warning("%s: dropped %d rows with out-of-vocabulary labels", spec.name, dropped)
    if frame.empty:
        raise ValueError(f"{spec.name}: no rows left after label filtering; check {spec.dataset_path}")

    label2id = {label: index for index, label in enumerate(spec.labels)}
    id2label = {index: label for label, index in label2id.items()}
    frame["labels"] = frame[spec.label_column].map(label2id)

    LOGGER.info("%s: %d rows, class counts %s",
                spec.name, len(frame), frame[spec.label_column].value_counts().to_dict())
    return frame, label2id, id2label


def _stratified_splits(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """90/10 then 89/11 -> roughly 80/10/10, stratified on the label.

    Utterance-level (not conversation-level) splitting is correct here: these
    classifiers score isolated utterances at reward time, so that is the
    distribution they should be validated on.
    """
    train_val, test = train_test_split(
        frame, test_size=0.1, random_state=seed, stratify=frame["labels"]
    )
    train, validation = train_test_split(
        train_val, test_size=0.11, random_state=seed, stratify=train_val["labels"]
    )
    return train, validation, test


def _compute_metrics(eval_pred):
    """Accuracy and macro-F1 -- macro because the class distribution is skewed."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
    }


def train_classifier(
    spec: ClassifierSpec,
    *,
    base_model_id: str = "bert-base-uncased",
    output_dir: str | Path,
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 512,
    seed: int = DEFAULT_SEED,
    fp16: bool = True,
) -> dict[str, float]:
    """Fine-tune a sequence classifier and save it to ``output_dir``.

    Returns the held-out test metrics (the ones reported in Table 9).
    """
    output_dir = Path(output_dir)
    frame, label2id, id2label = _prepare_frame(spec)
    train_frame, eval_frame, test_frame = _stratified_splits(frame, seed)
    LOGGER.info("%s split: train=%d val=%d test=%d",
                spec.name, len(train_frame), len(eval_frame), len(test_frame))

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=len(spec.labels),
        id2label=id2label,
        label2id=label2id,
    )

    from datasets import Dataset

    def encode(frame_: pd.DataFrame):
        dataset = Dataset.from_pandas(frame_[[spec.text_column, "labels"]], preserve_index=False)
        return dataset.map(
            lambda batch: tokenizer(batch[spec.text_column], truncation=True, max_length=max_length),
            batched=True,
            remove_columns=[spec.text_column],
        )

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=fp16 and torch.cuda.is_available(),
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=encode(train_frame),
        eval_dataset=encode(eval_frame),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate(eval_dataset=encode(test_frame))
    LOGGER.info("%s test accuracy=%.4f macro-F1=%.4f",
                spec.name, metrics["eval_accuracy"], metrics["eval_f1"])

    # Prototypes must come from this exact checkpoint, so build them here rather
    # than in a separate script that could be pointed at a different directory.
    build_prototypes(
        model_dir=output_dir,
        train_frame=train_frame,
        text_column=spec.text_column,
        output_path=output_dir / "prototypes.pt",
        num_classes=len(spec.labels),
    )
    return {"accuracy": metrics["eval_accuracy"], "f1": metrics["eval_f1"]}


@torch.no_grad()
def build_prototypes(
    *,
    model_dir: str | Path,
    train_frame: pd.DataFrame,
    text_column: str,
    output_path: str | Path,
    num_classes: int,
    batch_size: int = 32,
) -> torch.Tensor:
    """Mean encoder embedding per class -> ``(num_classes, hidden_dim)`` tensor.

    Row ``i`` is class ``i``, matching the classifier's ``id2label``. A class
    with no training utterances would break that alignment, so we fill it with
    zeros and warn -- a zero prototype yields cosine similarity 0, i.e. "no
    evidence", which is the right neutral behaviour.
    """
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    vectors: list[torch.Tensor] = []
    hidden_size = int(model.config.hidden_size)

    for class_id in range(num_classes):
        utterances = train_frame.loc[train_frame["labels"] == class_id, text_column].astype(str).tolist()
        if not utterances:
            LOGGER.warning("class %d has no training utterances; using a zero prototype", class_id)
            vectors.append(torch.zeros(hidden_size))
            continue

        embeddings = mean_pooled_embedding(utterances, model, tokenizer, batch_size=batch_size)
        vectors.append(embeddings.mean(dim=0).cpu())
        LOGGER.info("class %d prototype from %d utterances", class_id, len(utterances))

    prototypes = torch.stack(vectors, dim=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prototypes, str(output_path))
    LOGGER.info("saved prototypes %s -> %s", tuple(prototypes.shape), output_path)
    return prototypes


__all__ = ["ClassifierSpec", "build_prototypes", "train_classifier"]
