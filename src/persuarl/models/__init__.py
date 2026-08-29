"""Backbone loading, LoRA plumbing and constrained decoding."""

from .decoding import (
    AllowedSequencesProcessor,
    SingleTokenChoiceProcessor,
    build_selector_logits_processor,
    patch_generate_with_constraint,
    route_token_ids,
)
from .loader import (
    QuantizationSpec,
    attach_lora,
    describe_devices,
    load_causal_lm,
    load_sequence_classifier,
    load_tokenizer,
    load_with_adapter,
    quantization_from_config,
    resolve_dtype,
    resolve_lora_targets,
)

__all__ = [
    "AllowedSequencesProcessor",
    "QuantizationSpec",
    "SingleTokenChoiceProcessor",
    "attach_lora",
    "build_selector_logits_processor",
    "describe_devices",
    "load_causal_lm",
    "load_sequence_classifier",
    "load_tokenizer",
    "load_with_adapter",
    "patch_generate_with_constraint",
    "quantization_from_config",
    "resolve_dtype",
    "resolve_lora_targets",
    "route_token_ids",
]
