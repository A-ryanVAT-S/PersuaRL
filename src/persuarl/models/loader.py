"""Backbone-agnostic model loading.

This module is the reason ``train_sft`` and ``train_selector`` accept *any*
causal LM without a per-model script. Two things vary across backbones and both
are handled here:

* **LoRA target modules.** Llama/Qwen/Mistral use separate ``q_proj``/``k_proj``/
  ``v_proj``; Phi-3 fuses them into ``qkv_proj``; GPT-NeoX-style models use
  ``query_key_value``. Hard-coding one set is exactly why the original scripts
  needed a copy per backbone. :func:`resolve_lora_targets` inspects the loaded
  module names instead.
* **Quantisation / dtype / device placement**, which differ between the 3B
  models that fit on one GPU and the 24-30B ones that do not.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

#: Attention/MLP projection names we know how to adapt, most specific first.
#: Order matters: a Phi-3 model exposes ``qkv_proj`` *and* ``o_proj``, so we
#: collect every family member that is actually present rather than stopping at
#: the first hit.
_KNOWN_LORA_TARGETS: tuple[str, ...] = (
    # separate QKV (Llama, Qwen, Mistral, Gemma)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # fused QKV (Phi-3, some Falcon builds)
    "qkv_proj", "query_key_value", "dense",
    # MLP
    "gate_proj", "up_proj", "down_proj",
    "gate_up_proj",              # Phi-3 fuses gate and up
    "fc1", "fc2",                # GPT-2 / OPT style
)

#: Attention-only subset, used when ``lora.attention_only: true``. Cheaper and
#: closer to the original SFT script (which adapted just q_proj/v_proj).
_ATTENTION_ONLY_TARGETS: frozenset[str] = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "query_key_value", "dense"}
)


@dataclass
class QuantizationSpec:
    """4-/8-bit loading options. ``bits=None`` means full precision."""

    bits: int | None = None
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True

    def to_bnb_config(self) -> BitsAndBytesConfig | None:
        if self.bits is None:
            return None
        if self.bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.quant_type,
                bnb_4bit_compute_dtype=resolve_dtype(self.compute_dtype),
                bnb_4bit_use_double_quant=self.double_quant,
            )
        if self.bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(f"unsupported quantization bits: {self.bits} (use 4, 8 or null)")


def resolve_dtype(name: str | torch.dtype | None) -> Any:
    """Map a config string to a torch dtype. ``"auto"`` is passed through."""
    if name is None or isinstance(name, torch.dtype):
        return name
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(mapping)} or 'auto'")
    return mapping[name]


def load_tokenizer(model_id: str, *, padding_side: str = "right", trust_remote_code: bool = False):
    """Load a tokenizer with a usable pad token.

    ``padding_side`` matters: ``"right"`` for SFT (so labels line up with
    inputs), ``"left"`` for anything that generates, because right-padding a
    batch pushes the generation cue away from the end of the sequence and the
    model continues from padding instead.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        LOGGER.info("%s has no pad token; reusing eos_token %r", model_id, tokenizer.eos_token)
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_causal_lm(
    model_id: str,
    *,
    dtype: str = "auto",
    device_map: str | dict | None = "auto",
    quantization: QuantizationSpec | None = None,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
):
    """Load a causal LM, optionally quantised, with sane defaults."""
    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype is not None:
        # transformers>=4.56 renamed torch_dtype -> dtype; the new name is what
        # the pinned version expects.
        kwargs["dtype"] = resolved_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    bnb_config = quantization.to_bnb_config() if quantization else None
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        LOGGER.info("loading %s in %d-bit", model_id, quantization.bits)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model


def load_sequence_classifier(path: str, *, device: str | torch.device = "cpu"):
    """Load one of the reward classifiers (BERT engagement / intent) in eval mode."""
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def resolve_lora_targets(
    model,
    requested: Sequence[str] | None = None,
    *,
    attention_only: bool = False,
) -> list[str]:
    """Pick LoRA target module names that actually exist in ``model``.

    ``requested`` wins if given (and is validated against the model, because a
    typo there silently trains nothing). Otherwise we intersect the model's
    linear-layer names with :data:`_KNOWN_LORA_TARGETS`.
    """
    present = {name.split(".")[-1] for name, _ in model.named_modules()}

    if requested:
        unknown = [name for name in requested if name not in present]
        if unknown:
            raise ValueError(
                f"LoRA target module(s) {unknown} do not exist in this backbone. "
                f"Candidates present: {sorted(present & set(_KNOWN_LORA_TARGETS))}"
            )
        return list(requested)

    candidates = [name for name in _KNOWN_LORA_TARGETS if name in present]
    if attention_only:
        candidates = [name for name in candidates if name in _ATTENTION_ONLY_TARGETS]

    if not candidates:
        raise ValueError(
            "could not auto-detect LoRA target modules for this backbone; "
            "set lora.target_modules explicitly in your config"
        )

    LOGGER.info("auto-detected LoRA targets: %s", candidates)
    return candidates


def attach_lora(
    model,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Sequence[str] | None = None,
    attention_only: bool = False,
    prepare_for_kbit: bool = False,
    task_type: str = "CAUSAL_LM",
):
    """Wrap ``model`` in a fresh LoRA adapter and report the trainable fraction."""
    if prepare_for_kbit:
        # Casts layernorms to fp32, enables gradient checkpointing and makes the
        # embedding output require grad -- required before LoRA on a 4-bit model.
        model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
        target_modules=resolve_lora_targets(model, target_modules, attention_only=attention_only),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def load_with_adapter(
    model_id: str,
    adapter_path: str | None,
    *,
    merge: bool = False,
    **load_kwargs: Any,
):
    """Load a base model and optionally apply (and merge) a trained adapter.

    ``merge=True`` folds the adapter into the base weights. Do that for the
    *warm start* of a second training stage -- GRPO on top of an SFT checkpoint
    needs the SFT behaviour baked in so the new adapter starts from zero rather
    than fighting the old one. Keep ``merge=False`` for plain inference.
    """
    model = load_causal_lm(model_id, **load_kwargs)
    if not adapter_path:
        return model

    LOGGER.info("applying adapter %s (merge=%s)", adapter_path, merge)
    model = PeftModel.from_pretrained(model, adapter_path)
    if merge:
        model = model.merge_and_unload()
    return model


def quantization_from_config(section) -> QuantizationSpec | None:
    """Build a :class:`QuantizationSpec` from a ``model.quantization`` config block."""
    bits = section.get("bits", None)
    if bits in (None, 0, "none", "null"):
        return None
    return QuantizationSpec(
        bits=int(bits),
        quant_type=section.get("quant_type", "nf4"),
        compute_dtype=section.get("compute_dtype", "bfloat16"),
        double_quant=bool(section.get("double_quant", True)),
    )


def describe_devices() -> str:
    """One-line summary of the visible GPUs, logged at the top of every run."""
    if not torch.cuda.is_available():
        return "cuda unavailable (running on CPU -- expect this to be very slow)"
    parts: Iterable[str] = (
        f"{index}:{torch.cuda.get_device_name(index)} "
        f"({torch.cuda.get_device_properties(index).total_memory / 1e9:.0f}GB)"
        for index in range(torch.cuda.device_count())
    )
    return f"{torch.cuda.device_count()} GPU(s) | " + ", ".join(parts)


__all__ = [
    "QuantizationSpec",
    "attach_lora",
    "describe_devices",
    "load_causal_lm",
    "load_sequence_classifier",
    "load_tokenizer",
    "load_with_adapter",
    "quantization_from_config",
    "resolve_dtype",
    "resolve_lora_targets",
]
