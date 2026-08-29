"""YAML configs with CLI overrides.

Every entry point follows the same contract::

    python -m persuarl.cli.<task> --config configs/<...>.yaml [--set a.b=c ...]

The YAML file carries the experiment; ``--set`` carries the one-off tweak you
would otherwise make by editing (and forgetting to revert) the file. Configs
may declare ``defaults: [path, ...]`` to inherit from a base file, which is how
the per-backbone files in ``configs/models/`` stay three lines long.

Values are kept as plain nested dicts wrapped in :class:`Config`, deliberately
*not* as dataclasses: TRL's ``GRPOConfig``/``TrainingArguments`` already own the
schema for the training knobs, and duplicating it here only creates drift.
"""

from __future__ import annotations

import ast
import copy
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

#: Sentinel distinguishing "no default given" from an explicit ``default=None``.
_MISSING = object()


class ConfigError(RuntimeError):
    """Raised for a missing key, a bad override or an unresolvable ``${VAR}``."""


class Config:
    """Read-only, dotted-path view over a nested dict."""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data or {}))

    # -- access ------------------------------------------------------------

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Fetch ``a.b.c``; raise :class:`ConfigError` if absent and no default."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"missing required config key: {path!r}")
                return default
            node = node[part]
        return node

    def section(self, path: str) -> Config:
        """Return a sub-tree as its own :class:`Config` (empty if absent)."""
        node = self.get(path, {})
        if not isinstance(node, Mapping):
            raise ConfigError(f"config key {path!r} is a value, not a section")
        return Config(node)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __contains__(self, path: str) -> bool:
        return self.get(path, None) is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({yaml.safe_dump(self._data, sort_keys=False)})"


def _expand_env(value: Any) -> Any:
    """Resolve ``${VAR}`` / ``${VAR:-fallback}`` inside strings, recursively."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name, fallback = match.group(1), match.group(2)
            resolved = os.environ.get(name, fallback)
            if resolved is None:
                raise ConfigError(
                    f"config references ${{{name}}} but it is not set "
                    f"(export it or add it to your .env)"
                )
            return resolved

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, Mapping):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; scalars and lists from ``override`` win outright."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_override(text: str) -> tuple[str, Any]:
    """``"train.lr=2e-5"`` -> ``("train.lr", 2e-05)``.

    Values go through ``literal_eval`` so ints, floats, bools, ``None`` and
    lists keep their type; anything that fails to parse stays a string (which
    is what you want for model ids and paths).
    """
    if "=" not in text:
        raise ConfigError(f"--set expects key=value, got {text!r}")
    key, _, raw = text.partition("=")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return key.strip(), value


def _apply_override(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"--set {key}: {part!r} is a value, not a section")
    node[parts[-1]] = value


def load_config(
    path: str | Path,
    overrides: Iterable[str] = (),
    *,
    _seen: set[Path] | None = None,
) -> Config:
    """Load ``path``, resolve its ``defaults:`` chain, then apply ``--set`` overrides."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    seen = _seen or set()
    if path in seen:
        raise ConfigError(f"circular defaults chain at {path}")
    seen.add(path)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    merged: dict[str, Any] = {}
    for parent in raw.pop("defaults", []) or []:
        parent_path = (path.parent / parent).resolve()
        merged = _deep_merge(merged, load_config(parent_path, _seen=seen).as_dict())
    merged = _deep_merge(merged, raw)

    for override in overrides:
        key, value = _parse_override(override)
        _apply_override(merged, key, value)

    # Env expansion happens last so that ``--set`` can inject ``${...}`` too.
    return Config(_expand_env(merged))


def add_config_args(parser) -> None:
    """Attach the standard ``--config`` / ``--set`` pair to an ArgumentParser."""
    parser.add_argument(
        "--config",
        required=True,
        help="path to the experiment YAML (see configs/)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. --set model.id=Qwen/Qwen2.5-3B-Instruct",
    )


__all__ = ["Config", "ConfigError", "load_config", "add_config_args"]
