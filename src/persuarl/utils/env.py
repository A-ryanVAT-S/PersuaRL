"""Repo-level ``.env`` support, mirroring what the shell scripts expect.

We do not depend on ``python-dotenv``: the file format we need is two lines of
parsing, and one fewer install is one fewer thing to pin.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until a directory containing ``pyproject.toml``."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=value`` pairs from ``.env`` into ``os.environ``.

    Existing environment variables win unless ``override=True`` -- so an
    ``export`` in your shell or a Slurm job script always beats the file.
    Returns the parsed mapping (handy for logging what was picked up).
    """
    env_path = Path(path) if path else find_repo_root() / ".env"
    parsed: dict[str, str] = {}
    if not env_path.is_file():
        return parsed

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def require_env(name: str, hint: str = "") -> str:
    """Fetch a required environment variable or fail with an actionable message."""
    value = os.environ.get(name)
    if not value:
        suffix = f" {hint}" if hint else ""
        raise RuntimeError(f"environment variable {name} is not set.{suffix}")
    return value
