"""Make `src/` importable without requiring an editable install.

CI and `make test` both run straight from a clone, so the test suite should not
depend on `pip install -e .` having happened first.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
