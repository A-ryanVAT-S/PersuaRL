"""Reproducibility.

Note the scope of what this buys you: the *data split* becomes deterministic
(which is what actually matters for a fair comparison across runs), and so does
model init. Sampling-based GRPO rollouts on multiple GPUs still vary run to
run -- the paper's numbers are single-run, seed 10 for decoding.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_cudnn: bool = False) -> int:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # the data-prep CLIs run fine without torch installed
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        # Costs throughput; only worth it when chasing a nondeterminism bug.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
