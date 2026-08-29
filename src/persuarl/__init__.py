"""PersuaRL -- RL-driven multi-expert selection for persuasive dialogue.

The package is organised around the three modules of the paper:

    persuarl.experts    the four task-specific expert LMs (T_i)
    persuarl.training   the Selector policy (pi_theta) and the Generator (A_phi)
    persuarl.rewards    the composite reward R = sum_k beta_k * R_k, minus penalties

Everything else is plumbing: `persuarl.data` turns InsureDial CSVs into
turn-level training examples, `persuarl.models` hides the
backbone-specific loading/LoRA details, and `persuarl.cli` exposes the
entry points that the shell scripts in `scripts/` call.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
