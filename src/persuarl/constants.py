"""Names, column headers and label sets that the whole pipeline agrees on.

Keeping these in one place matters more than it looks: the expert CSVs, the
Selector prompt, the Generator prompt and the reward classifiers all have to
use the *same* expert keys in the *same* order, otherwise the route bitmask
and the prototype tensors silently drift out of alignment.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Experts
# --------------------------------------------------------------------------

#: Canonical expert order. The Selector's binary mask o_t in {0,1}^n is indexed
#: by this list, so DO NOT reorder it without regenerating every route table.
EXPERT_KEYS: Final[tuple[str, ...]] = ("engagement", "intent", "keyterm", "sentiment")

#: Column that holds an expert's textual output inside `data/*/expert_outputs/*.csv`.
EXPERT_ANSWER_COLUMN: Final[str] = "{expert}_answer"

# --------------------------------------------------------------------------
# InsureDial schema
# --------------------------------------------------------------------------

COL_CONVERSATION_ID: Final[str] = "conversation_id"
COL_TURN_NO: Final[str] = "turn_no"
COL_USER_UTTERANCE: Final[str] = "user_utterance"
COL_AGENT_REPLY: Final[str] = "new_agent_reply"

#: The expert CSVs were exported with a slightly different header than the main
#: dialogue file ("utterance" instead of "user_utterance"). We normalise on load.
COL_EXPERT_UTTERANCE: Final[str] = "utterance"

# --------------------------------------------------------------------------
# Label spaces (Appendix B.1 of the paper)
# --------------------------------------------------------------------------

#: Engagement Strategy Consistency Reward (R1) is computed over these classes.
ENGAGEMENT_LABELS: Final[tuple[str, ...]] = (
    "logical",
    "emotional",
    "credibility",
    "personal",
    "persona",
    "default",
)

#: Intent Consistency Reward (R2) is computed over these classes.
INTENT_LABELS: Final[tuple[str, ...]] = (
    "Request_Insurance_Quote",
    "Ask_Coverage_Details",
    "Express_Concern",
    "Request_Additional_Info",
    "Confirm_Interest",
    "Ask_Price_or_Premium",
)

SENTIMENT_LABELS: Final[tuple[str, ...]] = ("positive", "neutral", "negative")

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

#: HuggingFace's convention for "do not compute loss on this token".
IGNORE_INDEX: Final[int] = -100

#: Default RNG seed. Used for the conversation-level split *and* for training,
#: which is why every split helper takes it explicitly rather than reading a global.
DEFAULT_SEED: Final[int] = 42
