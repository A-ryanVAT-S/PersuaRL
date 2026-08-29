"""The four expert modules, declared as data.

The original repo had four fine-tuning scripts (``Intent_expert_finetuning.py``,
``Keyterm_expert_finetuning.py``, ``Persuassion_Expert_finetuning.py``,
``Sentiment_expert_finetuning.py``) that were byte-identical apart from a system
prompt, an output template and a filename. That is a registry, not four
programs -- so here it is, and :mod:`persuarl.experts.training` is the one
program that consumes it.

Adding a fifth expert means adding an :class:`ExpertSpec` here and a key to
:data:`persuarl.constants.EXPERT_KEYS`; the route table, the Selector prompt and
the Generator's analysis block all widen automatically.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ..constants import ENGAGEMENT_LABELS, EXPERT_KEYS, INTENT_LABELS, SENTIMENT_LABELS


@dataclass(frozen=True)
class ExpertSpec:
    """Declarative description of one expert LM."""

    key: str
    """Must be a member of :data:`~persuarl.constants.EXPERT_KEYS`."""

    display_name: str
    system_prompt: str

    completion_template: str
    """Target string for SFT, e.g. ``"Label: {label}\\nReason: {reason}"``."""

    answer_pattern: re.Pattern[str]
    """Parses a rendered answer back into ``{label, reason}``.

    Used both to rebuild training data from the shipped expert-output CSVs and
    to sanity-check inference output.
    """

    labels: tuple[str, ...] = ()
    """Closed label set, empty for open-vocabulary experts like keyterm."""

    objective: str = "nll"
    """``"nll"`` for the generative experts, ``"cross_entropy"`` for sentiment
    when trained as a classifier (Appendix C.2)."""

    metadata: Mapping[str, str] = field(default_factory=dict)

    def render_completion(self, label: str, reason: str) -> str:
        return self.completion_template.format(label=str(label).strip(), reason=str(reason).strip())

    def parse_answer(self, text: str) -> dict[str, str] | None:
        """Split a rendered answer into its label and reason, or ``None``."""
        match = self.answer_pattern.match(str(text).strip())
        if not match:
            return None
        groups = match.groupdict()
        return {
            "label": (groups.get("label") or "").strip(),
            "reason": (groups.get("reason") or "").strip(),
        }


# --------------------------------------------------------------------------
# Engagement -- which persuasion strategy the agent should use next
# --------------------------------------------------------------------------

ENGAGEMENT = ExpertSpec(
    key="engagement",
    display_name="Engagement Strategy Expert",
    system_prompt="""You are an Engagement Strategy Selector for a motor insurance dialogue system. Based on the user's most recent utterance and the conversation history, you must recommend the most suitable persuasion strategy the agent should use next to move the conversation forward.

You must choose from the following six engagement strategies:

1. Credibility Appeal: Emphasize reputation and trust. Example: "New India Assurance has one of the widest repair networks."

2. Logical Appeal: Use facts, pricing, or benefits. Use when: User is analytical or budget-conscious.

3. Emotional Appeal: Focus on peace of mind and safety. Example: "Drive worry-free knowing your EV is protected."

4. Persona Appeal: Align with the user's identity or values. Example: "Built for modern EV owners."

5. Personal Appeal: Address the user empathetically and directly. Example: "This keeps your EV protected."

6. Default: Provide neutral, factual information. Example: "Let me explain the EV coverage options."

Your output must follow this exact format:

Label: [Selected Strategy]
Reason: [1-2 line explanation]

Focus solely on the user's final message.
""",
    completion_template="Label: {label}\nReason: {reason}",
    # The shipped CSVs use prose ("The persuasion strategy is X and the reason
    # is Y"); accept both that and the canonical "Label:/Reason:" form.
    answer_pattern=re.compile(
        r"^(?:The persuasion strategy is\s+(?P<label>.+?)\s+and the reason is\s+(?P<reason>.*)"
        r"|Label:\s*(?P<label2>.+?)\s*\nReason:\s*(?P<reason2>.*))$",
        re.DOTALL | re.IGNORECASE,
    ),
    labels=ENGAGEMENT_LABELS,
    objective="nll",
)

# --------------------------------------------------------------------------
# Intent -- what the user is trying to do this turn
# --------------------------------------------------------------------------

INTENT = ExpertSpec(
    key="intent",
    display_name="Intent Expert",
    system_prompt="""You are an Intent Expert for a virtual assistant specializing in motor insurance. Your job is to analyze the current user utterance, using the conversation history for context, and determine the single most relevant intent expressed by the user.

You must select from a fixed set of six pre-defined intents:

- Request_Insurance_Quote
  User initiates interest in getting a motor insurance quote or policy.

- Ask_Coverage_Details
  User asks about types of protection or what is covered (e.g., battery, theft, accident).

- Express_Concern
  User shares a priority or worry about coverage (e.g., battery damage).

- Request_Additional_Info
  User requests clarification or deeper explanation about features or terms.

- Confirm_Interest
  User explicitly agrees or indicates they want to proceed.

- Ask_Price_or_Premium
  User asks about the cost, premium, or pricing breakdown.

Instructions:
1. Determine the single most relevant intent based on the current user utterance and conversation context.
2. Provide a brief 1-2 line justification citing why this intent matches.

Your output must follow this exact format:

Label: [One of the six predefined intents]
Reason: [1-2 line explanation of why this intent matches the user's message]

Focus solely on the user's final message.
""",
    completion_template="Label: {label}\nReason: {reason}",
    answer_pattern=re.compile(
        r"^(?:The intent is\s+(?P<label>.+?)\s+and the reason is\s+(?P<reason>.*)"
        r"|Label:\s*(?P<label2>.+?)\s*\nReason:\s*(?P<reason2>.*))$",
        re.DOTALL | re.IGNORECASE,
    ),
    labels=INTENT_LABELS,
    objective="nll",
)

# --------------------------------------------------------------------------
# Keyterm -- domain vocabulary grounding
# --------------------------------------------------------------------------

KEYTERM = ExpertSpec(
    key="keyterm",
    display_name="Keyterm Expert",
    system_prompt="""You are a Keyterm Expert specializing in the motor insurance domain. Your job is to analyze the user's most recent utterance, using the conversation history for context, and identify one or more important motor insurance-related keyterms mentioned (explicitly or implicitly).

Examples of Common Keyterms (not limited to):
- Comprehensive coverage
- Third-party liability
- Roadside assistance
- Zero depreciation
- Deductibles
- Policy renewal
- Personal accident cover
- IDV (Insured Declared Value)

You may also extract user-specific or vehicle-specific keyterms (e.g., "Tesla Model 3," "EV," "2024 vehicle").

Instructions:
1. Extract all relevant keyterms mentioned or implied.
2. For each, provide a 1-line justification for its insurance relevance.

Your output must follow this exact format:

Extracted Keyterm: [Term]
Justification: [Brief reason]

Focus solely on the user's final message.
""",
    completion_template="Extracted Keyterm: {label}\nJustification: {reason}",
    answer_pattern=re.compile(
        r"^(?:The keyterms? extracted (?:are|is)\s+(?P<label>.*?)(?:\.\s*(?P<reason>.*))?"
        r"|Extracted Keyterm:\s*(?P<label2>.+?)\s*\nJustification:\s*(?P<reason2>.*))$",
        re.DOTALL | re.IGNORECASE,
    ),
    labels=(),  # open vocabulary
    objective="nll",
)

# --------------------------------------------------------------------------
# Sentiment -- emotional tone of the user's turn
# --------------------------------------------------------------------------

SENTIMENT = ExpertSpec(
    key="sentiment",
    display_name="Sentiment Expert",
    system_prompt="""You are trained to act solely as a Sentiment Expert. Your job is to analyze the emotional tone of the input text and classify it into one of the following categories:

- Positive : Expresses happiness, excitement, appreciation, or other positive emotions.
- Negative : Expresses disappointment, frustration, anger, sadness, or criticism.
- Neutral : Emotionally balanced, factual, or without strong emotional content.

Rules:
- Only focus on emotional tone, word choice, or sentiment-laden phrases.
- Do not summarize or infer intent beyond emotional expression.

Your output must follow this exact format:

Sentiment: [Positive / Negative / Neutral]
Explanation: [Concise reasoning based on emotional tone]

Focus solely on the user's final message.
""",
    completion_template="Sentiment: {label}\nExplanation: {reason}",
    answer_pattern=re.compile(
        r"^(?:The sentiment is\s+(?P<label>\w+)\.?\s*(?P<reason>.*)"
        r"|Sentiment:\s*(?P<label2>\w+)\s*\nExplanation:\s*(?P<reason2>.*))$",
        re.DOTALL | re.IGNORECASE,
    ),
    labels=SENTIMENT_LABELS,
    objective="cross_entropy",
)


EXPERT_REGISTRY: dict[str, ExpertSpec] = {
    spec.key: spec for spec in (ENGAGEMENT, INTENT, KEYTERM, SENTIMENT)
}

# Fail at import time rather than three hours into a training run.
assert set(EXPERT_REGISTRY) == set(EXPERT_KEYS), (
    f"registry {sorted(EXPERT_REGISTRY)} does not match EXPERT_KEYS {sorted(EXPERT_KEYS)}"
)


def get_expert(key: str) -> ExpertSpec:
    """Look up an expert by key, with a helpful error for typos."""
    try:
        return EXPERT_REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown expert {key!r}; available: {sorted(EXPERT_REGISTRY)}") from None


def parse_answer(key: str, text: str) -> dict[str, str] | None:
    """Parse a rendered expert answer using that expert's pattern.

    Handles both alternation branches of the registry patterns (the ``label``
    and ``label2`` groups) so callers never see the regex plumbing.
    """
    spec = get_expert(key)
    match = spec.answer_pattern.match(str(text).strip())
    if not match:
        return None
    groups = match.groupdict()
    label = groups.get("label") or groups.get("label2") or ""
    reason = groups.get("reason") or groups.get("reason2") or ""
    return {"label": label.strip(), "reason": reason.strip()}


ParseFn = Callable[[str], "dict[str, str] | None"]

__all__ = [
    "ENGAGEMENT",
    "EXPERT_REGISTRY",
    "ExpertSpec",
    "INTENT",
    "KEYTERM",
    "SENTIMENT",
    "get_expert",
    "parse_answer",
]
