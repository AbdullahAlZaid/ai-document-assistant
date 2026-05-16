"""Backend gate that decides whether computational tools should be exposed to the LLM.

Design rules:
- The gate is purely deterministic: same input → same output, no randomness, no LLM.
- It is independently testable without any external dependencies.
- It does NOT decide which tool to use — that remains the LLM's responsibility.
- It only answers: "does this question appear to need computation?"
- Returning True exposes tools to the LLM. Returning False skips them entirely.

This approach trades recall (some computational questions may be missed) for
precision and reliability — the model is never given tools it shouldn't use,
and the gate itself cannot fail in a way that produces wrong answers.
"""

import logging
import re

_log = logging.getLogger(__name__)


# ████████████████████████████████████████████████████████████████████
# ██ TRIGGER PATTERNS
# ████████████████████████████████████████████████████████████████████

# Each pattern is a compiled regex applied case-insensitively to the question.
# Word boundaries (\b) prevent partial matches (e.g. "day" inside "yesterday").
# Patterns are grouped by intent for readability and future extensibility.
_TRIGGERS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [

    # ── Date / time remaining ────────────────────────────────────────
    r"\bhow many days\b",
    r"\bdays (left|remaining|until|till|before)\b",
    r"\btime (left|remaining|until|till)\b",
    r"\bhow long (until|till|before|is left)\b",
    r"\bdays? from (now|today)\b",
    r"\bexpir(es?|ed|ation|ing)\b.{0,40}\b(when|how long|how many)\b",

    # ── Day of week ──────────────────────────────────────────────────
    r"\bwhat (day|weekday)\b",
    r"\bwhich (day|weekday)\b",
    r"\bday of (the )?week\b",
    r"\bfall(s)? on\b",

    # ── Arithmetic / calculation ─────────────────────────────────────
    r"\bcalculat(e|ion|or)\b",
    r"\bcomput(e|ation)\b",
    r"\btotal\b.{0,30}\b(is|cost|amount|value|price)\b",
    r"\bhow much\b.{0,30}\b(is|cost|total|amount)\b",
    r"\bsum (of|up)\b",
    r"\bpercentage\b",
    r"\bmultipl(y|ied)\b",

    # ── Financial ────────────────────────────────────────────────────
    r"\bpercent(age)?\s+(change|increase|decrease|growth|difference)\b",
    r"\b(increase|decrease|grew?|grew|grown|changed?)\s+by\b",
    r"\bprorat(e|ed|ion)\b",
    r"\bpartial.{0,20}(period|payment|amount|fee)\b",
    r"\b(markup|discount|tax|vat|fee)\b.{0,30}\b(on|of|applied)\b",
    r"\bhow much.{0,30}(owe|due|pay|charge|cost)\b",
    r"\b(after|with|including|plus).{0,20}\b(tax|vat|discount|markup|fee|rate)\b",
    r"\b(net|gross)\s+(amount|price|value|total)\b",
]]


# ████████████████████████████████████████████████████████████████████
# ██ GATE
# ████████████████████████████████████████████████████████████████████

def should_use_tools(question: str) -> bool:
    """Return True if the question appears to require a computational tool.

    The gate uses keyword/regex heuristics only — no LLM, no network calls.
    A False result means the pipeline falls through to normal RAG generation.
    A True result means tool schemas are passed to the LLM alongside the prompt;
    the LLM still decides whether and how to call a tool.
    """
    result = any(pattern.search(question) for pattern in _TRIGGERS)
    _log.debug(
        "ℹ️  [TOOL-GATE]: tools=%s | question: %.60r",
        "enabled" if result else "disabled",
        question,
    )
    return result
