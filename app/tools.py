"""Deterministic computational tools available to the LLM via tool calling.

Design rules:
- Every tool is a pure function: same input → same output, no side effects.
- Tools operate on values extracted from retrieved document chunks by the LLM.
  They never access external data, the internet, or application state.
- Each tool has a strict JSON schema used to register it with the Groq API.
- All tools return a plain string so the result can be appended directly to
  the conversation as a tool message without further transformation.
"""

import ast
import math
import operator
from datetime import date, datetime


# ████████████████████████████████████████████████████████████████████
# ██ DATE PARSING
# ████████████████████████████████████████████████████████████████████

# Formats tried in order. Earlier entries match more specific patterns first.
_DATE_FORMATS = (
    "%Y-%m-%d",    # 2032-02-06
    "%B %d, %Y",   # February 6, 2032
    "%b %d, %Y",   # Feb 6, 2032
    "%d %B %Y",    # 6 February 2032
    "%d %b %Y",    # 6 Feb 2032
    "%B %d %Y",    # February 6 2032  (no comma)
    "%b %d %Y",    # Feb 6 2032
    "%d %B, %Y",   # 6 February, 2032
    "%d %b, %Y",   # 6 Feb, 2032
)

_DATE_FORMAT_HINT = "YYYY-MM-DD, 'January 1, 2025', '1 January 2025', or similar."


def _parse_date(date_string: str) -> date:
    """Try each supported format and return the first match, or raise ValueError."""
    cleaned = date_string.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: '{date_string}'")


# ████████████████████████████████████████████████████████████████████
# ██ TOOL IMPLEMENTATIONS
# ████████████████████████████████████████████████████████████████████

def days_until(date_string: str) -> str:
    """Return the number of days from today until the given date."""
    try:
        target = _parse_date(date_string)
        today = date.today()
        delta = (target - today).days
        if delta > 0:
            return f"{delta} days remain until {target.isoformat()} (from today, {today.isoformat()})."
        if delta == 0:
            return f"The date {target.isoformat()} is today."
        return f"The date {target.isoformat()} was {abs(delta)} days ago (today is {today.isoformat()})."
    except ValueError:
        return f"Could not parse date '{date_string}'. Supported formats: {_DATE_FORMAT_HINT}"


def get_weekday(date_string: str) -> str:
    """Return the day of the week for the given date."""
    try:
        target = _parse_date(date_string)
        return f"{target.isoformat()} falls on a {target.strftime('%A')}."
    except ValueError:
        return f"Could not parse date '{date_string}'. Supported formats: {_DATE_FORMAT_HINT}"


# Allowed operators for the safe expression evaluator.
# Restricting to these prevents arbitrary code execution.
_SAFE_OPERATORS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.expr) -> float:
    """Recursively evaluate a parsed AST node using only safe arithmetic operators."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_fn = _SAFE_OPERATORS[type(node.op)]
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero.")
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression and return the result.

    Supports: +, -, *, /, ** and parentheses.
    Does NOT support: function calls, variable names, imports, or any Python builtins.
    This prevents code injection while still covering all practical use cases.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        # Format: drop the decimal point for whole numbers.
        formatted = int(result) if result == math.floor(result) else round(result, 6)
        return f"Result of {expression.strip()} = {formatted}"
    except ZeroDivisionError as exc:
        return f"Calculation error: {exc}"
    except Exception:
        return f"Could not evaluate expression '{expression}'. Use only numbers and + - * / ** operators."


# ████████████████████████████████████████████████████████████████████
# ██ FINANCIAL TOOLS
# ████████████████████████████████████████████████████████████████████

def percentage_change(old_value: float, new_value: float) -> str:
    """Calculate the percentage change between two values.

    Exists as a named tool because the LLM frequently constructs the wrong
    expression (new/old*100 instead of (new-old)/old*100) when using the
    generic calculator for this operation.
    """
    try:
        old_value = float(old_value)
        new_value = float(new_value)
        if old_value == 0:
            return "Cannot calculate percentage change from a base value of zero."
        change = ((new_value - old_value) / abs(old_value)) * 100
        sign = "+" if change >= 0 else ""
        return (
            f"Percentage change from {old_value:,.2f} to {new_value:,.2f} "
            f"= {sign}{change:.2f}%."
        )
    except (TypeError, ValueError):
        return "Could not calculate percentage change — values must be numbers."


def prorated_amount(full_amount: float, days_used: float, total_days: float) -> str:
    """Calculate a prorated (partial-period) amount based on days.

    Handles partial-period billing, refunds, or allocations common in contracts.
    Named tool prevents the LLM from constructing the wrong fraction.
    """
    try:
        full_amount = float(full_amount)
        days_used = float(days_used)
        total_days = float(total_days)
        if total_days <= 0:
            return "Cannot prorate — total_days must be greater than zero."
        if days_used < 0 or days_used > total_days:
            return f"days_used ({days_used}) must be between 0 and total_days ({total_days})."
        result = full_amount * (days_used / total_days)
        return (
            f"Prorated amount: {result:,.2f} "
            f"({days_used:.0f} of {total_days:.0f} days used from {full_amount:,.2f})."
        )
    except (TypeError, ValueError):
        return "Could not calculate prorated amount — all inputs must be numbers."


def amount_after_rate(base_amount: float, rate_percent: float) -> str:
    """Apply a percentage rate on top of a base amount (tax, VAT, markup, fee).

    Named tool prevents the LLM from getting the decimal conversion wrong
    (e.g. multiplying by 15 instead of 1.15 for a 15% rate).
    """
    try:
        base_amount = float(base_amount)
        rate_percent = float(rate_percent)
        if rate_percent < 0:
            return "rate_percent must be non-negative. For a reduction use amount_after_discount."
        result = base_amount * (1 + rate_percent / 100)
        return (
            f"Total after {rate_percent:.2f}% rate applied to {base_amount:,.2f} "
            f"= {result:,.2f}."
        )
    except (TypeError, ValueError):
        return "Could not apply rate — both inputs must be numbers."


def amount_after_discount(base_amount: float, discount_percent: float) -> str:
    """Apply a percentage discount to a base amount (sale price, reduction, rebate).

    Named tool prevents the LLM from adding instead of subtracting the discount.
    """
    try:
        base_amount = float(base_amount)
        discount_percent = float(discount_percent)
        if not (0 <= discount_percent <= 100):
            return f"discount_percent ({discount_percent}) must be between 0 and 100."
        result = base_amount * (1 - discount_percent / 100)
        return (
            f"Amount after {discount_percent:.2f}% discount on {base_amount:,.2f} "
            f"= {result:,.2f}."
        )
    except (TypeError, ValueError):
        return "Could not apply discount — both inputs must be numbers."


# ████████████████████████████████████████████████████████████████████
# ██ TOOL REGISTRY
# ████████████████████████████████████████████████████████████████████

# Maps tool name → callable. Used by the executor in llm.py to dispatch calls.
TOOL_FUNCTIONS: dict[str, callable] = {
    "days_until": days_until,
    "get_weekday": get_weekday,
    "calculator": calculator,
    "percentage_change": percentage_change,
    "prorated_amount": prorated_amount,
    "amount_after_rate": amount_after_rate,
    "amount_after_discount": amount_after_discount,
}

# OpenAI-compatible tool schemas sent to the Groq API alongside the prompt.
# The model uses these descriptions to decide which tool to call and what to pass.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "days_until",
            "description": (
                "Calculate the number of days from today until a specific date "
                "found in the document. Use this when the user asks how many days "
                "remain, how long until something happens, or similar questions "
                "about time remaining. The date must come from the document context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_string": {
                        "type": "string",
                        "description": "The target date exactly as it appears in the document (e.g. '2032-02-06', 'February 6, 2032', '6 February 2032'). Do not reformat it.",
                    }
                },
                "required": ["date_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weekday",
            "description": (
                "Return the day of the week for a specific date found in the document. "
                "Use this when the user asks what day of the week a date falls on. "
                "The date must come from the document context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_string": {
                        "type": "string",
                        "description": "The date exactly as it appears in the document (e.g. '2032-02-06', 'February 6, 2032', '6 February 2032'). Do not reformat it.",
                    }
                },
                "required": ["date_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a simple arithmetic expression derived from values in the document. "
                "Use this when the user asks to calculate, compute, or perform arithmetic "
                "on numbers mentioned in the document. "
                "Supports: + - * / ** and parentheses only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A safe arithmetic expression using only numbers and operators, e.g. '12500 * 1.15'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "percentage_change",
            "description": (
                "Calculate the percentage change between two numeric values from the document. "
                "Use this when the user asks about increase, decrease, growth, or percentage difference "
                "between two amounts. Do NOT use the calculator for this — use this tool instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_value": {
                        "type": "number",
                        "description": "The original or starting value, extracted from the document.",
                    },
                    "new_value": {
                        "type": "number",
                        "description": "The new or final value, extracted from the document.",
                    },
                },
                "required": ["old_value", "new_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prorated_amount",
            "description": (
                "Calculate a prorated (partial-period) amount based on days used out of total days. "
                "Use this for partial-period billing, refunds, or allocations mentioned in contracts. "
                "Example: monthly fee prorated for days remaining in a period."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "full_amount": {
                        "type": "number",
                        "description": "The full amount for the complete period, extracted from the document.",
                    },
                    "days_used": {
                        "type": "number",
                        "description": "The number of days used or elapsed.",
                    },
                    "total_days": {
                        "type": "number",
                        "description": "The total number of days in the full period.",
                    },
                },
                "required": ["full_amount", "days_used", "total_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "amount_after_rate",
            "description": (
                "Calculate the total after adding a percentage rate to a base amount. "
                "Use this for tax, VAT, markup, or fee calculations where the rate is added ON TOP of the base. "
                "Example: price + 15% VAT, cost + 10% markup. "
                "Do NOT use the calculator for this — use this tool instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "base_amount": {
                        "type": "number",
                        "description": "The original amount before the rate is applied, extracted from the document.",
                    },
                    "rate_percent": {
                        "type": "number",
                        "description": "The percentage rate to add (e.g. 15 for 15%). Do NOT pass 0.15 — pass 15.",
                    },
                },
                "required": ["base_amount", "rate_percent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "amount_after_discount",
            "description": (
                "Calculate the final price after subtracting a percentage discount from a base amount. "
                "Use this for discounts, reductions, rebates, or sale prices. "
                "Example: original price minus 20% discount. "
                "Do NOT use the calculator for this — use this tool instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "base_amount": {
                        "type": "number",
                        "description": "The original amount before the discount, extracted from the document.",
                    },
                    "discount_percent": {
                        "type": "number",
                        "description": "The discount percentage (e.g. 20 for 20%). Do NOT pass 0.20 — pass 20.",
                    },
                },
                "required": ["base_amount", "discount_percent"],
            },
        },
    },
]
