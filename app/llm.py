"""Helpers for calling a large language model."""

import json
import logging
import os
import re
import uuid
from typing import Sequence

from openai import OpenAI

from app.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

_log = logging.getLogger(__name__)

# ████████████████████████████████████████████████████████████████████
# ██ CLIENT
# ████████████████████████████████████████████████████████████████████

# Set GROQ_API_KEY in your .env file. Get a free key at console.groq.com
_MODEL = "llama-3.1-8b-instant"
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Create the Groq client once and reuse it across calls."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


# ████████████████████████████████████████████████████████████████████
# ██ PROMPT
# ████████████████████████████████████████████████████████████████████

def build_prompt(context_chunks: Sequence[str], question: str, language: str = "English") -> str:
    """Build a prompt from retrieved document chunks and the user question."""
    context = "\n\n".join(context_chunks)
    return (
        f"You are a helpful assistant that answers questions based only on the provided document context.\n\n"
        f"The answer may appear in any part of the context, including earlier or later sections. "
        f"Carefully consider all provided context before answering.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the context above. If the answer is not in the context, say so clearly.\n"
        f"Always respond in {language}. "
        f"Keep all proper nouns, character names, place names, and titles in their original English spelling — do not transliterate them. "
        f"Format your answer clearly: use a numbered list or bullet points when listing multiple items, and write in complete sentences. "
        f"{_SCOPE_INSTRUCTION}"
    )


# ████████████████████████████████████████████████████████████████████
# ██ GENERATION
# ████████████████████████████████████████████████████████████████████

def generate_answer(
    context_chunks: Sequence[str],
    question: str,
    language: str = "English",
    history: list[dict] | None = None,
) -> str:
    """Send context, optional conversation history, and a question to the LLM."""
    recent_history = (history or [])[-_MAX_HISTORY_TURNS:]

    if not recent_history:
        # Single turn — existing behaviour unchanged.
        messages = [{"role": "user", "content": build_prompt(context_chunks, question, language)}]
    else:
        # Multi-turn — context goes in a system message once; history follows as
        # alternating user/assistant turns so the model can resolve follow-ups.
        context = "\n\n".join(context_chunks)
        system_content = (
            f"You are a helpful assistant that answers questions based only on the provided document context.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer based only on the context above. If the answer is not in the context, say so clearly. "
            f"Keep all proper nouns, character names, place names, and titles in their original English spelling — do not transliterate them. "
            f"Format your answer clearly: use a numbered list or bullet points when listing multiple items, and write in complete sentences. "
            f"Always respond in {language}."
        )
        messages: list[dict] = [{"role": "system", "content": system_content}]
        for turn in recent_history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user", "content": question})

    completion = _get_client().chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        messages=messages,
    )
    return completion.choices[0].message.content


# ████████████████████████████████████████████████████████████████████
# ██ TOOL-AUGMENTED GENERATION
# ████████████████████████████████████████████████████████████████████

# Appended to the system message immediately before API call 2.
# Prevents the 8b model from re-reasoning over a value the tool already computed.
_VERBATIM_INSTRUCTION = (
    "\n\nCRITICAL INSTRUCTION FOR THIS RESPONSE: "
    "The tool result(s) above are exact and authoritative. "
    "You MUST include the exact numerical value(s) and date(s) from the tool result verbatim. "
    "Do NOT recompute, approximate, round, convert to other units, or paraphrase any number or date. "
    "Your only role is to frame the tool result naturally using the document context. "
    "Do not verify or recalculate — trust the tool output completely."
)

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# Tools whose output IS the complete answer — no LLM framing needed.
# For these, API call 2 is skipped entirely to eliminate drift risk.
_PURE_DETERMINISTIC_TOOLS = frozenset({"days_until", "get_weekday"})

# Added to every system message to prevent unsolicited elaboration.
_SCOPE_INSTRUCTION = (
    "Answer only what was specifically asked. "
    "Do not add unsolicited calculations, alternative scenarios, or extra information the user did not request."
)


def _extract_key_value(tool_name: str, tool_result: str) -> str | None:
    """Extract the primary verifiable value from a tool result string.

    Used by the validation layer to confirm the LLM included the deterministic
    result in its final response without modification.
    """
    if tool_name == "days_until":
        # e.g. "2,080 days remain until..." → "2080"
        match = re.search(r"([\d,]+)\s+day", tool_result)
        return match.group(1).replace(",", "") if match else None
    if tool_name == "get_weekday":
        # e.g. "...falls on a Thursday." → "Thursday"
        for day in _WEEKDAYS:
            if day in tool_result:
                return day
        return None
    if tool_name == "calculator":
        # e.g. "Result of ... = 14375" → "14375"
        match = re.search(r"=\s*([\d.]+)", tool_result)
        return match.group(1) if match else None
    if tool_name == "percentage_change":
        # e.g. "...= +20.00%." → "20.00"
        match = re.search(r"=\s*[+-]?([\d.]+)%", tool_result)
        return match.group(1) if match else None
    if tool_name == "prorated_amount":
        # e.g. "Prorated amount: 500.00 (...)" → "500.00"
        match = re.search(r"Prorated amount:\s*([\d,.]+)", tool_result)
        return match.group(1).replace(",", "") if match else None
    if tool_name in ("amount_after_rate", "amount_after_discount"):
        # e.g. "Total after 15.00% rate applied to 1,000.00 = 1,150.00." → "1150.00"
        match = re.search(r"=\s*([\d,.]+)", tool_result)
        return match.group(1).replace(",", "") if match else None
    return None


def _key_value_in_response(key_value: str, response: str) -> bool:
    """Return True if key_value appears in response after normalizing number formatting.

    Strips digit-group commas so "2,080" matches "2080" and vice versa.
    """
    normalize = lambda s: re.sub(r"(\d),(\d)", r"\1\2", s)
    return normalize(key_value).lower() in normalize(response).lower()

def generate_answer_with_tools(
    context_chunks: Sequence[str],
    question: str,
    language: str = "English",
    history: list[dict] | None = None,
) -> str:
    """Run a tool-calling loop then generate a final grounded answer.

    Hardened against post-tool arithmetic drift via three layers:
    - _VERBATIM_INSTRUCTION injected into the system message before API call 2.
    - Validation check that the key numerical/named value appears in the response.
    - Deterministic fallback: if validation fails, the tool result is returned directly.
    """
    cid = uuid.uuid4().hex[:8]
    _log.debug("[%s] ℹ️  [LLM]: Tool-augmented generation | question: %.80r", cid, question)
    _log.debug("[%s] ℹ️  [LLM]: Context chunks: %d", cid, len(context_chunks))

    recent_history = (history or [])[-_MAX_HISTORY_TURNS:]
    context = "\n\n".join(context_chunks)

    system_content = (
        f"You are a helpful assistant that answers questions based only on the provided document context.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer based only on the context above. If the answer is not in the context, say so clearly. "
        f"Keep all proper nouns, character names, place names, and titles in their original English spelling — do not transliterate them. "
        f"Format your answer clearly: use a numbered list or bullet points when listing multiple items, and write in complete sentences. "
        f"Always respond in {language}.\n\n"
        f"You have access to tools for date calculations and arithmetic. "
        f"Only call a tool if the user's question requires a computation that cannot be answered by reading the document text alone. "
        f"Tool inputs must be values found in the document context — do not invent values.\n\n"
        f"{_SCOPE_INSTRUCTION}"
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]
    for turn in recent_history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})

    # ── API call 1: LLM decides whether to call a tool ──────────────
    _log.debug("[%s] ℹ️  [LLM]: API call 1 — tool selection", cid)
    completion = _get_client().chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
    )
    response_message = completion.choices[0].message

    if not response_message.tool_calls:
        _log.debug("[%s] ℹ️  [LLM]: No tool called — returning direct answer", cid)
        return response_message.content

    # ── Execute each tool call ───────────────────────────────────────
    messages.append(response_message)
    executed: list[tuple[str, str]] = []  # (tool_name, tool_result) for validation

    for tool_call in response_message.tool_calls:
        name = tool_call.function.name
        args_str = tool_call.function.arguments
        tool_fn = TOOL_FUNCTIONS.get(name)

        _log.debug("[%s] ✅ [TOOL-CALL]: tool=%r args=%s", cid, name, args_str)

        if tool_fn is None:
            tool_result = f"Tool '{name}' is not available."
            _log.warning("[%s] ⚠️  [TOOL-CALL]: Unknown tool requested: %r", cid, name)
        else:
            try:
                args = json.loads(args_str)
                tool_result = tool_fn(**args)
                _log.debug("[%s] ✅ [TOOL-RESULT]: %r → %r", cid, name, tool_result)
            except Exception as exc:
                tool_result = f"Tool '{name}' failed: {exc}"
                _log.warning("[%s] ⚠️  [TOOL-RESULT]: %r failed: %s", cid, name, exc)

        executed.append((name, tool_result))
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        })

    # ── Pure deterministic bypass ────────────────────────────────────
    # If every called tool is in _PURE_DETERMINISTIC_TOOLS, the tool output
    # IS the complete answer. Skip API call 2 entirely — no LLM framing,
    # no drift risk.
    if executed and all(name in _PURE_DETERMINISTIC_TOOLS for name, _ in executed):
        result = " ".join(result for _, result in executed)
        _log.debug("[%s] ✅ [LLM]: Pure deterministic bypass — returning tool result directly", cid)
        return result

    # ── Harden system message before API call 2 ─────────────────────
    # Appends the verbatim instruction to prevent post-tool arithmetic drift.
    messages[0]["content"] += _VERBATIM_INSTRUCTION

    # ── API call 2: LLM generates final answer with tool results ─────
    _log.debug("[%s] ℹ️  [LLM]: API call 2 — final answer", cid)
    final_completion = _get_client().chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        messages=messages,
    )
    response = final_completion.choices[0].message.content

    # ── Validation: confirm key value appears in the response ────────
    for tool_name, tool_result in executed:
        key_value = _extract_key_value(tool_name, tool_result)
        if key_value is None:
            continue
        if _key_value_in_response(key_value, response):
            _log.debug("[%s] ✅ [VALIDATION]: Key value %r found in response", cid, key_value)
        else:
            _log.warning(
                "[%s] ⚠️  [VALIDATION]: Key value %r NOT in response — activating fallback",
                cid, key_value,
            )
            return tool_result

    _log.debug("[%s] ✅ [LLM]: Final answer generated | %d chars", cid, len(response))
    return response


# ████████████████████████████████████████████████████████████████████
# ██ SUMMARIZATION
# ████████████████████████████████████████████████████████████████████

# Each prompt uses {context} as the only placeholder.
# Grounding rules and output structure are tailored per document type.
_SUMMARY_PROMPTS: dict[str, str] = {

    # ── STORY ────────────────────────────────────────────────────────
    # Follows the narrative arc: overview → events → characters → theme.
    # Flexible: skips sections that don't apply to the document.
    "Story": (
        "You are a helpful assistant. Summarize the following story using only "
        "the information provided.\n\n"
        "Use this structure where applicable:\n"
        "- Overview: 2–3 sentences describing the overall story.\n"
        "- Key Events: The most important events in the order they happen.\n"
        "- Characters: The main characters and what happens to them.\n"
        "- Main Message: One sentence stating the central theme or lesson.\n\n"
        "If the document does not fully follow a narrative structure, adapt your "
        "summary to what is available and keep it logical. Skip any section that "
        "does not apply.\n\n"
        "Use ONLY the provided context. Do NOT add or assume information. "
        "If something is unclear or missing, do not guess — simply omit that point.\n\n"
        "Document:\n{context}"
    ),

    # ── ARTICLE ──────────────────────────────────────────────────────
    # Extracts the main idea, evidence, and conclusion from informational text.
    # Flexible: skips sections that don't appear in the document.
    "Article": (
        "You are a helpful assistant. Summarize the following article using only "
        "the information provided.\n\n"
        "Use this structure where applicable:\n"
        "- Main Idea: One sentence stating what the article is about.\n"
        "- Key Points: The most important points, as a short bullet list.\n"
        "- Supporting Details: Important facts, examples, or data mentioned.\n"
        "- Conclusion: What the article concludes or recommends, if stated.\n\n"
        "If the document does not follow this structure exactly, adapt your summary "
        "to what is present. Skip any section that does not apply.\n\n"
        "Use ONLY the provided context. Do NOT add or assume information. "
        "If something is unclear or missing, do not guess — simply omit that point.\n\n"
        "Document:\n{context}"
    ),

    # ── CONTRACT ─────────────────────────────────────────────────────
    # Reports facts only — no legal interpretation, no assumptions.
    # Flexible: skips sections absent from the document.
    "Contract": (
        "You are a helpful assistant. Summarize the following contract or formal "
        "document using only the information provided.\n\n"
        "Use this structure where applicable:\n"
        "- Parties Involved: Who are the parties named in this document.\n"
        "- Key Terms: Important definitions or terms as stated in the document.\n"
        "- Obligations: What each party is required to do.\n"
        "- Conditions: Any deadlines, requirements, or special conditions mentioned.\n"
        "- Important Notes: Any penalties, limitations, or special clauses stated.\n\n"
        "If a section does not appear in the document, skip it. "
        "Do not interpret legal meaning — report only what is explicitly stated.\n\n"
        "Use ONLY the provided context. Do NOT add or assume information. "
        "If something is unclear or missing, do not guess — simply omit that point.\n\n"
        "Document:\n{context}"
    ),

    # ── TECHNICAL ────────────────────────────────────────────────────
    # Surfaces concepts, mechanisms, and findings without adding jargon.
    # Flexible: skips sections not present in the document.
    "Technical": (
        "You are a helpful assistant. Summarize the following technical document "
        "using only the information provided.\n\n"
        "Use this structure where applicable:\n"
        "- Topic: What this document is about in one sentence.\n"
        "- Key Concepts: The main ideas, terms, or definitions introduced.\n"
        "- How It Works: A brief description of the main method, process, or mechanism described.\n"
        "- Findings or Conclusions: What the document concludes or recommends.\n\n"
        "If a section does not apply to this document, skip it. "
        "Write clearly and avoid unnecessary jargon.\n\n"
        "Use ONLY the provided context. Do NOT add or assume information. "
        "If something is unclear or missing, do not guess — simply omit that point.\n\n"
        "Document:\n{context}"
    ),
}


# Groq free tier: 6 000 TPM. Prompt template ≈ 150 tokens, output capped at 1 024.
# That leaves ~4 800 tokens for context → ~19 200 chars. 16 000 gives a safe buffer.
_MAX_CONTEXT_CHARS = 16_000

# Cap history to avoid exceeding the free-tier token limit on long conversations.
_MAX_HISTORY_TURNS = 5


def summarize_document(chunks: Sequence[str], doc_type: str = "Article", language: str = "English") -> str:
    """Send all document chunks to the LLM and return a summary shaped by doc_type."""
    context = "\n\n".join(chunks)
    if len(context) > _MAX_CONTEXT_CHARS:
        context = context[:_MAX_CONTEXT_CHARS]
    template = _SUMMARY_PROMPTS.get(doc_type, _SUMMARY_PROMPTS["Article"])
    prompt = template.format(context=context)
    prompt += f"\n\nAlways write your response in {language}."
    completion = _get_client().chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content
