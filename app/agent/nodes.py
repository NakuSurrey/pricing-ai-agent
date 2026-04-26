"""langgraph nodes — each one reads AgentState, does one job, returns a partial update.

node order in the graph:
  input_guardrail -> intent -> tool_call -> compose -> output_guardrail
"""

from __future__ import annotations

import json
import os
from typing import Any

from app.agent.prompts import (
    COMPOSE_SYSTEM_PROMPT,
    COMPOSE_USER_TEMPLATE,
    INPUT_REFUSAL,
    INTENT_SYSTEM_PROMPT,
    OUTPUT_REFUSAL,
    PRICING_EXTRACTION_SYSTEM_PROMPT,
)
from app.guardrails.input_filter import check_input
from app.guardrails.output_filter import check_output
from app.logging_config import logger
from app.schemas import AgentState, PolicyChunk, PricingRequest
from app.tools.policy_lookup import lookup as policy_lookup
from app.tools.pricing_api import PricingError, get_quote

# defaults used when the user's question doesn't supply every pricing slot
PRICING_DEFAULTS: dict[str, Any] = {
    "flood_zone": "standard",
    "claims_last_5_years": "0",
    "security": "standard_locks",
    "property_age_band": "1945_to_2000",
    "policy_type": "standard",
}


# ---------------------------------------------------------------------------
# llm helper — one place that knows how to call groq
# ---------------------------------------------------------------------------


def _call_groq_json(system_prompt: str, user_prompt: str) -> dict | None:
    """call groq with json response format. returns parsed dict or None on failure."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — llm node will return None")
        return None

    try:
        from groq import Groq
    except ImportError:
        logger.warning("groq sdk not installed")
        return None

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        logger.error("groq json call failed: {}", e)
        return None


def _call_groq_text(system_prompt: str, user_prompt: str) -> str | None:
    """call groq for free-text output. returns string or None on failure."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error("groq text call failed: {}", e)
        return None


# ---------------------------------------------------------------------------
# node 1 — input guardrail
# ---------------------------------------------------------------------------


def input_guardrail_node(state: AgentState) -> dict:
    """check the question against regex rules. block early if anything fires."""
    result = check_input(state.question)
    update: dict[str, Any] = {"input_guardrail": result}

    if result.verdict == "block":
        update["refused"] = True
        update["refusal_reason"] = "; ".join(result.reasons) or "input blocked"
        update["final_answer"] = INPUT_REFUSAL
        logger.info("input guardrail blocked the question")

    return update


# ---------------------------------------------------------------------------
# node 2 — intent classification
# ---------------------------------------------------------------------------


def intent_node(state: AgentState) -> dict:
    """ask the llm what the user wants. returns one of four intents."""
    if state.refused:
        # already blocked upstream — pass through
        return {}

    response = _call_groq_json(INTENT_SYSTEM_PROMPT, state.question)

    if response is None:
        # llm unavailable — keyword-based fallback so the graph still runs in dev
        q = state.question.lower()
        if any(w in q for w in ("price", "cost", "quote", "premium", "how much")):
            intent = "pricing"
        elif any(w in q for w in ("cover", "covered", "claim", "exclusion", "policy")):
            intent = "policy"
        else:
            intent = "policy"  # safe default — try to answer from policy docs
        logger.info("intent fallback (no llm) -> {}", intent)
        return {"intent": intent}

    intent = response.get("intent", "policy")
    if intent not in ("pricing", "policy", "both", "out_of_scope"):
        intent = "policy"

    logger.info("intent classified as {}", intent)
    return {"intent": intent}


# ---------------------------------------------------------------------------
# node 3 — tool call (pricing and/or policy lookup)
# ---------------------------------------------------------------------------


def _build_pricing_request(question: str) -> PricingRequest | None:
    """ask the llm to extract pricing slots, fill defaults, build the request."""
    response = _call_groq_json(PRICING_EXTRACTION_SYSTEM_PROMPT, question)
    if response is None:
        return None

    # required fields — if any of these are null we cannot price
    required = ("property_type", "bedrooms", "risk_profile")
    if not all(response.get(k) not in (None, "null", "") for k in required):
        logger.info("pricing extraction missing required slots: {}", response)
        return None

    payload: dict[str, Any] = {**PRICING_DEFAULTS}
    for key in (
        "property_type",
        "bedrooms",
        "risk_profile",
        "flood_zone",
        "claims_last_5_years",
        "security",
        "property_age_band",
        "policy_type",
    ):
        val = response.get(key)
        if val not in (None, "null", ""):
            payload[key] = val

    # bedrooms comes back as a string from the llm — coerce to int
    try:
        payload["bedrooms"] = int(payload["bedrooms"])
    except (TypeError, ValueError):
        return None

    try:
        return PricingRequest(**payload)
    except Exception as e:
        logger.warning("pricing request validation failed: {}", e)
        return None


def tool_call_node(state: AgentState) -> dict:
    """run pricing tool, policy lookup, or both — depending on intent."""
    if state.refused:
        return {}

    update: dict[str, Any] = {}

    if state.intent == "out_of_scope":
        # nothing to fetch — compose node will produce a polite redirect
        return {}

    if state.intent in ("pricing", "both"):
        pricing_req = _build_pricing_request(state.question)
        if pricing_req is not None:
            update["pricing_request"] = pricing_req
            try:
                update["quote"] = get_quote(pricing_req)
            except PricingError as e:
                logger.warning("pricing tool returned no row: {}", e)

    if state.intent in ("policy", "both", "pricing"):
        # pricing questions still benefit from policy context (e.g. exclusions)
        result = policy_lookup(state.question, top_k=4)
        update["policy_chunks"] = result.chunks

    return update


# ---------------------------------------------------------------------------
# node 4 — compose the answer
# ---------------------------------------------------------------------------


def _format_quote_block(state: AgentState) -> str:
    if state.quote is None:
        return "(no quote available)"
    q = state.quote
    lines = [
        f"Base premium: £{q.base_annual_premium_gbp:.2f}",
        f"Final indicative premium: £{q.final_annual_premium_gbp:.2f}",
        "Modifiers applied:",
    ]
    for m in q.modifiers_applied:
        lines.append(f"  - {m.name} = {m.value} (x{m.multiplier})")
    return "\n".join(lines)


def _format_context_block(chunks: list[PolicyChunk]) -> str:
    if not chunks:
        return "(no policy context retrieved)"
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {c.source_file}) {c.section_title}\n{c.text}")
    return "\n\n".join(parts)


def compose_node(state: AgentState) -> dict:
    """produce the draft answer — llm call when available, deterministic otherwise."""
    if state.refused:
        return {}

    if state.intent == "out_of_scope":
        return {
            "draft_answer": (
                "This assistant only covers UK home insurance — pricing and policy "
                "questions. For other topics please contact the relevant provider."
            )
        }

    user_prompt = COMPOSE_USER_TEMPLATE.format(
        question=state.question,
        quote_block=_format_quote_block(state),
        context_block=_format_context_block(state.policy_chunks),
    )

    answer = _call_groq_text(COMPOSE_SYSTEM_PROMPT, user_prompt)

    if answer is None:
        # no llm — build a deterministic stub that still passes guardrails
        parts = [
            "Here is what we found based on your question.",
        ]
        if state.quote is not None:
            parts.append(
                f"Indicative estimate: £{state.quote.final_annual_premium_gbp:.2f} "
                "per year. This is not a binding quote — actual price is subject to "
                "underwriting."
            )
        if state.policy_chunks:
            parts.append("Relevant policy notes:")
            for c in state.policy_chunks[:2]:
                parts.append(f"- (source: {c.source_file}) {c.section_title}")
        parts.append("Let me know if you'd like more detail on any point.")
        answer = "\n\n".join(parts)

    return {"draft_answer": answer}


# ---------------------------------------------------------------------------
# node 5 — output guardrail
# ---------------------------------------------------------------------------


def output_guardrail_node(state: AgentState) -> dict:
    """run FCA rubric over the draft. on second failure, return safe refusal."""
    if state.refused:
        # input guardrail already produced final_answer — keep it
        return {"final_answer": state.final_answer or INPUT_REFUSAL}

    draft = state.draft_answer or ""
    result = check_output(draft, state.policy_chunks)
    update: dict[str, Any] = {"output_guardrail": result}

    if result.verdict == "allow":
        update["final_answer"] = draft
        return update

    # blocked — one retry allowed per ARCHITECTURE.md
    if state.retry_count < 1:
        logger.info("output blocked — will retry compose with stricter prompt")
        update["retry_count"] = state.retry_count + 1
        update["draft_answer"] = None  # force compose node to rerun
        return update

    # second failure — safe refusal, do not show the bad draft
    logger.warning("output blocked twice — returning safe refusal")
    update["refused"] = True
    update["refusal_reason"] = "; ".join(result.reasons) or "output failed FCA rubric"
    update["final_answer"] = OUTPUT_REFUSAL
    return update
