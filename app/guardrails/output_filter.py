"""output guardrail — runs after the agent drafts an answer.

what it checks (the FCA Consumer Duty rubric, simplified):
  1. clear     — answer is understandable, no undefined jargon
  2. fair      — no absolutes, no guaranteed-cheapest claims
  3. not misleading — no firm price stated as binding, includes 'indicative' caveat
  4. grounded  — claims about policy match the retrieved context (no hallucination)

two paths:
  - if GROQ_API_KEY is set -> ask the llm to judge against the rubric (json out)
  - if not -> fall back to deterministic regex rules (good enough for dev/tests)

the deterministic path is what runs in pytest. the llm path is what runs in prod.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from app.logging_config import logger
from app.schemas import GuardrailResult, PolicyChunk


# ---------------------------------------------------------------------------
# deterministic rules — used when no llm is available
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputRule:
    name: str
    pattern: re.Pattern
    reason: str


ABSOLUTES_RULES: tuple[OutputRule, ...] = (
    OutputRule(
        name="guaranteed_language",
        pattern=re.compile(
            r"\b(guaranteed|always\s+(cover|covered|covers|approved|refused|accepted)|"
            r"never\s+(refused|denied|rejected)|no\s+exceptions|100%\s+approved)\b",
            re.IGNORECASE,
        ),
        reason="uses absolute language banned under FCA Consumer Duty",
    ),
    OutputRule(
        name="cheapest_claim",
        pattern=re.compile(
            r"\b(cheapest|lowest\s+price|best\s+price\s+anywhere)\b", re.IGNORECASE
        ),
        reason="claims to be cheapest — not a fair, evidenced statement",
    ),
    OutputRule(
        name="binding_quote_language",
        pattern=re.compile(
            r"\b(this\s+is\s+your\s+(final|binding)\s+(price|quote))\b", re.IGNORECASE
        ),
        reason="presents an indicative price as a binding quote",
    ),
)


def _has_indicative_caveat(text: str) -> bool:
    """check if the answer mentions 'indicative' or similar caveat wording."""
    pattern = re.compile(
        r"\b(indicative|estimate|not\s+a\s+binding\s+quote|subject\s+to\s+underwriting)\b",
        re.IGNORECASE,
    )
    return bool(pattern.search(text))


def _mentions_a_price(text: str) -> bool:
    """detect a £-prefixed number — used to decide if a caveat is even required."""
    return bool(re.search(r"£\s?\d", text))


def _check_deterministic(answer: str) -> GuardrailResult:
    """no-llm path — runs the regex rules and the caveat check."""
    reasons: list[str] = []
    hits: list[str] = []

    for rule in ABSOLUTES_RULES:
        if rule.pattern.search(answer):
            reasons.append(rule.reason)
            hits.append(rule.name)

    if _mentions_a_price(answer) and not _has_indicative_caveat(answer):
        reasons.append("mentions a price without an 'indicative estimate' caveat")
        hits.append("missing_caveat")

    if hits:
        return GuardrailResult(verdict="block", reasons=reasons, rule_hits=hits)
    return GuardrailResult(verdict="allow", reasons=[], rule_hits=[])


# ---------------------------------------------------------------------------
# llm-as-judge path
# ---------------------------------------------------------------------------


JUDGE_SYSTEM_PROMPT = """You are an FCA Consumer Duty compliance reviewer.

Score the assistant's draft answer against this rubric. For each dimension,
return "pass" or "fail" with a one-line reason.

Dimensions:
  1. clear           — language is plain, jargon defined, sentences short
  2. fair            — no absolutes ("guaranteed", "always", "cheapest")
  3. not_misleading  — any price is clearly indicative, never a binding quote
  4. grounded        — every policy claim is supported by the provided context

Output JSON only, this exact shape:
{
  "clear":          {"verdict": "pass|fail", "reason": "..."},
  "fair":           {"verdict": "pass|fail", "reason": "..."},
  "not_misleading": {"verdict": "pass|fail", "reason": "..."},
  "grounded":       {"verdict": "pass|fail", "reason": "..."}
}
"""


def _format_context(chunks: list[PolicyChunk]) -> str:
    """flatten retrieved chunks into a single string the judge can read."""
    if not chunks:
        return "(no policy context retrieved)"
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] from {c.source_file} - {c.section_title}\n{c.text}")
    return "\n\n".join(parts)


def _call_groq_judge(answer: str, context: str) -> dict | None:
    """
    call groq with the rubric prompt. returns parsed json, or None if anything fails.
    failure is silent — caller falls back to the deterministic path.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        # importing inside the function keeps groq a soft dependency for tests
        from groq import Groq
    except ImportError:
        logger.warning("groq sdk not installed — skipping llm judge")
        return None

    user_prompt = (
        f"DRAFT ANSWER:\n{answer}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        "Return JSON only."
    )

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        # never let the judge crash the request — log and fall back
        logger.warning("groq judge call failed: {}", e)
        return None


def _verdict_from_judge(judge_json: dict) -> GuardrailResult:
    """convert the judge's per-dimension verdicts into a single GuardrailResult."""
    reasons: list[str] = []
    hits: list[str] = []

    for dim in ("clear", "fair", "not_misleading", "grounded"):
        d = judge_json.get(dim, {})
        if d.get("verdict", "").lower() == "fail":
            hits.append(dim)
            reasons.append(f"{dim}: {d.get('reason', 'no reason given')}")

    if hits:
        return GuardrailResult(verdict="block", reasons=reasons, rule_hits=hits)
    return GuardrailResult(verdict="allow", reasons=[], rule_hits=[])


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def check_output(answer: str, policy_chunks: list[PolicyChunk] | None = None) -> GuardrailResult:
    """
    public entry point. tries llm judge first; falls back to deterministic rules.
    deterministic rules also run on top of an llm 'allow' as a belt-and-braces check.
    """
    if not answer or not answer.strip():
        return GuardrailResult(
            verdict="block",
            reasons=["empty answer"],
            rule_hits=["empty_output"],
        )

    chunks = policy_chunks or []
    judge_json = _call_groq_judge(answer, _format_context(chunks))

    if judge_json is not None:
        llm_result = _verdict_from_judge(judge_json)
        if llm_result.verdict == "block":
            logger.warning("output blocked by llm judge — hits={}", llm_result.rule_hits)
            return llm_result
        # llm said allow — still run the deterministic rules as a hard backstop
        det_result = _check_deterministic(answer)
        if det_result.verdict == "block":
            logger.warning(
                "output blocked by deterministic rules after llm allow — hits={}",
                det_result.rule_hits,
            )
        return det_result

    # no llm available — deterministic only
    result = _check_deterministic(answer)
    if result.verdict == "block":
        logger.warning("output blocked by deterministic rules — hits={}", result.rule_hits)
    return result


if __name__ == "__main__":
    # quick manual check — only runs when this file is executed directly
    samples = [
        "Based on the policy, your indicative estimate is around £420 per year. "
        "This is not a binding quote — actual price is subject to underwriting.",
        "Your final binding quote is £420. This is the cheapest price anywhere, guaranteed.",
        "We always cover flood damage, no exceptions.",
    ]
    for s in samples:
        r = check_output(s, [])
        print(f"verdict={r.verdict:5s} hits={r.rule_hits}")
