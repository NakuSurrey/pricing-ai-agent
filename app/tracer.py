"""tracer — one json line per request, written to logs/trace.jsonl.

each line captures the question, the routing path through the graph,
guardrail verdicts, tool calls, latency, and the final answer length.

format choice: jsonl (one json object per line) — easy to grep, easy to load
into a notebook for analysis, easy to ship to a log collector later.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.logging_config import logger
from app.schemas import AgentState

# trace file lives next to the project root, not inside app/
TRACE_PATH = Path(__file__).resolve().parents[1] / "logs" / "trace.jsonl"


def _ensure_log_dir() -> None:
    """make sure logs/ exists before any write — idempotent."""
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _state_to_trace(state: AgentState) -> dict[str, Any]:
    """pull the fields worth keeping out of AgentState — drop heavy text fields."""
    return {
        "intent": state.intent,
        "refused": state.refused,
        "refusal_reason": state.refusal_reason,
        "retry_count": state.retry_count,
        "input_guardrail_verdict": (
            state.input_guardrail.verdict if state.input_guardrail else None
        ),
        "input_guardrail_hits": (
            state.input_guardrail.rule_hits if state.input_guardrail else []
        ),
        "output_guardrail_verdict": (
            state.output_guardrail.verdict if state.output_guardrail else None
        ),
        "output_guardrail_hits": (
            state.output_guardrail.rule_hits if state.output_guardrail else []
        ),
        "had_quote": state.quote is not None,
        "policy_chunks_count": len(state.policy_chunks),
        "answer_length": len(state.final_answer or ""),
    }


@contextmanager
def trace_request(question: str, session_id: str | None = None):
    """
    context manager — wraps one request, writes one trace line on exit.
    use:
        with trace_request(q) as t:
            final_state = run_graph(q)
            t["state"] = final_state
    """
    trace_id = str(uuid.uuid4())
    started = time.perf_counter()
    record: dict[str, Any] = {
        "trace_id": trace_id,
        "session_id": session_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "question_preview": question[:200],
        "question_length": len(question),
        "state": None,  # caller sets this
        "error": None,  # caller sets this on failure
    }

    try:
        yield record
    except Exception as e:
        # capture the error so the trace line still goes out
        record["error"] = repr(e)
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)

        # flatten the AgentState if the caller stored it
        state = record.pop("state", None)
        if isinstance(state, AgentState):
            record.update(_state_to_trace(state))

        try:
            _ensure_log_dir()
            with TRACE_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # never let a logging failure crash the request — log and move on
            logger.error("failed to write trace line: {}", e)


if __name__ == "__main__":
    # quick manual check — only runs when this file is executed directly
    from app.schemas import GuardrailResult

    fake_state = AgentState(
        question="test",
        intent="policy",
        input_guardrail=GuardrailResult(verdict="allow"),
        output_guardrail=GuardrailResult(verdict="allow"),
        final_answer="ok",
    )
    with trace_request("test question", session_id="sandbox") as t:
        t["state"] = fake_state
    print(f"wrote a trace line to {TRACE_PATH}")
