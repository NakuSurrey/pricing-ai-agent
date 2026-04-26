"""api routes — POST /ask runs the graph, GET /health returns service status."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.agent.graph import run as run_graph
from app.logging_config import logger
from app.schemas import AskRequest, AskResponse
from app.tracer import trace_request

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """liveness probe — used by docker-compose and any platform's healthcheck."""
    return {"status": "ok", "service": "pricing-ai-agent"}


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """
    main endpoint — takes a question, runs the agent graph, returns the answer
    plus the audit trail (citations, quote, refusal info).
    every call writes one line to logs/trace.jsonl.
    """
    logger.info(
        "POST /ask received — len={} session={}",
        len(payload.question),
        payload.session_id,
    )

    try:
        with trace_request(payload.question, payload.session_id) as trace:
            final_state = run_graph(payload.question)
            trace["state"] = final_state
    except Exception as e:
        # graph blew up — never leak internals, but log them server-side
        logger.exception("graph execution failed: {}", e)
        raise HTTPException(status_code=500, detail="agent failed to process request")

    if final_state.final_answer is None:
        # graph returned without setting a final answer — should not happen, but guard it
        logger.error("graph finished with no final_answer — state={}", final_state.model_dump())
        raise HTTPException(status_code=500, detail="agent produced no answer")

    return AskResponse(
        answer=final_state.final_answer,
        citations=final_state.policy_chunks,
        quote=final_state.quote,
        refused=final_state.refused,
        refusal_reason=final_state.refusal_reason,
    )
