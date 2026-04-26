"""langgraph state machine — wires the five nodes with the output retry loop."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    compose_node,
    input_guardrail_node,
    intent_node,
    output_guardrail_node,
    tool_call_node,
)
from app.logging_config import logger
from app.schemas import AgentState

# ---------------------------------------------------------------------------
# routing functions — return the name of the next node based on state
# ---------------------------------------------------------------------------


def _route_after_input_guardrail(state: AgentState) -> Literal["intent", "end"]:
    """if the input was blocked, skip everything and finish."""
    if state.refused:
        return "end"
    return "intent"


def _route_after_output_guardrail(
    state: AgentState,
) -> Literal["compose", "end"]:
    """blocked output with a retry left -> back to compose. otherwise end."""
    if state.refused:
        return "end"
    if state.final_answer is None and state.retry_count >= 1:
        # safety net — should not happen, but never loop forever
        return "end"
    if state.final_answer is None:
        return "compose"
    return "end"


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def build_graph():
    """build and compile the langgraph state machine. cached — only built once."""
    logger.info("building agent graph")

    builder = StateGraph(AgentState)

    builder.add_node("input_guardrail_check", input_guardrail_node)
    builder.add_node("intent_classify", intent_node)
    builder.add_node("tool_call", tool_call_node)
    builder.add_node("compose", compose_node)
    builder.add_node("output_guardrail_check", output_guardrail_node)

    builder.set_entry_point("input_guardrail_check")

    builder.add_conditional_edges(
        "input_guardrail_check",
        _route_after_input_guardrail,
        {"intent": "intent_classify", "end": END},
    )
    builder.add_edge("intent_classify", "tool_call")
    builder.add_edge("tool_call", "compose")
    builder.add_edge("compose", "output_guardrail_check")
    builder.add_conditional_edges(
        "output_guardrail_check",
        _route_after_output_guardrail,
        {"compose": "compose", "end": END},
    )

    return builder.compile()


def run(question: str) -> AgentState:
    """one-shot helper — run the graph on a question, return the final state."""
    graph = build_graph()
    initial = AgentState(question=question)
    # langgraph returns a dict; rebuild the model so callers get a typed object
    final_dict = graph.invoke(initial)
    return AgentState(**final_dict)


if __name__ == "__main__":
    # quick manual check — only runs when this file is executed directly
    import sys

    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What does my standard policy cover for flood damage?"
    )
    final = run(question)
    print("=" * 60)
    print("QUESTION :", question)
    print("INTENT   :", final.intent)
    print("REFUSED  :", final.refused)
    if final.quote:
        print("QUOTE    :", f"£{final.quote.final_annual_premium_gbp:.2f}")
    print("CHUNKS   :", len(final.policy_chunks))
    print("-" * 60)
    print("ANSWER   :")
    print(final.final_answer)
    print("=" * 60)
