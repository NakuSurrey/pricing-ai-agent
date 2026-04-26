"""streamlit chat ui — talks to the fastapi /ask endpoint.

run with:
    streamlit run ui/streamlit_app.py

requires the api to be running:
    uvicorn app.main:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# config — every setting overridable via env so docker can pass real values
# ---------------------------------------------------------------------------

API_URL = os.getenv("PRICING_AGENT_API_URL", "http://127.0.0.1:8765")
ASK_ENDPOINT = f"{API_URL.rstrip('/')}/ask"
HEALTH_ENDPOINT = f"{API_URL.rstrip('/')}/health"
REQUEST_TIMEOUT_SECONDS = 60

# ---------------------------------------------------------------------------
# page setup — runs once per session
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pricing AI Agent — Home Insurance",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Pricing AI Agent")
st.caption(
    "ask about UK home insurance — policy cover, exclusions, or an indicative price. "
    "every answer includes the policy section it came from. prices shown are indicative, "
    "not binding quotes."
)

# ---------------------------------------------------------------------------
# session state — keeps the chat history across reruns
# ---------------------------------------------------------------------------

# every streamlit interaction reruns this whole script — use session_state
# to hold the chat across reruns instead of losing it every keystroke
if "messages" not in st.session_state:
    # each entry: {"role": "user"|"assistant", "content": str, "extra": dict|None}
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# sidebar — health probe + reset + tips
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("connection")
    st.code(API_URL, language="text")

    if st.button("check api health"):
        try:
            r = requests.get(HEALTH_ENDPOINT, timeout=5)
            if r.status_code == 200:
                st.success(f"api ok — {r.json()}")
            else:
                st.error(f"api returned {r.status_code}")
        except requests.RequestException as e:
            st.error(f"could not reach api: {e}")

    st.divider()

    if st.button("clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "**try asking:**\n\n"
        "- what does my standard policy cover for flood damage?\n"
        "- how much for a 3-bed semi, medium risk?\n"
        "- am I covered if my flat is empty for 90 days?\n"
        "- what's the excess on subsidence claims?"
    )

# ---------------------------------------------------------------------------
# rendering helpers — keep the chat loop below clean
# ---------------------------------------------------------------------------


def _render_assistant_extras(extra: dict[str, Any] | None) -> None:
    """render citations, quote, and refusal info that came back with an answer."""
    if not extra:
        return

    # refusal banner sits above everything — must not be missed
    if extra.get("refused"):
        reason = extra.get("refusal_reason") or "request was blocked by the agent"
        st.warning(f"**refused** — {reason}")

    # indicative quote panel
    quote = extra.get("quote")
    if quote:
        with st.container(border=True):
            st.markdown("**indicative price**")
            cols = st.columns(2)
            cols[0].metric(
                "base premium",
                f"£{quote.get('base_annual_premium_gbp', 0):.2f}",
            )
            cols[1].metric(
                "final premium",
                f"£{quote.get('final_annual_premium_gbp', 0):.2f}",
            )
            modifiers = quote.get("modifiers_applied") or []
            if modifiers:
                with st.expander("how the price was built"):
                    for m in modifiers:
                        st.write(
                            f"- **{m.get('name')}** = `{m.get('value')}`  "
                            f"→ ×{m.get('multiplier')}"
                        )
            disclaimer = quote.get("disclaimer")
            if disclaimer:
                st.caption(disclaimer)

    # policy citations — every chunk the rag layer surfaced
    citations = extra.get("citations") or []
    if citations:
        with st.expander(f"sources ({len(citations)})"):
            for i, c in enumerate(citations, start=1):
                src = c.get("source_file", "?")
                section = c.get("section_title", "?")
                text = c.get("text", "")
                st.markdown(f"**[{i}] {src} — {section}**")
                # truncate long chunks so the panel stays readable
                shown = text if len(text) <= 600 else text[:600] + "…"
                st.markdown(shown)
                if i < len(citations):
                    st.divider()


def _post_question(question: str) -> dict[str, Any] | None:
    """call POST /ask, return parsed json or None on failure (with an st.error shown)."""
    try:
        r = requests.post(
            ASK_ENDPOINT,
            json={"question": question},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        st.error(
            f"could not reach the api at {ASK_ENDPOINT} — is uvicorn running? "
            f"({e})"
        )
        return None

    if r.status_code != 200:
        # show the server's detail message so debugging is easy
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(f"api returned {r.status_code}: {detail}")
        return None

    try:
        return r.json()
    except ValueError:
        st.error("api response was not valid json")
        return None


# ---------------------------------------------------------------------------
# replay the chat history every rerun — newest at the bottom
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_assistant_extras(msg.get("extra"))

# ---------------------------------------------------------------------------
# main input — the chat box pinned at the bottom
# ---------------------------------------------------------------------------

prompt = st.chat_input("ask about cover, exclusions, or get an indicative price...")

if prompt:
    # echo the user message immediately so the ui feels responsive
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "extra": None}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # call the api with a spinner so the user knows something is happening
    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            data = _post_question(prompt)

        if data is None:
            # error already shown via st.error inside _post_question
            answer = "_no answer — see error above._"
            extra = None
        else:
            answer = data.get("answer", "_(empty answer)_")
            extra = {
                "refused": data.get("refused", False),
                "refusal_reason": data.get("refusal_reason"),
                "quote": data.get("quote"),
                "citations": data.get("citations") or [],
            }

        st.markdown(answer)
        _render_assistant_extras(extra)

    # save the assistant turn so it survives the next rerun
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "extra": extra}
    )
