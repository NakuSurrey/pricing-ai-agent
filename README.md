# Pricing AI Agent

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1c3d5a.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-ff4b4b.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-2a9d8f.svg)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed.svg)](https://www.docker.com/)
[![Live](https://img.shields.io/badge/status-live-brightgreen.svg)](http://46.225.208.197:8090)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

A guardrailed agentic AI assistant that answers UK home insurance questions and produces indicative prices, with FCA Consumer Duty wording on every response and a citation back to the policy section it came from.

## Live demo

http://46.225.208.197:8090

No login. Click any sample question in the sidebar or type your own. Try the prompt-injection sample to see the input guardrail block it. Try the car-insurance one to see the out-of-scope refusal.

## What it does

- Answers questions about cover, exclusions, and claims for UK home insurance, citing the exact policy section the answer came from.
- Produces an indicative price for a property, applying modifiers for risk profile, claims history, security, property age, and policy type.
- Blocks prompt-injection attempts and out-of-scope questions before they reach the model.
- Refuses to handle PII like card numbers or NI numbers.
- Runs every reply through an output guardrail that checks for FCA-incompatible language (no "guaranteed", no "cheapest", no "always covered").

## Why I built it

Home insurance pricing sits inside one of the most heavily regulated chat surfaces in the UK. Saying the wrong thing — "you're guaranteed cover", "this is the cheapest", "your claim is always covered" — is a Consumer Duty violation, not a typo. Most agent demos ignore this. This project treats it as the core constraint and builds the whole system around it: input guardrails first, intent classification second, deterministic tools third, LLM composition fourth, output guardrail last. The price you see is always indicative, the answer always cites a source, the refusal always has a reason.

## Tech stack

| Layer | Technology | Why this choice |
|---|---|---|
| Agent runtime | LangGraph 0.2 | State-machine model fits the 5-node guardrails-first design better than a free-form ReAct loop |
| LLM | Groq (Llama 3.1 8B) | Free tier, fast inference, deterministic JSON output via `response_format=json_object` |
| Vector store | ChromaDB 0.5 (local persist) | Embedded, no extra service, persisted via Docker named volume |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small, fast, runs on CPU, ships with the container |
| API | FastAPI 0.115 | Auto OpenAPI docs, Pydantic native, used for `/ask` during local dev |
| UI | Streamlit 1.40 | Single Python file, chat layout out of the box, deploys without a build step |
| Validation | Pydantic 2.10 | Every node input/output is a model — no dict-based typing |
| Logging | Loguru + JSONL tracer | Loguru handles app logs; a separate JSONL tracer captures node-by-node decisions |
| Tests | Pytest 8.3 | 23 tests across pricing, guardrails, graph wiring, and end-to-end flows |
| Container | Docker + docker-compose | Single image, single port (8090), named volume for ChromaDB persistence |
| Deploy target | Hetzner VPS (Ubuntu 24.04) | Existing box, full control, no platform vendor lock-in |

## Architecture

```
                       user types question
                              |
                              v
                    +---------+---------+
                    |   Streamlit chat   |   in-container, imports graph directly
                    |   (port 8090)      |
                    +---------+---------+
                              |
                              v
                    +---------+---------+
                    |  LangGraph state   |
                    |     machine        |
                    +---------+---------+
                              |
                              v
   +--------------------------+--------------------------+
   |                                                     |
   |   NODE 1  input_guardrail_check                     |
   |   regex first (injection patterns, PII), then LLM   |
   |   refused -> exit with refusal_reason               |
   |                                                     |
   v                                                     |
   NODE 2  intent_classify                               |
   pricing | policy | both | out_of_scope                |
   out_of_scope -> exit with refusal                     |
   |                                                     |
   v                                                     |
   NODE 3  tool_call                                     |
   pricing_api  -> mock pricing table + modifiers        |
   policy_lookup -> ChromaDB top-k retrieval             |
   |                                                     |
   v                                                     |
   NODE 4  compose                                       |
   LLM writes the answer using the FCA-aware template    |
   |                                                     |
   v                                                     |
   NODE 5  output_guardrail_check                        |
   regex pass for banned phrases ("guaranteed",          |
   "cheapest", "always covered"...)                      |
   blocked -> loop back to compose with stricter prompt  |
   passed -> emit final_answer + citations + quote       |
   |                                                     |
   +-----------------------------------------------------+
                              |
                              v
                    +---------+---------+
                    |  rendered in chat  |
                    |  + sources expander|
                    |  + price panel     |
                    +-------------------+
```

Out-of-scope questions exit after Node 2. Injection attempts exit after Node 1. A failed output guardrail loops back to Node 4 once with a stricter prompt before giving up.

## How to run locally

Prerequisites: Python 3.11, Docker, a free Groq API key from https://console.groq.com.

```bash
# clone
git clone https://github.com/NakuSurrey/pricing-ai-agent.git
cd pricing-ai-agent

# env vars — copy the example and paste a real key into the new file
cp .env.example .env
# open .env and set GROQ_API_KEY=<your key>

# build + start
docker-compose up -d --build

# open
# http://localhost:8090
```

First boot will run the ChromaDB ingest automatically (37 chunks from 5 mock policies). The named volume keeps that data across restarts, so subsequent boots skip the ingest and start in seconds.

To run without Docker:

```bash
python -m venv .venv
source .venv/bin/activate            # mac / linux
.venv\Scripts\activate               # windows powershell

pip install -r requirements.txt
python -m app.rag.ingest             # one-time — populates chroma_db/

# terminal 1 — fastapi
uvicorn app.main:app --port 8765

# terminal 2 — streamlit (local-dev version, talks to fastapi over http)
streamlit run ui/streamlit_app.py
```

To run the test suite:

```bash
pytest -v
# expect 23 passed
```

## Key decisions

- **LangGraph over a free-form ReAct loop.** A state machine makes the guardrails-first design explicit. Every request hits the input guardrail before any LLM call, every reply hits the output guardrail before reaching the user. A ReAct loop would have to rely on the model remembering not to skip those steps.
- **In-container Streamlit, no FastAPI in the deployed container.** Two surfaces — `app/streamlit_app.py` for the deployed single-process container, `ui/streamlit_app.py` for local dev against the FastAPI server. Keeps the production container to one process on one port, matches the rest of the projects on the deploy host.
- **Regex-first input guardrail.** Injection patterns and PII (card numbers, NI numbers) are caught with regex before any LLM call. Cheap, deterministic, testable. Saves a model call on every blocked request.
- **Deterministic-fallback path for tests.** The graph runs end-to-end without `GROQ_API_KEY` set — keyword-based intent and slot extraction kick in. Means CI and reviewers can run the whole test suite without an API key.
- **Pydantic everywhere.** Every node input and output is a typed model. No `dict[str, Any]` floating between nodes. Caught real schema bugs early — the in-container Streamlit was reading `state.answer` instead of `state.final_answer` and Pydantic raised a clean `AttributeError` instead of a silent `None`.
- **JSONL tracer separate from app logs.** Loguru writes human-readable app logs to `logs/app.log`. The tracer writes one JSON object per node decision to `logs/trace.jsonl`. Easy to grep, easy to feed into an eval harness.
- **Hetzner VPS over a managed PaaS.** Reuses an existing box hardened after a previous cryptojacking incident — full control of resource limits (`cpus: 2.0`, `mem_limit: 2g`, `tmpfs /tmp:noexec,nosuid`) and firewall rules. No platform vendor opinions on what "free tier" means.

## What I learned

- LangGraph node names cannot collide with state field names — silent constraint that took two errors to surface. Renamed `intent` to `intent_classify` and `*_guardrail` nodes to `*_guardrail_check`.
- Streamlit launched against a file path puts the file's parent directory at `sys.path[0]`, not the working directory. Setting `PYTHONPATH=/app` in the Dockerfile is the cleanest fix.
- `docker-compose` 1.29.2 cannot read image metadata produced by Docker engine 29.x — `KeyError: 'ContainerConfig'` on recreate. `docker-compose down` followed by `docker-compose up -d` avoids the recreate code path entirely.
- The output guardrail's regex for "always covered" had to handle every verb tense and intervening words. "always be covered" slipped past the first version because the pattern required `always` directly followed by `cover/covered`.
- Pydantic field names matter at every boundary, not just at the API. The Streamlit renderer guessing field names ("answer", "indicative_price", "citations") instead of reading the real schema (`final_answer`, `quote`, `policy_chunks`) caused a working agent to look broken in the browser.

## License

MIT. See [`LICENSE`](LICENSE) if present, or treat the code as MIT until one is added.
