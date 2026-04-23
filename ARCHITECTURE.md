# ARCHITECTURE — Pricing AI Agent

Design document for the agentic AI pricing assistant. Written before the code
so the system has a shape, not just files.

---

## 1. What the system does

Takes a natural-language question about home-insurance pricing. Returns an
FCA-compliant, explainable answer that cites the underlying policy context.

Example input: "I have a 3-bed semi in a flood zone, what would cover cost?"

Example output (shape): a plain-English price range, the inputs used, the
relevant policy clauses, and a clear caveat that this is an indicative
estimate, not a binding quote.

---

## 2. Why it exists

Real pricing engines are actuarial software. That is not in scope here.
What **is** in scope: showing that an LLM-led agent can sit in front of a
pricing function and a policy knowledge base, pick the right tool for the
question, compose an answer that reads like a regulated firm wrote it, and
refuse cleanly when the question is out of scope or adversarial.

The FCA Consumer Duty language ("clear, fair, not misleading") is treated
as a hard output constraint, not a stylistic hint.

---

## 3. Tech stack — one sentence each

| Layer            | Choice                                     | Why                                                     |
| ---------------- | ------------------------------------------ | ------------------------------------------------------- |
| Orchestration    | LangGraph                                  | Stateful graph, conditional edges, guardrail loop fits  |
| LLM              | Groq (Llama 3.3 70B)                       | Free tier, low latency, good instruction following      |
| Embeddings       | sentence-transformers all-MiniLM-L6-v2     | Local, free, 384-dim, fast on CPU                       |
| Vector DB        | ChromaDB                                   | Local, one line to persist, no external service         |
| Validation       | Pydantic v2                                | Catches bad tool inputs at the boundary                 |
| API              | FastAPI                                    | Pydantic-native, auto OpenAPI docs, async               |
| UI               | Streamlit                                  | One file, deploys in minutes, plan says "minimal UI"    |
| Logging          | loguru + JSON trace file                   | Human logs in the console, machine logs for eval        |
| Container        | Docker + compose                           | Reproducible, runs the same locally and on Hetzner      |
| Tests            | pytest                                     | Fixtures, simple syntax, industry standard              |
| CI               | GitHub Actions                             | Free minutes, repo is on GitHub, one YAML file          |

---

## 4. Runtime flow — what happens on one request

```
[ user types question in streamlit ]
              |
              v
[ POST /ask  ->  fastapi endpoint ]
              |
              v
[ pydantic validates request body ]
              |
              v
[ langgraph stategraph starts ]
              |
              v
  +-------------------------------+
  | NODE 1  input guardrail       |  prompt-injection + PII check
  +-------------------------------+
              |
              v
  +-------------------------------+
  | NODE 2  classify intent       |  pricing | policy | both | out-of-scope
  +-------------------------------+
              |
              v
  +-------------------------------+
  | NODE 3  call tools            |  pricing_api + policy_lookup (RAG)
  +-------------------------------+
              |
              v
  +-------------------------------+
  | NODE 4  compose answer        |  llm writes FCA-worded reply
  +-------------------------------+
              |
              v
  +-------------------------------+
  | NODE 5  output guardrail      |  llm-as-judge: clear, fair, not misleading
  +-------------------------------+
              |
              v
[ fastapi returns json response ]
              |
              v
[ streamlit renders answer + loguru writes trace ]
```

Out-of-scope questions exit after Node 2 with a canned refusal. Injection
attempts exit after Node 1. Compose failures re-loop once through Node 4
with a stricter prompt.

---

## 5. Components

### 5.1  app/agent
- `graph.py` — wires the five nodes into a LangGraph state machine
- `nodes.py` — each node is a pure function of state
- `prompts.py` — system prompt + FCA response template

### 5.2  app/tools
- `pricing_api.py` — mock pricing function. Takes property + risk inputs,
  looks up base price in `data/mock_pricing_table.json`, applies modifiers,
  returns a `PriceQuote`
- `policy_lookup.py` — wraps the Chroma retriever. Takes a query string,
  returns top-k policy chunks

### 5.3  app/guardrails
- `input_filter.py` — regex first (cheap), then one LLM call for
  prompt-injection patterns
- `output_filter.py` — LLM-as-judge against the FCA Consumer Duty rubric

### 5.4  app/rag
- `ingest.py` — chunks policy markdown, embeds with MiniLM, writes to Chroma
- `retriever.py` — thin query interface used by `policy_lookup.py`

### 5.5  app/schemas.py
All Pydantic models in one file — `PricingRequest`, `PriceQuote`, `PolicyChunk`,
`AgentState`, `AskRequest`, `AskResponse`.

### 5.6  app/main.py
FastAPI with a single `POST /ask` endpoint. Delegates to the graph.

### 5.7  app/streamlit_app.py
Chat-style UI. Sends questions to the FastAPI endpoint (or calls the graph
directly for the simplest deployment).

---

## 6. Data

- `data/mock_policies/` — 3-5 short markdown files (standard, landlord,
  high-value, flood-zone, etc.). Hand-written, clearly labelled as mock.
- `data/mock_pricing_table.json` — ~20 rows, keyed by (property_type,
  risk_profile), each row has base price + modifiers.

No real customer data. No scraped policy wording. Everything is synthetic.

---

## 7. Guardrails — the contract

### Input guardrail
Refuses if the input contains:
- Prompt-injection patterns (instructions targeting the system prompt)
- PII a customer should not paste (card numbers, full NI numbers)
- Requests for advice outside home insurance

### Output guardrail
Refuses to return if the draft answer:
- Asserts a firm price as a binding quote
- Uses absolutes ("guaranteed cheapest", "always covered")
- Omits the "indicative estimate" caveat
- Contains content the RAG context does not support (hallucination check)

Failure of the output guardrail re-runs Node 4 once with a stricter prompt.
Second failure returns a generic safe refusal.

---

## 8. Evaluation

`eval/golden_questions.json` holds 20 labelled questions across five
categories: in-scope pricing, in-scope policy, both, out-of-scope, injection.

`eval/run_eval.py` runs the agent over all 20, scores each with an
LLM-as-judge (`eval/judge_prompts.py`), and reports:
- Task success rate
- Hallucination rate
- FCA-compliance rate
- Refusal rate on out-of-scope + injection

Target for v1: >= 85% success on in-scope, 100% refusal on injection.

---

## 9. Deployment

- Built and pushed from the local PC
- Pulled on a Hetzner VPS over SSH
- Docker compose brings up one container
- Streamlit serves on host port 8090
- URL: `http://46.225.208.197:8090`

No TLS at v1 — HTTP direct IP + port, free subdomain (sslip.io) is an
optional polish step after the first deploy works.

### 9.1 Security posture (from the NHS incident on the same host)
- Container has `cpus: 2.0` and `mem_limit: 2g`
- `/tmp` is mounted tmpfs `noexec,nosuid,size=64M`
- Secrets live in `.env`, never in code, never in git
- Only port 8090 is exposed; no database port is published
- SSH to the host is key-only (`PermitRootLogin prohibit-password`)

---

## 10. What this is not

- Not a real pricing engine
- Not a multi-tenant product
- Not a replacement for a licensed broker
- Not a chat about anything other than home insurance

These boundaries are enforced by the input guardrail and the system prompt,
not by good manners.
