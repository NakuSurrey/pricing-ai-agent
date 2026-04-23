# Pricing AI Agent

Agentic AI pricing assistant for home insurance. FCA-compliant responses,
RAG over mock policy documents, guardrailed input and output, evaluation
harness, dockerised deployment.

> Work in progress — this README is a placeholder during the 7-day build.
> The final version (badges, live URL, screenshots, decisions log) ships at
> the end of the build.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design.

## Run locally (will be filled in fully at the end)

```bash
cp .env.example .env
# edit .env and paste your free groq key
docker compose up --build
# open http://localhost:8090
```

## Repo layout

See `ARCHITECTURE.md` section 5 for the module-by-module breakdown.
