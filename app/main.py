"""fastapi app entry point — `uvicorn app.main:app` to run."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# load .env before anything else imports settings — rule: env first, code second
load_dotenv()

from app.api.routes import router  # noqa: E402  (import after load_dotenv on purpose)
from app.logging_config import logger  # noqa: E402

app = FastAPI(
    title="Pricing AI Agent",
    description=(
        "UK home-insurance assistant — policy lookup over a vector store, "
        "indicative pricing from a mock table, FCA Consumer Duty guardrails."
    ),
    version="0.1.0",
)

# cors — open in dev, locked down via env var in prod
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def _on_startup() -> None:
    """warm the graph on boot so the first request is not slow."""
    logger.info("api starting — pre-warming agent graph")
    from app.agent.graph import build_graph

    build_graph()
    logger.info("api ready")
