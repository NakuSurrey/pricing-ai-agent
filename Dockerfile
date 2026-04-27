# pricing ai agent — single-process container, streamlit only
# matches the rest of the projects on the same hetzner host.
#
# changes from phase 1 dockerfile:
#  - installs curl so the docker-compose healthcheck can hit /_stcore/health
#  - bakes an entrypoint that runs `python -m app.rag.ingest` on first boot
#    only (idempotent — chroma_data named volume persists across restarts)
#  - silences streamlit's first-run telemetry prompt

FROM python:3.11-slim

# system deps — curl for healthcheck, build-essential nuked at the end of pip install
# to keep the image small. PYTHONDONTWRITEBYTECODE keeps __pycache__ out of layers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps first (cached layer) before copying the rest of the source
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# copy the project last so source edits don't bust the deps layer
COPY . /app

# entrypoint — runs ingest only when chroma_db is empty, then launches streamlit.
# inlined here so the image is self-contained (no separate entrypoint.sh on disk).
RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -e' \
    '' \
    '# ingest only on first boot — chroma_data volume persists, so subsequent' \
    '# starts skip this entirely. checks if the chroma_db dir is empty.' \
    'if [ -z "$(ls -A /app/chroma_db 2>/dev/null)" ]; then' \
    '  echo "[entrypoint] chroma_db empty — running ingest..."' \
    '  python -m app.rag.ingest' \
    'else' \
    '  echo "[entrypoint] chroma_db already populated — skipping ingest"' \
    'fi' \
    '' \
    'echo "[entrypoint] launching streamlit on port ${APP_PORT:-8090}"' \
    'exec streamlit run app/streamlit_app.py \' \
    '  --server.port="${APP_PORT:-8090}" \' \
    '  --server.address=0.0.0.0 \' \
    '  --server.headless=true \' \
    '  --browser.gatherUsageStats=false' \
    > /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh

# the default chroma persist dir lives inside /app — mount the named volume here
# in docker-compose so the embeddings survive container restarts.
RUN mkdir -p /app/chroma_db /app/logs

EXPOSE 8090

ENTRYPOINT ["/app/entrypoint.sh"]
