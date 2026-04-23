"""
retriever — given a question, returns the top-k most relevant policy chunks.

design:
- lazy singleton: the embedding model and chroma client load once on first call
- returns a simple list of dicts, not chroma-specific objects, so tool code
  stays decoupled from the vector store choice
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.logging_config import logger

load_dotenv()

CHROMA_DIR = Path(os.getenv("CHROMA_DB_DIR", "./chroma_db")).resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "home_insurance_policies")
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class RetrievedChunk:
    """structured retrieval result — what the tool layer consumes."""
    chunk_id: str
    text: str
    source_file: str
    section_title: str
    distance: float


@lru_cache(maxsize=1)
def _load_embedder() -> SentenceTransformer:
    """load the sentence-transformers model once and keep it in memory."""
    logger.info("loading embedding model {}", EMBED_MODEL_NAME)
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _load_collection():
    """open the persistent chroma collection once and reuse it."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        coll = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"chroma collection '{COLLECTION_NAME}' not found at {CHROMA_DIR}. "
            f"did you run `python -m app.rag.ingest`?"
        ) from e
    return coll


def retrieve(query: str, top_k: int = 4) -> list[RetrievedChunk]:
    """
    embed the query, ask chroma for the top_k nearest chunks, return them.
    distance is cosine distance (smaller = closer).
    """
    if not query or not query.strip():
        return []

    embedder = _load_embedder()
    collection = _load_collection()

    query_vec = embedder.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    result = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # chroma returns list-of-lists keyed by query; we only sent one query
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    ids = result["ids"][0]

    chunks = [
        RetrievedChunk(
            chunk_id=cid,
            text=doc,
            source_file=meta.get("source_file", ""),
            section_title=meta.get("section_title", ""),
            distance=float(dist),
        )
        for cid, doc, meta, dist in zip(ids, docs, metas, dists)
    ]

    logger.debug("retrieved {} chunks for query: {!r}", len(chunks), query[:80])
    return chunks


if __name__ == "__main__":
    # quick manual check — only runs when you run this file directly
    import sys

    q = " ".join(sys.argv[1:]) or "what is covered under the standard home policy?"
    logger.info("test query: {}", q)
    for c in retrieve(q, top_k=3):
        logger.info(
            "[{:.3f}] {} / {} :: {}",
            c.distance,
            c.source_file,
            c.section_title,
            c.text[:120].replace("\n", " "),
        )
