"""
ingest mock policy markdown files into ChromaDB.

run this once before the retriever can find anything:
    python -m app.rag.ingest

design choices in this file:
- chunks are built section by section from markdown headings, with a fallback
  sliding-window for any section that's too long
- embeddings use sentence-transformers all-MiniLM-L6-v2, 384 dims, cpu-only
- chroma collection is recreated every run so the ingest is idempotent
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.logging_config import logger

load_dotenv()

# paths and config, read from env so docker and local agree
POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "mock_policies"
CHROMA_DIR = Path(os.getenv("CHROMA_DB_DIR", "./chroma_db")).resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "home_insurance_policies")
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# chunking knobs — kept small because policy docs are short and dense
MAX_CHARS_PER_CHUNK = 900
OVERLAP_CHARS = 120


@dataclass
class PolicyChunk:
    """one chunk of text plus where it came from — becomes one chroma row."""
    chunk_id: str
    source_file: str
    section_title: str
    text: str


def _split_markdown_by_h2(md_text: str) -> list[tuple[str, str]]:
    """
    break the markdown into (section_title, section_body) pairs using `## ` headings.
    anything before the first `## ` is grouped under the first `# ` heading if present,
    otherwise under "intro".
    """
    lines = md_text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "intro"
    current_body: list[str] = []

    for line in lines:
        if line.startswith("## "):
            # flush whatever we had before starting a new section
            if current_body:
                sections.append((current_title, current_body))
            current_title = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)

    # don't forget the last section
    if current_body:
        sections.append((current_title, current_body))

    # drop empty sections and stringify
    return [(t, "\n".join(body).strip()) for t, body in sections if "".join(body).strip()]


def _sliding_window(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    cut a long string into overlapping windows.
    used only when a single section is bigger than MAX_CHARS_PER_CHUNK.
    """
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    step = max_chars - overlap
    while start < len(text):
        chunks.append(text[start : start + max_chars])
        start += step
    return chunks


def build_chunks_from_file(path: Path) -> list[PolicyChunk]:
    """read one markdown file and turn it into a list of PolicyChunks."""
    md_text = path.read_text(encoding="utf-8")
    sections = _split_markdown_by_h2(md_text)

    chunks: list[PolicyChunk] = []
    for sect_title, sect_body in sections:
        # clean up multiple blank lines so embeddings aren't diluted by whitespace
        sect_body = re.sub(r"\n{3,}", "\n\n", sect_body).strip()
        if not sect_body:
            continue

        windows = _sliding_window(sect_body, MAX_CHARS_PER_CHUNK, OVERLAP_CHARS)
        for i, window_text in enumerate(windows):
            chunk_id = f"{path.stem}::{sect_title}::{i}"
            # prefix the text with the section title so retrieval keeps the context
            embed_text = f"[{path.stem}] {sect_title}\n\n{window_text}"
            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    source_file=path.name,
                    section_title=sect_title,
                    text=embed_text,
                )
            )
    return chunks


def _get_chroma_client() -> chromadb.api.ClientAPI:
    """persistent chroma client that writes to CHROMA_DIR on disk."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def ingest() -> int:
    """
    full ingest pipeline. returns the number of chunks written.
    idempotent: deletes and re-creates the collection every run.
    """
    if not POLICY_DIR.exists():
        raise FileNotFoundError(f"policy dir not found: {POLICY_DIR}")

    logger.info("loading embedding model {}", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # gather chunks across every markdown file in POLICY_DIR
    all_chunks: list[PolicyChunk] = []
    for md_path in sorted(POLICY_DIR.glob("*.md")):
        file_chunks = build_chunks_from_file(md_path)
        logger.info("{} -> {} chunks", md_path.name, len(file_chunks))
        all_chunks.extend(file_chunks)

    if not all_chunks:
        raise RuntimeError(f"no chunks produced — is {POLICY_DIR} empty?")

    # embed everything in one batch — miniLM is fast enough on cpu for this size
    logger.info("embedding {} chunks", len(all_chunks))
    embeddings = model.encode(
        [c.text for c in all_chunks],
        normalize_embeddings=True,   # cosine-friendly vectors
        show_progress_bar=False,
    ).tolist()

    # drop and recreate the collection so re-runs start clean
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("deleted existing collection {}", COLLECTION_NAME)
    except Exception:
        # collection didn't exist — fine on first run
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # match our normalized embeddings
    )

    collection.add(
        ids=[c.chunk_id for c in all_chunks],
        documents=[c.text for c in all_chunks],
        embeddings=embeddings,
        metadatas=[
            {"source_file": c.source_file, "section_title": c.section_title}
            for c in all_chunks
        ],
    )

    logger.info(
        "ingest complete — {} chunks written to collection '{}' at {}",
        len(all_chunks),
        COLLECTION_NAME,
        CHROMA_DIR,
    )
    return len(all_chunks)


if __name__ == "__main__":
    ingest()
