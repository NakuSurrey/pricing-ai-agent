"""policy lookup tool — wraps the rag retriever, returns pydantic objects."""

from __future__ import annotations

from app.logging_config import logger
from app.rag.retriever import retrieve
from app.schemas import PolicyChunk, PolicyLookupResult


def lookup(query: str, top_k: int = 4) -> PolicyLookupResult:
    """
    take a free-text query, ask the retriever for top_k chunks,
    box them into pydantic objects so the agent gets a stable contract.
    """
    raw_chunks = retrieve(query=query, top_k=top_k)

    # converting the dataclass results into pydantic models — same fields, validated
    chunks = [
        PolicyChunk(
            chunk_id=rc.chunk_id,
            text=rc.text,
            source_file=rc.source_file,
            section_title=rc.section_title,
            distance=rc.distance,
        )
        for rc in raw_chunks
    ]

    logger.info("policy lookup returned {} chunks for query: {!r}", len(chunks), query[:80])

    return PolicyLookupResult(query=query, chunks=chunks, count=len(chunks))


if __name__ == "__main__":
    # quick manual check — only runs when this file is executed directly
    result = lookup("what happens if I'm in a flood zone?", top_k=3)
    print(result.model_dump_json(indent=2))
