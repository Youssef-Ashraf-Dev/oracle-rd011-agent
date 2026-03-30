"""
RD.011 Agent — RAG retriever.

All LLM calls for generation and intro use this module to retrieve
relevant examples from the RD.011 knowledge base.

Configure embeddings and Chroma location via rag/config_rag.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from config import RAG_ENABLED, TOP_K_RETRIEVAL
from rag.config_rag import CHROMA_DB_PATH, COLLECTION_NAME, get_embedding_function

logger = logging.getLogger(__name__)

_RAG_READY: bool | None = None
_RAG_BLOCK_REASON: str | None = None


@dataclass
class Document:
    """A single retrieved chunk from the knowledge base."""

    page_content: str
    metadata: Dict[str, str] = field(default_factory=dict)
    # Expected metadata keys: source, section, chunk_index


def _ensure_rag_ready() -> bool:
    """Check RAG prerequisites once and cache the result."""
    global _RAG_READY, _RAG_BLOCK_REASON

    if _RAG_READY is not None:
        return _RAG_READY

    if not RAG_ENABLED:
        _RAG_READY = False
        _RAG_BLOCK_REASON = "RAG disabled (RAG_ENABLED=0)"
        return _RAG_READY

    try:
        import langchain_chroma  # noqa: F401
    except Exception as exc:
        _RAG_READY = False
        _RAG_BLOCK_REASON = f"langchain_chroma not installed ({exc})"
        return _RAG_READY

    _RAG_READY = True
    _RAG_BLOCK_REASON = None
    return _RAG_READY


def retrieve(query: str, top_k: int | None = None) -> List[Document]:
    """
    Retrieve relevant chunks from the RD.011 knowledge base.

    Queries the Chroma vector store for documents similar to the query.
    Returns examples sorted by semantic similarity.

    Parameters
    ----------
    query
        Natural-language query describing desired context.
        Example: "general ledger month end closing process steps"
    top_k
        Maximum number of chunks to return (default: 5).

    Returns
    -------
    List[Document]
        List of Document objects with page_content and metadata.
        Returns empty list if Chroma DB unavailable or empty.

    Notes
    -----
    If Chroma DB is not available or not yet built, logs a warning
    and returns an empty list (graceful degradation).
    """
    if not _ensure_rag_ready():
        logger.info("RAG unavailable: %s. Returning empty results.", _RAG_BLOCK_REASON)
        return []

    if not CHROMA_DB_PATH.exists():
        logger.warning(
            "Chroma DB not found at %s. Returning empty results. "
            "Run: python rag/ingest.py --docs-dir path/to/examples/",
            CHROMA_DB_PATH,
        )
        return []

    try:
        embedding_func = get_embedding_function()
    except Exception as e:
        logger.warning("Could not initialize embedding function: %s. Returning empty results.", e)
        return []

    try:
        from langchain_chroma import Chroma

        vector_db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=embedding_func,
        )

        # Query Chroma
        if top_k is None:
            top_k = TOP_K_RETRIEVAL
        logger.debug("Querying Chroma: %s (top_k=%d)", query[:60], top_k)
        results = vector_db.similarity_search(query, k=top_k)

        # Convert LangChain Document to our Document dataclass
        docs = []
        for result in results:
            doc = Document(
                page_content=result.page_content,
                metadata=result.metadata or {},
            )
            docs.append(doc)

        logger.debug("Retrieved %d results from Chroma", len(docs))
        return docs

    except Exception as e:
        logger.warning("Error querying Chroma: %s. Returning empty results.", e)
        return []
