"""
RD.011 Agent — RAG Configuration.

Embedding provider setup for Chroma vector store.
Delegates to the same provider routing as LLM models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

# ── RAG paths ──────────────────────────────────────────────────────────────

RAG_DIR = Path(__file__).parent
CHROMA_DB_PATH = RAG_DIR / "chroma_db"
IMPLICIT_PROCESSES_CONFIG = RAG_DIR.parent / "config_implicit_processes.json"

# ── Embedding configuration ────────────────────────────────────────────────

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

# Chroma collection name for example documents
COLLECTION_NAME = "rd011_examples"

# RAG retrieval parameters
DEFAULT_TOP_K = 5


def get_embedding_function():
    """
    Return a Chroma-compatible embedding function.

    Supports:
    - "google": Google Generative AI embeddings (free, fast)
    - None/fallback: No embedding (returns MockEmbedding that raises on use)

    Returns
    -------
    Callable
        An embedding function that Chroma can use for vectorization.
        Raises NotImplementedError if provider is not configured or API key missing.
    """
    if EMBEDDING_PROVIDER == "google":
        return _get_google_embedding()
    else:
        logger.warning(
            "Embedding provider '%s' not recognized. Falling back to no-op.",
            EMBEDDING_PROVIDER,
        )
        return _get_noop_embedding()


def _get_google_embedding():
    """Get Google Generative AI embedding function."""
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set; cannot use Google embeddings.")
        raise ValueError("GOOGLE_API_KEY is required for embedding provider 'google'.")

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embedding_func = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
        logger.info("Initialized Google embedding function (model=%s)", EMBEDDING_MODEL)
        return embedding_func
    except ImportError:
        logger.error("langchain_google_genai not installed; cannot use Google embeddings.")
        raise


def _get_noop_embedding():
    """Get a no-op embedding function for graceful degradation."""

    class NoOpEmbedding:
        """Placeholder embedding that raises if actually used."""

        def embed_documents(self, texts):
            raise NotImplementedError(
                "RAG embedding not configured. "
                "Set EMBEDDING_PROVIDER and required API keys."
            )

        def embed_query(self, text):
            raise NotImplementedError(
                "RAG embedding not configured. "
                "Set EMBEDDING_PROVIDER and required API keys."
            )

    return NoOpEmbedding()
