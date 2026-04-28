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

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

_LOCAL_PROVIDER_ALIASES = {
    "local",
    "huggingface",
    "sentence-transformers",
    "sentence_transformers",
}

# Chroma collection name for example documents.
# Use a separate default for local embeddings to avoid dimension conflicts
# with existing cloud-embedded collections.
_default_collection = "rd011_examples_local" if EMBEDDING_PROVIDER in _LOCAL_PROVIDER_ALIASES else "rd011_examples"
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", _default_collection)

# RAG retrieval parameters
DEFAULT_TOP_K = 5


def get_embedding_function():
    """
    Return a Chroma-compatible embedding function.

    Supports:
    - "local": Local sentence-transformers embeddings (no API limits)
    - "google": Google Generative AI embeddings (free, fast)
    - None/fallback: No embedding (returns MockEmbedding that raises on use)

    Returns
    -------
    Callable
        An embedding function that Chroma can use for vectorization.
        Raises NotImplementedError if provider is not configured or API key missing.
    """
    if EMBEDDING_PROVIDER in {
        "local",
        "huggingface",
        "sentence-transformers",
        "sentence_transformers",
    }:
        return _get_local_embedding()
    if EMBEDDING_PROVIDER == "google":
        return _get_google_embedding()
    else:
        logger.warning(
            "Embedding provider '%s' not recognized. Falling back to no-op.",
            EMBEDDING_PROVIDER,
        )
        return _get_noop_embedding()


def _get_local_embedding():
    """Get local sentence-transformers embedding function."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Local embedding provider dependencies are missing or incompatible. "
            "Install/update with: pip install -U sentence-transformers transformers huggingface-hub. "
            f"Original error: {exc}"
        ) from exc

    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    logger.info(
        "Initialized local embedding function (model=%s, device=%s)",
        EMBEDDING_MODEL,
        EMBEDDING_DEVICE,
    )

    class LocalSentenceTransformerEmbedding:
        """Adapter exposing LangChain-compatible embedding methods."""

        def __init__(self, st_model):
            self._st_model = st_model

        def embed_documents(self, texts):
            if not texts:
                return []
            vectors = self._st_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vectors.tolist()

        def embed_query(self, text):
            vectors = self._st_model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vectors[0].tolist()

    return LocalSentenceTransformerEmbedding(model)


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
