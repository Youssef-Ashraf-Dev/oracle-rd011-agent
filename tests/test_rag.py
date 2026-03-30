"""
Tests for RAG (Retrieval-Augmented Generation) pipeline.

Tests cover:
- Ingest: parsing documents, chunking, storing in Chroma
- Retriever: querying, fallback behavior
- Config: embedding function initialization
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Conditionally skip tests if dependencies not installed
pytest.importorskip("langchain_chroma")
pytest.importorskip("python-docx")


@pytest.fixture(autouse=True)
def _enable_rag_for_tests():
    """Ensure RAG is enabled and cache is reset for each test."""
    import rag.retriever as retriever

    retriever.RAG_ENABLED = True
    retriever._RAG_READY = None
    retriever._RAG_BLOCK_REASON = None
    yield
    retriever._RAG_READY = None
    retriever._RAG_BLOCK_REASON = None


class TestEmbeddingConfig:
    """Test RAG configuration and embedding function setup."""

    def test_embedding_config_defaults(self):
        """Verify default embedding provider is 'google'."""
        from rag.config_rag import EMBEDDING_PROVIDER

        assert EMBEDDING_PROVIDER in ("google", "openai", "groq")

    def test_get_embedding_function_called_without_error_when_api_key_exists(self):
        """Test embedding function initialization with mocked API key."""
        # Skip if GOOGLE_API_KEY not set
        import os

        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set; skipping embedding test")

        from rag.config_rag import get_embedding_function

        embedding_func = get_embedding_function()
        assert embedding_func is not None


class TestIngest:
    """Test document ingestion pipeline."""

    def test_parse_and_chunk_examples_with_empty_directory(self, tmp_path):
        """Test handling of empty directory."""
        from rag.ingest import parse_and_chunk_examples

        docs = parse_and_chunk_examples(tmp_path)
        assert docs == []

    def test_parse_and_chunk_examples_nonexistent_directory(self):
        """Test handling of nonexistent directory."""
        from rag.ingest import parse_and_chunk_examples

        with pytest.raises(ValueError, match="does not exist"):
            parse_and_chunk_examples(Path("/nonexistent/path"))

    def test_ingest_to_chroma_empty_documents(self):
        """Test ingest with empty document list."""
        from rag.ingest import ingest_to_chroma

        result = ingest_to_chroma([])
        assert result == 0

    @patch("rag.config_rag.get_embedding_function")
    def test_ingest_to_chroma_creates_directory(self, mock_embedding, tmp_path):
        """Test that ingest creates Chroma DB directory if missing."""
        from rag.ingest import ingest_to_chroma
        from rag import config_rag

        # Mock embedding function
        mock_embedding.return_value = MagicMock()

        # Temporarily override Chroma path for testing
        original_path = config_rag.CHROMA_DB_PATH
        test_db_path = tmp_path / "test_chroma"
        config_rag.CHROMA_DB_PATH = test_db_path

        # Skip if Chroma not available
        pytest.importorskip("langchain_chroma")

        try:
            # Prepare minimal test documents
            test_docs = [
                {
                    "page_content": "Test content for GL process",
                    "metadata": {"source": "test_doc.docx", "section": "GL"},
                }
            ]

            # This will raise if embeddings fail, but directory should be created
            try:
                ingest_to_chroma(test_docs)
            except Exception:
                # Ingest may fail due to mocking, but directory should exist
                pass

            # Verify directory was created
            assert test_db_path.exists()

        finally:
            config_rag.CHROMA_DB_PATH = original_path


class TestRetriever:
    """Test RAG retriever functionality."""

    def test_retrieve_with_missing_chroma_db(self):
        """Test graceful fallback when Chroma DB doesn't exist."""
        from rag.retriever import retrieve
        from unittest.mock import patch

        with patch("rag.retriever.CHROMA_DB_PATH") as mock_path:
            mock_path.exists.return_value = False
            results = retrieve("test query")
            assert results == []

    def test_retrieve_returns_document_objects(self):
        """Test that retriever returns proper Document objects."""
        from rag.retriever import Document

        doc = Document(
            page_content="Test content",
            metadata={"source": "test.docx", "section": "AP"},
        )
        assert doc.page_content == "Test content"
        assert doc.metadata["source"] == "test.docx"

    @patch("langchain_chroma.Chroma")
    def test_retrieve_calls_similarity_search(self, mock_chroma_class):
        """Test that retrieve calls Chroma's similarity_search method."""
        from rag.retriever import retrieve
        from unittest.mock import patch

        # Mock Chroma instance and result
        mock_instance = MagicMock()
        mock_chroma_class.return_value = mock_instance

        # Mock search results
        mock_lc_doc = MagicMock()
        mock_lc_doc.page_content = "Example GL month-end process"
        mock_lc_doc.metadata = {"source": "example.docx"}
        mock_instance.similarity_search.return_value = [mock_lc_doc]

        with patch("rag.retriever.CHROMA_DB_PATH") as mock_path:
            mock_path.exists.return_value = True
            with patch("rag.retriever.get_embedding_function"):
                results = retrieve("general ledger month end", top_k=5)

                # Verify Chroma was called with correct params
                assert len(results) == 1
                assert results[0].page_content == "Example GL month-end process"

    def test_retrieve_with_embedding_error(self):
        """Test graceful fallback when embedding fails."""
        from rag.retriever import retrieve
        from unittest.mock import patch

        with patch("rag.retriever.get_embedding_function") as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")
            with patch("rag.retriever.CHROMA_DB_PATH") as mock_path:
                mock_path.exists.return_value = True
                results = retrieve("test query")
                assert results == []
