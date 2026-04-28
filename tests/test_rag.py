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
        """Verify embedding provider supports configured defaults/providers."""
        from rag.config_rag import EMBEDDING_PROVIDER

        assert EMBEDDING_PROVIDER in ("local", "google", "openai", "groq")

    @patch("rag.config_rag._get_local_embedding")
    def test_get_embedding_function_routes_to_local_provider(self, mock_local):
        """Test local embedding provider dispatch without loading model weights."""
        from rag.config_rag import get_embedding_function

        mock_local.return_value = MagicMock()
        with patch("rag.config_rag.EMBEDDING_PROVIDER", "local"):
            embedding_func = get_embedding_function()

        assert embedding_func is mock_local.return_value


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

    def test_build_exemplar_blocks_uses_adaptive_candidate_k(self):
        """Candidate pool size should adapt to the number of needed types."""
        import rag.retriever as retriever

        with patch("rag.retriever.select_exemplars", return_value=([], [])):
            with patch("rag.retriever.retrieve_candidates", return_value=[]) as mock_retrieve:
                retriever.build_exemplar_blocks(
                    query="AP invoice processing",
                    needed_types=["process_narrative", "process_steps"],
                    module="AP",
                )

                first_k = mock_retrieve.call_args_list[0].kwargs["k"]
                assert first_k == max(retriever.TOP_K_RETRIEVAL, 8)

            with patch("rag.retriever.retrieve_candidates", return_value=[]) as mock_retrieve:
                retriever.build_exemplar_blocks(
                    query="Finance intro",
                    needed_types=["intro", "enterprise_structure", "ledger", "coa"],
                    module="GL",
                )

                first_k = mock_retrieve.call_args_list[0].kwargs["k"]
                assert first_k == max(retriever.TOP_K_RETRIEVAL, 16)

    def test_build_exemplar_blocks_escalates_when_type_missing(self):
        """Retriever should escalate candidate size before type-specific fallbacks."""
        import rag.retriever as retriever

        narrative_doc = retriever.Document(
            page_content="Narrative example",
            metadata={"source": "ex.docx", "section_type": "process_narrative", "chunk_index": "1"},
            score=0.9,
        )

        retrieve_calls: list[dict] = []

        def _fake_retrieve(query, k, filter=None, strategy="mmr"):
            retrieve_calls.append({"query": query, "k": k, "filter": filter, "strategy": strategy})
            return [narrative_doc]

        with patch("rag.retriever.retrieve_candidates", side_effect=_fake_retrieve):
            with patch(
                "rag.retriever.select_exemplars",
                side_effect=[([narrative_doc], ["process_steps"]), ([narrative_doc], ["process_steps"])],
            ):
                retriever.build_exemplar_blocks(
                    query="AP invoice processing",
                    needed_types=["process_narrative", "process_steps"],
                    module="AP",
                )

        assert len(retrieve_calls) >= 2
        assert retrieve_calls[1]["k"] > retrieve_calls[0]["k"]

    def test_build_exemplar_blocks_fallback_avoids_combined_filter(self):
        """Fallback calls should avoid combined module+section_type Chroma filters."""
        import rag.retriever as retriever

        narrative_doc = retriever.Document(
            page_content="Narrative example",
            metadata={"source": "ex.docx", "section_type": "process_narrative", "chunk_index": "1", "module": "FA"},
            score=0.9,
        )
        steps_doc = retriever.Document(
            page_content="Steps example",
            metadata={"source": "ex.docx", "section_type": "process_steps", "chunk_index": "2", "module": "FA"},
            score=0.8,
        )

        retrieve_calls: list[dict] = []

        def _fake_retrieve(query, k, filter=None, strategy="mmr"):
            retrieve_calls.append({"query": query, "k": k, "filter": filter, "strategy": strategy})
            query_l = str(query).lower()
            if "process steps" in query_l:
                return [steps_doc]
            return [narrative_doc]

        with patch("rag.retriever.retrieve_candidates", side_effect=_fake_retrieve):
            with patch(
                "rag.retriever.select_exemplars",
                side_effect=[([narrative_doc], ["process_steps"]), ([narrative_doc], ["process_steps"])],
            ):
                style_block, step_block = retriever.build_exemplar_blocks(
                    query="FA invoice accounting",
                    needed_types=["process_narrative", "process_steps"],
                    module="FA",
                )

        assert style_block
        assert step_block
        for call in retrieve_calls:
            filt = call["filter"]
            assert not (isinstance(filt, dict) and "module" in filt and "section_type" in filt)
