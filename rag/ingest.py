"""
RD.011 Agent — RAG Ingest Pipeline.

Parse example RD.011 documents, chunk by section, embed, and store in Chroma.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rag.config_rag import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    get_embedding_function,
)

logger = logging.getLogger(__name__)


def parse_and_chunk_examples(docs_dir: Path) -> list:
    """
    Parse all .docx files in docs_dir, chunk by section, and return Document objects.

    Chunking strategy:
    - Split at double newlines (section boundaries in Docx2txt)
    - Min chunk size: 200 chars
    - Max chunk size: 2000 chars
    - Overlap: 100 chars (preserve some context across chunks)

    Each chunk will have metadata:
    - source: filename
    - section: First heading in the chunk
    - chunk_index: Sequential position
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise ValueError(f"Documents directory does not exist: {docs_dir}")

    logger.info("Scanning %s for .docx files...", docs_dir)
    docx_files = sorted(docs_dir.glob("**/*.docx"))
    logger.info("Found %d .docx files", len(docx_files))

    if not docx_files:
        logger.warning("No .docx files found in %s", docs_dir)
        return []

    all_documents = []

    for docx_file in docx_files:
        logger.info("Loading: %s", docx_file.name)

        try:
            try:
                from langchain_community.document_loaders import Docx2txtLoader
            except Exception as exc:
                raise RuntimeError(
                    "langchain_community is required for RAG ingest. "
                    "Install with: pip install langchain-community"
                ) from exc

            loader = Docx2txtLoader(str(docx_file))
            docs = loader.load()

            if docs:
                # Docx2txtLoader returns list with one Document containing full text
                full_text = docs[0].page_content

                # Chunk with recursion (tries double newline, then newline, then word boundary)
                try:
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                except Exception as exc:
                    raise RuntimeError(
                        "langchain is required for RAG ingest. "
                        "Install with: pip install langchain"
                    ) from exc

                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", " ", ""],
                    chunk_size=2000,
                    chunk_overlap=100,
                    length_function=len,
                )
                chunks = splitter.split_text(full_text)

                logger.info("  → Split into %d chunks", len(chunks))

                # Create Document objects with metadata
                for idx, chunk_text in enumerate(chunks):
                    # Infer section name from first line if it looks like a heading
                    lines = chunk_text.split("\n")
                    section_name = lines[0][:80] if lines else "Unknown"

                    doc_dict = {
                        "page_content": chunk_text,
                        "metadata": {
                            "source": docx_file.name,
                            "source_path": str(docx_file),
                            "section": section_name,
                            "chunk_index": idx,
                        },
                    }
                    all_documents.append(doc_dict)

            else:
                logger.warning("  → No content loaded from %s", docx_file.name)

        except Exception as e:
            logger.error("  → Failed to load %s: %s", docx_file.name, e)
            continue

    logger.info("Total chunks created: %d", len(all_documents))
    return all_documents


def ingest_to_chroma(documents: list, collection_name: str = COLLECTION_NAME) -> int:
    """
    Ingest documents into Chroma vector database.

    If collection already exists, upsert (update/insert) to avoid duplicates.

    Parameters
    ----------
    documents : list
        List of document dicts with 'page_content' and 'metadata'.
    collection_name : str
        Name of the Chroma collection.

    Returns
    -------
    int
        Number of documents ingested.
    """
    if not documents:
        logger.info("No documents to ingest.")
        return 0

    # Ensure Chroma DB directory exists
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Using Chroma DB at: %s", CHROMA_DB_PATH)

    try:
        embedding_func = get_embedding_function()
    except Exception as e:
        logger.error("Failed to initialize embedding function: %s", e)
        raise

    # Initialize Chroma with upsert capability
    logger.info("Initializing Chroma collection '%s'...", collection_name)

    try:
        # Upsert: will create collection if not exists, or update existing docs
        try:
            from langchain_chroma import Chroma
        except Exception as exc:
            raise RuntimeError(
                "langchain_chroma is required for RAG ingest. "
                "Install with: pip install langchain-chroma"
            ) from exc

        vector_db = Chroma.from_documents(
            documents=[
                type("Document", (), {
                    "page_content": doc["page_content"],
                    "metadata": doc["metadata"],
                })()
                for doc in documents
            ],
            embedding=embedding_func,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DB_PATH),
        )

        # Persist to disk
        vector_db.persist()

        logger.info("✓ Successfully ingested %d documents into Chroma.", len(documents))
        logger.info("  Collection: %s", collection_name)
        logger.info("  Location: %s", CHROMA_DB_PATH)

        # Print sample metadata
        if documents:
            logger.info("\nSample metadata from first chunk:")
            logger.info("  %s", json.dumps(documents[0]["metadata"], indent=2))

        return len(documents)

    except Exception as e:
        logger.error("Failed to ingest into Chroma: %s", e)
        raise


def main():
    """CLI entry point for ingest pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest example RD.011 documents into Chroma vector store.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        required=True,
        help="Directory containing .docx example documents.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help="Chroma collection name (default: rd011_examples).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting RAG ingest pipeline...")
    logger.info("Examples directory: %s", args.docs_dir)

    try:
        documents = parse_and_chunk_examples(args.docs_dir)
        ingested = ingest_to_chroma(documents, collection_name=args.collection)
        logger.info("\n✓ Ingestion complete: %d documents stored.", ingested)
    except Exception as e:
        logger.error("Ingest pipeline failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
