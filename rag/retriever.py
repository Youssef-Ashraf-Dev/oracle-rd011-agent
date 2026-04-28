"""
RD.011 Agent — RAG retriever.

All LLM calls for generation and intro use this module to retrieve
relevant examples from the RD.011 knowledge base.

Configure embeddings and Chroma location via rag/config_rag.py.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from config import (
    RAG_ALLOWED_SOURCES,
    RAG_ENABLED,
    RAG_MAX_CHUNK_CHARS,
    RAG_MAX_CHUNKS_PER_SOURCE,
    RAG_MAX_CONTEXT_CHARS,
    RAG_RETRIEVE_CACHE_MAX_ENTRIES,
    TOP_K_RETRIEVAL,
)
from rag.config_rag import CHROMA_DB_PATH, COLLECTION_NAME, get_embedding_function

logger = logging.getLogger(__name__)

_RAG_READY: bool | None = None
_RAG_BLOCK_REASON: str | None = None
_RETRIEVE_CACHE: dict[tuple, list["Document"]] = {}
_RETRIEVE_CALL_COUNTER = 0

_TRACE_ON_VALUES = {"1", "true", "yes", "y", "on"}
RAG_TRACE_RETRIEVAL = str(os.getenv("RAG_TRACE_RETRIEVAL", "false")).strip().lower() in _TRACE_ON_VALUES
RAG_TRACE_PREVIEW_CHARS = int(os.getenv("RAG_TRACE_PREVIEW_CHARS", "160"))


@dataclass
class Document:
    """A single retrieved chunk from the knowledge base."""

    page_content: str
    metadata: Dict[str, str] = field(default_factory=dict)
    score: float | None = None
    # Expected metadata keys: source, section, chunk_index


def _next_retrieve_call_id() -> int:
    global _RETRIEVE_CALL_COUNTER
    _RETRIEVE_CALL_COUNTER += 1
    return _RETRIEVE_CALL_COUNTER


def _trace_retrieval_call(
    call_id: int,
    query: str,
    strategy: str,
    requested_k: int,
    search_k: int | None,
    filter: dict[str, str] | None,
    from_cache: bool,
    docs: list["Document"],
) -> None:
    """Emit detailed retrieval traces when RAG_TRACE_RETRIEVAL is enabled."""
    if not RAG_TRACE_RETRIEVAL:
        return

    logger.info(
        "RAG_TRACE call=%d strategy=%s requested_k=%d search_k=%s cache=%s filter=%s query=%s results=%d",
        call_id,
        strategy,
        requested_k,
        search_k,
        from_cache,
        filter,
        query,
        len(docs),
    )

    for idx, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        preview = " ".join((d.page_content or "").split())
        if len(preview) > RAG_TRACE_PREVIEW_CHARS:
            preview = preview[:RAG_TRACE_PREVIEW_CHARS].rstrip() + "..."

        logger.info(
            (
                "RAG_TRACE call=%d result=%d score=%s source=%s section_type=%s "
                "module=%s process_id=%s chunk_index=%s preview=%s"
            ),
            call_id,
            idx,
            d.score,
            meta.get("source"),
            meta.get("section_type"),
            meta.get("module"),
            meta.get("process_id"),
            meta.get("chunk_index"),
            preview,
        )


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


def get_rag_readiness_status() -> tuple[bool, str | None]:
    """Return whether RAG is ready and the blocking reason (if any)."""
    ready = _ensure_rag_ready()
    return ready, _RAG_BLOCK_REASON


def _allowed_sources_list() -> list[str] | None:
    if not RAG_ALLOWED_SOURCES:
        return None
    return [s.strip() for s in str(RAG_ALLOWED_SOURCES).split(",") if s.strip()]


def _cache_key(
    query: str,
    k: int,
    filter: dict[str, str] | None,
    strategy: str,
    allowed_sources: Sequence[str] | None,
) -> tuple:
    filt = tuple(sorted((filter or {}).items()))
    allow = tuple(allowed_sources or ())
    return (query, k, strategy, filt, allow)


def _get_vector_db(embedding_func):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embedding_func,
    )


def retrieve_candidates(
    query: str,
    k: int,
    filter: dict[str, str] | None = None,
    strategy: str = "mmr",
) -> List[Document]:
    """
    Retrieve candidate chunks. Returns a ranked list.

    strategy:
    - "mmr": diversify results; score is None; order is treated as ranking
    - "scores": returns similarity scores when supported; higher is better
    """
    call_id = _next_retrieve_call_id()

    if not _ensure_rag_ready():
        _trace_retrieval_call(
            call_id=call_id,
            query=query,
            strategy=strategy,
            requested_k=k,
            search_k=None,
            filter=filter,
            from_cache=False,
            docs=[],
        )
        return []

    if not CHROMA_DB_PATH.exists():
        _trace_retrieval_call(
            call_id=call_id,
            query=query,
            strategy=strategy,
            requested_k=k,
            search_k=None,
            filter=filter,
            from_cache=False,
            docs=[],
        )
        return []

    try:
        embedding_func = get_embedding_function()
    except Exception:
        _trace_retrieval_call(
            call_id=call_id,
            query=query,
            strategy=strategy,
            requested_k=k,
            search_k=None,
            filter=filter,
            from_cache=False,
            docs=[],
        )
        return []

    allowed_sources = _allowed_sources_list()
    cache_key = _cache_key(query, k, filter, strategy, allowed_sources)
    if cache_key in _RETRIEVE_CACHE:
        cached_docs = list(_RETRIEVE_CACHE[cache_key])
        _trace_retrieval_call(
            call_id=call_id,
            query=query,
            strategy=strategy,
            requested_k=k,
            search_k=None,
            filter=filter,
            from_cache=True,
            docs=cached_docs,
        )
        return cached_docs

    vector_db = _get_vector_db(embedding_func)

    # If we post-filter by source, fetch more candidates first.
    search_k = k
    if allowed_sources:
        search_k = min(60, max(k * 10, k))

    docs: list[Document] = []

    try:
        if strategy == "scores":
            # Try to get relevance scores; not all backends accept filter in the signature.
            results = vector_db.similarity_search_with_relevance_scores(
                query,
                k=search_k,
                filter=filter,
            )
            for d, score in results:
                docs.append(Document(page_content=d.page_content, metadata=d.metadata or {}, score=score))
        else:
            results = vector_db.max_marginal_relevance_search(
                query,
                k=search_k,
                fetch_k=min(200, search_k * 4),
                filter=filter,
            )
            for d in results:
                docs.append(Document(page_content=d.page_content, metadata=d.metadata or {}, score=None))
    except Exception:
        try:
            results = vector_db.similarity_search(query, k=search_k, filter=filter)
            for d in results:
                docs.append(Document(page_content=d.page_content, metadata=d.metadata or {}, score=None))
        except Exception as exc:
            logger.warning("Error querying Chroma: %s. Returning empty results.", exc)
            _trace_retrieval_call(
                call_id=call_id,
                query=query,
                strategy=strategy,
                requested_k=k,
                search_k=search_k,
                filter=filter,
                from_cache=False,
                docs=[],
            )
            return []

    if allowed_sources:
        docs = [d for d in docs if (d.metadata or {}).get("source") in allowed_sources]

    docs = docs[:k]
    _trace_retrieval_call(
        call_id=call_id,
        query=query,
        strategy=strategy,
        requested_k=k,
        search_k=search_k,
        filter=filter,
        from_cache=False,
        docs=docs,
    )
    if RAG_RETRIEVE_CACHE_MAX_ENTRIES > 0:
        while len(_RETRIEVE_CACHE) >= RAG_RETRIEVE_CACHE_MAX_ENTRIES:
            _RETRIEVE_CACHE.pop(next(iter(_RETRIEVE_CACHE)))
    _RETRIEVE_CACHE[cache_key] = list(docs)
    return docs


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
            "Run: python -m rag.ingest --docs-dir path/to/examples/",
            CHROMA_DB_PATH,
        )
        return []

    if top_k is None:
        top_k = TOP_K_RETRIEVAL
    return retrieve_candidates(query, k=top_k, filter=None, strategy="mmr")


def _rank_key(doc: Document) -> float:
    # Higher score = more relevant; None means keep original order.
    return float(doc.score) if doc.score is not None else 0.0


def _doc_identity(doc: Document) -> tuple[str, str, str, str]:
    """Build a stable identity key for deduplicating retrieved docs."""
    meta = doc.metadata or {}
    return (
        str(meta.get("source", "") or ""),
        str(meta.get("section_type", "") or ""),
        str(meta.get("chunk_index", "") or ""),
        (doc.page_content or "")[:160],
    )


def _merge_unique_docs(existing: Sequence[Document], incoming: Sequence[Document]) -> list[Document]:
    """Append incoming docs while preserving order and removing duplicates."""
    merged = list(existing)
    seen = {_doc_identity(d) for d in merged}
    for d in incoming:
        key = _doc_identity(d)
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)
    return merged


def _adaptive_candidate_k(needed_types: Sequence[str]) -> int:
    """Compute an adaptive candidate pool size based on requested section types."""
    unique_types = {str(t).strip() for t in needed_types if str(t).strip()}
    # 2 types -> 8 candidates, 4 types -> 16 candidates.
    adaptive = max(8, min(16, len(unique_types) * 4))
    return max(TOP_K_RETRIEVAL, adaptive)


def _escalated_candidate_k(initial_k: int) -> int:
    """Compute a larger pool size used only when type coverage is still missing."""
    return min(24, max(initial_k + 4, initial_k * 2))


def select_exemplars(
    candidates: list[Document],
    needed_types: list[str],
    per_type_limit: int = 1,
    per_source_limit: int = 2,
    budget_chars: int | None = None,
    per_chunk_chars: int | None = None,
    min_relevance_score: float | None = None,
) -> tuple[list[Document], list[str]]:
    """
    Deterministically select chunks for the required section types.

    Returns: (selected_docs, missing_types)
    """
    if not candidates:
        return [], list(needed_types)

    if budget_chars is None:
        budget_chars = RAG_MAX_CONTEXT_CHARS
    if per_chunk_chars is None:
        per_chunk_chars = RAG_MAX_CHUNK_CHARS

    # Stable ranking: if scores exist, sort by score desc, else keep retrieval order.
    # Optionally filter out low-quality candidates.
    ranked = list(candidates)
    if min_relevance_score is not None:
        ranked = [d for d in ranked if d.score is None or d.score >= min_relevance_score]

    if any(d.score is not None for d in ranked):
        ranked.sort(key=_rank_key, reverse=True)

    by_type: dict[str, list[Document]] = {}
    for d in ranked:
        st = str((d.metadata or {}).get("section_type", "") or "").strip()
        if not st:
            continue
        by_type.setdefault(st, []).append(d)

    selected: list[Document] = []
    missing: list[str] = []

    for st in needed_types:
        pool = by_type.get(st, [])
        if not pool:
            missing.append(st)
            continue
        selected.extend(pool[:per_type_limit])

    # Enforce per-source limit (drop lowest-ranked chunks from over-represented sources).
    def _source_key(d: Document) -> str:
        meta = d.metadata or {}
        return str(meta.get("style_family") or meta.get("source") or "")

    if per_source_limit > 0 and selected:
        # Build rank index for deterministic dropping when no scores.
        rank_index = {id(d): i for i, d in enumerate(ranked)}
        per_source: dict[str, list[Document]] = {}
        for d in selected:
            per_source.setdefault(_source_key(d), []).append(d)

        pruned: list[Document] = []
        for src, docs in per_source.items():
            if len(docs) <= per_source_limit:
                pruned.extend(docs)
                continue
            # Keep the best; drop the rest.
            docs_sorted = sorted(
                docs,
                key=lambda x: (_rank_key(x), -rank_index.get(id(x), 0)),
                reverse=True,
            )
            pruned.extend(docs_sorted[:per_source_limit])
        selected = pruned

    # Enforce overall budget by removing lowest-ranked chunks first.
    def _chunk_len(d: Document) -> int:
        txt = (d.page_content or "").strip()
        return min(len(txt), per_chunk_chars)

    def _selected_rank_tuple(d: Document) -> tuple[float, int]:
        # Smaller tuple = worse, so we can sort ascending for dropping.
        idx = ranked_index.get(id(d), 0)
        return (_rank_key(d), -idx)

    ranked_index = {id(d): i for i, d in enumerate(ranked)}

    # Preserve order while de-duping by stable content/metadata key.
    deduped: list[Document] = []
    seen: set[tuple] = set()
    for d in selected:
        meta = d.metadata or {}
        key = (
            str(meta.get("source", "")),
            str(meta.get("section_type", "")),
            str(meta.get("chunk_index", "")),
            (d.page_content or "")[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)
    selected = deduped
    used = sum(_chunk_len(d) for d in selected)
    if used > budget_chars:
        # Sort worst-first for dropping.
        drop_order = sorted(selected, key=_selected_rank_tuple)
        keep: set[int] = set(id(d) for d in selected)
        for d in drop_order:
            if used <= budget_chars:
                break
            if id(d) not in keep:
                continue
            keep.remove(id(d))
            used -= _chunk_len(d)
        selected = [d for d in selected if id(d) in keep]

    # Recompute missing types after selection (we don't drop type-critical chunks first).
    present_types = {str((d.metadata or {}).get("section_type", "") or "") for d in selected}
    missing = [st for st in needed_types if st not in present_types]

    return selected, missing


def build_exemplar_blocks(
    query: str,
    needed_types: list[str],
    module: str | None = None,
    process_id: str | None = None,
) -> tuple[str, str]:
    """
    Retrieve and build (style_block, step_block) with explicit fallbacks.
    """
    candidates: list[Document] = []
    module_filter = {"module": module} if module else None
    initial_k = _adaptive_candidate_k(needed_types)

    # Primary retrieval: semantic query within module scope.
    candidates = retrieve_candidates(query, k=initial_k, filter=module_filter, strategy="scores")

    # Optional strict narrowing by process_id as a secondary attempt only.
    if not candidates and process_id:
        id_filter = {"process_id": process_id}
        candidates = retrieve_candidates(query, k=initial_k, filter=id_filter, strategy="scores")

    selected, missing = select_exemplars(
        candidates=candidates,
        needed_types=needed_types,
        per_type_limit=1,
        per_source_limit=RAG_MAX_CHUNKS_PER_SOURCE,
        min_relevance_score=0.45,  # Relevance gating to block bad noise
    )

    # If type coverage is missing, widen the candidate pool before hard fallbacks.
    if missing:
        expanded_k = _escalated_candidate_k(initial_k)
        if expanded_k > initial_k:
            expanded_candidates = retrieve_candidates(
                query,
                k=expanded_k,
                filter=module_filter,
                strategy="scores",
            )
            if expanded_candidates:
                candidates = _merge_unique_docs(candidates, expanded_candidates)
                selected, missing = select_exemplars(
                    candidates=candidates,
                    needed_types=needed_types,
                    per_type_limit=1,
                    per_source_limit=RAG_MAX_CHUNKS_PER_SOURCE,
                    min_relevance_score=0.45,
                )

    # Fallback per missing type: broaden query and post-filter by section_type.
    # Avoid passing multi-key metadata filters to Chroma here (backend rejects them).
    if missing:
        broaden = query
        if module:
            broaden = f"{module} {query}"
        for st in list(missing):
            fallback_query = f"{module or ''} {st.replace('_', ' ')}".strip()
            fallback_k = max(TOP_K_RETRIEVAL, 8)

            # First fallback attempt: module-scoped retrieval then local section_type filter.
            more = retrieve_candidates(
                fallback_query,
                k=fallback_k,
                filter=module_filter,
                strategy="mmr",
            )
            typed = [d for d in more if str((d.metadata or {}).get("section_type", "") or "") == st]

            # Last-resort fallback: unscoped retrieval then local section_type filter.
            if not typed and module_filter is not None:
                more = retrieve_candidates(
                    fallback_query,
                    k=fallback_k,
                    filter=None,
                    strategy="mmr",
                )
                typed = [d for d in more if str((d.metadata or {}).get("section_type", "") or "") == st]

            if typed:
                selected = _merge_unique_docs(selected, [typed[0]])

    # Split into style vs step exemplars by section_type.
    style_types = {
        "document_control",
        "intro",
        "how_organized",
        "enterprise_structure",
        "ledger",
        "coa",
        "module_overview",
        "process_outline",
        "process_narrative",
        "key_requirements",
        "issues_open",
        "issues_closed",
    }
    step_types = {"process_steps", "process_diagram", "journal_entries"}

    style_docs: list[Document] = []
    step_docs: list[Document] = []

    for d in selected:
        st = str((d.metadata or {}).get("section_type", "") or "")
        if st in step_types:
            step_docs.append(d)
        elif st in style_types:
            style_docs.append(d)

    style_block = format_rag_context(style_docs, max_chars=RAG_MAX_CONTEXT_CHARS // 2)
    step_block = format_rag_context(step_docs, max_chars=RAG_MAX_CONTEXT_CHARS // 2)
    return style_block, step_block


def format_rag_context(
    docs: list[Document],
    max_chars: int | None = None,
    per_chunk_chars: int | None = None,
    max_chunks_per_source: int | None = None,
) -> str:
    """
    Render retrieved docs into a prompt-safe context string.

    This prevents 413/request-too-large failures and reduces TPM pressure by
    bounding the injected context.
    """
    if not docs:
        return ""

    if max_chars is None:
        max_chars = RAG_MAX_CONTEXT_CHARS
    if per_chunk_chars is None:
        per_chunk_chars = RAG_MAX_CHUNK_CHARS
    if max_chunks_per_source is None:
        max_chunks_per_source = RAG_MAX_CHUNKS_PER_SOURCE

    out_parts: list[str] = []
    used = 0
    per_source: dict[str, int] = {}

    for d in docs:
        meta = d.metadata or {}
        src = str(meta.get("source", "") or "")
        section = str(meta.get("section", "") or "")

        if src:
            count = per_source.get(src, 0)
            if count >= max_chunks_per_source:
                continue
            per_source[src] = count + 1

        text = (d.page_content or "").strip()
        if not text:
            continue
        if len(text) > per_chunk_chars:
            text = text[:per_chunk_chars].rstrip() + "..."

        header_bits = []
        if src:
            header_bits.append(f"source={src}")
        style_family = str(meta.get("style_family", "") or "")
        if style_family and style_family != src:
            header_bits.append(f"style_family={style_family}")

        section_type = str(meta.get("section_type", "") or "")
        if section_type:
            header_bits.append(f"section_type={section_type}")
        if section:
            header_bits.append(f"section={section[:60]}")

        module = str(meta.get("module", "") or "")
        if module:
            header_bits.append(f"module={module}")
        process_id = str(meta.get("process_id", "") or "")
        if process_id:
            header_bits.append(f"process_id={process_id}")
        process_name = str(meta.get("process_name", "") or "")
        if process_name:
            header_bits.append(f"process_name={process_name[:48]}")
        header = f"[{', '.join(header_bits)}]" if header_bits else "[example]"
        chunk = f"{header}\n{text}"

        # +2 for the double newline separators we add between chunks
        projected = used + len(chunk) + 2
        if projected > max_chars and out_parts:
            break
        if projected > max_chars and not out_parts:
            # If even the first chunk is too large, hard-truncate to max_chars.
            chunk = chunk[:max_chars].rstrip()
            out_parts.append(chunk)
            break

        out_parts.append(chunk)
        used = projected

    return "\n\n".join(out_parts)
