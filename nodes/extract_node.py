"""
RD.011 Agent — Extract node.

Sends all parsed text to a large-context LLM and produces a structured
ExtractionResult with client details, modules, actors, and requirements.
"""

from __future__ import annotations

import logging
import re

from config import TaskType, normalize_business_actor
from llm.retry import call_with_retry
from models.schemas import ExtractionResult
from prompts.extraction_prompt import build_extraction_prompt

logger = logging.getLogger(__name__)


_EMPTY_CONFLICT_PHRASES = {
    "not specified",
    "not mentioned",
    "not stated",
    "not provided",
    "not documented",
    "not available",
    "n/a",
    "na",
    "tbd",
    "unknown",
    "unclear",
    "silent",
    "missing",
}

_MAX_CANDIDATE_PROCESSES_PER_MODULE = 12


def _normalize_search_text(value: str) -> str:
    text = (value or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_source_evidence(role: str, source_text: str) -> bool:
    role_norm = _normalize_search_text(role)
    if not role_norm:
        return False

    if role_norm in source_text:
        return True

    role_tokens = [tok for tok in role_norm.split() if len(tok) >= 3]
    if len(role_tokens) >= 2:
        return all(tok in source_text for tok in role_tokens)

    return False


def _is_empty_side(value: str) -> bool:
    text = (value or "").strip().lower()
    if not text:
        return True
    return any(phrase in text for phrase in _EMPTY_CONFLICT_PHRASES)


def _normalized_conflict_value(value: str) -> str:
    text = (value or "").lower()
    text = re.sub(r"\(source:.*?\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("three-way", "3-way").replace("four-way", "4-way")
    text = text.replace("two-way", "2-way")
    return text


def _is_conflict_like_requirement(text: str) -> bool:
    value = (text or "").lower()
    patterns = (
        r"\bolder\b",
        r"\bnewer\b",
        r"\bcontradict",
        r"\bconflict",
        r"\bvs\b",
        r"\bversus\b",
        r"\bsupersed",
        r"\bopen question",
        r"\bnot specified\b",
        r"\bnot mentioned\b",
        r"\bsource order\b",
    )
    return any(re.search(pattern, value) for pattern in patterns)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = (item or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _sanitize_extraction_result(data: dict) -> dict:
    """Normalize extracted facts to avoid propagating conflict noise downstream."""
    cleaned = dict(data)

    # Keep only meaningful conflicts with two explicit incompatible sides.
    conflicts = []
    for conflict in cleaned.get("conflicts_between_documents", []) or []:
        older = str(conflict.get("older_value", ""))
        newer = str(conflict.get("newer_value", ""))

        if _is_empty_side(older) or _is_empty_side(newer):
            continue

        if _normalized_conflict_value(older) == _normalized_conflict_value(newer):
            continue

        conflicts.append(conflict)
    cleaned["conflicts_between_documents"] = conflicts

    # Remove conflict/open-question prose from module requirements.
    reqs = cleaned.get("requirements_per_module", {}) or {}
    reqs_out = {}
    for module, module_reqs in reqs.items():
        safe_reqs = [
            r for r in (module_reqs or [])
            if not _is_conflict_like_requirement(r)
        ]
        reqs_out[module] = _dedupe_preserve_order(safe_reqs)
    cleaned["requirements_per_module"] = reqs_out

    # Normalize candidate process hints from Key Points Discussed bullets.
    candidates = cleaned.get("candidate_processes", {}) or {}
    candidates_out: dict[str, list[str]] = {}
    for module, items in candidates.items():
        cleaned_items = _dedupe_preserve_order([
            str(item).strip() for item in (items or []) if str(item).strip()
        ])
        if not cleaned_items:
            continue
        if len(cleaned_items) > _MAX_CANDIDATE_PROCESSES_PER_MODULE:
            logger.info(
                "Trimming candidate_processes for %s from %d to %d",
                module,
                len(cleaned_items),
                _MAX_CANDIDATE_PROCESSES_PER_MODULE,
            )
            cleaned_items = cleaned_items[:_MAX_CANDIDATE_PROCESSES_PER_MODULE]
        candidates_out[module] = cleaned_items
    cleaned["candidate_processes"] = candidates_out

    return cleaned


def extract_node(state: dict) -> dict:
    """
    Extract structured information from all parsed input files.

    Uses LARGE_CONTEXT model to handle the full concatenation of all
    input documents in a single call.
    """
    raw_texts = state.get("raw_texts", {})
    errors = list(state.get("errors", []))

    if not raw_texts:
        errors.append("No raw texts available for extraction.")
        return {
            "extraction_result": None,
            "errors": errors,
            "last_completed_node": "extract",
        }

    prompt = build_extraction_prompt(raw_texts)
    source_corpus = _normalize_search_text("\n".join(raw_texts.values()))

    try:
        result = call_with_retry(TaskType.LARGE_CONTEXT, prompt, ExtractionResult)

        # Normalize actor context with priority to client-confirmed org_roles.
        normalized = result.model_dump()
        org_roles_raw = normalized.get("org_roles", {}) or {}
        org_roles: dict[str, list[str]] = {}
        for module, role_list in org_roles_raw.items():
            cleaned_roles: list[str] = []
            for role in role_list or []:
                role_name = " ".join(str(role).split()).strip()
                if not role_name:
                    continue
                if _has_source_evidence(role_name, source_corpus) and role_name not in cleaned_roles:
                    cleaned_roles.append(role_name)
            if cleaned_roles:
                org_roles[module] = cleaned_roles
            elif role_list:
                # Keep extracted roles as fallback when source matching is inconclusive.
                org_roles[module] = _dedupe_preserve_order([str(r).strip() for r in role_list if str(r).strip()])

        actors_raw = normalized.get("business_actors", {}) or {}
        resolved_actors: dict[str, list[str]] = {}
        module_keys = set(normalized.get("modules_in_scope", []) or [])
        module_keys.update(org_roles.keys())
        module_keys.update(actors_raw.keys())

        for module in module_keys:
            mapped_list: list[str] = []
            if org_roles.get(module):
                for role in org_roles[module]:
                    mapped = normalize_business_actor(role)
                    if mapped and mapped not in mapped_list:
                        mapped_list.append(mapped)
            else:
                for actor in actors_raw.get(module, []):
                    actor_name = " ".join(str(actor).split()).strip()
                    if not actor_name:
                        continue
                    mapped = normalize_business_actor(actor_name)
                    if mapped and mapped not in mapped_list:
                        mapped_list.append(mapped)
            resolved_actors[module] = mapped_list

        normalized["org_roles"] = org_roles
        normalized["business_actors"] = resolved_actors
        normalized = _sanitize_extraction_result(normalized)

        logger.info(
            "Extraction complete: %d modules, %d actors groups",
            len(result.modules_in_scope),
            len(result.business_actors),
        )
        return {
            "extraction_result": normalized,
            "last_completed_node": "extract",
        }
    except RuntimeError as exc:
        error_msg = f"Extraction failed after retries: {exc}"
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            "extraction_result": None,
            "errors": errors,
            "last_completed_node": "extract",
        }
