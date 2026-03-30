"""
RD.011 Agent — Extract node.

Sends all parsed text to a large-context LLM and produces a structured
ExtractionResult with client details, modules, actors, and requirements.
"""

from __future__ import annotations

import logging

from config import CANONICAL_BUSINESS_ACTORS, TaskType, normalize_business_actor
from llm.retry import call_with_retry
from models.schemas import ExtractionResult
from prompts.extraction_prompt import build_extraction_prompt

logger = logging.getLogger(__name__)


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

    try:
        result = call_with_retry(TaskType.LARGE_CONTEXT, prompt, ExtractionResult)

        # Normalize business actor names to canonical list for consistency
        normalized = result.model_dump()
        actors = normalized.get("business_actors", {})
        for module, actor_list in actors.items():
            mapped_list = []
            for actor in actor_list:
                mapped = normalize_business_actor(actor)
                if mapped not in mapped_list:
                    mapped_list.append(mapped)
            actors[module] = mapped_list
        normalized["business_actors"] = actors

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
