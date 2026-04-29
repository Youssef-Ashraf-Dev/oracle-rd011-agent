"""
RD.011 Agent — Update Plan node.

Revises the DocumentPlan based on consultant feedback using a reasoning
model, then loops back to the present_plan node.
"""

from __future__ import annotations

import logging

from config import TaskType
from llm.retry import call_with_retry
from models.schemas import DocumentConflict, DocumentPlan, ExtractionResult
from prompts.update_prompt import build_update_prompt
from nodes.plan_node import (
    _load_implicit_processes,
    _augment_plan_with_implicit_processes,
    _order_and_renumber_processes,
    _apply_section_actor_context,
    _enforce_fixed_section_order,
)

logger = logging.getLogger(__name__)


def update_plan_node(state: dict) -> dict:
    """
    Revise the document plan based on consultant feedback.

    Passes structured document conflicts to the update prompt so the LLM
    can apply any consultant override decisions. Rebuilds section_queue
    after plan update.
    """
    plan_data = state.get("document_plan")
    feedback = state.get("consultant_feedback", "")
    extraction_data = state.get("extraction_result")
    errors = list(state.get("errors", []))

    if not plan_data:
        errors.append("No document plan to update.")
        return {"errors": errors, "last_completed_node": "update_plan"}

    # Extract structured document conflicts so the LLM can apply resolutions
    conflicts: list[DocumentConflict] = []
    extraction: ExtractionResult | None = None
    if extraction_data:
        try:
            extraction = ExtractionResult.model_validate(extraction_data)
            conflicts = extraction.conflicts_between_documents
        except Exception:
            pass

    current_plan = DocumentPlan.model_validate(plan_data)
    prompt = build_update_prompt(current_plan, feedback, conflicts=conflicts, extraction=extraction)

    try:
        updated_plan = call_with_retry(TaskType.REASONING, prompt, DocumentPlan)

        # Re-apply implicit processes and renumber in logical order
        implicit_procs = _load_implicit_processes()
        updated_plan = _augment_plan_with_implicit_processes(updated_plan, implicit_procs)
        updated_plan = _order_and_renumber_processes(updated_plan, implicit_procs)
        updated_plan = _apply_section_actor_context(updated_plan, extraction)
        updated_plan = _enforce_fixed_section_order(updated_plan)

        # Rebuild section queue
        section_queue = [
            f"{section.section_id}.{proc.process_id}"
            for section in updated_plan.sections
            for proc in section.processes
        ]

        logger.info("Plan updated based on feedback. New queue: %d items", len(section_queue))

        return {
            "document_plan": updated_plan.model_dump(),
            "section_queue": section_queue,
            "consultant_approved": False,
            "last_completed_node": "update_plan",
        }
    except RuntimeError as exc:
        error_msg = f"Plan update failed: {exc}"
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            "errors": errors,
            "last_completed_node": "update_plan",
        }
