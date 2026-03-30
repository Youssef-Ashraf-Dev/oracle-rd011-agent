"""
RD.011 Agent — Generate Section node.

Generates content for a single Oracle Finance Cloud process. Iterates
through the section_queue, producing SectionContent for each process
with optional RAG context.
"""

from __future__ import annotations

import logging
import time

from config import RAG_ENABLED, TOP_K_RETRIEVAL, TaskType
from llm.retry import call_with_retry
from models.schemas import (
    DocumentPlan,
    ExtractionResult,
    SectionContent,
    SectionPlan,
    ProcessEntry,
)
from prompts.generation_prompt import build_generation_prompt

logger = logging.getLogger(__name__)


def _find_section_and_process(
    plan: DocumentPlan, section_queue_key: str
) -> tuple[SectionPlan | None, ProcessEntry | None]:
    """
    Find the SectionPlan and ProcessEntry for a section_queue key.

    The key format is "{section_id}.{process_id}" e.g. "AP.AP.01".
    """
    parts = section_queue_key.split(".", 1)
    if len(parts) < 2:
        return None, None

    target_section_id = parts[0]
    target_process_id = parts[1]

    for section in plan.sections:
        if section.section_id == target_section_id:
            for proc in section.processes:
                if proc.process_id == target_process_id:
                    return section, proc

    return None, None


def generate_section_node(state: dict) -> dict:
    """
    Generate content for the current process in the section queue.

    On success: stores the SectionContent and increments the index.
    On failure: appends to failed_sections and increments the index.
    """
    idx = state.get("current_section_index", 0)
    section_queue = state.get("section_queue", [])
    plan_data = state.get("document_plan")
    extraction_data = state.get("extraction_result", {})
    generated_sections = dict(state.get("generated_sections", {}))
    failed_sections = list(state.get("failed_sections", []))
    errors = list(state.get("errors", []))

    if idx >= len(section_queue):
        logger.info("All sections generated")
        return {"last_completed_node": "generate_section"}

    section_key = section_queue[idx]
    logger.info("Generating section %d/%d: %s", idx + 1, len(section_queue), section_key)

    if not plan_data:
        errors.append(f"No document plan for section {section_key}")
        return {
            "current_section_index": idx + 1,
            "errors": errors,
            "last_completed_node": "generate_section",
        }

    plan = DocumentPlan.model_validate(plan_data)
    section_plan, process = _find_section_and_process(plan, section_key)

    if not section_plan or not process:
        error_msg = f"Could not find section/process for key: {section_key}"
        logger.error(error_msg)
        errors.append(error_msg)
        failed_sections.append(section_key)
        return {
            "current_section_index": idx + 1,
            "generated_sections": generated_sections,
            "failed_sections": failed_sections,
            "errors": errors,
            "last_completed_node": "generate_section",
        }

    # Build requirements text
    extraction = ExtractionResult.model_validate(extraction_data) if extraction_data else None
    requirements_list = []
    if extraction:
        requirements_list = extraction.requirements_per_module.get(
            section_plan.section_id, []
        )
    requirements_text = "\n".join(f"- {r}" for r in requirements_list) or "No specific requirements captured."

    # RAG context (optional)
    rag_context = ""
    if RAG_ENABLED:
        try:
            from rag.retriever import retrieve
            query = f"{section_plan.module_name} {process.process_name} Oracle Finance Cloud process steps"
            docs = retrieve(query, top_k=TOP_K_RETRIEVAL)
            rag_context = "\n\n".join(d.page_content for d in docs)
        except NotImplementedError:
            logger.info("RAG not available - generating without reference examples")
        except Exception as exc:
            logger.warning("RAG query failed: %s", exc)
    else:
        logger.info("RAG disabled - generating without reference examples")

    # Build prompt and call LLM
    prompt = build_generation_prompt(
        section_plan=section_plan,
        process=process,
        requirements_text=requirements_text,
        rag_context=rag_context,
    )

    try:
        content = call_with_retry(TaskType.GENERATION, prompt, SectionContent)
        generated_sections[section_key] = content.model_dump()
        logger.info(
            "Generated %s: %d steps, %d journal entries",
            section_key,
            len(content.process_steps),
            len(content.journal_entries),
        )
    except Exception as exc:
        error_msg = f"Generation failed for {section_key}: {exc}"
        logger.error(error_msg)
        errors.append(error_msg)
        failed_sections.append(section_key)

    # Throttle to stay under Groq free-tier TPM limit
    time.sleep(4)

    return {
        "current_section_index": idx + 1,
        "generated_sections": generated_sections,
        "failed_sections": failed_sections,
        "errors": errors,
        "last_completed_node": "generate_section",
    }
