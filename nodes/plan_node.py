"""
RD.011 Agent — Plan node.

Uses a reasoning model to create a complete DocumentPlan from the
extraction results, defining processes for each Oracle module in scope.
Augments with implicit required processes post-LLM.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from config import TaskType
from llm.retry import call_with_retry
from models.schemas import DocumentPlan, ExtractionResult, ProcessEntry
from prompts.planning_prompt import build_planning_prompt

logger = logging.getLogger(__name__)

# Path to implicit processes config (sibling to config.py)
IMPLICIT_PROCESSES_CONFIG = Path(__file__).parent.parent / "config_implicit_processes.json"

def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return " ".join(cleaned.split())


def _load_implicit_processes() -> list:
    """Load implicit processes from config file (ordered list)."""
    if not IMPLICIT_PROCESSES_CONFIG.exists():
        logger.warning("Implicit processes config not found at %s", IMPLICIT_PROCESSES_CONFIG)
        return []

    try:
        with open(IMPLICIT_PROCESSES_CONFIG) as f:
            data = json.load(f)
            return list(data.get("implicit_processes", []))
    except Exception as e:
        logger.warning("Failed to load implicit processes: %s", e)
        return []


def _augment_plan_with_implicit_processes(plan: DocumentPlan, implicit_procs: list) -> DocumentPlan:
    """
    Merge implicit processes into the plan while preserving MoM order.

    - MoM (explicit) processes keep their relative order.
    - Implicit processes are inserted in their logical order.
    - If a process exists in both, the MoM version wins (implicit skipped).
    """
    if not implicit_procs:
        return plan

    # Derive client name prefix for process IDs (e.g. "Contoso" → "Contoso.")
    clean_name = plan.client_name.rstrip(" .") if plan.client_name else ""
    client_prefix = f"{clean_name}." if clean_name else ""

    # Normalize legacy CE to CM
    module_aliases = {"CE": "CM"}

    # Precompute implicit entries with global order index
    implicit_entries = [
        (idx, proc) for idx, proc in enumerate(implicit_procs)
        if proc.get("module")
    ]

    for section in plan.sections:
        module_id = module_aliases.get(section.section_id, section.section_id)
        section.section_id = module_id
        module_entries = [
            (idx, proc) for idx, proc in implicit_entries
            if proc.get("module") == module_id
        ]
        if not module_entries:
            continue

        existing_names = [p.process_name for p in section.processes if p.process_name]
        existing_norms = {_normalize_name(n) for n in existing_names if n}

        def _is_present(imp: dict) -> bool:
            """
            Strict baseline: only treat as present if the name matches exactly
            after normalization (case/whitespace/punctuation-insensitive).
            """
            imp_name = imp.get("process_name") or ""
            imp_norm = _normalize_name(imp_name)
            return bool(imp_norm and imp_norm in existing_norms)

        def _mark_present(imp: dict) -> None:
            imp_name = imp.get("process_name") or ""
            imp_norm = _normalize_name(imp_name)
            if imp_name:
                existing_names.append(imp_name)
            if imp_norm:
                existing_norms.add(imp_norm)

        def _implicit_index_for_proc(proc: ProcessEntry) -> int | None:
            name = _normalize_name(proc.process_name or "")
            if not name:
                return None
            for idx, imp in module_entries:
                if name == _normalize_name(imp.get("process_name") or ""):
                    return idx
            return None

        def _make_process(imp: dict) -> ProcessEntry:
            return ProcessEntry(
                process_id=f"{client_prefix}{imp.get('process_id')}",
                process_name=imp.get("process_name"),
                process_description=imp.get("description"),
                output=imp.get("process_name"),
                confidence=imp.get("default_confidence", "high"),
                missing_info=[],
            )

        merged: list[ProcessEntry] = []
        last_idx = -1

        for proc in section.processes:
            cur_idx = _implicit_index_for_proc(proc)
            if cur_idx is not None:
                # Insert missing implicit processes between last_idx and cur_idx
                for idx, imp in module_entries:
                    if idx <= last_idx or idx >= cur_idx:
                        continue
                    if _is_present(imp):
                        continue
                    merged.append(_make_process(imp))
                    _mark_present(imp)
                    last_idx = idx

            merged.append(proc)
            if cur_idx is not None and cur_idx > last_idx:
                last_idx = cur_idx

        # Append remaining implicit processes after the last anchor
        for idx, imp in module_entries:
            if idx <= last_idx:
                continue
            if _is_present(imp):
                continue
            merged.append(_make_process(imp))
            _mark_present(imp)
            last_idx = idx

        section.processes = merged

    return plan


def _order_and_renumber_processes(plan: DocumentPlan, implicit_procs: list) -> DocumentPlan:
    """
    Renumber process_id sequentially per module while preserving current order.

    MoM order is preserved; implicit processes are inserted earlier in the
    merge step. This function only assigns stable sequential IDs.
    """
    # Derive client name prefix for process IDs
    clean_name = plan.client_name.rstrip(" .") if plan.client_name else ""
    client_prefix = f"{clean_name}." if clean_name else ""

    for section in plan.sections:
        module_id = section.section_id
        for i, proc in enumerate(section.processes, start=1):
            proc.process_id = (
                f"{client_prefix}{module_id}.{i:02d}"
                if client_prefix
                else f"{module_id}.{i:02d}"
            )

    return plan


def plan_node(state: dict) -> dict:
    """
    Create a DocumentPlan with ordered processes for each module.

    Builds the section_queue for downstream generation loop.
    Augments the plan with implicit required processes post-LLM.
    """
    extraction_data = state.get("extraction_result")
    errors = list(state.get("errors", []))

    if not extraction_data:
        errors.append("No extraction result available for planning.")
        return {
            "document_plan": None,
            "section_queue": [],
            "errors": errors,
            "last_completed_node": "plan",
        }

    extraction = ExtractionResult.model_validate(extraction_data)
    prompt = build_planning_prompt(extraction)

    try:
        plan = call_with_retry(TaskType.REASONING, prompt, DocumentPlan)

        # Augment with implicit required processes
        implicit_procs = _load_implicit_processes()
        plan = _augment_plan_with_implicit_processes(plan, implicit_procs)
        plan = _order_and_renumber_processes(plan, implicit_procs)

        # Build section queue: "{section_id}.{process_id}" for each process
        section_queue = [
            f"{section.section_id}.{proc.process_id}"
            for section in plan.sections
            for proc in section.processes
        ]

        logger.info(
            "Planning complete: %d sections, %d total processes (including implicit)",
            len(plan.sections),
            len(section_queue),
        )

        return {
            "document_plan": plan.model_dump(),
            "section_queue": section_queue,
            "last_completed_node": "plan",
        }
    except RuntimeError as exc:
        error_msg = f"Planning failed after retries: {exc}"
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            "document_plan": None,
            "section_queue": [],
            "errors": errors,
            "last_completed_node": "plan",
        }
