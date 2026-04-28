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

from config import TaskType, normalize_business_actor
from llm.retry import call_with_retry
from models.schemas import DocumentPlan, ExtractionResult, ProcessEntry
from prompts.planning_prompt import build_planning_prompt

logger = logging.getLogger(__name__)

# Path to implicit processes config (sibling to config.py)
IMPLICIT_PROCESSES_CONFIG = Path(__file__).parent.parent / "config_implicit_processes.json"

_PLURAL_TOKEN_MAP = {
    "customers": "customer",
    "suppliers": "supplier",
    "entries": "entry",
    "invoices": "invoice",
    "assets": "asset",
    "processes": "process",
}

_NAME_ALIAS_PATTERNS = (
    (r"\bcreate journal entries?\b", "manual journal entry"),
    (r"\bjournal entry creation\b", "manual journal entry"),
    (r"\bmonth[- ]end close\b", "month end closing"),
    (r"\bperiod clos(?:e|ing)\b", "month end closing"),
    (r"\bmaintain customers data\b", "maintain customer data"),
    (r"\bmaintain suppliers data\b", "maintain supplier data"),
)

_FIXED_MODULE_ORDER = ["AP", "AR", "GL", "FA", "CM"]

def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return " ".join(cleaned.split())


def _canonical_process_name(name: str) -> str:
    """Normalize process names for duplicate detection."""
    normalized = _normalize_name(name)
    if not normalized:
        return ""

    tokens = [_PLURAL_TOKEN_MAP.get(tok, tok) for tok in normalized.split()]
    normalized = " ".join(tokens)

    for pattern, replacement in _NAME_ALIAS_PATTERNS:
        normalized = re.sub(pattern, replacement, normalized)

    return " ".join(normalized.split())


def _process_suffix(process_id: str) -> str:
    """Extract the trailing MODULE.NN portion from any process_id format."""
    if not process_id:
        return ""
    matches = re.findall(r"([A-Z]{2,3}\.\d{2})", str(process_id).upper())
    return matches[-1] if matches else ""


def _names_equivalent(left: str, right: str) -> bool:
    """
    Process-name equivalence used for dedup and implicit matching.

    IMPORTANT:
    We intentionally keep this strict to avoid accidentally dropping implicit
    processes (user requirement: implicit list should be a minimum baseline).
    More tolerant semantic matching (LLM-judged duplicates) is deferred.
    """
    left_norm = _canonical_process_name(left)
    right_norm = _canonical_process_name(right)

    if not left_norm or not right_norm:
        return False
    return left_norm == right_norm


def _unique_preserve_order(values: list[str]) -> list[str]:
    """Remove duplicates while preserving original order."""
    seen = set()
    unique = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _enforce_fixed_section_order(plan: DocumentPlan) -> DocumentPlan:
    """Enforce a deterministic module chapter order across all runs."""
    module_aliases = {"CE": "CM"}
    rank = {module: idx for idx, module in enumerate(_FIXED_MODULE_ORDER)}

    indexed_sections = []
    for original_idx, section in enumerate(plan.sections):
        section_id = module_aliases.get(section.section_id, section.section_id)
        section.section_id = section_id
        indexed_sections.append((rank.get(section_id, len(rank) + original_idx), original_idx, section))

    indexed_sections.sort(key=lambda item: (item[0], item[1]))
    plan.sections = [section for _, _, section in indexed_sections]
    return plan


def _apply_section_actor_context(
    plan: DocumentPlan,
    extraction: ExtractionResult | None,
) -> DocumentPlan:
    """Attach client-confirmed actor lists to each section for generation constraints."""
    if extraction is None:
        return plan

    module_aliases = {"CE": "CM"}
    business_actor_map = extraction.business_actors or {}
    org_roles_map = extraction.org_roles or {}

    for section in plan.sections:
        section_id = module_aliases.get(section.section_id, section.section_id)
        section.section_id = section_id

        extracted_business_actors = business_actor_map.get(section_id, [])
        extracted_org_roles = org_roles_map.get(section_id, [])

        if extracted_org_roles:
            section.org_roles = _unique_preserve_order(extracted_org_roles)
            normalized_roles = [normalize_business_actor(role) for role in section.org_roles]
            section.business_actors = _unique_preserve_order(normalized_roles)
        elif extracted_business_actors:
            normalized_actors = [normalize_business_actor(actor) for actor in extracted_business_actors]
            section.business_actors = _unique_preserve_order(normalized_actors)
        else:
            section.business_actors = _unique_preserve_order(section.business_actors)
            section.org_roles = _unique_preserve_order(section.org_roles)

    return plan


def _deduplicate_processes(processes: list[ProcessEntry]) -> list[ProcessEntry]:
    """Remove duplicate process entries by id or near-equivalent name, preserving order."""
    deduped: list[ProcessEntry] = []
    seen_suffixes: set[str] = set()
    seen_names: list[str] = []

    for proc in processes:
        suffix = _process_suffix(proc.process_id)
        name = proc.process_name or ""

        if suffix and suffix in seen_suffixes:
            continue
        if name and any(_names_equivalent(name, existing) for existing in seen_names):
            continue

        deduped.append(proc)
        if suffix:
            seen_suffixes.add(suffix)
        if name:
            seen_names.append(name)

    return deduped


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
        existing_suffixes = {
            _process_suffix(p.process_id)
            for p in section.processes
            if _process_suffix(p.process_id)
        }

        def _is_present(imp: dict) -> bool:
            """
            Treat as present if process_id suffix matches OR names are equivalent.
            """
            imp_name = imp.get("process_name") or ""
            imp_suffix = _process_suffix(imp.get("process_id") or "")

            if imp_suffix and imp_suffix in existing_suffixes:
                return True

            return bool(
                imp_name and any(_names_equivalent(imp_name, existing) for existing in existing_names)
            )

        def _mark_present(imp: dict) -> None:
            imp_name = imp.get("process_name") or ""
            imp_suffix = _process_suffix(imp.get("process_id") or "")
            if imp_name:
                existing_names.append(imp_name)
            if imp_suffix:
                existing_suffixes.add(imp_suffix)

        def _implicit_index_for_proc(proc: ProcessEntry) -> int | None:
            name = proc.process_name or ""
            suffix = _process_suffix(proc.process_id)
            if not name and not suffix:
                return None
            for idx, imp in module_entries:
                imp_suffix = _process_suffix(imp.get("process_id") or "")
                if suffix and imp_suffix and suffix == imp_suffix:
                    return idx
                if _names_equivalent(name, imp.get("process_name") or ""):
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

        section.processes = _deduplicate_processes(merged)

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
        plan = _apply_section_actor_context(plan, extraction)
        plan = _enforce_fixed_section_order(plan)

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
