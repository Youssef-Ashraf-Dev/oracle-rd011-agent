"""
RD.011 Agent — Generate Section node.

Generates content for a single Oracle Finance Cloud process. Iterates
through the section_queue, producing SectionContent for each process
with optional RAG context.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import time
from difflib import SequenceMatcher

from config import (
    RAG_ENABLED,
    TOP_K_RETRIEVAL,
    CANONICAL_BUSINESS_ACTORS,
    ENABLE_REPAIR_PASS,
    GENERATION_THROTTLE_ON_FAILURE_ONLY,
    GENERATION_THROTTLE_SECONDS,
    TaskType,
    normalize_business_actor,
)
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

_MODULE_CANONICAL_NAME = {
    "AP": "Oracle Payables",
    "AR": "Oracle Receivables",
    "GL": "Oracle General Ledger",
    "FA": "Oracle Fixed Assets",
    "CM": "Oracle Cash Management",
}

_STOPWORDS = {
    "and", "the", "for", "with", "from", "that", "this", "into",
    "will", "are", "was", "were", "been", "have", "has", "had",
    "process", "module", "oracle", "cloud", "entry", "data", "management",
}

_REQUIREMENT_NOISE_PATTERNS = (
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

_PROCESS_HINTS = {
    "direct invoice": {"direct", "non", "po", "invoice"},
    "po invoice": {"po", "purchase", "order", "invoice", "matching"},
    "payment": {"payment", "disbursement", "bank", "run", "remittance"},
    "receipt": {"receipt", "cash", "collection", "remittance"},
    "reversal": {"reversal", "cancel", "cancellation", "void", "reverse"},
    "memo": {"memo", "debit", "credit"},
    "supplier": {"supplier", "vendor", "onboarding", "master"},
    "customer": {"customer", "client", "master", "credit"},
    "journal": {"journal", "entry", "posting", "ledger"},
    "asset": {"asset", "fixed", "depreciation", "retirement", "cip"},
    "reconciliation": {"reconciliation", "statement", "bank"},
    "month end": {"month", "close", "closing", "period", "cutoff"},
}

_IMPLICIT_PROCESSES_CONFIG = Path(__file__).parent.parent / "config_implicit_processes.json"
_ROLE_LIKE_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s(?:Accountant|Manager|Controller|Clerk|Specialist|Administrator|Department))\b"
)
_ROLE_STRIP_PREFIX_PATTERN = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_ORACLE_PRODUCT_PATTERN = re.compile(r"\bOracle\s+[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,4}\b")
_NUMERIC_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])\$?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z0-9])")
_IDENTIFIER_NUMBER_PATTERN = re.compile(r"\b[A-Z]{1,5}[.\-]?0*(\d{1,3})\b")
_NODE_HARDENED_RETRY_ATTEMPTS = 2
_HARDENED_RETRY_MARKER = "RETRY HARDENING ACTIVE"


def _load_implicit_actors_by_module() -> dict[str, list[str]]:
    """Load module actor hints from implicit process config for allowlist expansion."""
    if not _IMPLICIT_PROCESSES_CONFIG.exists():
        return {}

    try:
        with open(_IMPLICIT_PROCESSES_CONFIG, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load implicit actors from %s: %s", _IMPLICIT_PROCESSES_CONFIG, exc)
        return {}

    actor_map: dict[str, list[str]] = {}
    for proc in payload.get("implicit_processes", []) or []:
        module = str(proc.get("module", "")).strip().upper()
        if not module:
            continue
        module_actors = actor_map.setdefault(module, [])
        for actor in proc.get("business_actors", []) or []:
            normalized = normalize_business_actor(str(actor).strip())
            if normalized and normalized not in module_actors:
                module_actors.append(normalized)

    return actor_map


_IMPLICIT_ACTORS_BY_MODULE = _load_implicit_actors_by_module()


def _tokenize(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    }


def _strip_source_suffix(text: str) -> str:
    return re.sub(r"\s*\[source:.*?\]\s*$", "", text or "", flags=re.IGNORECASE).strip()


def _is_generation_noise_requirement(text: str) -> bool:
    value = (text or "").lower()
    return any(re.search(pattern, value) for pattern in _REQUIREMENT_NOISE_PATTERNS)


def _process_hint_tokens(process: ProcessEntry) -> set[str]:
    name_desc = f"{process.process_name} {process.process_description}".lower()
    tokens = set()
    for key, hints in _PROCESS_HINTS.items():
        if key in name_desc:
            tokens.update(hints)
    return tokens


def _select_relevant_requirements(process: ProcessEntry, requirements_list: list[str]) -> list[str]:
    """Select the most process-relevant module requirements to reduce cross-process leakage."""
    if not requirements_list:
        return []

    safe_requirements = [
        _strip_source_suffix(req)
        for req in requirements_list
        if req and not _is_generation_noise_requirement(req)
    ]
    if not safe_requirements:
        return []

    process_tokens = _tokenize(f"{process.process_name} {process.process_description}")
    process_tokens.update(_process_hint_tokens(process))
    scored: list[tuple[int, str]] = []

    for req in safe_requirements:
        req_tokens = _tokenize(req)
        overlap = len(process_tokens & req_tokens)
        phrase_match = process.process_name.lower() in req.lower()
        bonus = 3 if phrase_match else 0
        scored.append((overlap + bonus, req))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_non_zero = [req for score, req in scored if score > 0][:10]
    if top_non_zero:
        return top_non_zero

    # No relevant requirement match found: avoid leaking module-wide noise.
    return []


def _allowed_actors(section_plan: SectionPlan) -> list[str]:
    """Build allowed actors from org roles, section actors, and implicit config actors."""
    actors: list[str] = []
    section_id = str(getattr(section_plan, "section_id", "") or "").upper()
    org_roles = list(getattr(section_plan, "org_roles", []) or [])
    business_actors = list(getattr(section_plan, "business_actors", []) or [])
    implicit_actors = list(_IMPLICIT_ACTORS_BY_MODULE.get(section_id, []) or [])

    for actor in [*org_roles, *business_actors, *implicit_actors]:
        normalized = normalize_business_actor(actor)
        if normalized and normalized not in actors:
            actors.append(normalized)

    # System is always permitted for fully automated steps.
    if "System" not in actors:
        actors.append("System")
    return actors


def _allowed_oracle_product_phrases(section_plan: SectionPlan) -> set[str]:
    """Build allowed Oracle product/module phrases for this section."""
    section_id = str(getattr(section_plan, "section_id", "") or "").upper()
    section_module_name = _MODULE_CANONICAL_NAME.get(section_id, "")
    allowed = {
        "Oracle Cloud",
        "Oracle Finance Cloud",
    }
    if section_module_name:
        allowed.add(section_module_name)
    return allowed


def _find_disallowed_oracle_phrases(text: str, section_plan: SectionPlan) -> list[str]:
    """
    Detect Oracle product phrases that are outside this section's allowed module scope.

    Examples that should be flagged in AP section: "Oracle Receivables", "Oracle Inventory".
    """
    allowed = _allowed_oracle_product_phrases(section_plan)
    section_id = str(getattr(section_plan, "section_id", "") or "").upper()
    section_module_name = _MODULE_CANONICAL_NAME.get(section_id, "")

    found: set[str] = set()
    for match in _ORACLE_PRODUCT_PATTERN.findall(text or ""):
        phrase = " ".join(match.split())
        if phrase in allowed:
            continue
        # Allow module feature phrasing that starts with the current module, e.g. "Oracle Payables AutoMatch"
        if section_module_name and phrase.startswith(section_module_name + " "):
            continue
        found.add(phrase)

    # Explicitly block other canonical module names in this section.
    if section_module_name:
        for module_name in _MODULE_CANONICAL_NAME.values():
            if module_name == section_module_name:
                continue
            if module_name in (text or ""):
                found.add(module_name)

    return sorted(found)


def _find_disallowed_role_mentions(text: str, allowed_actor_set: set[str]) -> list[str]:
    """Find role-like mentions in free text that are not in the allowed actor set."""
    if not text:
        return []

    hits: set[str] = set()

    # 1) Canonical role names used in narrative but not allowed for this section.
    lower_text = text.lower()
    for canon in CANONICAL_BUSINESS_ACTORS:
        if canon.lower() in lower_text and canon not in allowed_actor_set:
            hits.add(canon)

    # 2) Role-like title phrases that normalize outside the allowlist.
    for role_like in _ROLE_LIKE_PATTERN.findall(text):
        cleaned_role = _ROLE_STRIP_PREFIX_PATTERN.sub("", role_like).strip()
        normalized = normalize_business_actor(cleaned_role)
        if normalized not in allowed_actor_set:
            hits.add(normalized or cleaned_role)

    return sorted(hits)


def _extract_numeric_tokens(text: str) -> set[str]:
    return {match.group(0) for match in _NUMERIC_TOKEN_PATTERN.finditer(text or "")}


def _extract_identifier_number_tokens(*values: str) -> set[str]:
    """Collect numeric parts from process-like identifiers (for example AP.04, AP04, CM.03)."""
    numbers: set[str] = set()
    for value in values:
        if not value:
            continue
        for match in _IDENTIFIER_NUMBER_PATTERN.finditer(str(value).upper()):
            raw = (match.group(1) or "").strip()
            if not raw:
                continue
            canonical = raw.lstrip("0") or "0"
            numbers.add(raw)
            numbers.add(canonical)
            if len(raw) < 2:
                numbers.add(canonical.zfill(2))
    return numbers


def _strip_disallowed_numeric_tokens(text: str, disallowed_numbers: set[str]) -> str:
    """Remove exact numeric tokens from text while keeping punctuation reasonably clean."""
    if not text or not disallowed_numbers:
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in disallowed_numbers:
            return ""
        return token

    cleaned = _NUMERIC_TOKEN_PATTERN.sub(_replace, text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    return cleaned.strip()


def _rewrite_ungrounded_threshold_terms(text: str) -> str:
    """Replace threshold wording with neutral criteria wording when MoM has no thresholds."""
    if not text:
        return text

    rewritten = re.sub(r"\bthresholds\b", "approval criteria", text, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bthreshold\b", "approval criteria", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\s+", " ", rewritten)
    return rewritten.strip()


def _normalized_sentence(sentence: str) -> str:
    cleaned = re.sub(r"\s+", " ", (sentence or "").strip().lower())
    cleaned = re.sub(r"[^a-z0-9 ]+", "", cleaned)
    return cleaned


def _collect_quality_violations(
    content: SectionContent,
    section_plan: SectionPlan,
    process: ProcessEntry,
    requirements_text: str,
) -> list[str]:
    """Deterministic checks that catch common quality regressions beyond schema validation."""
    violations: list[str] = []
    allowed_actor_set = set(_allowed_actors(section_plan))
    bad_actors = sorted(
        {
            step.business_actor
            for step in content.process_steps
            if normalize_business_actor(step.business_actor) not in allowed_actor_set
        }
    )
    if bad_actors:
        violations.append(
            "Disallowed actors used: " + ", ".join(bad_actors)
        )

    text_blob = "\n".join(
        [content.narrative, *content.key_requirements]
    )

    disallowed_oracle_phrases = _find_disallowed_oracle_phrases(text_blob, section_plan)
    if disallowed_oracle_phrases:
        violations.append("Out-of-scope Oracle product names: " + ", ".join(disallowed_oracle_phrases))

    disallowed_roles_in_narrative = _find_disallowed_role_mentions(text_blob, allowed_actor_set)
    if disallowed_roles_in_narrative:
        violations.append("Out-of-scope role mentions in narrative/requirements: " + ", ".join(disallowed_roles_in_narrative))

    module_name = _MODULE_CANONICAL_NAME.get(section_plan.section_id, "")
    if module_name and module_name not in text_blob and "Oracle" in text_blob:
        violations.append(
            f"Expected canonical module name '{module_name}' not found in Oracle references"
        )

    requirements_norm = (requirements_text or "").strip().lower()
    no_grounding = requirements_norm in {
        "",
        "no specific requirements captured.",
        "no specific requirements captured",
    }
    if not no_grounding:
        content_numbers = _extract_numeric_tokens(text_blob)
        source_numbers = _extract_numeric_tokens(requirements_text)
        identifier_number_tokens = _extract_identifier_number_tokens(
            process.process_id,
            process.process_name,
            content.process_id,
            content.process_name,
        )
        invented_numbers = sorted(
            number
            for number in (content_numbers - source_numbers)
            if number not in identifier_number_tokens
        )
        if invented_numbers:
            violations.append("Potential invented numeric values: " + ", ".join(invented_numbers))

        if re.search(r"\bthreshold\b", text_blob, flags=re.IGNORECASE) and not re.search(
            r"\bthreshold\b", requirements_text, flags=re.IGNORECASE
        ):
            violations.append("Threshold mentioned but not present in MoM requirements")

    # Generic anti-boilerplate check: require process-specific anchor overlap.
    process_tokens = _tokenize(f"{process.process_name} {process.process_description}")
    narrative_tokens = _tokenize(content.narrative)
    token_overlap = len(process_tokens & narrative_tokens)
    if process_tokens and token_overlap == 0:
        violations.append("Narrative appears generic and lacks process-specific anchors")

    sentences = [
        s for s in re.split(r"(?<=[.!?])\s+", content.narrative or "")
        if len(s.strip()) >= 25
    ]
    norm_seen: set[str] = set()
    for sentence in sentences:
        norm = _normalized_sentence(sentence)
        if not norm:
            continue
        if norm in norm_seen:
            violations.append("Narrative repeats the same sentence pattern")
            break
        norm_seen.add(norm)

    pname = (process.process_name or "").lower()
    if "direct invoice" in pname:
        if re.search(r"po\s*matching|match\s+po|procurement", text_blob, flags=re.IGNORECASE):
            violations.append("Direct Invoice content references PO/procurement logic")

    if "debit memo" in pname or "credit memo" in pname:
        if re.search(r"invoice approval workflow", text_blob, flags=re.IGNORECASE):
            violations.append("Memo process references invoice approval workflow")

    if "payment cancellation" in pname or "receipt reversal" in pname:
        if re.search(r"invoice approval workflow", text_blob, flags=re.IGNORECASE):
            violations.append("Reversal process references invoice approval workflow")

    return violations


def _is_hard_violation(msg: str) -> bool:
    """Classify which quality-gate violations must block section output."""
    hard_prefixes = (
        "Disallowed actors used:",
        "Out-of-scope Oracle product names:",
        "Potential invented numeric values:",
        "Threshold mentioned but not present in MoM requirements",
        "Direct Invoice content references PO/procurement logic",
        "Memo process references invoice approval workflow",
        "Reversal process references invoice approval workflow",
    )
    return msg.startswith(hard_prefixes)


def _partition_violations(violations: list[str]) -> tuple[list[str], list[str]]:
    """Split violations into hard blockers vs soft quality warnings."""
    hard = [v for v in violations if _is_hard_violation(v)]
    soft = [v for v in violations if not _is_hard_violation(v)]
    return hard, soft


def _pick_closest_actor(actor: str, allowed: list[str]) -> str:
    if not allowed:
        return actor
    if actor in allowed:
        return actor

    non_system = [a for a in allowed if a != "System"]
    candidates = non_system or allowed

    best = candidates[0]
    best_score = -1.0
    for cand in candidates:
        score = SequenceMatcher(a=actor.lower(), b=cand.lower()).ratio()
        if score > best_score:
            best = cand
            best_score = score
    # If the similarity is weak, choose the first approved role for consistency.
    if best_score < 0.45:
        return candidates[0]
    return best


def _sanitize_content(
    content: SectionContent,
    section_plan: SectionPlan,
    requirements_text: str,
) -> SectionContent:
    """Last-resort deterministic sanitation when repair still leaves violations."""
    data = content.model_dump()
    allowed = _allowed_actors(section_plan)

    for step in data.get("process_steps", []):
        actor = step.get("business_actor", "")
        normalized_actor = normalize_business_actor(actor)
        if normalized_actor in allowed:
            step["business_actor"] = normalized_actor
            continue
        step["business_actor"] = _pick_closest_actor(normalized_actor or actor, allowed)

    section_module_name = _MODULE_CANONICAL_NAME.get(section_plan.section_id, "")

    def _rewrite_oracle_phrases(value: str) -> str:
        if not section_module_name or not value:
            return value
        out = value
        for phrase in _find_disallowed_oracle_phrases(value, section_plan):
            out = re.sub(re.escape(phrase), section_module_name, out, flags=re.IGNORECASE)
        return out

    data["narrative"] = _rewrite_oracle_phrases(data.get("narrative", ""))

    requirements_norm = (requirements_text or "").strip().lower()
    no_grounding = requirements_norm in {
        "",
        "no specific requirements captured.",
        "no specific requirements captured",
    }

    if not no_grounding:
        text_blob = "\n".join([data.get("narrative", ""), *data.get("key_requirements", [])])
        content_numbers = _extract_numeric_tokens(text_blob)
        source_numbers = _extract_numeric_tokens(requirements_text)
        identifier_numbers = _extract_identifier_number_tokens(
            str(data.get("process_id", "")),
            str(data.get("process_name", "")),
        )
        disallowed_numbers = {
            number
            for number in (content_numbers - source_numbers)
            if number not in identifier_numbers
        }

        if disallowed_numbers:
            data["narrative"] = _strip_disallowed_numeric_tokens(
                data.get("narrative", ""),
                disallowed_numbers,
            )
            data["key_requirements"] = [
                _strip_disallowed_numeric_tokens(req, disallowed_numbers)
                for req in data.get("key_requirements", [])
                if _strip_disallowed_numeric_tokens(req, disallowed_numbers)
            ]

            missing_info = list(data.get("missing_info", []))
            msg = (
                "Removed ungrounded numeric values: "
                + ", ".join(sorted(disallowed_numbers))
            )
            if msg not in missing_info:
                missing_info.append(msg)
            data["missing_info"] = missing_info

    requirements_has_threshold = bool(
        re.search(r"\bthreshold\b", requirements_text, flags=re.IGNORECASE)
    )
    if not requirements_has_threshold:
        data["narrative"] = _rewrite_ungrounded_threshold_terms(data.get("narrative", ""))

    fixed_reqs = []
    for req in data.get("key_requirements", []):
        req_fixed = _rewrite_oracle_phrases(req)
        if not requirements_has_threshold:
            req_fixed = _rewrite_ungrounded_threshold_terms(req_fixed)
        req_fixed = req_fixed.strip()
        if req_fixed:
            fixed_reqs.append(req_fixed)
    data["key_requirements"] = fixed_reqs

    if not requirements_has_threshold:
        missing_info = list(data.get("missing_info", []))
        msg = "Approval threshold amount not confirmed in MoM"
        if msg not in missing_info:
            missing_info.append(msg)
        data["missing_info"] = missing_info

    return SectionContent.model_validate(data)


def _repair_content_with_violations(
    content: SectionContent,
    section_plan: SectionPlan,
    process: ProcessEntry,
    requirements_text: str,
    violations: list[str],
) -> SectionContent:
    """One-shot LLM repair pass focused on deterministic quality violations."""
    allowed_actors = ", ".join(_allowed_actors(section_plan))
    module_name = _MODULE_CANONICAL_NAME.get(section_plan.section_id, section_plan.module_name)
    violations_block = "\n".join(f"- {v}" for v in violations)
    content_json = json.dumps(content.model_dump(), ensure_ascii=False, indent=2)

    repair_prompt = f"""You must repair a generated SectionContent JSON for Oracle RD.011.

Fix ALL violations below while preserving process meaning and schema.
Violations:
{violations_block}

Hard constraints:
- process_id must stay exactly: {content.process_id}
- process_name must stay exactly: {content.process_name}
- Use only this canonical module name when referenced: {module_name}
- Allowed business actors: {allowed_actors}
- 'System' is allowed only for fully automated steps.
- Any number/threshold must exist in MoM requirements below; otherwise remove it and add a missing_info item.

Requirements from MoM:
{requirements_text}

Current JSON to repair:
{content_json}

Return ONLY valid JSON matching the SectionContent schema."""

    return call_with_retry(
        TaskType.GENERATION,
        repair_prompt,
        SectionContent,
        max_retries=2,
    )


def _build_hardened_retry_prompt(
    base_prompt: str,
    section_key: str,
    process: ProcessEntry,
    error: str,
) -> str:
    """Append strict corrective instructions for one same-node regeneration retry."""
    error_text = (error or "").strip()
    if len(error_text) > 500:
        error_text = error_text[:500] + "..."

    hardening_block = f"""

## {_HARDENED_RETRY_MARKER}
Previous generation attempt failed for section {section_key} ({process.process_id} - {process.process_name}).
Last failure summary:
{error_text or 'Unknown generation failure.'}

For this retry, apply these strict output constraints:
- Return ONLY one valid JSON object matching SectionContent.
- No markdown fences, no prose, no comments, no trailing commas.
- Use double quotes for all JSON keys and string values.
- Keep process_id and process_name exactly as provided in context.
- If uncertain about a detail, keep the statement conservative and record it in missing_info.
- Do not emit partial JSON. Ensure the final byte is the closing brace of the JSON object.
"""
    return f"{base_prompt}{hardening_block}"


def _generate_and_validate_section_content(
    *,
    prompt: str,
    section_key: str,
    section_plan: SectionPlan,
    process: ProcessEntry,
    requirements_text: str,
) -> SectionContent:
    """Generate one section and apply deterministic quality gate checks."""
    content = call_with_retry(TaskType.GENERATION, prompt, SectionContent)

    violations = _collect_quality_violations(
        content=content,
        section_plan=section_plan,
        process=process,
        requirements_text=requirements_text,
    )
    if violations:
        hard_violations, soft_violations = _partition_violations(violations)
        if soft_violations:
            logger.warning(
                "Soft quality warnings for %s: %s",
                section_key,
                "; ".join(soft_violations),
            )

        if hard_violations:
            logger.warning("Hard quality violations for %s: %s", section_key, "; ".join(hard_violations))
            if ENABLE_REPAIR_PASS:
                try:
                    repaired = _repair_content_with_violations(
                        content=content,
                        section_plan=section_plan,
                        process=process,
                        requirements_text=requirements_text,
                        violations=hard_violations,
                    )
                    content = repaired
                except Exception as repair_exc:
                    logger.warning(
                        "Repair pass failed for %s (%s). Applying deterministic sanitation.",
                        section_key,
                        repair_exc,
                    )

            post_repair_violations = _collect_quality_violations(
                content=content,
                section_plan=section_plan,
                process=process,
                requirements_text=requirements_text,
            )
            post_repair_hard, _ = _partition_violations(post_repair_violations)
            if post_repair_hard:
                logger.warning(
                    "Hard violations remain for %s after repair: %s. Applying deterministic sanitation.",
                    section_key,
                    "; ".join(post_repair_hard),
                )
                content = _sanitize_content(
                    content=content,
                    section_plan=section_plan,
                    requirements_text=requirements_text,
                )

    final_violations = _collect_quality_violations(
        content=content,
        section_plan=section_plan,
        process=process,
        requirements_text=requirements_text,
    )
    final_hard_violations, final_soft_violations = _partition_violations(final_violations)
    if final_hard_violations:
        raise RuntimeError(
            "Hard quality gate failed after repair/sanitization: " + "; ".join(final_hard_violations)
        )
    if final_soft_violations:
        logger.warning(
            "Section %s generated with soft quality warnings: %s",
            section_key,
            "; ".join(final_soft_violations),
        )

    return content


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
    section_failed = False

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
    requirements_list = _select_relevant_requirements(process, requirements_list)
    requirements_text = "\n".join(f"- {r}" for r in requirements_list) or "No specific requirements captured."

    # RAG context (optional): exemplars for style + step granularity.
    rag_style_context = ""
    rag_step_context = ""
    if RAG_ENABLED:
        try:
            from rag.retriever import build_exemplar_blocks

            module_code = str(section_plan.section_id or "").strip().upper()
            # Use semantic process identity (name/description) instead of ordinal process_id,
            # since process numbering varies across projects and example sets.
            query = f"{module_code} {process.process_name} {process.process_description} process narrative steps"
            rag_style_context, rag_step_context = build_exemplar_blocks(
                query=query,
                needed_types=["process_narrative", "process_steps"],
                module=module_code,
            )
            if rag_style_context or rag_step_context:
                logger.info(
                    "RAG exemplars selected for %s: style_chars=%d step_chars=%d",
                    section_key,
                    len(rag_style_context or ""),
                    len(rag_step_context or ""),
                )
        except Exception as exc:
            logger.warning("RAG query failed: %s", exc)
    else:
        logger.info("RAG disabled - generating without reference examples")

    # Build prompt and call LLM
    prompt = build_generation_prompt(
        section_plan=section_plan,
        process=process,
        requirements_text=requirements_text,
        rag_style_context=rag_style_context,
        rag_step_context=rag_step_context,
    )

    content: SectionContent | None = None
    last_exc: Exception | None = None
    attempt_prompt = prompt

    for attempt_idx in range(1, _NODE_HARDENED_RETRY_ATTEMPTS + 1):
        try:
            content = _generate_and_validate_section_content(
                prompt=attempt_prompt,
                section_key=section_key,
                section_plan=section_plan,
                process=process,
                requirements_text=requirements_text,
            )
            if attempt_idx > 1:
                logger.info("Recovered %s on hardened same-node retry", section_key)
            break
        except Exception as exc:
            last_exc = exc
            if attempt_idx >= _NODE_HARDENED_RETRY_ATTEMPTS:
                break
            logger.warning(
                "Generation attempt %d failed for %s: %s. Retrying once with hardened prompt.",
                attempt_idx,
                section_key,
                exc,
            )
            attempt_prompt = _build_hardened_retry_prompt(
                base_prompt=prompt,
                section_key=section_key,
                process=process,
                error=str(exc),
            )

    if content is not None:
        generated_sections[section_key] = content.model_dump()
        logger.info(
            "Generated %s: %d steps, %d journal entries",
            section_key,
            len(content.process_steps),
            len(content.journal_entries),
        )
    else:
        section_failed = True
        error_msg = f"Generation failed for {section_key}: {last_exc}"
        logger.error(error_msg)
        errors.append(error_msg)
        failed_sections.append(section_key)

    # Optional throttling to control burstiness on free tiers.
    if GENERATION_THROTTLE_SECONDS > 0:
        if not GENERATION_THROTTLE_ON_FAILURE_ONLY or section_failed:
            time.sleep(GENERATION_THROTTLE_SECONDS)

    return {
        "current_section_index": idx + 1,
        "generated_sections": generated_sections,
        "failed_sections": failed_sections,
        "errors": errors,
        "last_completed_node": "generate_section",
    }
