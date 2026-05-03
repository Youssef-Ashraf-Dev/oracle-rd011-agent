"""
RD.011 Agent — Structured-output retry loop.

Calls an LLM, parses the response as JSON into a Pydantic model, and
retries with injected error context on validation failure.  All validation
is deterministic Python — no second LLM pass for validation.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import (
    MAX_RETRIES,
    CAPABILITY_MAP,
    LLM_TELEMETRY_ENABLED,
    LLM_TELEMETRY_PATH,
    FAIL_FAST_JSONDECODE_GENERATION,
    TaskType,
)
from llm.router import get_client

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
_LOG_PREVIEW_CHARS = 280
_TELEMETRY_LOCK = threading.Lock()


def _emit_telemetry(event: str, **fields) -> None:
    """Append one structured retry/routing event to JSONL telemetry."""
    if not LLM_TELEMETRY_ENABLED:
        return
    try:
        path = Path(LLM_TELEMETRY_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        # Multiple concurrent generation workers may emit telemetry; serialize writes
        # so JSONL lines are not interleaved.
        with _TELEMETRY_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("Telemetry write failed: %s", exc)


def _safe_preview(text: str, max_chars: int = _LOG_PREVIEW_CHARS) -> str:
    """
    Build a compact, redacted single-line preview for logs.

    Redacts common credential patterns and escapes newlines so log entries stay
    readable. Output is truncated to ``max_chars``.
    """
    if not text:
        return "<empty>"

    preview = str(text)

    # Common secret patterns that can appear in model echoes.
    redactions = [
        (r"(?i)(api[_-]?key\s*[:=]\s*)(['\"]?)[^\s,;\"']+", r"\1\2[REDACTED]"),
        (r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;\"']+", r"\1[REDACTED]"),
        (r"\bsk-[A-Za-z0-9_-]{16,}\b", "[REDACTED]"),
        (r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]"),
    ]
    for pattern, replacement in redactions:
        preview = re.sub(pattern, replacement, preview)

    preview = preview.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    if len(preview) > max_chars:
        return f"{preview[:max_chars]}..."
    return preview


def _strip_leading_code_fence(text: str) -> str:
    """Remove a leading markdown code fence line if present (even if unclosed)."""
    if not text:
        return text
    stripped = text.lstrip()
    if not stripped.startswith("```"):
        return text

    # Remove the first fence line (``` or ```json). Keep the rest for JSON scanning.
    nl = stripped.find("\n")
    if nl == -1:
        return ""
    return stripped[nl + 1 :]


def _extract_balanced_json(text: str) -> str:
    """
    Extract the first balanced JSON object/array from a string.

    This handles common LLM failure modes:
    - Prose before/after the JSON
    - Unclosed markdown fences
    - Multiple JSON snippets (takes the first complete one)

    Returns empty string if nothing JSON-like is found.
    """
    if not text:
        return ""

    s = text
    # Prefer scanning content after a leading code fence when present.
    s = _strip_leading_code_fence(s)

    # Find earliest plausible JSON start.
    starts = [idx for idx in (s.find("{"), s.find("[")) if idx != -1]
    if not starts:
        return ""
    start = min(starts)

    stack: list[str] = []
    in_string = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        # Not in string
        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            stack.append("}")
            continue
        if ch == "[":
            stack.append("]")
            continue

        if stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return s[start : i + 1].strip()

    # Unbalanced; fall back to best-effort slice first->last close.
    last_obj = s.rfind("}")
    last_arr = s.rfind("]")
    last = max(last_obj, last_arr)
    if last != -1 and last > start:
        return s[start : last + 1].strip()
    return ""


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences and messy wrappers."""
    text = (text or "").strip()
    if not text:
        return text

    # Prefer a properly closed fenced block if present.
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        balanced = _extract_balanced_json(candidate)
        return balanced or candidate

    # Otherwise, extract the first balanced JSON object/array from the whole response.
    balanced = _extract_balanced_json(text)
    if balanced:
        return balanced

    # Last resort: keep old heuristic (first { to last }).
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()
    return text


def _repair_json_candidate(text: str) -> str:
    """Apply lightweight repairs for common LLM JSON formatting mistakes."""
    repaired = (text or "").strip()
    if not repaired:
        return repaired

    # Normalize smart quotes and strip control characters.
    repaired = (
        repaired
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    repaired = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", repaired)

    # Remove trailing commas before JSON object/array closure.
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _build_retry_prompt(
    original_prompt: str,
    error_message: str,
    previous_response: str,
) -> str:
    """Build the retry prompt with error injection."""
    return (
        f"Previous attempt failed with this error:\n"
        f"{error_message}\n\n"
        f"Please fix the issue and return valid JSON matching the schema.\n"
        f"Your previous (invalid) response was:\n"
        f"{previous_response}\n\n"
        f"Original instructions:\n"
        f"{original_prompt}"
    )


def _sanitize_step_ids(data: dict) -> dict:
    """
    Silently correct common small-model step_id formatting errors
    before Pydantic validation. Does not count as a retry attempt.

    Known patterns fixed:
      AP.04-01-01  ->  AP-04-01   (dot separator + extra segment)
      AP-04-01-01  ->  AP-04-01   (extra trailing segment)
      ap-04-01     ->  AP-04-01   (lowercase prefix)
      AP-4-1       ->  AP-04-01   (missing zero padding)
      AP.04.01     ->  AP-04-01   (dots instead of dashes)
      AP09-01      ->  AP-09-01   (module + proc number merged)
    """
    if "process_steps" not in data:
        return data
    for step in data.get("process_steps", []):
        sid = step.get("step_id", "")
        if not sid:
            continue
        sid = sid.replace(".", "-").replace("_", "-")
        parts = sid.split("-")
        if parts:
            parts[0] = parts[0].upper()
            # Split merged module+proc like "AP09" -> "AP", "09"
            merged = re.match(r'^([A-Z]{2,3})(\d+)$', parts[0])
            if merged:
                parts = [merged.group(1), merged.group(2)] + parts[1:]
        padded = [parts[0]] if parts else []
        for p in parts[1:]:
            padded.append(p.zfill(2) if p.isdigit() else p)
        if len(padded) > 3:
            padded = padded[:3]
        step["step_id"] = "-".join(padded)
    return data


def _sanitize_process_id(pid: str) -> str:
    """
    Extract and clean process_id from malformed values.

    Handles cases like:
            "Client Name.AP.01"    ->  "AP.01"
            "Client.Region.AP.01"  ->  "AP.01"
      "AP-05-01"             ->  "AP.05"  (step_id format confused as process_id)
      "AP.01"                ->  "AP.01"  (already correct)

    Extracts the last part matching MODULE.NN pattern (e.g. AP.01).
    Returns original if no match found.
    """
    if not pid:
        return pid

    # Try to find the module code pattern (2-3 uppercase letters + dot + 2 digits)
    match = re.search(r'([A-Z]{2,3}\.\d{2})(?:\s|$|\.)', pid)
    if match:
        return match.group(1)

    # Fallback: dash-separated step_id format used as process_id (e.g. AP-05-01 → AP.05)
    match = re.search(r'^([A-Z]{2,3})-(\d{2})(?:-\d+)?$', pid.upper().strip())
    if match:
        return f"{match.group(1)}.{match.group(2)}"

    return pid


def _model_id(provider: str, model_name: str) -> str:
    """Return normalized provider/model identifier used across logs and policy maps."""
    return f"{provider}/{model_name}"





def call_with_retry(
    task_type: TaskType,
    prompt: str,
    schema: Type[T],
    max_retries: int = MAX_RETRIES,
) -> T:
    """
    Call an LLM via a cascade/waterfall of models, parsing JSON response into *schema*.

    Cascade order: primary model first, then fallback_chain entries.
    Models in ``MODEL_POLICY_BLOCKLIST`` are skipped before first attempt.
    Skip to next model immediately on 429 (rate limit) or 400 (decommissioned).
    Retry the same model on ValidationError with a 3-second cooldown.

    Parameters
    ----------
    task_type
        TaskType.REASONING, GENERATION, or LARGE_CONTEXT — determines the model cascade
    prompt
        The full prompt string to send
    schema
        The Pydantic v2 ``BaseModel`` subclass to validate response against
    max_retries
        Maximum validation retries per model (default from config)

    Returns
    -------
    An instance of *schema* with validated data.

    Raises
    ------
    RuntimeError
        If all models in the fallback chain are exhausted.
    """
    if task_type not in CAPABILITY_MAP:
        raise ValueError(f"Unknown TaskType: {task_type}")

    cfg = CAPABILITY_MAP[task_type]

    # Build cascade: primary model first, then fallback_chain entries
    if "provider" in cfg and "model" in cfg:
        primary = {"provider": cfg["provider"], "model": cfg["model"]}
        cascade = [primary] + cfg.get("fallback_chain", [])
    else:
        # Bare fallback_chain with no primary (legacy support)
        cascade = cfg["fallback_chain"]

    schema_name = schema.__name__


    tried = []

    for cascade_idx, model_cfg in enumerate(cascade, start=1):
        provider = model_cfg["provider"]
        model_name = model_cfg["model"]
        tried.append(_model_id(provider, model_name))

        logger.info(
            "Cascade attempt %d/%d for %s: trying %s/%s",
            cascade_idx, len(cascade), task_type.value, provider, model_name,
        )
        _emit_telemetry(
            "cascade_attempt",
            task_type=task_type.value,
            schema=schema_name,
            cascade_index=cascade_idx,
            cascade_total=len(cascade),
            provider=provider,
            model=model_name,
        )

        try:
            client = get_client(provider, model_name, task_type)
            result = _call_with_retry_single(
                client,
                prompt,
                schema,
                max_retries,
                task_type=task_type.value,
                provider=provider,
                model_name=model_name,
            )
            logger.info("Successfully generated %s via %s", schema_name, _model_id(provider, model_name))
            _emit_telemetry(
                "model_success",
                task_type=task_type.value,
                schema=schema_name,
                provider=provider,
                model=model_name,
                cascade_index=cascade_idx,
            )
            return result

        except Exception as exc:
            exc_str = str(exc).lower()

            is_429 = (
                "429" in exc_str
                or "too many requests" in exc_str
                or "rate_limit" in exc_str
                or "resource_exhausted" in exc_str
            )
            if is_429:
                logger.warning(
                    "Rate limit (429) for %s/%s on %s",
                    provider,
                    model_name,
                    schema_name,
                )
                _emit_telemetry(
                    "rate_limited",
                    task_type=task_type.value,
                    schema=schema_name,
                    provider=provider,
                    model=model_name,
                    cascade_index=cascade_idx,
                    error_type=type(exc).__name__,
                    error_text=str(exc)[:220],
                )

            # Rate-limit, request-size, or model-availability errors — skip immediately.
            is_skip = (
                "429" in exc_str
                or "400" in exc_str
                or "413" in exc_str
                or "503" in exc_str
                or "rate_limit" in exc_str
                or "too many requests" in exc_str
                or "payload too large" in exc_str
                or "request too large" in exc_str
                or "context length" in exc_str
                or "maximum context" in exc_str
                or "resource_exhausted" in exc_str
                or "decommissioned" in exc_str
                or "model_not_found" in exc_str
                or "unavailable" in exc_str
                or "overloaded" in exc_str
            )

            if is_skip and cascade_idx < len(cascade):
                next_m = cascade[cascade_idx]
                logger.warning(
                    "Model %s/%s skipped (%s). Falling back to %s/%s.",
                    provider, model_name,
                    "rate/payload/model error",
                    next_m["provider"], next_m["model"],
                )
                _emit_telemetry(
                    "model_skipped",
                    task_type=task_type.value,
                    schema=schema_name,
                    provider=provider,
                    model=model_name,
                    reason="rate/payload/model error",
                    error_type=type(exc).__name__,
                    error_text=str(exc)[:220],
                )
                continue

            # Last model or non-skip error
            if cascade_idx < len(cascade):
                logger.warning(
                    "Model %s/%s failed (%s: %s) — trying next fallback.",
                    provider, model_name, type(exc).__name__, str(exc)[:120],
                )
                _emit_telemetry(
                    "model_failed",
                    task_type=task_type.value,
                    schema=schema_name,
                    provider=provider,
                    model=model_name,
                    error_type=type(exc).__name__,
                    error_text=str(exc)[:220],
                )
                continue

    # All models exhausted
    raise RuntimeError(
        f"All models in cascade for {task_type.value} exhausted. "
        f"Models tried: {', '.join(tried)}."
    )


def _call_with_retry_single(
    client,
    prompt: str,
    schema: Type[T],
    max_retries: int,
    *,
    task_type: str,
    provider: str,
    model_name: str,
) -> T:
    """
    Inner retry loop for a single client (no fallback).

    Retries with error injection on JSON/schema validation failures.
    Adds a 3-second cooldown between validation retries to avoid TPM spikes.
    Transport/provider errors are raised immediately so the cascade can skip models.
    """
    current_prompt = prompt
    last_raw_response = ""

    for attempt in range(1, max_retries + 1):
        logger.info("LLM call attempt %d/%d for %s", attempt, max_retries, schema.__name__)

        try:
            started = time.perf_counter()
            response = client.invoke(current_prompt)
            latency_ms = int((time.perf_counter() - started) * 1000)
        except Exception:
            # Transport/provider errors must not be retried as validation failures.
            raise

        try:
            # LangChain chat models return AIMessage; extract content.
            # Newer Gemini models return content as a list of parts
            # e.g. [{"type": "text", "text": "..."}] — flatten to a string.
            raw_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            if isinstance(raw_text, list):
                raw_text = "".join(
                    part["text"] if isinstance(part, dict) and "text" in part
                    else str(part)
                    for part in raw_text
                )
            last_raw_response = raw_text

            json_str = _extract_json(raw_text)
            try:
                raw_dict = json.loads(json_str)
            except json.JSONDecodeError:
                repaired_json = _repair_json_candidate(json_str)
                if repaired_json == json_str:
                    raise
                raw_dict = json.loads(repaired_json)
            raw_dict = _sanitize_step_ids(raw_dict)
            if "process_id" in raw_dict:
                raw_dict["process_id"] = _sanitize_process_id(raw_dict["process_id"])
            result = schema.model_validate(raw_dict)
            logger.info("Validation passed on attempt %d", attempt)
            _emit_telemetry(
                "attempt_success",
                task_type=task_type,
                schema=schema.__name__,
                provider=provider,
                model=model_name,
                attempt=attempt,
                latency_ms=latency_ms,
                prompt_chars=len(current_prompt),
                response_chars=len(raw_text),
            )
            return result

        except json.JSONDecodeError as exc:
            error_message = str(exc)
            raw_preview = _safe_preview(last_raw_response)
            json_candidate = _extract_json(last_raw_response) if last_raw_response else ""
            json_preview = _safe_preview(json_candidate)

            logger.warning(
                "Attempt %d failed (%s): %s",
                attempt,
                type(exc).__name__,
                error_message[:200],
            )
            _emit_telemetry(
                "attempt_validation_failed",
                task_type=task_type,
                schema=schema.__name__,
                provider=provider,
                model=model_name,
                attempt=attempt,
                error_type=type(exc).__name__,
                error_text=error_message[:220],
                prompt_chars=len(current_prompt),
                response_chars=len(last_raw_response),
            )
            logger.warning(
                "Invalid JSON output preview for %s (attempt %d): "
                "raw_len=%d, json_candidate_len=%d, raw_preview=%s, json_candidate_preview=%s",
                schema.__name__,
                attempt,
                len(last_raw_response),
                len(json_candidate),
                raw_preview,
                json_preview,
            )

            if (
                FAIL_FAST_JSONDECODE_GENERATION
                and task_type == TaskType.GENERATION.value
                and schema.__name__ == "SectionContent"
                and attempt < max_retries
            ):
                raise RuntimeError(
                    "Fail-fast JSONDecodeError for generation/SectionContent; "
                    f"skipping retries for {provider}/{model_name}"
                ) from exc

            if attempt < max_retries:
                # 3-second cooldown between validation retries to avoid TPM spike
                logger.debug("Validation retry cooldown: 3s to avoid TPM spike")
                time.sleep(3)
                current_prompt = _build_retry_prompt(
                    original_prompt=prompt,
                    error_message=error_message,
                    previous_response=last_raw_response,
                )
                _emit_telemetry(
                    "attempt_retry_prompt_built",
                    task_type=task_type,
                    schema=schema.__name__,
                    provider=provider,
                    model=model_name,
                    attempt=attempt,
                )
            else:
                raise RuntimeError(
                    f"All {max_retries} retries exhausted for {schema.__name__}. "
                    f"Last error: {error_message}"
                ) from exc

        except (ValidationError, ValueError) as exc:
            error_message = str(exc)
            logger.warning(
                "Attempt %d failed (%s): %s",
                attempt, type(exc).__name__, error_message[:200],
            )
            _emit_telemetry(
                "attempt_validation_failed",
                task_type=task_type,
                schema=schema.__name__,
                provider=provider,
                model=model_name,
                attempt=attempt,
                error_type=type(exc).__name__,
                error_text=error_message[:220],
                prompt_chars=len(current_prompt),
                response_chars=len(last_raw_response),
            )

            if attempt < max_retries:
                # 3-second cooldown between validation retries to avoid TPM spike
                logger.debug("Validation retry cooldown: 3s to avoid TPM spike")
                time.sleep(3)
                current_prompt = _build_retry_prompt(
                    original_prompt=prompt,
                    error_message=error_message,
                    previous_response=last_raw_response,
                )
                _emit_telemetry(
                    "attempt_retry_prompt_built",
                    task_type=task_type,
                    schema=schema.__name__,
                    provider=provider,
                    model=model_name,
                    attempt=attempt,
                )
            else:
                raise RuntimeError(
                    f"All {max_retries} retries exhausted for {schema.__name__}. "
                    f"Last error: {error_message}"
                ) from exc

    # Should not reach here, but satisfy type checker
    raise RuntimeError(f"Retry loop exited unexpectedly for {schema.__name__}")
