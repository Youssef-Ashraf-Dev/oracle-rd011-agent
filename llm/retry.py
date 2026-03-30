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
import time
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import MAX_RETRIES, CAPABILITY_MAP, TaskType
from llm.router import get_client

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    # Strip ```json ... ``` wrappers
    if text.startswith("```"):
        # Find end of first line (```json or ```)
        first_newline = text.index("\n")
        # Find closing ```
        last_fence = text.rfind("```")
        if last_fence > first_newline:
            text = text[first_newline + 1 : last_fence].strip()
    return text


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
      "AL Gosaibi Co.AP.01"  ->  "AP.01"
      "AL.GO.AP.01"          ->  "AP.01"
      "AL.GOSAIBI.CO.AP.01"  ->  "AP.01"
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


def _sanitize_step_types(data: dict) -> dict:
    """
    Normalize legacy step_type values to the new vocabulary.
    Does not count as a retry attempt.

    Mappings:
      "Manual" -> "Manual Step"
      "System" -> "System Assisted"
      "System Generated" -> "System Automated"
      Already-correct values are passed through unchanged.
    """
    if "process_steps" not in data:
        return data

    type_map = {
        "Manual": "Manual Step",
        "System": "System Assisted",
        "System Generated": "System Automated",
        "Manual Step": "Manual Step",  # Already correct
        "System Assisted": "System Assisted",  # Already correct
        "System Automated": "System Automated",  # Already correct
        "Decision": "Decision",  # No change
    }

    for step in data.get("process_steps", []):
        old_type = step.get("step_type", "")
        if old_type:
            step["step_type"] = type_map.get(old_type, old_type)

    return data


def call_with_retry(
    task_type: TaskType,
    prompt: str,
    schema: Type[T],
    max_retries: int = MAX_RETRIES,
) -> T:
    """
    Call an LLM via a cascade/waterfall of models, parsing JSON response into *schema*.

    Cascade order: primary model first, then fallback_chain entries.
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

    tried = []

    for cascade_idx, model_cfg in enumerate(cascade, start=1):
        provider = model_cfg["provider"]
        model_name = model_cfg["model"]
        tried.append(f"{provider}/{model_name}")

        logger.info(
            "Cascade attempt %d/%d for %s: trying %s/%s",
            cascade_idx, len(cascade), task_type.value, provider, model_name,
        )

        try:
            client = get_client(provider, model_name, task_type)
            result = _call_with_retry_single(client, prompt, schema, max_retries)
            logger.info(
                "Successfully generated %s via %s/%s",
                schema.__name__, provider, model_name,
            )
            return result

        except Exception as exc:
            exc_str = str(exc).lower()

            # 429 (rate limit / quota) or 400 (decommissioned model) — skip immediately
            is_skip = (
                "429" in exc_str
                or "400" in exc_str
                or "rate_limit" in exc_str
                or "too many requests" in exc_str
                or "resource_exhausted" in exc_str
                or "decommissioned" in exc_str
                or "model_not_found" in exc_str
            )

            if is_skip and cascade_idx < len(cascade):
                next_m = cascade[cascade_idx]
                logger.warning(
                    "Model %s/%s skipped (%s). Falling back to %s/%s.",
                    provider, model_name,
                    "rate limit" if "429" in exc_str else "bad request",
                    next_m["provider"], next_m["model"],
                )
                continue

            # Last model or non-skip error
            if cascade_idx < len(cascade):
                logger.warning(
                    "Model %s/%s failed (%s: %s) — trying next fallback.",
                    provider, model_name, type(exc).__name__, str(exc)[:120],
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
) -> T:
    """
    Inner retry loop for a single client (no fallback).

    Retries with error injection on ValidationError.
    Adds a 3-second cooldown between validation retries to avoid TPM spikes.
    429/400 errors are raised immediately so the cascade can skip to next model.
    """
    current_prompt = prompt
    last_raw_response = ""

    for attempt in range(1, max_retries + 1):
        logger.info("LLM call attempt %d/%d for %s", attempt, max_retries, schema.__name__)

        try:
            response = client.invoke(current_prompt)
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
            raw_dict = json.loads(json_str)
            raw_dict = _sanitize_step_ids(raw_dict)
            raw_dict = _sanitize_step_types(raw_dict)
            if "process_id" in raw_dict:
                raw_dict["process_id"] = _sanitize_process_id(raw_dict["process_id"])
            result = schema.model_validate(raw_dict)
            logger.info("Validation passed on attempt %d", attempt)
            return result

        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            error_message = str(exc)
            logger.warning(
                "Attempt %d failed (%s): %s",
                attempt, type(exc).__name__, error_message[:200],
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
            else:
                raise RuntimeError(
                    f"All {max_retries} retries exhausted for {schema.__name__}. "
                    f"Last error: {error_message}"
                ) from exc

    # Should not reach here, but satisfy type checker
    raise RuntimeError(f"Retry loop exited unexpectedly for {schema.__name__}")
