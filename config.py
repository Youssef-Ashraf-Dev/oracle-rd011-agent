"""
RD.011 Agent — Central configuration.

All environment variables, LLM capability mapping, and project-wide
constants live here.  No other module reads os.environ directly.
"""

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── LLM capability tiers ──────────────────────────────────────────────────

class TaskType(Enum):
    LARGE_CONTEXT = "large_context"
    REASONING = "reasoning"
    GENERATION = "generation"


CAPABILITY_MAP = {
    TaskType.LARGE_CONTEXT: {
        "provider": "google",
        "model": "gemini-3.1-flash-lite-preview",
        "fallback_chain": [
            {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        ],
        # Leverages the 1M context window for the initial document ingestion.
        # Single model (no cascade) — runs once per document.
    },
    TaskType.REASONING: {
        "provider":       "groq",
        "model":          "llama-3.3-70b-versatile",
        "fallback_chain": [
            {"provider": "mistral", "model": "mistral-medium-latest"},
            {"provider": "google", "model": "gemini-3.1-flash-lite-preview"},
        ],
        # For planning and cross-checking facts.
        # Waterfall: primary → fallback_chain on (429: Rate Limit Exceeded) or (400: Bad Request).
    },
    TaskType.GENERATION: {
        "provider":       "groq",
        "model":          "meta-llama/llama-4-scout-17b-16e-instruct",
        "fallback_chain": [
            {"provider": "google", "model": "gemini-3.1-flash-lite-preview"},
        ],
        # High rate limit for writing 15+ sections quickly.
        # Waterfall: primary → fallback_chain on (429: Rate Limit Exceeded) or (400: Bad Request).
    },
}


def _load_capability_overrides() -> dict:
    """
    Load optional task routing overrides from JSON.

    Expected format:
    {
      "reasoning": {
        "provider": "mistral",
        "model": "mistral-medium-latest",
        "fallback_chain": [{"provider": "google", "model": "gemini-3.1-flash-lite-preview"}]
      },
      "generation": { ... }
    }
    """
    override_path = os.getenv("CAPABILITY_OVERRIDES_PATH", "").strip()
    if not override_path:
        return {}

    path = Path(override_path)
    if not path.exists():
        return {}

    try:
        import json

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    task_key_map = {
        TaskType.LARGE_CONTEXT.value: TaskType.LARGE_CONTEXT,
        TaskType.REASONING.value: TaskType.REASONING,
        TaskType.GENERATION.value: TaskType.GENERATION,
    }

    parsed: dict[TaskType, dict] = {}
    for key, value in raw.items():
        task = task_key_map.get(str(key).strip().lower())
        if not task or not isinstance(value, dict):
            continue
        provider = value.get("provider")
        model = value.get("model")
        fallback_chain = value.get("fallback_chain", [])
        if not provider or not model or not isinstance(fallback_chain, list):
            continue
        parsed[task] = {
            "provider": provider,
            "model": model,
            "fallback_chain": fallback_chain,
        }

    return parsed


_CAPABILITY_OVERRIDES = _load_capability_overrides()
if _CAPABILITY_OVERRIDES:
    CAPABILITY_MAP.update(_CAPABILITY_OVERRIDES)

# Optional model policy rules keyed by TaskType -> schema -> provider/model.
# Example:
# MODEL_POLICY_BLOCKLIST = {
#     TaskType.REASONING: {
#         "DocumentPlan": {"groq/llama-3.3-70b-versatile": "high JSON drift"}
#     }
# }
MODEL_POLICY_BLOCKLIST = {
    TaskType.REASONING: {
        "DocumentPlan": {
            "groq/llama-3.3-70b-versatile": "payload-sensitive for large planning prompts",
            "mistral/mistral-medium-latest": "frequent timeouts on large planning prompts",
        },
        "PlanValidationResult": {
            "groq/llama-3.3-70b-versatile": "payload-sensitive for large validation prompts",
            "mistral/mistral-medium-latest": "frequent timeouts on large validation prompts",
        },
    }
}

# ── API keys ──────────────────────────────────────────────────────────────

# Feature flags

def _env_flag(name: str, default: str = "false") -> bool:
    """Parse a boolean flag from environment variables."""
    val = os.getenv(name, default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

RAG_ENABLED = _env_flag("RAG_ENABLED", "false")
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000"))
RAG_MAX_CHUNK_CHARS = int(os.getenv("RAG_MAX_CHUNK_CHARS", "2000"))
RAG_MAX_CHUNKS_PER_SOURCE = int(os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", "2"))
RAG_ALLOWED_SOURCES = os.getenv("RAG_ALLOWED_SOURCES")  # comma-separated basenames, optional
RAG_RETRIEVE_CACHE_MAX_ENTRIES = int(os.getenv("RAG_RETRIEVE_CACHE_MAX_ENTRIES", "512"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── Paths ─────────────────────────────────────────────────────────────────

CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "checkpoints/rd011_checkpoints.db")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/")
TEMPLATE_PATH = os.getenv("TEMPLATE_PATH", "templates/RD011_TEMPLATE.docx")
DIAGRAMS_DIR = os.path.join(OUTPUT_DIR, "diagrams")

# ── Generation config ────────────────────────────────────────────────────

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
MAX_APPROVAL_ITERATIONS = int(os.getenv("MAX_APPROVAL_ITERATIONS", "3"))
ENABLE_REPAIR_PASS = _env_flag("ENABLE_REPAIR_PASS", "true")
GENERATION_THROTTLE_SECONDS = float(os.getenv("GENERATION_THROTTLE_SECONDS", "0"))
GENERATION_THROTTLE_ON_FAILURE_ONLY = _env_flag("GENERATION_THROTTLE_ON_FAILURE_ONLY", "true")

# If a generation model returns invalid JSON for SectionContent, retries often
# repeat the same formatting error (especially around large string fields like
# diagram_code). When enabled, the retry loop will fail fast and move to the
# next fallback model after the first JSONDecodeError.
FAIL_FAST_JSONDECODE_GENERATION = _env_flag("FAIL_FAST_JSONDECODE_GENERATION", "true")

# Generate multiple process sections in parallel to reduce wall time.
# Keep this small (2–3) to avoid 429 rate limits on smaller provider tiers.
SECTION_GENERATION_CONCURRENCY = int(os.getenv("SECTION_GENERATION_CONCURRENCY", "2"))
if SECTION_GENERATION_CONCURRENCY < 1:
    SECTION_GENERATION_CONCURRENCY = 1
if SECTION_GENERATION_CONCURRENCY > 8:
    SECTION_GENERATION_CONCURRENCY = 8

# Retry/route telemetry (JSONL)
LLM_TELEMETRY_ENABLED = _env_flag("LLM_TELEMETRY_ENABLED", "true")
LLM_TELEMETRY_PATH = os.getenv("LLM_TELEMETRY_PATH", "outputs/llm_telemetry.jsonl")

# ── Word paragraph style names (as found in RD011_TEMPLATE.docx) ─────────

WORD_STYLES = {
    # These names must match the style names embedded in `templates/RD011_TEMPLATE.docx`.
    "table_heading": "Table Heading",
    "table_text": "Table Text",
    "body_text": "Body Text",
    "heading1": "Heading 1",
    "heading2": "Heading 2",
    "heading3": "Heading 3",
    "number_list": "Number List",
    "bullet": "Bullet",
    "title": "Title",
}

# ── Canonical business actor naming ──────────────────────────────────────────

# These are the ONLY actor labels allowed in generated process steps.
# Use these exact spellings for consistency across all documents.
CANONICAL_BUSINESS_ACTORS = [
    "AP Accountant",
    "Treasury Accountant",
    "Budget Controller",
    "AR Accountant",
    "GL Accountant",
    "FA Accountant",
    "FA Manager",
    "Chief Accountant",
    "Finance Manager",
    "Treasury Manager",
    "System",
]

# Common synonyms → canonical actor names
ACTOR_SYNONYMS = {
    "ap clerk": "AP Accountant",
    "accounts payable manager": "Finance Manager",
    "treasurer": "Treasury Manager",
    "general ledger manager": "Chief Accountant",
    "ar clerk": "AR Accountant",
    "credit manager": "Finance Manager",
    "billing specialist": "AR Accountant",
    "collections manager": "Treasury Manager",
    "sales manager": "Finance Manager",
    "accounts receivable manager": "Finance Manager",
    "treasury analyst": "Treasury Accountant",
    "cash manager": "Treasury Manager",
    "fixed asset clerk": "FA Accountant",
    "asset manager": "FA Manager",
    "project manager": "Finance Manager",
    "internal auditor": "Chief Accountant",
    "controller": "Chief Accountant",
    "financial controller": "Chief Accountant",
    "general ledger accountant": "GL Accountant",
    "accounting manager": "Chief Accountant",
    "finance / accounting": "Finance Manager",
    "finance/ accounting": "Finance Manager",
    "finance department": "Finance Manager",
}


def normalize_business_actor(name: str) -> str:
    """Map an actor name to its canonical label (case-insensitive)."""
    if not name:
        return name
    raw = " ".join(str(name).strip().split())
    key = raw.lower()
    # Direct canonical match (case-insensitive)
    for canon in CANONICAL_BUSINESS_ACTORS:
        if key == canon.lower():
            return canon
    # Synonym mapping
    mapped = ACTOR_SYNONYMS.get(key)
    return mapped if mapped else raw

# ── Diagram dimensions ───────────────────────────────────────────────────

DIAGRAM_WIDTH_INCHES = 6.0
DIAGRAM_FALLBACK_WIDTH = 1200
DIAGRAM_FALLBACK_HEIGHT = 800
