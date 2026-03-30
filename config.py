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
        "provider": "groq",
        "model":    "llama-3.3-70b-versatile",
        "fallback_chain": [
            {"provider": "google", "model": "gemini-3.1-flash-lite-preview"}
        ],
        # Leverages the 1M context window for the initial document ingestion.
        # Single model (no cascade) — runs once per document.
    },
    TaskType.REASONING: {
        "provider":       "groq",
        "model":          "llama-3.3-70b-versatile",
        "fallback_chain": [
            {"provider": "groq",   "model": "openai/gpt-oss-120b"},
            {"provider": "google", "model": "gemini-3.1-flash-lite-preview"},
        ],
        # For planning and cross-checking facts.
        # Waterfall: primary → fallback_chain on 429 or 400.
    },
    TaskType.GENERATION: {
        "provider":       "groq",
        "model":          "meta-llama/llama-4-scout-17b-16e-instruct",
        "fallback_chain": [
            {"provider": "groq",   "model": "qwen/qwen3-32b"},
            {"provider": "google", "model": "gemini-3.1-flash-lite-preview"},
        ],
        # High rate limit for writing 15+ sections quickly.
        # Waterfall: primary → fallback_chain on 429 or 400.
    },
}

# ── API keys ──────────────────────────────────────────────────────────────

# Feature flags

def _env_flag(name: str, default: str = "false") -> bool:
    """Parse a boolean flag from environment variables."""
    val = os.getenv(name, default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

RAG_ENABLED = _env_flag("RAG_ENABLED", "false")

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

# ── Word paragraph style names (as found in RD011_TEMPLATE.docx) ─────────

WORD_STYLES = {
    "table_heading": "TableHeading",
    "table_text": "TableText",
    "body_text": "BodyText",
    "heading1": "Heading 1",
    "heading2": "Heading 2",
    "heading3": "Heading 3",
    "number_list": "NumberList",
    "bullet": "Bullet",
    "title": "Title",
}

# ── Canonical business actor naming ──────────────────────────────────────────

# These are the ONLY actor labels allowed in generated process steps.
# Use these exact spellings for consistency across all documents.
CANONICAL_BUSINESS_ACTORS = [
    "Supplier User",
    "Supplier Administrator",
    "AP Accountant",
    "AR Accountant",
    "GL Accountant",
    "FA Accountant",
    "FA Manager",
    "Treasury Accountant",
    "Treasury Manager",
    "Budget Controller",
    "Finance Manager",
    "Chief Accountant",
    "Revenue Accountant",
    "Key User",
    "Procurement",
    "Purchasing",
    "Inventory Department",
    "Cashier",
    "Finance Department",
    "System",
]

# Common synonyms → canonical actor names
ACTOR_SYNONYMS = {
    "ap clerk": "AP Accountant",
    "accounts payable manager": "Finance Manager",
    "procurement specialist": "Procurement",
    "treasurer": "Treasury Manager",
    "general ledger manager": "Chief Accountant",
    "ar clerk": "AR Accountant",
    "credit manager": "Finance Manager",
    "billing specialist": "AR Accountant",
    "collections manager": "Treasury Manager",
    "sales manager": "Revenue Accountant",
    "accounts receivable manager": "Finance Manager",
    "treasury analyst": "Treasury Accountant",
    "cash manager": "Treasury Manager",
    "fixed asset clerk": "FA Accountant",
    "asset manager": "FA Manager",
    "project manager": "Finance Manager",
    "internal auditor": "Inventory Department",
    "controller": "Chief Accountant",
    "general ledger accountant": "GL Accountant",
    "accounting manager": "Chief Accountant",
    "finance / accounting": "Finance Department",
    "finance/ accounting": "Finance Department",
    "finance department": "Finance Department",
    "procurement department": "Procurement",
    "purchasing department": "Purchasing",
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
