"""
RD.011 Agent — LangGraph state graph definition.

Builds the complete processing pipeline with optional SQLite checkpointing.
If `langgraph-checkpoint-sqlite` is not installed, falls back automatically
to MemorySaver (in-memory, non-persistent).  State will not survive process
restarts in that case — install the extra package when you need durability:

    pip install langgraph-checkpoint-sqlite
"""

from __future__ import annotations

import logging
import os

import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from config import CHECKPOINT_DB_PATH
from nodes import (
    assemble_document_node,
    await_approval_node,
    detect_issues_node,
    error_handler_node,
    extract_node,
    generate_intro_node,
    generate_section_node,
    ingest_node,
    plan_node,
    present_plan_node,
    render_diagrams_node,
    update_plan_node,
)
from state import RD011State

logger = logging.getLogger(__name__)

# Optional SQLite checkpointer — gracefully absent when the extra package is
# not installed.  We check at module import time so build_graph() can branch
# without try/except noise inside the function.
try:
    from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver  # type: ignore[import]
    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False


def route_after_approval(state: RD011State) -> str:
    """Route to generation or plan update based on approval status."""
    if state.get("approval_maxed"):
        return "error_handler"
    if state["consultant_approved"]:
        return "generate_intro"
    return "update_plan"


def route_after_section(state: RD011State) -> str:
    """Route to next section or to diagram rendering when queue is exhausted."""
    if state["current_section_index"] < len(state["section_queue"]):
        return "generate_section"
    return "render_diagrams"


def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Uses SQLite checkpointing when langgraph-checkpoint-sqlite is installed;
    falls back to MemorySaver (in-memory) otherwise.
    """
    builder = StateGraph(RD011State)

    # ── Add all nodes ────────────────────────────────────────────────────
    builder.add_node("ingest", ingest_node)
    builder.add_node("extract", extract_node)
    builder.add_node("plan", plan_node)
    builder.add_node("detect_issues", detect_issues_node)
    builder.add_node("present_plan", present_plan_node)
    builder.add_node("await_approval", await_approval_node)
    builder.add_node("update_plan", update_plan_node)
    builder.add_node("generate_intro", generate_intro_node)
    builder.add_node("generate_section", generate_section_node)
    builder.add_node("render_diagrams", render_diagrams_node)
    builder.add_node("assemble_document", assemble_document_node)
    builder.add_node("error_handler", error_handler_node)

    # ── Entry point ──────────────────────────────────────────────────────
    builder.set_entry_point("ingest")

    # ── Linear edges ─────────────────────────────────────────────────────
    builder.add_edge("ingest", "extract")
    builder.add_edge("extract", "plan")
    builder.add_edge("plan", "detect_issues")
    builder.add_edge("detect_issues", "present_plan")
    builder.add_edge("present_plan", "await_approval")

    # ── Approval conditional ─────────────────────────────────────────────
    builder.add_conditional_edges(
        "await_approval",
        route_after_approval,
        {
            "generate_intro": "generate_intro",
            "update_plan": "update_plan",
            "error_handler": "error_handler",
        },
    )

    # Update plan loops back to present_plan
    builder.add_edge("update_plan", "present_plan")

    # ── Section generation loop ──────────────────────────────────────────
    builder.add_edge("generate_intro", "generate_section")
    builder.add_conditional_edges(
        "generate_section",
        route_after_section,
        {"generate_section": "generate_section", "render_diagrams": "render_diagrams"},
    )

    # ── Render then assemble ─────────────────────────────────────────────
    builder.add_edge("render_diagrams", "assemble_document")
    builder.add_edge("assemble_document", END)
    builder.add_edge("error_handler", END)

    # ── Compile with checkpointer ─────────────────────────────────────────
    if _SQLITE_AVAILABLE:
        checkpoint_dir = os.path.dirname(CHECKPOINT_DB_PATH)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        # In langgraph-checkpoint-sqlite 3.x, use a direct sqlite3 connection
        # (from_conn_string() returns a context manager, not a saver instance)
        conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        checkpointer = _SqliteSaver(conn=conn)
        logger.info("Using SQLite checkpointer: %s", CHECKPOINT_DB_PATH)
    else:
        checkpointer = MemorySaver()
        logger.warning(
            "langgraph-checkpoint-sqlite is not installed; "
            "using in-memory checkpointing — state will NOT persist across restarts. "
            "To enable persistence: pip install langgraph-checkpoint-sqlite"
        )

    return builder.compile(checkpointer=checkpointer)
