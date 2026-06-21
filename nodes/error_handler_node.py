"""
RD.011 Agent — Error Handler node.

Logs error details and prints a human-readable summary.
Allows the graph to continue or terminate gracefully.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def error_handler_node(state: dict) -> dict:
    """
    Log and display accumulated errors.

    Returns state unchanged so the graph can continue or terminate
    gracefully depending on the edge configuration.
    """
    errors = state.get("errors", [])
    last_node = state.get("last_completed_node", "unknown")
    failed_sections = state.get("failed_sections", [])

    if not errors and not failed_sections:
        logger.info("Error handler invoked but no errors found.")
        return {}

    logger.info("=" * 55)
    logger.info("RD.011 Agent — Error Summary")
    logger.info("=" * 55)

    if errors:
        logger.info("Errors (%d total):", len(errors))
        for i, error in enumerate(errors, start=1):
            logger.error("Error %d: %s", i, error)

    if failed_sections:
        logger.info("Failed Sections (%d total):", len(failed_sections))
        for section in failed_sections:
            logger.warning("Failed section: %s", section)

    logger.info("Last completed node: %s", last_node)
    logger.info("Thread ID: %s", state.get("thread_id", "N/A"))
    logger.info("To resume from the last checkpoint, run: python main.py --resume --thread-id %s", state.get("thread_id", "<thread_id>"))

    return {}
