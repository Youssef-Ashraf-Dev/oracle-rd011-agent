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

    print()
    print("\u2550" * 55)
    print("  RD.011 Agent \u2014 Error Summary")
    print("\u2550" * 55)
    print()

    if errors:
        print(f"  Errors ({len(errors)} total):")
        for i, error in enumerate(errors, start=1):
            logger.error("Error %d: %s", i, error)
            print(f"    {i}. {error}")
        print()

    if failed_sections:
        print(f"  Failed Sections ({len(failed_sections)} total):")
        for section in failed_sections:
            print(f"    \u2022 {section}")
        print()

    print(f"  Last completed node: {last_node}")
    print(f"  Thread ID: {state.get('thread_id', 'N/A')}")
    print()
    print("  To resume from the last checkpoint, run:")
    print(f"    python main.py --resume --thread-id {state.get('thread_id', '<thread_id>')}")
    print()
    print("\u2500" * 55)

    return {}
