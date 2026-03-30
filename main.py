"""
RD.011 Future Process Model Generator

Usage:
    python main.py --mom path/to/mom.docx --scope path/to/scope.docx [--questionnaire path/to/q.xlsx]
    python main.py --resume --thread-id <thread_id>

On first run: generates a new thread_id and prints it.
On --resume: loads checkpoint from SQLite and continues from last completed node.

At the approval step, the plan is printed and you are prompted to type:
  APPROVE           — accept the plan and start document generation
  <any other text>  — send feedback to revise the plan (loop repeats)
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

from langgraph.types import Command

from graph import build_graph
from state import RD011State


def _setup_logging():
    """Configure logging for the agent."""
    # Force UTF-8 on stdout/stderr so Unicode box-drawing characters in the
    # plan display don't raise UnicodeEncodeError on Windows CP1252 terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("rd011_agent.log", mode="a", encoding="utf-8"),
        ],
    )


def _run_with_approval_loop(graph, first_input, config: dict, thread_id: str) -> dict:
    """
    Invoke the graph and handle LangGraph interrupt() calls.

    In LangGraph 1.0.x, interrupt() does NOT raise an exception.  Instead,
    graph.invoke() returns normally with an ``__interrupt__`` key in the
    result dict.  We detect this, prompt the consultant on stdin, and resume
    via ``Command(resume=...)``.  This loop continues until the graph
    completes without an interrupt or a real error occurs.
    """
    current_input = first_input
    result: dict = {}

    while True:
        try:
            result = graph.invoke(current_input, config=config)
        except Exception as exc:
            import traceback
            logging.getLogger(__name__).error("Pipeline failed: %s", exc)
            logging.getLogger(__name__).error("Traceback:\n%s", traceback.format_exc())
            print(f"\nPipeline error: {exc}")
            traceback.print_exc()
            print(f"Resume with: python main.py --resume --thread-id {thread_id}")
            sys.exit(1)

        # Graph suspended at await_approval_node — ask the consultant
        if "__interrupt__" in result:
            sys.stdout.flush()
            def _looks_like_resume_cmd(text: str) -> bool:
                t = text.strip().lower()
                return (
                    t.startswith("python main.py")
                    and "--resume" in t
                    and "--thread-id" in t
                )

            while True:
                try:
                    user_response = input("  >>> ").strip()
                except EOFError:
                    # Non-interactive environment (e.g. piped input exhausted)
                    print("  [No input received — aborting]")
                    print(f"  Rerun with: python main.py --resume --thread-id {thread_id}")
                    sys.exit(1)

                if not user_response:
                    print("  [Please type APPROVE or feedback]")
                    continue
                if _looks_like_resume_cmd(user_response):
                    print("  [Detected resume command text — please type APPROVE or feedback]")
                    continue
                break

            current_input = Command(resume=user_response)
        else:
            break  # Graph completed normally

    return result


def main():
    """Entry point for the RD.011 document generator."""
    _setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="RD.011 Future Process Model Document Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mom",
        nargs="+",
        action="append",
        metavar="FILE",
        help=(
            "Path(s) to Minutes of Meeting .docx files. "
            "Use one --mom per file (--mom A.docx --mom B.docx) "
            "OR list all files after a single --mom flag (--mom A.docx B.docx C.docx)."
        ),
    )
    parser.add_argument(
        "--scope",
        help="Path to Scope of Solution .docx",
    )
    parser.add_argument(
        "--questionnaire",
        help="Path to Oracle Questionnaire .xlsx (optional)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--thread-id",
        help="Thread ID for resume",
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.resume and not args.mom:
        parser.error("--mom is required for new runs (use --resume to continue an existing run)")

    graph = build_graph()

    if args.resume:
        if not args.thread_id:
            parser.error("--thread-id is required when using --resume")

        thread_id = args.thread_id
        logger.info("Resuming thread: %s", thread_id)
        print(f"Resuming thread: {thread_id}")
        print()

        config = {"configurable": {"thread_id": thread_id}}

        # Detect whether the checkpoint is paused at a human-in-the-loop interrupt.
        # graph.invoke(None) returns the saved state without re-running any nodes
        # when the graph is at an interrupt — GraphInterrupt is never raised that way.
        # We must detect it upfront and pass Command(resume=...) as the first input.
        snapshot = graph.get_state(config)
        has_interrupt = any(t.interrupts for t in snapshot.tasks)

        if has_interrupt:
            print("  Graph is paused at the approval step.")
            print()
            print("  ─────────────────────────────────────────────────────")
            print("  Type APPROVE to proceed, or type feedback to revise:")
            print("  ─────────────────────────────────────────────────────")
            try:
                user_response = input("  >>> ").strip()
            except EOFError:
                print("  [No input received — aborting]")
                print(f"  Rerun with: python main.py --resume --thread-id {thread_id}")
                sys.exit(1)
            first_input = Command(resume=user_response)
        else:
            # Not at an interrupt — resume normally from last checkpoint
            first_input = None

        result = _run_with_approval_loop(graph, first_input, config, thread_id)

    else:
        thread_id = str(uuid.uuid4())[:8]

        input_files: list[str] = []
        if args.mom:
            # nargs="+" + action="append" → [[A, B], [C], [D, E]] — flatten it
            for group in args.mom:
                input_files.extend(group)
        if args.scope:
            input_files.append(args.scope)
        if args.questionnaire:
            input_files.append(args.questionnaire)

        initial_state: RD011State = {
            "thread_id": thread_id,
            "input_files": input_files,
            "raw_texts": {},
            "extraction_result": None,
            "document_plan": None,
            "issue_report": None,
            "consultant_approved": False,
            "consultant_feedback": "",
            "approval_iteration": 0,
            "approval_maxed": False,
            "intro_content": None,
            "section_queue": [],
            "current_section_index": 0,
            "generated_sections": {},
            "failed_sections": [],
            "diagram_registry": {},
            "output_path": None,
            "errors": [],
            "last_completed_node": "",
        }

        print(f"Starting new run. Thread ID: {thread_id}")
        print(f"Save this ID to resume if interrupted: {thread_id}")
        print()

        config = {"configurable": {"thread_id": thread_id}}
        result = _run_with_approval_loop(graph, initial_state, config, thread_id)

    # Report results
    if result.get("output_path"):
        print(f"\nDocument generated: {result['output_path']}")
    if result.get("failed_sections"):
        print(f"Failed sections (check manually): {result['failed_sections']}")
    if result.get("errors"):
        print(f"Errors encountered: {len(result['errors'])}")
        for err in result["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
