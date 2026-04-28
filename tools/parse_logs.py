"""Parse RD011 agent logs into per-run summaries.

This is a developer utility (not part of the runtime pipeline).

The parser splits a log file into runs using the "Using SQLite checkpointer" marker
and prints a summary for the last N runs.

Usage:
  python tools/parse_logs.py rd011_agent.log
  python tools/parse_logs.py rd011_agent.log --runs 2
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


def _split_runs(lines: list[str]) -> list[list[str]]:
    run_boundaries = [i for i, line in enumerate(lines) if "Using SQLite checkpointer" in line]
    if not run_boundaries:
        return []

    runs: list[list[str]] = []
    for idx, start_idx in enumerate(run_boundaries):
        end_idx = run_boundaries[idx + 1] if idx + 1 < len(run_boundaries) else len(lines)
        runs.append(lines[start_idx:end_idx])
    return runs


def parse_log(path: Path, runs_to_print: int = 2) -> int:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return 2

    runs = _split_runs(lines)
    if not runs:
        print("No run boundaries found (missing 'Using SQLite checkpointer').")
        return 1

    selected = runs[-runs_to_print:] if runs_to_print > 0 else runs

    results = []
    for run_id, run in enumerate(selected, start=max(1, len(runs) - len(selected) + 1)):
        run_data: dict = {"run_id": run_id, "total_lines": len(run)}

        ts_pattern = re.compile(r"^(\d{2}:\d{2}:\d{2})")
        timestamps = [ts_pattern.match(line).group(1) for line in run if ts_pattern.match(line)]
        if timestamps:
            run_data["start_time"] = timestamps[0]
            run_data["end_time"] = timestamps[-1]
            try:
                fmt = "%H:%M:%S"
                tdelta = datetime.strptime(run_data["end_time"], fmt) - datetime.strptime(run_data["start_time"], fmt)
                run_data["duration"] = str(tdelta)
            except Exception:
                run_data["duration"] = "Unknown"
        else:
            run_data["start_time"] = "Unknown"
            run_data["end_time"] = "Unknown"
            run_data["duration"] = "Unknown"

        log_levels = Counter()
        for line in run:
            if "[INFO]" in line:
                log_levels["INFO"] += 1
            elif "[WARNING]" in line:
                log_levels["WARNING"] += 1
            elif "[ERROR]" in line:
                log_levels["ERROR"] += 1
        run_data["log_levels"] = log_levels

        run_data["success"] = any("Document assembled" in line for line in run)
        run_data["files_parsed"] = len([line for line in run if "doc_parser.file_parser: Parsed" in line])

        ingest_lines = [line for line in run if "doc_parser.ingest: Ingested" in line]
        total_ingest_lines_count = 0
        total_ingest_chars = 0
        for line in ingest_lines:
            match = re.search(r"Ingested (\d+) lines \((\d+) chars\)", line)
            if match:
                total_ingest_lines_count += int(match.group(1))
                total_ingest_chars += int(match.group(2))
        run_data["total_ingest_lines"] = total_ingest_lines_count
        run_data["total_ingest_chars"] = total_ingest_chars

        extract_match = next(
            (
                re.search(r"Extracted metadata: (\d+) modules, (\d+) actor groups", line)
                for line in run
                if "Extracted metadata" in line
            ),
            None,
        )
        run_data["modules"] = extract_match.group(1) if extract_match else "0"
        run_data["actor_groups"] = extract_match.group(2) if extract_match else "0"

        plan_match = next((re.search(r"Planned (\d+) sections across (\d+) modules", line) for line in run if "Planned" in line), None)
        run_data["total_sections"] = plan_match.group(1) if plan_match else "0"

        run_data["contradictions"] = len([line for line in run if "contradiction" in line.lower()])
        run_data["ambiguous"] = len([line for line in run if "ambiguous" in line.lower()])

        intro_match = next((re.search(r"Generated intro with (\d+) paragraphs", line) for line in run if "Generated intro" in line), None)
        run_data["intro_paragraphs"] = intro_match.group(1) if intro_match else "0"

        gen_lines = [line for line in run if "nodes.generate_section_node: Generating section" in line]
        run_data["sections_attempted"] = len(gen_lines)
        run_data["sections_generated"] = len([line for line in run if "nodes.generate_section_node: Generated" in line])
        run_data["sections_failed"] = len([line for line in run if "FAILED" in line and "section" in line.lower()])
        run_data["retries"] = len([line for line in run if "LLM call attempt" in line and "attempt 1" not in line])
        run_data["json_failures"] = len([line for line in run if "JSON" in line and ("invalid" in line.lower() or "failure" in line.lower())])

        run_data["rag_enabled"] = len([line for line in run if "RAG enabled" in line])
        run_data["rag_disabled"] = len([line for line in run if "RAG disabled" in line])
        run_data["rag_not_available"] = len([line for line in run if "RAG not available" in line])
        run_data["exemplars_selected"] = len([line for line in run if "exemplars selected" in line])
        run_data["step_chars_0"] = len([line for line in run if "step_chars=0" in line])
        run_data["style_chars_0"] = len([line for line in run if "style_chars=0" in line])
        run_data["rag_query_failures"] = len([line for line in run if "RAG query failed" in line])
        rag_trace_matches = [re.search(r"results=(\d+)", line) for line in run if "RAG_TRACE" in line]
        run_data["rag_trace_results_gt_0"] = sum(1 for m in rag_trace_matches if m and int(m.group(1)) > 0)

        results.append(run_data)

    for data in results:
        print(f"\n--- Run {data['run_id']} ---")
        print(f"Time: {data['start_time']} to {data['end_time']} (Duration: {data['duration']})")
        print(f"Total Lines: {data['total_lines']}")
        print(
            "Log Counts: "
            f"INFO={data['log_levels']['INFO']}, "
            f"WARNING={data['log_levels']['WARNING']}, "
            f"ERROR={data['log_levels']['ERROR']}"
        )
        print(f"Status: {'SUCCESS' if data['success'] else 'FAILED'}")
        print(f"Ingest: {data['files_parsed']} files, {data['total_ingest_lines']} lines, {data['total_ingest_chars']} chars")
        print(f"Extraction: {data['modules']} modules, {data['actor_groups']} actor groups")
        print(f"Planning: {data['total_sections']} sections")
        print(f"Issues: Contradictions={data['contradictions']}, Ambiguous={data['ambiguous']}")
        print(f"Intro: {data['intro_paragraphs']} paragraphs")
        print(
            "Section Gen: "
            f"Attempted={data['sections_attempted']}, "
            f"Generated={data['sections_generated']}, "
            f"Failed={data['sections_failed']}"
        )
        print(f"Generation Retries: {data['retries']}, JSON Failures: {data['json_failures']}")
        print(f"RAG: Enabled={data['rag_enabled']}, Disabled={data['rag_disabled']}, NotAvail={data['rag_not_available']}")
        print(
            "RAG Details: "
            f"Exemplars={data['exemplars_selected']}, "
            f"StepChars0={data['step_chars_0']}, "
            f"StyleChars0={data['style_chars_0']}, "
            f"Failures={data['rag_query_failures']}, "
            f"TraceResults>0={data['rag_trace_results_gt_0']}"
        )

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse RD011 agent logs")
    ap.add_argument("log_file", help="Log file path (e.g. rd011_agent.log)")
    ap.add_argument("--runs", type=int, default=2, help="Number of last runs to summarize")
    args = ap.parse_args()
    return parse_log(Path(args.log_file), runs_to_print=args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
