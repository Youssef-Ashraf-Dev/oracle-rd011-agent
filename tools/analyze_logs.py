"""Analyze RD011 agent run logs.

This is a developer utility (not part of the runtime pipeline).

Usage:
  python tools/analyze_logs.py rd011_agent.log
  python tools/analyze_logs.py run_last1.log run_last2.log
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


def analyze_log(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return 2

    metrics = {
        "start_time": None,
        "end_time": None,
        "ingest_complete": 0,
        "extraction_complete": 0,
        "planning_complete": 0,
        "issue_detection_complete": 0,
        "approval": 0,
        "intro_generation_complete": 0,
        "generating_section": 0,
        "generated_section": 0,
        "render_diagrams": 0,
        "assemble_document": 0,
        "total_processes": 0,
        "llm_attempts": 0,
        "validation_passed": 0,
        "llm_failed": 0,
        "rag_not_available": 0,
        "rag_disabled": 0,
        "rag_query_failed": 0,
        "rag_exemplars": 0,
        "step_chars_0": 0,
        "error_chroma": 0,
        "rag_trace_0": 0,
        "rag_trace_gt0": 0,
        "http_200": 0,
        "http_429": 0,
        "http_404": 0,
        "http_503": 0,
        "warnings": 0,
        "errors": [],
        "rendered_diagrams": 0,
        "mmdc_not_found": 0,
    }

    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?)")

    for line in lines:
        ts_match = ts_pattern.search(line)
        if ts_match:
            ts_str = ts_match.group(1).replace(",", ".")
            try:
                fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in ts_str else "%Y-%m-%d %H:%M:%S"
                ts = datetime.strptime(ts_str, fmt)
                if metrics["start_time"] is None or ts < metrics["start_time"]:
                    metrics["start_time"] = ts
                if metrics["end_time"] is None or ts > metrics["end_time"]:
                    metrics["end_time"] = ts
            except Exception:
                pass

        if "Ingest complete" in line:
            metrics["ingest_complete"] += 1
        if "Extraction complete" in line:
            metrics["extraction_complete"] += 1
        if "Planning complete" in line:
            metrics["planning_complete"] += 1
            m = re.search(r"Planning complete:.* (\d+) total processes", line)
            if m:
                metrics["total_processes"] = int(m.group(1))
        if "Issue detection complete" in line:
            metrics["issue_detection_complete"] += 1
        if "Approval received" in line or "Skipping approval" in line:
            metrics["approval"] += 1
        if "Intro generation complete" in line:
            metrics["intro_generation_complete"] += 1
        if "Generating section" in line:
            metrics["generating_section"] += 1
        if "Generated section" in line:
            metrics["generated_section"] += 1
        if "render_diagrams" in line:
            metrics["render_diagrams"] += 1
        if "assemble_document" in line:
            metrics["assemble_document"] += 1
        if "LLM call attempt" in line:
            metrics["llm_attempts"] += 1
        if "Validation passed" in line:
            metrics["validation_passed"] += 1
        if re.search(r"Attempt .* failed", line):
            metrics["llm_failed"] += 1
        if "RAG not available" in line:
            metrics["rag_not_available"] += 1
        if "RAG disabled" in line:
            metrics["rag_disabled"] += 1
        if "RAG query failed" in line:
            metrics["rag_query_failed"] += 1
        if "RAG exemplars selected for" in line:
            metrics["rag_exemplars"] += 1
        if "step_chars=0" in line:
            metrics["step_chars_0"] += 1
        if "Error querying Chroma" in line:
            metrics["error_chroma"] += 1
        m_rag = re.search(r"RAG_TRACE .* results=(\d+)", line)
        if m_rag:
            if int(m_rag.group(1)) == 0:
                metrics["rag_trace_0"] += 1
            else:
                metrics["rag_trace_gt0"] += 1
        if "status=200" in line or " 200 OK" in line:
            metrics["http_200"] += 1
        if "status=429" in line or " 429 " in line:
            metrics["http_429"] += 1
        if "status=404" in line or " 404 " in line:
            metrics["http_404"] += 1
        if "status=503" in line or " 503 " in line:
            metrics["http_503"] += 1
        if "[WARNING]" in line or " WARNING " in line:
            metrics["warnings"] += 1
        if "[ERROR]" in line or " ERROR " in line:
            metrics["errors"].append(line.strip())
        if "Rendered diagram for" in line:
            metrics["rendered_diagrams"] += 1
        if "mmdc not found on PATH" in line:
            metrics["mmdc_not_found"] += 1

    duration = (
        (metrics["end_time"] - metrics["start_time"]).total_seconds()
        if metrics["start_time"] and metrics["end_time"]
        else 0
    )
    verdict = "SUCCESS" if not metrics["errors"] and metrics["assemble_document"] > 0 else "FAIL"

    print(f"--- {path} ---")
    print(f"Start: {metrics['start_time']} Dur: {duration}s")
    print(
        "Stages: "
        f"Ingest:{metrics['ingest_complete']}, "
        f"Extr:{metrics['extraction_complete']}, "
        f"Plan:{metrics['planning_complete']} (Procs:{metrics['total_processes']}), "
        f"Issue:{metrics['issue_detection_complete']}, "
        f"App:{metrics['approval']}, "
        f"Intro:{metrics['intro_generation_complete']}, "
        f"Gen:{metrics['generating_section']}/{metrics['generated_section']}, "
        f"Render:{metrics['render_diagrams']}, "
        f"Assemb:{metrics['assemble_document']}"
    )
    print(f"LLM: Att:{metrics['llm_attempts']}, Pass:{metrics['validation_passed']}, Fail:{metrics['llm_failed']}")
    print(
        "RAG: "
        f"NA:{metrics['rag_not_available']}, Dis:{metrics['rag_disabled']}, QFail:{metrics['rag_query_failed']}, "
        f"Exemp:{metrics['rag_exemplars']}, step0:{metrics['step_chars_0']}, Chroma:{metrics['error_chroma']}, "
        f"Trace0:{metrics['rag_trace_0']}, Trace>0:{metrics['rag_trace_gt0']}"
    )
    print(
        "HTTP: "
        f"200:{metrics['http_200']}, 429:{metrics['http_429']}, 404:{metrics['http_404']}, 503:{metrics['http_503']}"
    )
    print(f"Health: W:{metrics['warnings']}, E:{len(metrics['errors'])}")
    if metrics["errors"]:
        for e in metrics["errors"][:3]:
            print(f"  E: {e[:120]}...")
    print(f"Diags: Ren:{metrics['rendered_diagrams']}, NoMMDC:{metrics['mmdc_not_found']}")
    print(f"Verdict: {verdict}\n")
    return 0 if verdict == "SUCCESS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze RD011 agent logs")
    ap.add_argument("log_files", nargs="+", help="One or more log files to analyze")
    args = ap.parse_args()

    rc = 0
    for f in args.log_files:
        rc = max(rc, analyze_log(Path(f)))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
