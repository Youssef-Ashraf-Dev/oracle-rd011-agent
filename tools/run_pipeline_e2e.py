"""Run the RD011 pipeline end-to-end and optionally auto-approve.

This is a developer utility (not part of the runtime pipeline). It exists to
quickly smoke-test the CLI approval loop and verify a document is produced.

Usage:
  python tools/run_pipeline_e2e.py --auto-approve
  python tools/run_pipeline_e2e.py --mom samples/inputs_gosaibi_fin_2022/AP20_Formatted.docx --auto-approve
  python tools/run_pipeline_e2e.py --mom <mom1> --mom <mom2> --scope <scope.docx> --auto-approve
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _default_moms() -> list[str]:
    base = Path("samples") / "inputs_gosaibi_fin_2022"
    if not base.exists():
        return []
    return [str(p) for p in sorted(base.glob("*.docx"))]


def main() -> int:
    ap = argparse.ArgumentParser(description="RD011 E2E pipeline runner")
    ap.add_argument(
        "--mom",
        action="append",
        default=None,
        help="Path to a MoM .docx. Repeat --mom to pass multiple files.",
    )
    ap.add_argument("--scope", default=None, help="Optional scope .docx")
    ap.add_argument(
        "--questionnaire",
        action="append",
        default=None,
        help="Optional questionnaire .xlsx. Repeat to pass multiple files.",
    )
    ap.add_argument("--auto-approve", action="store_true", help="Send APPROVE to stdin")
    ap.add_argument("--cwd", default=".", help="Working directory (project root)")
    args = ap.parse_args()

    moms = args.mom if args.mom else _default_moms()
    if not moms:
        print("No MOMs provided and no defaults found under samples/inputs_gosaibi_fin_2022")
        return 2

    cwd = Path(args.cwd).resolve()

    cmd: list[str] = [sys.executable, "main.py"]
    if args.scope:
        cmd += ["--scope", args.scope]
    cmd += ["--mom"] + moms

    if args.questionnaire:
        for q in args.questionnaire:
            cmd += ["--questionnaire", q]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    print(f"Pipeline start: {time.strftime('%H:%M:%S')}")
    print(f"CWD: {cwd}")
    print("Command:")
    print("  " + " ".join(cmd))
    print(f"MOMs ({len(moms)}):")
    for m in moms:
        print(f"  - {m}")
    if args.scope:
        print(f"SCOPE: {args.scope}")
    if args.questionnaire:
        print(f"QUESTIONNAIRES ({len(args.questionnaire)}):")
        for q in args.questionnaire:
            print(f"  - {q}")
    print()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if args.auto_approve and proc.stdin is not None:
        proc.stdin.write("APPROVE\n")
        proc.stdin.flush()
        proc.stdin.close()

    approval_shown = False
    doc_generated = False
    thread_id = None

    assert proc.stdout is not None
    for line in proc.stdout:
        s = line.rstrip("\n")
        print(s)
        if "Thread ID:" in s:
            thread_id = s.split("Thread ID:")[-1].strip()
        if "Type APPROVE" in s:
            approval_shown = True
        if "Document saved" in s or "Document assembled" in s or (".docx" in s.lower() and "output" in s.lower()):
            doc_generated = True

    rc = proc.wait()

    print()
    print("=" * 60)
    print(f"Pipeline end:       {time.strftime('%H:%M:%S')}")
    print(f"Thread ID:          {thread_id}")
    print(f"Return code:        {rc}")
    print(f"Approval shown:     {approval_shown}")
    print(f"Document generated: {doc_generated}")
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
