"""
Full E2E pipeline test: fresh run with APPROVE piped in.
Uses all available MOM files (they accumulate on each other).
Streams output in real-time and checks for document generation.
"""
import subprocess
import sys
import os
import time

PYTHON = r"C:\Users\RayaIT-Admin\miniconda3\envs\rd011-env\python.exe"

# All MOMs to ingest (add up on each other to build complete requirements)
MOMS = [
    "samples/inputs/AP20_Formatted.docx",
    "samples/inputs/AR_Full.docx",
    "samples/inputs/CM_Jan1.docx",
    "samples/inputs/Dec_19_2022_GL_Analysis_MOM_Presentation_Ready.docx",
    "samples/inputs/FA_Dec28.docx",
]

# Scope document (single file defining requirements)
SCOPE = "samples/inputs/Oracle_Financials_Requirements_Agreed_PainPoints.docx"

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

print(f"Pipeline start: {time.strftime('%H:%M:%S')}")
print(f"MOMs ({len(MOMS)}):")
for mom in MOMS:
    print(f"  - {mom}")
print(f"SCOPE: {SCOPE}")
print()

# Build command: python main.py --scope <scope> --mom <mom1> <mom2> ... <mom5>
cmd = [PYTHON, "main.py", "--scope", SCOPE, "--mom"] + MOMS

proc = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=r"d:\Finance\rd011",
    env=env,
    text=True,
    encoding="utf-8",
    errors="replace",
)

# Send APPROVE then close stdin
proc.stdin.write("APPROVE\n")
proc.stdin.flush()
proc.stdin.close()

approval_shown = False
doc_generated = False
thread_id = None

for line in proc.stdout:
    s = line.rstrip()
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", errors="replace").decode("ascii"))

    if "Thread ID:" in s:
        thread_id = s.split("Thread ID:")[-1].strip()
    if "Type APPROVE" in s:
        approval_shown = True
    if "Document saved" in s or "Document generated" in s or ("output" in s.lower() and ".docx" in s.lower()):
        doc_generated = True

rc = proc.wait()

print()
print("=" * 60)
print(f"Pipeline end:       {time.strftime('%H:%M:%S')}")
print(f"Thread ID:          {thread_id}")
print(f"Return code:        {rc}")
print(f"Approval shown:     {approval_shown}")
print(f"Document generated: {doc_generated}")

# Check outputs dir
out_dir = r"d:\Finance\rd011\outputs"
if os.path.isdir(out_dir):
    for f in os.listdir(out_dir):
        full = os.path.join(out_dir, f)
        sz = os.path.getsize(full) if os.path.isfile(full) else 0
        print(f"  OUTPUT: {f}  ({sz:,} bytes)")
else:
    print("  No outputs directory found")
