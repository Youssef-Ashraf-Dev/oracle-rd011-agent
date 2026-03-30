"""
Report which processes in the final plan are implicit (from config) vs extracted.

Usage:
  python _implicit_report.py --thread-id <THREAD_ID>
  python _implicit_report.py --latest
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sqlite3
import struct
from collections import defaultdict
from pathlib import Path


class MsgpackDecoder:
    def __init__(self, data: bytes):
        self.data = data
        self.i = 0

    def read(self, n: int) -> bytes:
        b = self.data[self.i : self.i + n]
        self.i += n
        return b

    def unpack(self):
        b = self.data[self.i]
        self.i += 1

        if b <= 0x7F:
            return b
        if 0x80 <= b <= 0x8F:
            size = b & 0x0F
            return {self.unpack(): self.unpack() for _ in range(size)}
        if 0x90 <= b <= 0x9F:
            size = b & 0x0F
            return [self.unpack() for _ in range(size)]
        if 0xA0 <= b <= 0xBF:
            length = b & 0x1F
            return self.read(length).decode("utf-8", errors="replace")
        if b == 0xC0:
            return None
        if b == 0xC2:
            return False
        if b == 0xC3:
            return True
        if b == 0xC4:
            length = self.read(1)[0]
            return self.read(length)
        if b == 0xC5:
            length = struct.unpack(">H", self.read(2))[0]
            return self.read(length)
        if b == 0xC6:
            length = struct.unpack(">I", self.read(4))[0]
            return self.read(length)
        if b == 0xC7:
            length = self.read(1)[0]
            _type = self.read(1)[0]
            return self.read(length)
        if b == 0xC8:
            length = struct.unpack(">H", self.read(2))[0]
            _type = self.read(1)[0]
            return self.read(length)
        if b == 0xC9:
            length = struct.unpack(">I", self.read(4))[0]
            _type = self.read(1)[0]
            return self.read(length)
        if b == 0xCA:
            return struct.unpack(">f", self.read(4))[0]
        if b == 0xCB:
            return struct.unpack(">d", self.read(8))[0]
        if b == 0xCC:
            return self.read(1)[0]
        if b == 0xCD:
            return struct.unpack(">H", self.read(2))[0]
        if b == 0xCE:
            return struct.unpack(">I", self.read(4))[0]
        if b == 0xCF:
            return struct.unpack(">Q", self.read(8))[0]
        if b == 0xD0:
            return struct.unpack(">b", self.read(1))[0]
        if b == 0xD1:
            return struct.unpack(">h", self.read(2))[0]
        if b == 0xD2:
            return struct.unpack(">i", self.read(4))[0]
        if b == 0xD3:
            return struct.unpack(">q", self.read(8))[0]
        if b == 0xD4:
            _type = self.read(1)[0]
            return self.read(1)
        if b == 0xD5:
            _type = self.read(1)[0]
            return self.read(2)
        if b == 0xD6:
            _type = self.read(1)[0]
            return self.read(4)
        if b == 0xD7:
            _type = self.read(1)[0]
            return self.read(8)
        if b == 0xD8:
            _type = self.read(1)[0]
            return self.read(16)
        if b == 0xD9:
            length = self.read(1)[0]
            return self.read(length).decode("utf-8", errors="replace")
        if b == 0xDA:
            length = struct.unpack(">H", self.read(2))[0]
            return self.read(length).decode("utf-8", errors="replace")
        if b == 0xDB:
            length = struct.unpack(">I", self.read(4))[0]
            return self.read(length).decode("utf-8", errors="replace")
        if b == 0xDC:
            size = struct.unpack(">H", self.read(2))[0]
            return [self.unpack() for _ in range(size)]
        if b == 0xDD:
            size = struct.unpack(">I", self.read(4))[0]
            return [self.unpack() for _ in range(size)]
        if b == 0xDE:
            size = struct.unpack(">H", self.read(2))[0]
            return {self.unpack(): self.unpack() for _ in range(size)}
        if b == 0xDF:
            size = struct.unpack(">I", self.read(4))[0]
            return {self.unpack(): self.unpack() for _ in range(size)}
        if 0xE0 <= b <= 0xFF:
            return b - 0x100
        raise ValueError(f"Unknown byte: {b:#x}")


def unpack_msgpack(data: bytes):
    return MsgpackDecoder(data).unpack()


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return " ".join(cleaned.split())


def name_similarity(a: str, b: str) -> float:
    norm_a = normalize_name(a)
    norm_b = normalize_name(b)
    if not norm_a or not norm_b:
        return 0.0
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()


def get_latest_thread_id(con: sqlite3.Connection) -> str | None:
    cur = con.cursor()
    cur.execute(
        """
        select thread_id, max(rowid) as max_row
        from checkpoints
        group by thread_id
        order by max_row desc
        limit 1
        """
    )
    row = cur.fetchone()
    return row[0] if row else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Implicit vs extracted process report")
    parser.add_argument("--thread-id", dest="thread_id")
    parser.add_argument("--latest", action="store_true", help="Use latest thread in checkpoints DB")
    parser.add_argument(
        "--db",
        default="checkpoints/rd011_checkpoints.db",
        help="Path to checkpoints sqlite db",
    )
    parser.add_argument(
        "--config",
        default="config_implicit_processes.json",
        help="Path to implicit processes config",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Checkpoint DB not found: {db_path}")
        return 1

    con = sqlite3.connect(db_path)
    thread_id = args.thread_id
    if args.latest or not thread_id:
        thread_id = get_latest_thread_id(con)
    if not thread_id:
        print("No thread_id found in checkpoints DB.")
        return 1

    cur = con.cursor()
    cur.execute(
        "select checkpoint from checkpoints where thread_id=? order by rowid desc limit 1",
        (thread_id,),
    )
    row = cur.fetchone()
    if not row:
        print(f"No checkpoint found for thread_id: {thread_id}")
        return 1

    checkpoint = unpack_msgpack(row[0])
    plan = checkpoint.get("channel_values", {}).get("document_plan")
    if not plan:
        print(f"No document_plan found for thread_id: {thread_id}")
        return 1

    with open(args.config, "r", encoding="utf-8") as f:
        implicit = json.load(f).get("implicit_processes", [])

    implicit_by_module = defaultdict(list)
    for proc in implicit:
        implicit_by_module[proc.get("module")].append(proc)

    # Normalize legacy CE to CM
    aliases = {"CE": "CM"}

    implicit_added = defaultdict(list)
    implicit_present_extracted = defaultdict(list)
    implicit_missing = defaultdict(list)
    extracted_only = defaultdict(list)
    potential_duplicates = defaultdict(list)

    # Build implicit name sets by module for extracted-only detection
    implicit_name_set = defaultdict(set)
    for mod, items in implicit_by_module.items():
        for proc in items:
            implicit_name_set[mod].add(normalize_name(proc.get("process_name") or ""))

    for section in plan.get("sections", []):
        module = section.get("section_id")
        imp_module = aliases.get(module, module)
        imp_list = implicit_by_module.get(imp_module, [])

        plan_procs = section.get("processes", [])
        plan_by_norm = defaultdict(list)
        for proc in plan_procs:
            plan_by_norm[normalize_name(proc.get("process_name") or "")].append(proc)

        # Classify implicit presence
        for imp in imp_list:
            imp_name = imp.get("process_name") or ""
            imp_norm = normalize_name(imp_name)
            matches = plan_by_norm.get(imp_norm, [])
            if not matches:
                implicit_missing[module].append(imp_name)
                continue

            is_added = False
            for match in matches:
                if (
                    match.get("process_description") == imp.get("description")
                    and match.get("output") == imp.get("process_name")
                    and match.get("confidence") == imp.get("default_confidence", "high")
                    and match.get("missing_info") == []
                ):
                    is_added = True
                    break

            if is_added:
                implicit_added[module].append(imp_name)
            else:
                implicit_present_extracted[module].append(imp_name)

        # Extracted-only (not in implicit list by normalized name)
        imp_names = implicit_name_set.get(imp_module, set())
        for proc in plan_procs:
            name = proc.get("process_name") or ""
            if normalize_name(name) not in imp_names:
                extracted_only[module].append(name)

        # Potential duplicates: implicit names vs extracted-only names (name-similar)
        for imp in imp_list:
            imp_name = imp.get("process_name") or ""
            for ext_name in extracted_only.get(module, []):
                if normalize_name(imp_name) == normalize_name(ext_name):
                    continue
                if name_similarity(imp_name, ext_name) >= 0.65:
                    pair = f"{imp_name} ~ {ext_name}"
                    if pair not in potential_duplicates[module]:
                        potential_duplicates[module].append(pair)

    print(f"Thread ID: {thread_id}")
    print(f"Client: {plan.get('client_name')}")
    print("\nImplicit added (config-inserted):")
    for mod, items in implicit_added.items():
        print(f"  {mod}: {', '.join(items) if items else '-'}")

    print("\nImplicit present but extracted (name matched, content differs):")
    for mod, items in implicit_present_extracted.items():
        print(f"  {mod}: {', '.join(items) if items else '-'}")

    print("\nImplicit missing (not present in plan):")
    for mod, items in implicit_missing.items():
        print(f"  {mod}: {', '.join(items) if items else '-'}")

    print("\nExtracted-only (not in implicit list):")
    for mod, items in extracted_only.items():
        print(f"  {mod}: {', '.join(items) if items else '-'}")

    print("\nPotential duplicates (name-similar, review):")
    for mod, items in potential_duplicates.items():
        print(f"  {mod}: {', '.join(items) if items else '-'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
