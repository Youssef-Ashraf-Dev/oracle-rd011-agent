"""
List available LangGraph thread IDs from the SQLite checkpointer DB.

Usage:
  python tools/list_threads.py
  python tools/list_threads.py --db checkpoints/rd011_checkpoints.db --limit 20
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="checkpoints/rd011_checkpoints.db")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    cur.execute(
        """
        select thread_id, count(*) as checkpoints, max(rowid) as last_row
        from checkpoints
        group by thread_id
        order by last_row desc
        limit ?
        """,
        (args.limit,),
    )

    rows = cur.fetchall()
    if not rows:
        print("No threads found.")
        return 0

    print(f"DB: {db_path}")
    print("Most recent threads:")
    for thread_id, cnt, _ in rows:
        print(f"- {thread_id}  (checkpoints={cnt})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

