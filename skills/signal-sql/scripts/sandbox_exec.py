"""
scripts/sandbox_exec.py - Sandboxed Database Execution CLI
==========================================================
Usage:
  python sandbox_exec.py --db path/to/database.sqlite --sql "SELECT count(*) FROM table;"
"""

import sys
import os
import argparse
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, root_dir)

from agent.tools import DatabaseSandbox


def main():
    parser = argparse.ArgumentParser(description="Database Sandbox Execution Tool")
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite database file")
    parser.add_argument("--sql", type=str, required=True, help="SQL statement to execute")
    parser.add_argument("--max_rows", type=int, default=10, help="Max rows to display")
    args = parser.parse_args()

    sandbox = DatabaseSandbox(args.db, max_rows=args.max_rows)
    result = sandbox.execute(args.sql)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
