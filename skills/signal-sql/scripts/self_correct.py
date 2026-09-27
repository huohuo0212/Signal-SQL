"""
scripts/self_correct.py - Automated SQL Error Diagnosis & Self-Correction CLI
=============================================================================
Usage:
  python self_correct.py --db app.sqlite --sql "SELECT Names FROM stadium" --error "no such column: Names"
"""

import sys
import os
import argparse
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, root_dir)

from agent.state import AgentState
from agent.tools import SchemaExplorer
from agent.nodes.refiner import RefinerNode


def main():
    parser = argparse.ArgumentParser(description="SQL Self-Correction & Diagnosis CLI")
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite database file")
    parser.add_argument("--sql", type=str, required=True, help="Failed SQL query")
    parser.add_argument("--error", type=str, required=True, help="Error message returned by database")
    parser.add_argument("--question", type=str, default="", help="Original user question (optional)")
    args = parser.parse_args()

    # Discover schema
    ddls = SchemaExplorer.get_table_schema_ddl(args.db)
    state = AgentState(question=args.question, db_path=args.db)
    state.current_sql = args.sql
    state.execution_error = args.error
    state.schema_info = {"ddls": ddls}
    state.relevant_tables = list(ddls.keys())

    # Run refiner
    state = RefinerNode.run(state)

    print(json.dumps({
        "original_sql": args.sql,
        "error_message": args.error,
        "repaired_sql": state.current_sql,
        "correction_history": state.correction_history
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
