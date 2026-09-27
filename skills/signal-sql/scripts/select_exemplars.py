"""
scripts/select_exemplars.py - Standalone Few-Shot Exemplar Selector CLI
======================================================================
Usage:
  python select_exemplars.py --target "Show stadium name and capacity" --k 2
"""

import sys
import os
import argparse
import json

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, root_dir)

from agent.nodes.signal_memory import DEFAULT_EXEMPLAR_BANK, SignalMemoryNode
from agent.state import AgentState


def main():
    parser = argparse.ArgumentParser(description="Signal-SQL Exemplar Selection CLI")
    parser.add_argument("--target", type=str, required=True, help="Target natural language question")
    parser.add_argument("--k", type=int, default=2, help="Number of few-shot examples to retrieve")
    parser.add_argument("--candidates", type=str, default=None, help="Optional JSON file containing candidate pool")
    args = parser.parse_args()

    candidate_pool = None
    if args.candidates and os.path.exists(args.candidates):
        with open(args.candidates, "r", encoding="utf-8") as f:
            candidate_pool = json.load(f)

    state = AgentState(question=args.target, db_path="")
    state = SignalMemoryNode.run(state, candidate_pool=candidate_pool, k=args.k)

    print(json.dumps({
        "target": args.target,
        "k": args.k,
        "portfolio_sinr": state.portfolio_sinr,
        "selected_exemplars": state.retrieved_examples
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
