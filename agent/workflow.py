"""
agent/workflow.py - Multi-Agent State Machine Workflow
======================================================
Coordinates planning, Signal-SQL exemplar selection, generation, sandboxed execution,
and self-correction loops.
"""

from typing import Dict, Any, Optional
from agent.state import AgentState
from agent.nodes import (
    PlannerNode,
    SignalMemoryNode,
    GeneratorNode,
    ExecutorNode,
    RefinerNode,
    VisualizerNode
)


class SignalSQLAgent:
    """Production-grade Text-to-SQL Agent with Signal-SQL memory and self-correction."""

    def __init__(
        self,
        db_path: str,
        model: str = "gpt-4",
        max_iterations: int = 3,
        verbose: bool = True
    ):
        self.db_path = db_path
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose

    def _log(self, step: str, message: str):
        if self.verbose:
            print(f"\n[Agent: {step}] {message}")

    def run(self, question: str, candidate_pool: list = None, initial_sql_override: str = None) -> AgentState:
        """
        Execute the complete multi-agent workflow for a given question.
        """
        self._log("Start", f"Received natural language query: \"{question}\"")
        state = AgentState(
            question=question,
            db_path=self.db_path,
            max_iterations=self.max_iterations
        )

        # 1. Planning & Schema Inspection
        self._log("Planner", "Inspecting database schema and discovering relevant tables...")
        state = PlannerNode.run(state)
        self._log("Planner", f"Identified relevant tables: {state.relevant_tables}")

        # 2. Signal-SQL Exemplar Retrieval
        self._log("Signal-Memory", "Invoking MCTS & SINR to select optimal complementary few-shot examples...")
        state = SignalMemoryNode.run(state, candidate_pool=candidate_pool, k=2)
        self._log("Signal-Memory", f"Retrieved {len(state.retrieved_examples)} shots (Portfolio SINR: {state.portfolio_sinr:.4f})")

        # 3. SQL Generation
        self._log("Generator", "Assembling few-shot prompt and generating initial candidate SQL...")
        if initial_sql_override:
            state.current_sql = initial_sql_override
            self._log("Generator", f"[Fault Injection] Injected unverified SQL: {state.current_sql}")
        else:
            state = GeneratorNode.run(state, model=self.model)
            self._log("Generator", f"Generated SQL: {state.current_sql}")

        # 4. Sandboxed Execution & Self-Correction Loop
        while True:
            self._log("Executor", f"Executing SQL in database sandbox (Attempt {state.iteration_count + 1})...")
            state = ExecutorNode.run(state)

            if state.execution_success:
                self._log("Executor", f"Execution successful! Fetched {len(state.execution_result)} rows.")
                break

            self._log("Executor", f"[!] Execution Error: {state.execution_error}")

            if state.iteration_count >= state.max_iterations:
                self._log("Refiner", f"Max correction attempts ({state.max_iterations}) reached. Exiting loop.")
                break

            # Self-Correction
            self._log("Refiner", "Analyzing error trace and diagnosing root cause...")
            state = RefinerNode.run(state, model=self.model)
            self._log("Refiner", f"Repaired SQL: {state.current_sql}")

        # 5. Data Insight & Markdown Visualization
        self._log("Visualizer", "Formatting markdown table and synthesizing final response...")
        state = VisualizerNode.run(state)
        self._log("Complete", f"Workflow finished with status: {state.status}")

        return state
