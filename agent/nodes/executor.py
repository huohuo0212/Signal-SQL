"""
agent/nodes/executor.py - Sandbox Execution Node
================================================
Safely runs generated SQL against the DatabaseSandbox and captures results / error traces.
"""

from agent.state import AgentState
from agent.tools import DatabaseSandbox


class ExecutorNode:
    """Executes SQL in sandbox and updates state with results or error messages."""

    @staticmethod
    def run(state: AgentState) -> AgentState:
        state.status = "EXECUTING"
        sandbox = DatabaseSandbox(state.db_path)
        
        exec_res = sandbox.execute(state.current_sql)
        
        state.execution_success = exec_res["success"]
        state.execution_columns = exec_res["columns"]
        state.execution_result = exec_res["rows"]
        state.execution_error = exec_res["error"]
        
        return state
