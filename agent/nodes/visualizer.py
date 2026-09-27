"""
agent/nodes/visualizer.py - Data Analyst & Visualization Node
============================================================
Translates raw SQL execution results into formatted markdown tables and business insights.
"""

from typing import List, Any
from agent.state import AgentState


class VisualizerNode:
    """Formats execution tables and generates final answer summaries."""

    @staticmethod
    def _format_markdown_table(columns: List[str], rows: List[Any]) -> str:
        """Render SQL result rows into GitHub-flavored Markdown table."""
        if not columns or not rows:
            return "_[Empty Result Set]_"

        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        
        row_lines = []
        for r in rows:
            line = "| " + " | ".join(str(val) for val in r) + " |"
            row_lines.append(line)
            
        return "\n".join([header, separator] + row_lines)

    @staticmethod
    def run(state: AgentState) -> AgentState:
        state.status = "COMPLETED" if state.execution_success else "FAILED"
        
        if not state.execution_success:
            state.final_answer = f"Failed to execute SQL after {state.iteration_count} self-correction attempts.\n\nError: {state.execution_error}\n\nLast Attempted SQL:\n```sql\n{state.current_sql}\n```"
            return state

        table_md = VisualizerNode._format_markdown_table(state.execution_columns, state.execution_result)
        
        answer_parts = [
            f"### Query Results for: \"{state.question}\"\n",
            f"**Verified SQL Query:**\n```sql\n{state.current_sql}\n```\n",
            f"**Execution Data ({len(state.execution_result)} rows):**\n",
            table_md,
            f"\n\n*Executed successfully via Signal-SQL-Agent sandbox (Self-corrections: {state.iteration_count}, Portfolio SINR: {state.portfolio_sinr:.4f})*"
        ]
        
        state.final_answer = "\n".join(answer_parts)
        return state
