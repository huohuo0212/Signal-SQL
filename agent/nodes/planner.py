"""
agent/nodes/planner.py - Schema Inspection & Query Intent Planner Node
=====================================================================
Analyzes user questions, dynamically inspects schema, and identifies relevant tables.
"""

from agent.state import AgentState
from agent.tools import SchemaExplorer


class PlannerNode:
    """Explores database schema, extracts foreign keys, and prunes relevant tables."""

    @staticmethod
    def run(state: AgentState) -> AgentState:
        state.status = "PLANNING"
        
        # 1. Inspect all tables and DDLs
        ddls = SchemaExplorer.get_table_schema_ddl(state.db_path)
        foreign_keys = SchemaExplorer.get_foreign_keys(state.db_path)
        
        # 2. Simple table pruning based on token matching with user question
        q_lower = state.question.lower()
        relevant_tables = []
        for table in ddls.keys():
            t_lower = table.lower()
            # Match table name or singular form
            if t_lower in q_lower or t_lower.rstrip('s') in q_lower:
                relevant_tables.append(table)
        
        # If no explicit match, retain all tables (safe default for small-medium DBs)
        if not relevant_tables:
            relevant_tables = list(ddls.keys())
            
        # 3. Store in state
        state.schema_info = {
            "ddls": ddls,
            "foreign_keys": foreign_keys,
            "relevant_tables": relevant_tables
        }
        state.relevant_tables = relevant_tables
        
        return state
