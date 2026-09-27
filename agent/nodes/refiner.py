"""
agent/nodes/refiner.py - Self-Correction & Refiner Node
======================================================
Analyzes database error messages and attempts to self-correct the erroneous SQL.
"""

import os
import re
from agent.state import AgentState
from llm.chatgpt import ask_llm


class RefinerNode:
    """Self-correction agent that diagnoses execution errors and repairs SQL."""

    @staticmethod
    def run(state: AgentState, model: str = "gpt-4") -> AgentState:
        state.status = "REFINING"
        state.iteration_count += 1
        
        failed_sql = state.current_sql
        error_msg = state.execution_error or "Unknown error"
        
        # 1. Try LLM-based intelligent self-correction
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            ddls = [state.schema_info["ddls"][t] for t in state.relevant_tables if t in state.schema_info.get("ddls", {})]
            schema_context = "\n\n".join(ddls)

            refine_prompt = f"""You are an expert SQL debugger. The following SQLite SQL failed to execute on the database.
Your job is to diagnose the error and output ONLY the fixed, valid SQL query.

Target User Question: {state.question}
Database Schema DDL:
{schema_context}

Failed SQL Query:
{failed_sql}

Execution Error Message:
{error_msg}

Rules:
1. Output ONLY the repaired SQL query.
2. Absolutely no markdown fences, no explanations, no comments.
3. Make sure table and column names match the schema DDL 100% accurately.
"""
            try:
                res = ask_llm(model, [refine_prompt], temperature=0.0, n=1)
                fixed_sql = res["response"][0] if isinstance(res["response"], list) else res["response"]
                fixed_sql = fixed_sql.strip()
                if "```sql" in fixed_sql:
                    fixed_sql = fixed_sql.split("```sql")[1].split("```")[0].strip()
                elif "```" in fixed_sql:
                    fixed_sql = fixed_sql.split("```")[1].split("```")[0].strip()
                
                # Record in correction history
                state.correction_history.append({
                    "iteration": state.iteration_count,
                    "failed_sql": failed_sql,
                    "error": error_msg,
                    "repaired_sql": fixed_sql,
                    "method": "llm"
                })
                state.current_sql = fixed_sql
                return state
            except Exception as e:
                print(f"[RefinerNode] LLM Refinement call failed ({e}), using rule-based repair.")

        # 2. Rule-based / Heuristic self-correction fallback
        fixed_sql = RefinerNode._rule_based_repair(failed_sql, error_msg, state)
        state.correction_history.append({
            "iteration": state.iteration_count,
            "failed_sql": failed_sql,
            "error": error_msg,
            "repaired_sql": fixed_sql,
            "method": "rule_based"
        })
        state.current_sql = fixed_sql
        return state

    @staticmethod
    def _rule_based_repair(failed_sql: str, error_msg: str, state: AgentState) -> str:
        """Repairs common syntax and column hallucination errors."""
        repaired = failed_sql
        
        # Error Case A: "no such column: <col>"
        col_match = re.search(r'no such column:\s*([^\s,;]+)', error_msg, re.IGNORECASE)
        if col_match:
            bad_col = col_match.group(1).split('.')[-1].strip()
            # Search available schema columns for closest match
            all_cols = []
            for ddl in state.schema_info.get("ddls", {}).values():
                cols = re.findall(r'(\b[a-zA-Z_0-9]+\b)\s+(?:TEXT|INT|INTEGER|REAL|BLOB|VARCHAR)', ddl, re.IGNORECASE)
                all_cols.extend(cols)
            
            # 1. Exact or substring match
            matched_col = None
            for candidate in all_cols:
                if candidate.lower() == bad_col.lower() or candidate.lower() in bad_col.lower() or bad_col.lower() in candidate.lower():
                    matched_col = candidate
                    break
            
            # 2. Fuzzy match via difflib (handles inflections like Capacities -> Capacity)
            if not matched_col:
                import difflib
                close_matches = difflib.get_close_matches(bad_col, all_cols, n=1, cutoff=0.5)
                if close_matches:
                    matched_col = close_matches[0]
            
            if matched_col:
                repaired = re.sub(rf'\b{re.escape(bad_col)}\b', matched_col, repaired, flags=re.IGNORECASE)

        # Error Case B: "no such table: <table>"
        table_match = re.search(r'no such table:\s*([^\s,;]+)', error_msg, re.IGNORECASE)
        if table_match:
            bad_table = table_match.group(1).strip()
            all_tables = list(state.schema_info.get("ddls", {}).keys())
            matched_table = None
            for candidate in all_tables:
                if candidate.lower() == bad_table.lower() or candidate.lower() == bad_table.lower().rstrip('s'):
                    matched_table = candidate
                    break
            if not matched_table:
                import difflib
                close_tables = difflib.get_close_matches(bad_table, all_tables, n=1, cutoff=0.5)
                if close_tables:
                    matched_table = close_tables[0]
            if matched_table:
                repaired = re.sub(rf'\b{re.escape(bad_table)}\b', matched_table, repaired, flags=re.IGNORECASE)

        return repaired
