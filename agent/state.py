"""
agent/state.py - Context State Definition for Signal-SQL-Agent
==============================================================
Defines the typed data structures passed between agent nodes.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Represents the global state of the Text-to-SQL Agent workflow."""
    # User Request
    question: str
    db_path: str
    
    # Exploration & Schema Discovery
    schema_info: Dict[str, Any] = field(default_factory=dict)
    relevant_tables: List[str] = field(default_factory=list)
    atomic_sub_queries: List[str] = field(default_factory=list)
    
    # Signal-SQL Memory & Few-Shot Retrieval
    retrieved_examples: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_sinr: float = 0.0
    
    # SQL Generation & Sandbox Execution
    current_sql: str = ""
    candidate_sqls: List[str] = field(default_factory=list)
    execution_success: bool = False
    execution_result: List[Any] = field(default_factory=list)
    execution_columns: List[str] = field(default_factory=list)
    execution_error: Optional[str] = None
    
    # Self-Correction Loop Control
    iteration_count: int = 0
    max_iterations: int = 3
    correction_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Final Presentation
    final_answer: str = ""
    status: str = "INITIALIZED"  # INITIALIZED, PLANNING, RETRIEVING, GENERATING, EXECUTING, REFINING, COMPLETED, FAILED
