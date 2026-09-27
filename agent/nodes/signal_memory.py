"""
agent/nodes/signal_memory.py - Signal-SQL Exemplar Selection Node
================================================================
Invokes Signal-SQL (MCTS + SINR) to retrieve mutually complementary few-shot examples.
"""

from typing import List, Dict, Any
import numpy as np
from agent.state import AgentState
from utils.sinr_reward import extract_sql_features, calculate_sinr
from utils.mcts_optimizer import MCTS


# Default baseline exemplars for out-of-the-box agent queries
DEFAULT_EXEMPLAR_BANK = [
    {
        "question": "What is the total count of singers?",
        "query": "SELECT count(*) FROM singer",
        "db_id": "concert_singer"
    },
    {
        "question": "Find the names and capacities of stadiums with capacity over 5000.",
        "query": "SELECT Name, Capacity FROM stadium WHERE Capacity > 5000",
        "db_id": "stadium_db"
    },
    {
        "question": "List the names of all stadiums joined with their concert records where year is 2014.",
        "query": "SELECT T1.Name, T2.Year FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID = T2.Stadium_ID WHERE T2.Year = '2014'",
        "db_id": "stadium_db"
    },
    {
        "question": "Count concerts per stadium grouped by stadium ID and ordered by count descending.",
        "query": "SELECT Stadium_ID, count(*) FROM concert GROUP BY Stadium_ID ORDER BY count(*) DESC",
        "db_id": "stadium_db"
    },
    {
        "question": "Find the average capacity of all stadiums grouped by location having max capacity > 10000.",
        "query": "SELECT Location, avg(Capacity) FROM stadium GROUP BY Location HAVING max(Capacity) > 10000",
        "db_id": "stadium_db"
    }
]


class SignalMemoryNode:
    """Selects the best portfolio of few-shot examples using MCTS & SINR."""

    @staticmethod
    def run(state: AgentState, candidate_pool: List[Dict[str, Any]] = None, k: int = 2) -> AgentState:
        state.status = "RETRIEVING"
        pool = candidate_pool if candidate_pool else DEFAULT_EXEMPLAR_BANK
        
        # 1. Estimate target rough SQL or extract question keyword indicators
        target_features = set()
        q_lower = state.question.lower()
        if "count" in q_lower or "how many" in q_lower or "total number" in q_lower:
            target_features.add("count")
        if "join" in q_lower or "and their" in q_lower or len(state.relevant_tables) > 1:
            target_features.add("join")
        if "group by" in q_lower or "per" in q_lower or "each" in q_lower:
            target_features.add("group by")
        if "order by" in q_lower or "sorted by" in q_lower or "highest" in q_lower or "desc" in q_lower:
            target_features.add("order by")
            target_features.add("desc")
        if "average" in q_lower or "avg" in q_lower:
            target_features.add("avg")
        if "maximum" in q_lower or "max" in q_lower:
            target_features.add("max")
        target_features.update(["select", "from"])

        synthetic_target_sql = " ".join(target_features)

        # 2. Extract features & embeddings from candidates
        candidate_sqls = [c["query"] for c in pool]
        
        # Mock / Normalized embeddings for demo
        rng = np.random.RandomState(42)
        target_emb = rng.randn(384)
        target_emb /= np.linalg.norm(target_emb)
        cand_embs = rng.randn(len(pool), 384)
        cand_embs /= np.linalg.norm(cand_embs, axis=1, keepdims=True)

        # 3. MCTS Search for optimal combination
        best_indices = MCTS.search(
            target_sql=synthetic_target_sql,
            target_embedding=target_emb,
            candidate_pool=pool,
            candidate_sqls=candidate_sqls,
            candidate_embeddings=cand_embs,
            k=min(k, len(pool)),
            n_simulations=80,
            c_uct=1.41
        )

        selected_examples = [pool[i] for i in best_indices]
        selected_sqls = [candidate_sqls[i] for i in best_indices]
        sinr = calculate_sinr(synthetic_target_sql, selected_sqls, cand_embs[best_indices])

        state.retrieved_examples = selected_examples
        state.portfolio_sinr = sinr
        return state
