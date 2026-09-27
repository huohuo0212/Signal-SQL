"""
run_demo.py - Quick Start Demo for Signal-SQL
==================================================
Demonstrates the full end-to-end pipeline:
1. Feature extraction on candidate SQL queries
2. Signal-to-Interference-plus-Noise Ratio (SINR) reward evaluation
3. Monte Carlo Tree Search (MCTS) combinatorial example selection
4. Dynamic Prompt assembly via prompt_factory
"""

import os
import sys
import numpy as np

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sinr_reward import extract_sql_features, calculate_sinr
from utils.mcts_optimizer import MCTS
from prompt.prompt_builder import prompt_factory
from utils.enums import REPR_TYPE, EXAMPLE_TYPE, SELECTOR_TYPE


def run_demo():
    print("=" * 70)
    print("        Signal-SQL: Few-Shot Example Selection Demo")
    print("=" * 70)

    # 1. Target Query to generate SQL for
    target = {
        "question": "Show the name and capacity of each stadium hosting concerts in 2014, sorted by concert count descending.",
        "pre_skeleton": "SELECT _ , _ FROM _ JOIN _ ON _ = _ WHERE _ = _ GROUP BY _ ORDER BY count ( * ) DESC",
        "query": "SELECT T1.Name, T1.Capacity FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID = T2.Stadium_ID WHERE T2.Year = '2014' GROUP BY T1.Stadium_ID ORDER BY count(*) DESC",
        "db_id": "concert_singer"
    }

    # 2. Candidate Pool in training set
    candidates = [
        {
            "question": "What is the total number of singers?",
            "query": "SELECT count(*) FROM singer",
            "db_id": "concert_singer_2"
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
            "question": "Find the average age of all pets grouped by type having max weight > 10.",
            "query": "SELECT PetType, avg(pet_age) FROM Pets GROUP BY PetType HAVING max(weight) > 10",
            "db_id": "pets_db"
        },
        {
            "question": "List students who have pets and are older than 20 union all singers older than 30.",
            "query": "SELECT StuID FROM Student WHERE Age > 20 UNION ALL SELECT Singer_ID FROM singer WHERE Age > 30",
            "db_id": "school_db"
        }
    ]

    print(f"\n[1] Target Question: {target['question']}")
    print(f"    Target SQL:      {target['query']}")
    target_feats = extract_sql_features(target['query'])
    print(f"    Extracted SQL Features: {sorted(list(target_feats))}")

    print(f"\n[2] Candidate Pool Size: {len(candidates)} examples")
    for idx, c in enumerate(candidates):
        feats = extract_sql_features(c['query'])
        print(f"    [{idx}] Q: {c['question'][:55]}... | Features: {sorted(list(feats))}")

    # 3. Simulate or Compute Embeddings
    print("\n[3] Evaluating Candidate Set with SINR Reward & MCTS...")
    k_shot = 2
    candidate_sqls = [c["query"] for c in candidates]
    
    # Generate mock embeddings for quick demo (reproducible seed)
    rng = np.random.RandomState(42)
    target_emb = rng.randn(384)
    target_emb /= np.linalg.norm(target_emb)
    cand_embs = rng.randn(len(candidates), 384)
    cand_embs /= np.linalg.norm(cand_embs, axis=1, keepdims=True)

    # 4. Run MCTS Combinatorial Optimization
    best_indices = MCTS.search(
        target_sql=target["query"],
        target_embedding=target_emb,
        candidate_pool=candidates,
        candidate_sqls=candidate_sqls,
        candidate_embeddings=cand_embs,
        k=k_shot,
        n_simulations=100,
        c_uct=1.41,
        lambda1=0.5,
        lambda2=0.3
    )

    selected_sqls = [candidate_sqls[i] for i in best_indices]
    final_sinr = calculate_sinr(target["query"], selected_sqls, cand_embs[best_indices])

    print(f"\n[4] MCTS Search Result:")
    print(f"    Selected Optimal Indices: {best_indices}")
    print(f"    Final Portfolio SINR:     {final_sinr:.4f}")
    print("    Selected Complementary Examples:")
    for rank, idx in enumerate(best_indices, 1):
        print(f"      Shot {rank}: [Idx {idx}] \"{candidates[idx]['question']}\"")
        print(f"              SQL: {candidates[idx]['query']}")

    # 5. Format into few-shot Prompt
    print("\n[5] Generating Assembled ICL Prompt:")
    print("-" * 70)
    prompt_str = "/* Given the following database schema: */\n"
    prompt_str += "CREATE TABLE stadium (\n  Stadium_ID INT PRIMARY KEY,\n  Name TEXT,\n  Capacity INT\n);\n"
    prompt_str += "CREATE TABLE concert (\n  concert_ID INT PRIMARY KEY,\n  Stadium_ID INT,\n  Year TEXT,\n  FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)\n);\n\n"
    prompt_str += "/* Some SQL examples are provided based on similar problems: */\n\n"
    for rank, idx in enumerate(best_indices, 1):
        prompt_str += f"/* Answer the following: {candidates[idx]['question']} */\n{candidates[idx]['query']}\n\n"
    prompt_str += f"/* Answer the following: {target['question']} */\nSELECT "

    print(prompt_str)
    print("-" * 70)
    print("\n[OK] Demo executed successfully! Signal-SQL pipeline is ready.")


if __name__ == "__main__":
    run_demo()
