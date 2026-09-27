"""
run_agent_demo.py - Interactive Multi-Agent Self-Correction Demo
===============================================================
Demonstrates the full Signal-SQL-Agent loop:
1. Dynamic Schema Inspection
2. Signal-SQL Few-Shot Exemplar Selection (MCTS + SINR)
3. SQL Generation
4. Sandbox Execution & Error Interception
5. Intelligent Self-Correction Loop (Refiner Node)
6. Markdown Table & Data Insight Formatting
"""

import os
import sys
import sqlite3

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.workflow import SignalSQLAgent
from agent.tools import DatabaseSandbox


def setup_demo_database(db_path: str = "demo_stadium.sqlite"):
    """Create a demo database with tables and sample records."""
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE stadium (
        Stadium_ID INTEGER PRIMARY KEY,
        Name TEXT NOT NULL,
        Capacity INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE concert (
        concert_ID INTEGER PRIMARY KEY,
        Stadium_ID INTEGER,
        Year TEXT NOT NULL,
        FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)
    );
    """)

    # Insert sample rows
    stadiums = [
        (1, 'Wembley Stadium', 90000),
        (2, 'Camp Nou', 99354),
        (3, 'San Siro', 80018),
        (4, 'Stade de France', 81338)
    ]
    cur.executemany("INSERT INTO stadium VALUES (?, ?, ?);", stadiums)

    concerts = [
        (101, 1, '2014'),
        (102, 1, '2014'),
        (103, 1, '2015'),
        (104, 2, '2014'),
        (105, 3, '2013'),
        (106, 4, '2014'),
        (107, 4, '2014'),
        (108, 4, '2014')
    ]
    cur.executemany("INSERT INTO concert VALUES (?, ?, ?);", concerts)

    conn.commit()
    conn.close()
    return db_path


def main():
    print("=" * 75)
    print("             Signal-SQL-Agent: Multi-Agent Text-to-SQL Demo")
    print("=" * 75)

    # 1. Setup sample SQLite database
    db_file = setup_demo_database("demo_stadium.sqlite")
    print(f"\n[Environment] Initialized local SQLite sandbox at: {db_file}")

    # 2. Instantiate Agent
    agent = SignalSQLAgent(
        db_path=db_file,
        model="gpt-4",
        max_iterations=3,
        verbose=True
    )

    # 3. Scenario A: Standard High-Complexity Query
    print("\n" + "=" * 35 + " [Scenario 1: Complex Multi-Table Query] " + "=" * 35)
    user_query = "Show the name and capacity of each stadium hosting concerts in 2014, sorted by concert count descending."
    state = agent.run(user_query)

    print("\n" + "-" * 40 + " Agent Response (Scenario 1) " + "-" * 40)
    print(state.final_answer)

    # 4. Scenario B: Automated Self-Correction & Healing Demonstration
    print("\n\n" + "=" * 35 + " [Scenario 2: Automatic Error Self-Correction] " + "=" * 35)
    print("Testing Agent robustness when encountering schema hallucinations or column errors...")
    
    user_query_2 = "List the name and capacity of stadiums where capacity is over 85000."
    
    # We deliberately inject a hallucinated SQL with wrong column names ('Names', 'Capacities')
    faulty_sql = "SELECT Names, Capacities FROM stadium WHERE Capacities > 85000"
    state_2 = agent.run(user_query_2, initial_sql_override=faulty_sql)
    
    print("\n" + "-" * 40 + " Agent Response (Scenario 2) " + "-" * 40)
    print(state_2.final_answer)
    print("=" * 75)

    # Cleanup demo database
    try:
        if os.path.exists(db_file):
            os.remove(db_file)
    except Exception:
        pass


if __name__ == "__main__":
    main()
