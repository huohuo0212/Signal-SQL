---
name: signal-sql
description: >-
  Expert Text-to-SQL generation, prompt optimization, and self-healing agent skill powered by Signal Detection Theory,
  CFAR clutter calibration, MCTS exemplar selection, and sandboxed self-correction. Use when the user asks
  to write complex SQL queries, optimize Text-to-SQL prompts, select few-shot demonstrations, inspect
  database schemas, or diagnose and repair database execution errors.
---

# Signal-SQL: Expert Text-to-SQL & Self-Correction Skill

A specialized AI Agent Skill designed to solve complex Text-to-SQL queries with mathematical rigor and closed-loop execution verification. It transforms raw database schemas and natural language intents into verified, highly optimized SQL queries.

---

## 🎯 Activation Triggers (When to Use)

Activate this skill whenever:
1. **Generating Complex SQL**: The user requests SQL involving multi-table `JOIN`, nested subqueries, window functions, date math, or complex aggregation (`HAVING`, `GROUP BY`).
2. **Selecting Few-Shot Demonstrations**: Choosing the most complementary, noise-free $k$-shot examples from a candidate bank.
3. **Debugging Broken SQL**: Diagnosing and repairing database execution exceptions such as `OperationalError: no such column`, syntax mismatches, or table alias issues.
4. **Inspecting Databases & Grounding**: Exploring unknown SQLite databases, foreign keys, and distinct column values.

---

## 🏗️ Core Workflow

When this skill is active, execute the following 5-stage procedure:

### Stage 1: Schema Grounding & Intent Planning
- Inspect the schema DDL using [scripts/sandbox_exec.py](./scripts/sandbox_exec.py) or SQLite PRAGMA tools.
- Identify relevant tables, primary/foreign key connections, and column data types.
- Check column sample values if literal constraints in the question are ambiguous.

### Stage 2: MCTS & SINR Few-Shot Selection
- If Few-Shot In-Context Learning is required, run [scripts/select_exemplars.py](./scripts/select_exemplars.py) to select $k$ orthogonal examples.
- **Selection Principle**: Maximize Signal Coverage ($E_{\text{match}}$) while minimizing redundant clauses ($E_{\text{noise}}$) and stylistic conflicts ($E_{\text{inter}}$).
- For mathematical formulations, consult [references/sinr_theory.md](./references/sinr_theory.md).

### Stage 3: High-Fidelity Prompt Assembly
- Format database schema using standard SQL DDL format (`/* Given the following database schema: */`).
- Inject selected few-shot QA pairs.
- Add target question and instruct the model to produce ONLY executable SQL.

### Stage 4: Sandboxed Execution & Verification
- Test the candidate SQL against the isolated database sandbox using:
  ```bash
  python scripts/sandbox_exec.py --db <db_path> --sql "<candidate_sql>"
  ```
- Verify:
  - Did the query execute without syntax errors?
  - Did it return the expected column names and non-empty rows?

### Stage 5: Closed-Loop Self-Correction (If Errors Occur)
- If the database returns an error (e.g. `no such column`, `ambiguous column name`), invoke [scripts/self_correct.py](./scripts/self_correct.py):
  ```bash
  python scripts/self_correct.py --db <db_path> --sql "<failed_sql>" --error "<error_msg>"
  ```
- The self-correction engine inspects actual schema column names, repairs hallucinations (e.g. `Capacities` -> `Capacity`), and re-tests in the sandbox up to 3 attempts.

---

## 🛠️ Helper Scripts

All helper scripts are standalone, lightweight, and executable from the terminal:

| Script | Purpose | Usage Example |
| :--- | :--- | :--- |
| **`scripts/select_exemplars.py`** | MCTS + SINR Few-Shot Selector | `python scripts/select_exemplars.py --target "..." --k 2` |
| **`scripts/sandbox_exec.py`** | Safe Sandboxed SQL Execution | `python scripts/sandbox_exec.py --db app.sqlite --sql "SELECT ..."` |
| **`scripts/self_correct.py`** | Error Diagnosis & Schema Repair | `python scripts/self_correct.py --db app.sqlite --sql "..." --error "..."` |

---

## 📚 References

- [SINR Theory & Mathematical Derivations](./references/sinr_theory.md): Detailed formulation of $E_{\text{match}}$, $E_{\text{noise}}$, $E_{\text{inter}}$, and Covariance Pre-whitening $W = C^{-1/2}$.
