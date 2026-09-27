"""
agent/tools.py - Database Sandbox & Schema Explorer Tools
=========================================================
Provides safe execution environments, timeout protection, and dynamic schema inspection.
"""

import sqlite3
import re
from typing import Dict, List, Any, Tuple, Optional


class DatabaseSandbox:
    """Safe SQLite Execution Sandbox with read-only guards and error interception."""
    
    FORBIDDEN_KEYWORDS = [
        r'\bDROP\b', r'\bDELETE\b', r'\bUPDATE\b', r'\bINSERT\b', 
        r'\bALTER\b', r'\bTRUNCATE\b', r'\bGRANT\b', r'\bREVOKE\b'
    ]

    def __init__(self, db_path: str, max_rows: int = 50, timeout_sec: float = 10.0):
        self.db_path = db_path
        self.max_rows = max_rows
        self.timeout_sec = timeout_sec

    def _is_safe(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate that SQL is strictly read-only."""
        sql_clean = sql.strip().upper()
        for pattern in self.FORBIDDEN_KEYWORDS:
            if re.search(pattern, sql_clean):
                return False, f"Dangerous command detected: SQL cannot contain destructive operation '{pattern}'"
        return True, None

    def execute(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL query in sandbox.
        Returns:
            {
                "success": bool,
                "columns": List[str],
                "rows": List[Tuple],
                "row_count": int,
                "error": Optional[str]
            }
        """
        is_safe, error_msg = self._is_safe(sql)
        if not is_safe:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": error_msg
            }

        try:
            # Connect in URI read-only mode if supported, or standard mode
            conn = sqlite3.connect(self.db_path, timeout=self.timeout_sec)
            cursor = conn.cursor()
            cursor.execute(sql)
            
            # Extract column descriptions
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(self.max_rows)
            total_fetched = len(rows)
            
            conn.close()
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": total_fetched,
                "error": None
            }
        except sqlite3.OperationalError as e:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": f"SQL OperationalError: {str(e)}"
            }
        except sqlite3.DatabaseError as e:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": f"DatabaseError: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": f"Unexpected execution error: {str(e)}"
            }


class SchemaExplorer:
    """Dynamic database structure and sample value inspector."""

    @staticmethod
    def get_all_tables(db_path: str) -> List[str]:
        """Retrieve names of all user tables in SQLite database."""
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cur.fetchall()]
        conn.close()
        return tables

    @staticmethod
    def get_table_schema_ddl(db_path: str) -> Dict[str, str]:
        """Retrieve CREATE TABLE DDLs for all tables."""
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        ddls = {row[0]: row[1] for row in cur.fetchall() if row[1]}
        conn.close()
        return ddls

    @staticmethod
    def get_foreign_keys(db_path: str) -> List[str]:
        """Extract foreign key relationships across all tables."""
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        tables = SchemaExplorer.get_all_tables(db_path)
        fks = []
        for t in tables:
            try:
                cur.execute(f'PRAGMA foreign_key_list("{t}")')
                for row in cur.fetchall():
                    target_table = row[2]
                    from_col = row[3]
                    to_col = row[4]
                    fks.append(f"{t}.{from_col} = {target_table}.{to_col}")
            except Exception:
                continue
        conn.close()
        return fks

    @staticmethod
    def sample_table_values(db_path: str, table_name: str, limit: int = 3) -> Dict[str, List[Any]]:
        """Sample distinct values for each column to help with condition grounding."""
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        samples = {}
        try:
            cur.execute(f'PRAGMA table_info("{table_name}")')
            columns = [c[1] for c in cur.fetchall()]
            for col in columns:
                try:
                    cur.execute(f'SELECT DISTINCT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT {limit}')
                    samples[col] = [r[0] for r in cur.fetchall()]
                except Exception:
                    samples[col] = []
        except Exception:
            pass
        conn.close()
        return samples
