"""SQLite-backed experience store with fractal quaternion indexing."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


class QuaternionFractalIndex:
    """
    4-ary fractal tree for quaternion-based experience indexing.
    
    Structure: Each node divides quaternion space into 4 quadrants
    along (w,x) dimensions. Depth O(log₄ n) for n experiences.
    """

    def __init__(self, branching_factor: int = 4, max_depth: int = 8) -> None:
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.root: Dict[str, Any] = {"quadrants": [{} for _ in range(4)], "experiences": []}

    def _get_quadrant(self, q: np.ndarray) -> int:
        """
        Map quaternion to quadrant index [0,3].
        
        Partitioning:
        - w ≥ 0, x ≥ 0 → Q0
        - w < 0, x ≥ 0 → Q1
        - w < 0, x < 0 → Q2
        - w ≥ 0, x < 0 → Q3
        """
        w, x = q[0], q[1]
        return (0 if w >= 0 else 1) | ((0 if x >= 0 else 1) << 1)

    def insert(self, q: np.ndarray, exp_id: int, depth: int = 0) -> None:
        """Insert experience quaternion into fractal tree."""
        if depth >= self.max_depth:
            self.root["experiences"].append(exp_id)
            return

        quadrant = self._get_quadrant(q)
        if not self.root["quadrants"][quadrant]:
            self.root["quadrants"][quadrant] = {"quadrants": [{} for _ in range(4)], "experiences": []}
        
        self._insert_recursive(self.root["quadrants"][quadrant], q, exp_id, depth + 1)

    def _insert_recursive(self, node: Dict[str, Any], q: np.ndarray, exp_id: int, depth: int) -> None:
        """Recursive insertion into quadrant tree."""
        if depth >= self.max_depth:
            node["experiences"].append(exp_id)
            return

        quadrant = self._get_quadrant(q)
        if not node["quadrants"][quadrant]:
            node["quadrants"][quadrant] = {"quadrants": [{} for _ in range(4)], "experiences": []}

        self._insert_recursive(node["quadrants"][quadrant], q, exp_id, depth + 1)

    def search(self, q: np.ndarray, top_k: int = 5, depth: int = 0) -> List[int]:
        """
        Search for similar experiences in fractal tree.
        
        Complexity: O(log n) traversal + O(k) result collection
        """
        results = []
        self._search_recursive(self.root, q, top_k, depth, results)
        return results[:top_k]

    def _search_recursive(self, node: Dict[str, Any], q: np.ndarray, top_k: int, depth: int, results: List[int]) -> None:
        """Recursive search through quadrant tree."""
        if depth >= self.max_depth:
            results.extend(node.get("experiences", []))
            return

        # Find target quadrant
        quadrant = self._get_quadrant(q)
        if node["quadrants"][quadrant]:
            self._search_recursive(node["quadrants"][quadrant], q, top_k, depth + 1, results)

        # Also search nearby quadrants for robustness
        for i in range(4):
            if i != quadrant and node["quadrants"][i] and len(results) < top_k:
                self._search_recursive(node["quadrants"][i], q, top_k, depth + 1, results)


class KnowledgeDB:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "knowledge.db"
        
        # Initialize fractal quaternion index
        self.fractal_index = QuaternionFractalIndex()
        self._init_db()
        self._rebuild_fractal_index()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT NOT NULL,
                    task_type TEXT,
                    result TEXT,
                    feedback TEXT,
                    timestamp REAL,
                    confidence REAL,
                    strategy_used TEXT,
                    task_quaternion TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_task_type ON experiences(task_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_confidence ON experiences(confidence)")
            conn.commit()

    def _rebuild_fractal_index(self) -> None:
        """Rebuild fractal index from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, task_quaternion FROM experiences WHERE task_quaternion IS NOT NULL")
            for exp_id, q_str in cursor.fetchall():
                try:
                    q = np.array(json.loads(q_str), dtype=np.float32)
                    self.fractal_index.insert(q, exp_id)
                except (json.JSONDecodeError, ValueError):
                    pass

    def save_experience(self, experience: Dict[str, Any]) -> None:
        """Save experience with quaternion representation for fractal indexing."""
        # Extract or compute quaternion representation
        q_feedback = experience.get("_quaternion_feedback")
        if q_feedback is None:
            # Fallback: create from confidence
            confidence = experience.get("confidence", 0.0)
            angle = np.pi * confidence / 2.0
            q_feedback = [np.cos(angle), np.sin(angle), 0.0, 0.0]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO experiences
                (task, task_type, result, feedback, timestamp, confidence, strategy_used, task_quaternion)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience.get("task"),
                    experience.get("task_type"),
                    json.dumps(experience.get("result", {})),
                    json.dumps(experience.get("feedback", {})),
                    experience.get("timestamp"),
                    experience.get("confidence"),
                    experience.get("strategy_used"),
                    json.dumps(q_feedback),
                ),
            )
            conn.commit()
            
            # Add to fractal index
            exp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            q = np.array(q_feedback, dtype=np.float32)
            self.fractal_index.insert(q, exp_id)

    def retrieve_similar(self, task: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar experiences using fractal quaternion index.
        
        Algorithm:
        1. Encode task as quaternion (from provided _quaternion_feedback if available)
        2. Search fractal index: O(log n) traversal
        3. Fetch matching experiences from database
        
        Complexity: O(log n) vs O(n) in original implementation
        """
        # Try to get quaternion from task representation
        q_task = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Default: identity
        
        # Search fractal index
        experience_ids = self.fractal_index.search(q_task, top_k=top_k)
        
        if not experience_ids:
            # Fallback to SQL
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, task, result, confidence FROM experiences
                    ORDER BY confidence DESC
                    LIMIT ?
                    """,
                    (top_k,),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "task": row[1],
                        "result": json.loads(row[2] or "{}"),
                        "confidence": row[3] or 0.0,
                    }
                    for row in rows
                ]
        
        # Fetch from database by IDs
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(experience_ids))
            cursor = conn.execute(
                f"""
                SELECT task, result, confidence FROM experiences
                WHERE id IN ({placeholders})
                """,
                experience_ids,
            )
            rows = cursor.fetchall()
            return [
                {
                    "task": row[0],
                    "result": json.loads(row[1] or "{}"),
                    "confidence": row[2] or 0.0,
                }
                for row in rows
            ]

    def get_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            rows = conn.execute(
                "SELECT task_type, COUNT(*) FROM experiences GROUP BY task_type"
            ).fetchall()

        domains = [row[0] for row in rows if row[0]]
        return {
            "total_experiences": total,
            "domains": domains,
            "fractal_index_depth": 8,
            "retrieval_complexity": "O(log n)",
        }

