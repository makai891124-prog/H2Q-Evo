"""SQLite-backed experience store (minimal placeholder)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class KnowledgeDB:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "knowledge.db"
        self._init_db()

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
                    strategy_used TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_task_type ON experiences(task_type)")
            conn.commit()

    def save_experience(self, experience: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO experiences
                (task, task_type, result, feedback, timestamp, confidence, strategy_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience.get("task"),
                    experience.get("task_type"),
                    json.dumps(experience.get("result", {})),
                    json.dumps(experience.get("feedback", {})),
                    experience.get("timestamp"),
                    experience.get("confidence"),
                    experience.get("strategy_used"),
                ),
            )
            conn.commit()

    def retrieve_similar(self, task: str, top_k: int = 5) -> List[Dict[str, Any]]:
        _ = task
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT task, result, confidence FROM experiences
                ORDER BY id DESC
                LIMIT ?
                """,
                (top_k,),
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
        return {"total_experiences": total, "domains": domains}
