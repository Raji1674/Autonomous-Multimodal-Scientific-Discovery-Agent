
"""Persistent long-term memory for completed scientific research runs."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "with", "using", "by", "is", "are", "was", "were", "be", "from",
    "this", "that", "which", "what", "how", "based", "study",
}


def _tokens(value: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", (value or "").lower())
    return {word for word in words if len(word) > 2 and word not in _STOPWORDS}


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class ResearchMemory:
    """SQLite-backed research memory with simple keyword retrieval."""

    def __init__(self, db_path: str = "data/memory/research_memory.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_memories (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT UNIQUE,
                    research_question TEXT NOT NULL,
                    hypothesis TEXT,
                    findings TEXT,
                    verdict TEXT,
                    dataset TEXT,
                    target_column TEXT,
                    algorithm TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_research_question
                ON research_memories(research_question)
                """
            )

    @staticmethod
    def _extract_hypothesis(state: dict) -> str:
        hypothesis = state.get("selected_hypothesis") or {}
        if isinstance(hypothesis, dict):
            return str(hypothesis.get("hypothesis", ""))
        return str(hypothesis)

    @staticmethod
    def _extract_dataset(state: dict) -> str:
        selected = state.get("selected_dataset") or {}
        if isinstance(selected, dict):
            return str(
                selected.get("name")
                or selected.get("title")
                or selected.get("dataset_name")
                or state.get("dataset_path")
                or ""
            )
        return str(state.get("dataset_path") or selected or "")

    @staticmethod
    def _extract_algorithm(state: dict) -> str:
        plan = state.get("experiment_plan") or {}
        if isinstance(plan, dict):
            return str(
                plan.get("algorithm")
                or plan.get("model")
                or plan.get("model_name")
                or ""
            )
        return ""

    @staticmethod
    def _is_successful(state: dict) -> bool:
        if state.get("failure_state") or state.get("error"):
            return False

        # A completed report or a successful experiment is evidence that
        # the workflow produced a usable result.
        if state.get("final_report"):
            return True

        results = state.get("experiment_results") or {}
        if isinstance(results, dict) and results.get("success") is True:
            return True

        verdict = state.get("verdict")
        return bool(verdict and state.get("selected_hypothesis"))

    def save(self, state: dict, thread_id: str) -> str | None:
        """Save a successful research run. Return its memory record ID."""

        if not self._is_successful(state):
            return None

        question = str(state.get("research_question") or "").strip()
        if not question:
            return None

        experiment_results = state.get("experiment_results") or {}
        statistical_results = state.get("statistical_results") or {}
        critic_feedback = state.get("critic_feedback") or {}

        findings = {
            "experiment_results": experiment_results,
            "statistical_results": statistical_results,
            "critic_feedback": critic_feedback,
            "final_report": state.get("final_report", ""),
        }

        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        values = (
            memory_id,
            thread_id,
            question,
            self._extract_hypothesis(state),
            _json(findings),
            str(state.get("verdict") or ""),
            self._extract_dataset(state),
            str(state.get("target_column") or ""),
            self._extract_algorithm(state),
            now,
        )

        with self._connect() as connection:
            # A resumed workflow may be saved again. Update its existing
            # record instead of creating duplicate records.
            connection.execute(
                """
                INSERT INTO research_memories (
                    id, thread_id, research_question, hypothesis, findings,
                    verdict, dataset, target_column, algorithm, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    research_question = excluded.research_question,
                    hypothesis = excluded.hypothesis,
                    findings = excluded.findings,
                    verdict = excluded.verdict,
                    dataset = excluded.dataset,
                    target_column = excluded.target_column,
                    algorithm = excluded.algorithm,
                    created_at = excluded.created_at
                """,
                values,
            )

            row = connection.execute(
                "SELECT id FROM research_memories WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()

        return row["id"] if row else None

    def retrieve(
        self,
        research_question: str,
        limit: int = 3,
        exclude_thread_id: str | None = None,
    ) -> list[dict]:
        """Retrieve previous research using keyword overlap."""

        query_tokens = _tokens(research_question)
        if not query_tokens or limit <= 0:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM research_memories
                ORDER BY created_at DESC
                """
            ).fetchall()

        ranked = []

        for row in rows:
            if exclude_thread_id and row["thread_id"] == exclude_thread_id:
                continue

            question_tokens = _tokens(row["research_question"])
            hypothesis_tokens = _tokens(row["hypothesis"] or "")

            record_tokens = question_tokens | hypothesis_tokens

            if not record_tokens:
                continue

            overlap = len(query_tokens & record_tokens)

            if overlap == 0:
                continue

            similarity = overlap / len(query_tokens | record_tokens)

            findings = {}
            try:
                findings = json.loads(row["findings"] or "{}")
            except (TypeError, json.JSONDecodeError):
                findings = {}

            ranked.append(
                {
                    "research_question": row["research_question"],
                    "hypothesis": row["hypothesis"],
                    "findings": findings,
                    "verdict": row["verdict"],
                    "dataset": row["dataset"],
                    "target_column": row["target_column"],
                    "algorithm": row["algorithm"],
                    "created_at": row["created_at"],
                    "similarity": round(similarity, 4),
                }
            )

        ranked.sort(
            key=lambda item: (item["similarity"], item["created_at"]),
            reverse=True,
        )
        return ranked[:limit]

    def count(self) -> int:
        """Return the number of saved research records."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM research_memories"
            ).fetchone()
        return int(row["total"])
