"""Persistent reward-based debugging strategy selection."""
import random
import sqlite3
from pathlib import Path


class DebugPolicy:
    STRATEGIES = [
        "minimal_fix",
        "type_error_fix",
        "dependency_fix",
        "data_validation_fix",
    ]

    def __init__(self, db_path="data/checkpoints/debug_policy.sqlite", epsilon=0.2):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.epsilon = epsilon

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_rewards (
                    strategy TEXT PRIMARY KEY,
                    q_value REAL NOT NULL DEFAULT 0,
                    selections INTEGER NOT NULL DEFAULT 0
                )
            """)
            for strategy in self.STRATEGIES:
                conn.execute(
                    "INSERT OR IGNORE INTO strategy_rewards(strategy) VALUES (?)",
                    (strategy,),
                )

    def select_strategy(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT strategy, q_value, selections FROM strategy_rewards"
            ).fetchall()

        if random.random() < self.epsilon:
            return random.choice(self.STRATEGIES)

        return max(rows, key=lambda row: row[1])[0]

    def update(self, strategy, reward):
        if strategy not in self.STRATEGIES:
            raise ValueError("Unknown debugging strategy")

        if reward not in (-1.0, 1.0):
            raise ValueError("Reward must be -1.0 or 1.0")

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT q_value, selections FROM strategy_rewards WHERE strategy=?",
                (strategy,),
            ).fetchone()

            old_q, count = row
            new_q = old_q + (reward - old_q) / (count + 1)

            conn.execute(
                """UPDATE strategy_rewards
                   SET q_value=?, selections=?
                   WHERE strategy=?""",
                (new_q, count + 1, strategy),
            )

    def get_stats(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT strategy, q_value, selections FROM strategy_rewards"
            ).fetchall()

        return [
            {"strategy": name, "q_value": q, "selections": count}
            for name, q, count in rows
        ]
