"""Measure the Critic itself on SIMULATED data: false-accept rate (null) and power (real effect).

Scores are synthetic Gaussian draws, so this checks the decision rule's behaviour, not any
real model. Seed-based CV scores overlap, so real experiments are noisier than these draws.
"""
from __future__ import annotations

import numpy as np

from src.agents.statistical_critic import StatisticalCritic

_PLAN = dict(research_question="sim", hypothesis="sim", null_hypothesis="sim", dataset="iris", model="logreg",
             independent_variable="C", baseline_value=1.0, treatment_value=0.01, expected_direction="higher")


def simulate_accept_rate(true_diff: float, sd: float = 0.01, n: int = 20, n_sims: int = 1000, seed: int = 0,
                         critic: StatisticalCritic | None = None, n_train: int = 105, n_test: int = 45) -> dict:
    """Fraction of simulated experiments the critic accepts when the TRUE mean difference is true_diff."""
    critic = critic or StatisticalCritic()
    rng = np.random.default_rng(seed)
    accepted = 0
    for _ in range(n_sims):
        base = rng.normal(0.9, 0.02, n)
        treat = base + true_diff + rng.normal(0, sd, n)
        res = {"result_label": "executed", "baseline": base.tolist(), "treatment": treat.tolist(),
               "n_train": n_train, "n_test": n_test}
        accepted += critic.evaluate(_PLAN, res)["verdict"] == "accept"
    return {"true_diff": true_diff, "sd": sd, "n": n, "n_sims": n_sims, "accept_rate": accepted / n_sims}


if __name__ == "__main__":
    for diff in (0.0, 0.005, 0.01, 0.02):
        print(simulate_accept_rate(diff))
