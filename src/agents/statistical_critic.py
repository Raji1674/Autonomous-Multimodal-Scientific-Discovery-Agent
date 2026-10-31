
"""Statistical Critic: deterministic verdict from the numbers; an LLM may only explain them."""
from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent
from src.experiments.protocol import ExperimentPlan, ResultError, validate_result
from src.experiments.templates import run_plan
from src.llm.client import LLMClient
from src.schemas.state import FailureState, ResearchState
from src.statistics.significance import paired_analysis


EXPLAIN_SYSTEM = (
    "You explain statistical results to a researcher in 3-4 plain sentences. "
    "Use ONLY the numbers provided; never recompute, round differently, or invent "
    "values. Do not claim a discovery or novelty. 'accept' means the pre-set "
    "thresholds were met, not that the hypothesis is true."
)


class StatisticalCritic:
    """Pure logic (no LLM, no state), easy to unit test.

    Thresholds are conventions chosen BEFORE seeing results, to limit p-hacking.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_seeds: int = 10,
        min_abs_diff: float = 0.005,
        min_effect: float = 0.5,
        overfit_gap: float = 0.15,
    ):
        # Validate alpha.
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(alpha)
            or not 0 < alpha < 1
        ):
            raise ValueError("alpha must be finite and strictly between 0 and 1")

        # Validate the minimum number of paired runs.
        if (
            isinstance(min_seeds, bool)
            or not isinstance(min_seeds, (int, np.integer))
            or min_seeds < 2
        ):
            raise ValueError("min_seeds must be an integer >= 2")

        # Validate non-negative thresholds.
        for name, value in (
            ("min_abs_diff", min_abs_diff),
            ("min_effect", min_effect),
            ("overfit_gap", overfit_gap),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and >= 0")

        self.alpha = float(alpha)
        self.min_seeds = int(min_seeds)
        self.min_abs_diff = float(min_abs_diff)
        self.min_effect = float(min_effect)
        self.overfit_gap = float(overfit_gap)

    def evaluate(self, plan: dict, result: dict, n_tests: int = 1) -> dict:
        """Validate a built-in experiment plan, then evaluate its scores."""
        validated_plan = ExperimentPlan(**plan).model_dump()
        return self.evaluate_scores(
            result=result,
            expected_direction=validated_plan["expected_direction"],
            n_tests=n_tests,
        )

    def evaluate_scores(
        self,
        result: dict,
        expected_direction: str = "higher",
        n_tests: int = 1,
    ) -> dict:
        """Evaluate paired scores without requiring a built-in dataset plan."""
        if expected_direction not in {"higher", "lower"}:
            raise ValueError("expected_direction must be 'higher' or 'lower'")

        if (
            isinstance(n_tests, bool)
            or not isinstance(n_tests, (int, np.integer))
            or n_tests < 1
        ):
            raise ValueError("n_tests must be a positive integer")

        validate_result(result)

        if result["result_label"] != "executed":
            raise ResultError(
                f"cannot analyse a result labelled {result['result_label']!r}"
            )

        analysis = paired_analysis(
            result["baseline"],
            result["treatment"],
            result.get("n_train"),
            result.get("n_test"),
            alpha=self.alpha,
        )

        issues: list[dict] = []

        def flag(severity: str, message: str) -> None:
            issues.append({"severity": severity, "message": message})

        n = analysis["n"]

        if n < self.min_seeds:
            flag("critical", f"only {n} paired runs; need at least {self.min_seeds}")
        elif n < 20:
            flag("minor", f"{n} paired runs is small; 20+ is preferable")

        if analysis["test"] == "insufficient_data":
            return self._pack(
                analysis, "inconclusive", issues,
                ["not enough paired runs for any test"], n_tests,
            )

        if (analysis.get("normality") or {}).get("normal") is False:
            flag("minor", "differences are not normal; used Wilcoxon + bootstrap CI")

        if analysis["test"] == "paired_t":
            flag(
                "minor",
                "split sizes unknown: ordinary paired t-test may be optimistic "
                "for repeated splits",
            )

        baseline = analysis["baseline"]
        treatment = analysis["treatment"]

        if baseline["sd"] <= 1e-12 and treatment["sd"] <= 1e-12:
            flag("minor", "zero variance across seeds; check that splits vary")

        if baseline["mean"] > 0.99 and treatment["mean"] > 0.99:
            flag("minor", "ceiling effect: both arms are near-perfect")

        if n_tests > 1:
            flag("info", f"multiple comparisons: Bonferroni correction applied over {n_tests} tests")

        train_scores = result.get("treatment_train")
        test_scores = result.get("treatment")
        if train_scores and test_scores:
            gap = float(np.mean(train_scores) - np.mean(test_scores))
            if gap > self.overfit_gap:
                flag("minor", f"possible overfitting: train-test gap {gap:.3f}")

        p_value = analysis["p"]
        p_adjusted = None if p_value is None else min(1.0, float(p_value) * int(n_tests))
        analysis["p_adjusted"] = p_adjusted

        if any(issue["severity"] == "critical" for issue in issues):
            return self._pack(
                analysis, "inconclusive", issues,
                ["critical issue (see issues)"], n_tests,
            )

        difference = analysis["mean_diff"]
        ci = analysis["ci"]
        effect_size = analysis["effect_size"]

        if (
            p_adjusted is None
            or ci is None
            or difference is None
            or effect_size is None
            or not np.isfinite(effect_size)
):
            return self._pack(
        analysis,
        "inconclusive",
        issues,
        ["undefined or insufficient statistical output"],
        n_tests,
)
            

        significant = p_adjusted < self.alpha
        ci_excludes_zero = ci[0] > 0 or ci[1] < 0
        practical_ok = abs(difference) >= self.min_abs_diff
        effect_ok = (
    effect_size is not None
    and np.isfinite(effect_size)
    and abs(effect_size) >= self.min_effect
)
        direction_ok = (
            (difference > 0) == (expected_direction == "higher")
        )

        reasons: list[str] = []
        if not significant:
            reasons.append(f"adjusted p={p_adjusted:.4g} is not below alpha={self.alpha}")
        if not ci_excludes_zero:
            reasons.append("confidence interval includes 0")
        if not practical_ok:
            reasons.append(
                f"|mean difference| {abs(difference):.4f} is below "
                f"the practical threshold {self.min_abs_diff}"
            )
        if not effect_ok:
            reasons.append(f"|effect size| {abs(effect_size):.2f} is below {self.min_effect}")
        if significant and ci_excludes_zero and not direction_ok:
            reasons.append("significant effect in the OPPOSITE direction to the hypothesis")

        verdict = (
            "accept"
            if significant and ci_excludes_zero and practical_ok and effect_ok and direction_ok
            else "reject"
        )
        if verdict == "accept":
            reasons.append("all pre-set criteria met")

        return self._pack(analysis, verdict, issues, reasons, n_tests)

    @staticmethod
    def _pack(
        analysis: dict,
        verdict: str,
        issues: list[dict],
        reasons: list[str],
        n_tests: int,
    ) -> dict:
        """Return the evaluation with a heuristic quality score from 0 to 10.

        The quality score measures analysis reliability heuristically; it does
        not measure scientific truth.
        """
        weights = {
            "critical": 4,
            "minor": 1,
            "info": 0,
        }

        score = max(
            0,
            10 - sum(weights[issue["severity"]] for issue in issues),
        )

        return {
            "analysis": analysis,
            "verdict": verdict,
            "issues": issues,
            "reasons": reasons,
            "statistical_quality": score,
            "n_tests": n_tests,
        }


def check_reproducibility(
    plan: dict,
    result: dict,
    atol: float = 1e-9,
) -> dict:
    """Re-run the vetted template and compare its scores with the supplied result.

    This check is intended only for experiments produced by the built-in
    experiment template. It does not validate arbitrary external scripts.
    """
    if (
        isinstance(atol, bool)
        or not isinstance(atol, (int, float, np.integer, np.floating))
        or not np.isfinite(atol)
        or atol < 0
    ):
        raise ValueError("atol must be finite and >= 0")

    validate_result(result)

    if result["result_label"] != "executed":
        raise ResultError(
            "reproducibility check requires an executed result"
        )

    validated_plan = ExperimentPlan(**plan)
    repeated_result = run_plan(validated_plan)

    differences = []

    for key in ("baseline", "treatment"):
        original = np.asarray(result[key], dtype=float)
        repeated = np.asarray(repeated_result[key], dtype=float)

        if original.shape != repeated.shape:
            return {
                "reproduced": False,
                "max_abs_diff": None,
                "reason": f"{key} score lengths differ between runs",
            }

        differences.append(
            float(np.max(np.abs(original - repeated)))
        )

    max_difference = max(differences)

    return {
        "reproduced": bool(max_difference <= atol),
        "max_abs_diff": max_difference,
    }


class StatisticalCriticAgent(BaseAgent):
    """Workflow node that writes statistics, feedback and verdict.

    Reads experiment_plan and experiment_results. Results must be labelled
    'executed'; simulated or unexecuted outputs are refused.
    """

    name = "statistical_critic"
    failure_on_error = FailureState.STATISTICAL_ANALYSIS_FAILED

    def __init__(
        self,
        critic: StatisticalCritic | None = None,
        llm: LLMClient | None = None,
        n_tests: int = 1,
        check_repro: bool = False,
    ):
        super().__init__()

        if (
            isinstance(n_tests, bool)
            or not isinstance(n_tests, (int, np.integer))
            or n_tests < 1
        ):
            raise ValueError("n_tests must be a positive integer")

        self.critic = critic or StatisticalCritic()
        self.llm = llm
        self.n_tests = int(n_tests)
        self.check_repro = check_repro

    def run(self, state: ResearchState) -> dict:
        result = state.get("experiment_results", {})

        if not isinstance(result, dict) or result.get("result_label") != "executed":
            label = result.get("result_label") if isinstance(result, dict) else None
            return {
                "failure_state": FailureState.STATISTICAL_ANALYSIS_FAILED.value,
                "error": (
                    f"result is labelled {label!r}, not 'executed'; "
                    "nothing to analyse"
                ),
            }

        ev = self.critic.evaluate(
            state["experiment_plan"],
            result,
            self.n_tests,
        )

        feedback = {
            key: ev[key]
            for key in (
                "issues",
                "reasons",
                "statistical_quality",
                "verdict",
            )
        }

        if self.check_repro:
            feedback["reproducibility"] = check_reproducibility(
                state["experiment_plan"],
                result,
            )

        if self.llm is not None:
            import json

            feedback["interpretation"] = self.llm.complete(
                EXPLAIN_SYSTEM,
                json.dumps(ev, default=str),
            )

        return {
            "statistical_results": ev["analysis"],
            "critic_feedback": feedback,
            "verdict": ev["verdict"],
        }
