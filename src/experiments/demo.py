"""Run real experiments and statistically criticise them: python -m src.experiments.demo

Numbers printed are MEASURED by running scikit-learn here, not invented. They show how the
pipeline behaves on tiny public datasets; they are not research findings.
"""
from __future__ import annotations

from src.agents.statistical_critic import StatisticalCritic, check_reproducibility
from src.experiments.protocol import ExperimentPlan
from src.experiments.templates import run_plan
from src.statistics.significance import holm_adjust

PLANS = [
    dict(dataset="breast_cancer", model="logreg", independent_variable="C", baseline_value=1.0, treatment_value=0.01),
    dict(dataset="wine", model="knn", independent_variable="n_neighbors", baseline_value=5, treatment_value=1),
    dict(dataset="iris", model="tree", independent_variable="max_depth", baseline_value=3, treatment_value=2),
]


def main() -> None:
    critic, evals, plans = StatisticalCritic(), [], []
    for spec in PLANS:
        plan = ExperimentPlan(research_question="demo", hypothesis=f"{spec['independent_variable']} changes accuracy",
                              null_hypothesis="no change", n_seeds=20, **spec)
        res = run_plan(plan)
        ev = critic.evaluate(plan.model_dump(), res, n_tests=len(PLANS))
        a = ev["analysis"]
        print(f"\n{spec['dataset']}/{spec['model']} {spec['independent_variable']}: "
              f"{spec['baseline_value']} -> {spec['treatment_value']}  (runtime {res['runtime_s']}s)")
        print(f"  baseline {a['baseline']['mean']:.4f}  treatment {a['treatment']['mean']:.4f}  diff {a['mean_diff']:+.4f}")
        print(f"  test={a['test']}  p={a['p']:.4g}  p_bonferroni={a['p_adjusted']:.4g}  "
              f"CI95=[{a['ci'][0]:+.4f}, {a['ci'][1]:+.4f}]  {a['effect_size_name']}={a['effect_size']:.2f}")
        print(f"  verdict={ev['verdict']}  quality={ev['statistical_quality']}/10  reproduced={check_reproducibility(plan.model_dump(), res)['reproduced']}")
        for r in ev["reasons"]:
            print(f"    - {r}")
        for i in ev["issues"]:
            print(f"    [{i['severity']}] {i['message']}")
        evals.append(a["p"])
    print("\nHolm-adjusted p over the 3 tests:", [round(x, 4) for x in holm_adjust(evals)])


if __name__ == "__main__":
    main()
