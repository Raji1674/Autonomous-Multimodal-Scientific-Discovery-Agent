"""Vetted experiment code: one function, used in-process AND rendered as a sandbox script.

`_experiment` is deliberately self-contained (imports inside) so `render_script` can
copy its source into a standalone file. Both paths run identical code, so results match.
Design guarantees: split BEFORE fitting, scaling lives inside a Pipeline (no leakage),
both arms see identical train/test splits per seed (paired design), fixed seeds.
"""
from __future__ import annotations

import inspect

from src.experiments.protocol import ExperimentPlan, validate_result


def _experiment(plan: dict) -> dict:
    import platform
    import time

    import numpy as np
    import sklearn
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    loaders = {"iris": datasets.load_iris, "wine": datasets.load_wine,
               "breast_cancer": datasets.load_breast_cancer, "digits": datasets.load_digits}
    X, y = loaders[plan["dataset"]](return_X_y=True)

    def build(value, seed):
        kw = dict(plan.get("fixed_params") or {})
        kw[plan["independent_variable"]] = value
        name = plan["model"]
        if name == "logreg":
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **kw))
        if name == "svm":
            return make_pipeline(StandardScaler(), SVC(random_state=seed, **kw))
        if name == "knn":
            return make_pipeline(StandardScaler(), KNeighborsClassifier(**kw))
        if name == "tree":
            return DecisionTreeClassifier(random_state=seed, **kw)
        return RandomForestClassifier(random_state=seed, n_jobs=1, **kw)

    def score(model, Xs, ys):
        pred = model.predict(Xs)
        return float(accuracy_score(ys, pred) if plan["metric"] == "accuracy"
                     else f1_score(ys, pred, average="macro"))

    t0 = time.time()
    out = {"baseline": [], "treatment": [], "baseline_train": [], "treatment_train": []}
    for seed in range(plan["n_seeds"]):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=plan["test_size"], stratify=y, random_state=seed)
        for arm, value in (("baseline", plan["baseline_value"]), ("treatment", plan["treatment_value"])):
            model = build(value, seed).fit(Xtr, ytr)
            out[arm].append(score(model, Xte, yte))
            out[arm + "_train"].append(score(model, Xtr, ytr))
    return {
        "result_label": "executed", **out,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "seeds": list(range(plan["n_seeds"])),
        "runtime_s": round(time.time() - t0, 3),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
    }


def run_plan(plan: ExperimentPlan) -> dict:
    """Run a plan in-process. Only for OUR vetted template, never for LLM-written code."""
    return validate_result(_experiment(plan.model_dump(mode="json")))


def render_script(plan: ExperimentPlan) -> str:
    """Standalone script for the Docker sandbox; prints the result JSON as its last line."""
    return ("import json\n\n" + inspect.getsource(_experiment)
            + "\n\nprint(json.dumps(_experiment(" + repr(plan.model_dump(mode="json")) + ")))\n")
