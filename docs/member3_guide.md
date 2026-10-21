# Member 3 Guide: ML + Experimentation + Statistical Critic (3 weeks)

*Idhu unga role-ku step-by-step plan. All code below was actually run here: 36 pytest tests pass, and the outputs shown in Section 9 are real measured output.*

---

## 1. Your job in one picture

```
Hypothesis (from M1/M4 agents)
   ↓  ExperimentDesigner      -> ExperimentPlan (validated JSON)
   ↓  render_script / run_plan -> experiment code (runs in M4's Docker sandbox)
   ↓  parse_result            -> {baseline:[...], treatment:[...]}  per-seed scores
   ↓  StatisticalCritic       -> p-value, CI, effect size, issues, verdict
   ↓  (verdict "accept")      -> M5 synthesis / dashboard
```

You own: datasets, experiment design, experiment code, result format, statistics, Critic Agent, and measuring the Critic itself.

**Contracts you must agree with others (Week 1, Day 2):**

| With | Contract |
|---|---|
| M4 (sandbox) | The sandbox runs `render_script(plan)`; the script prints ONE JSON object on its LAST stdout line. M4 sets `result_label="executed"` only if the container exit code is 0 and `parse_result` succeeds. |
| M1 (orchestration) | State keys you read/write: `selected_hypothesis` → `experiment_plan` → `experiment_results` → `statistical_results`, `critic_feedback`, `verdict`. `verdict` is `accept` / `reject` / `inconclusive`. |
| M5 (synthesis/dashboard) | They read `statistical_results` (p, CI, effect size) and `critic_feedback`. They must never write results for anything not labelled `executed`. |

## 2. Three-week plan

| Day | Task | Output |
|---|---|---|
| **Week 1: experiment foundation** (team week 1 = skeleton, ingestion, Git) |||
| 1 | Setup, branch `feature/experiments-core`; learn pandas + scikit-learn basics (train/test split, accuracy) | Env works |
| 2 | Datasets + `ExperimentPlan` schema; agree contracts above | `datasets.py`, `protocol.py` |
| 3 | Experiment template with multi-seed paired design | `templates.py` (`_experiment`, `run_plan`) |
| 4 | Result protocol + parser + tests | `parse_result`, `validate_result` |
| 5 | `render_script` + hand to M4 for sandbox test | Script runs identically in-process and as a script |
| 6 | PR review, notes | Merged into `develop` |
| **Week 2: statistics + design** (team week 2 = RAG, sandbox, debug loop) |||
| 7 | Learn: p-value, CI, effect size, normality. Write stats functions | `significance.py` |
| 8 | Unit tests with known answers | `test_significance.py` |
| 9 | Critic decision rules + issue flags | `statistical_critic.py` |
| 10 | Experiment Designer (hypothesis → plan, retry on invalid) | `experiment_designer.py` |
| 11 | Integrate with M4's debug loop: "success" = valid result protocol, not just "no crash" | Reward definition agreed |
| 12 | Mid-review demo | Real run → critic verdict |
| **Week 3: Critic finalization + evaluation** (team week 3 = multimodal, critic, memory) |||
| 13 | Wire Critic node into the graph with M1/M5 (accept → synthesis, else end) | Routing works |
| 14 | Reproducibility check + Holm correction over a batch | `check_reproducibility`, `holm_adjust` |
| 15 | Calibrate the Critic: false-accept rate and power on simulated data | `calibration.py` |
| 16 | Run 10 to 20 real experiments, tabulate verdicts, document limits | Results table for report |
| 17 | Write your report chapters (stats critic, experiment architecture, limitations) | Docs |
| 18 | Buffer, viva prep, fix review comments | Final merge |

Team week 4 (synthesis, dashboard, holdout evaluation) is covered by M5, but you supply the statistical results tables and the Critic evaluation.

## 3. What to learn, and why

| Topic | Why you need it | Time |
|---|---|---|
| pandas/NumPy basics | Handle score arrays and tables | 3 h |
| scikit-learn: split, fit, score, Pipeline | Run experiments correctly (no leakage) | 4 h |
| p-value, CI, effect size | The Critic's core output | 4 h |
| Paired vs independent tests | Our design is paired (same split per seed) | 2 h |
| Normality checks, Wilcoxon, bootstrap | Choose a test by assumptions, not habit | 2 h |
| Multiple comparisons (Bonferroni, Holm) | Many hypotheses means many chances for false positives | 1 h |
| pytest | Prove the statistics are right | 2 h |

## 4. Setup (Day 1)

```bash
pip install numpy scipy scikit-learn pandas pydantic pytest
git checkout develop && git checkout -b feature/experiments-core
pytest -q          # all tests should pass before you change anything
```

## 5. Step-by-step with source code

### Step 1: Datasets (`src/experiments/datasets.py`)
**Why:** scikit-learn's built-in datasets (iris, wine, breast_cancer, digits) need no download, are free, tiny (experiments take under a second) and are versioned with the library. That keeps runs reproducible and the sandbox network-free.
```python
"""Small built-in datasets: offline, free, versioned with scikit-learn."""
from __future__ import annotations

import numpy as np
from sklearn import datasets as skd

REGISTRY = {
    "iris": skd.load_iris,
    "wine": skd.load_wine,
    "breast_cancer": skd.load_breast_cancer,
    "digits": skd.load_digits,
}


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return X, y and a small metadata dict (recorded in the experiment manifest)."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}; choose one of {sorted(REGISTRY)}")
    X, y = REGISTRY[name](return_X_y=True)
    meta = {"name": name, "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))), "source": "scikit-learn built-in"}
    return X, y, meta
```

### Step 2: Plan schema and result protocol (`src/experiments/protocol.py`)
**Why:** LLMs produce free text; the rest of the system needs exact data. `ExperimentPlan` rejects bad designs (a hyperparameter that doesn't exist for that model, identical baseline and treatment). `parse_result` takes only the LAST stdout line as JSON, because libraries print warnings first, and it refuses NaN/inf, unpaired lists and unknown labels.

Key idea: `result_label` is one of `executed` / `simulated` / `not_executed`. This enforces the project rule "never present simulated or unrun results as real".
```python
"""Experiment plan + result contracts: the JSON protocol between agents and the sandbox."""
from __future__ import annotations

import json
import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Which hyperparameters an experiment may vary, per model (keeps plans testable and safe).
ALLOWED_PARAMS: dict[str, set[str]] = {
    "logreg": {"C"},
    "svm": {"C"},
    "knn": {"n_neighbors"},
    "tree": {"max_depth"},
    "random_forest": {"n_estimators", "max_depth"},
}
RESULT_LABELS = {"executed", "simulated", "not_executed"}


class ExperimentPlan(BaseModel):
    """The 12-field design the Experiment Designer produces (subset that is machine-runnable)."""

    research_question: str
    hypothesis: str
    null_hypothesis: str
    dataset: Literal["iris", "wine", "breast_cancer", "digits"]
    model: Literal["logreg", "svm", "knn", "tree", "random_forest"]
    metric: Literal["accuracy", "f1_macro"] = "accuracy"
    independent_variable: str
    baseline_value: float | int | None
    treatment_value: float | int | None
    expected_direction: Literal["higher", "lower"] = "higher"
    n_seeds: int = Field(20, ge=2, le=100)
    test_size: float = Field(0.3, ge=0.1, le=0.5)
    fixed_params: dict[str, float | int | None] = Field(default_factory=dict)
    falsification_criteria: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ExperimentPlan":
        allowed = ALLOWED_PARAMS[self.model]
        if self.independent_variable not in allowed:
            raise ValueError(f"{self.independent_variable!r} is not allowed for {self.model}; use {sorted(allowed)}")
        bad = set(self.fixed_params) - allowed
        if bad:
            raise ValueError(f"fixed_params not allowed for {self.model}: {sorted(bad)}")
        if self.independent_variable in self.fixed_params:
            raise ValueError("independent_variable must not also appear in fixed_params")
        if self.baseline_value == self.treatment_value:
            raise ValueError("baseline_value and treatment_value must differ")
        return self


class ResultError(ValueError):
    """The experiment output does not follow the result protocol."""


def validate_result(data: dict) -> dict:
    """Check structure only (not scientific quality). Raises ResultError on problems."""
    if not isinstance(data, dict):
        raise ResultError("result must be a JSON object")
    if data.get("result_label") not in RESULT_LABELS:
        raise ResultError(f"result_label must be one of {sorted(RESULT_LABELS)}")
    for key in ("baseline", "treatment"):
        vals = data.get(key)
        if not isinstance(vals, list) or not vals:
            raise ResultError(f"{key!r} must be a non-empty list of scores")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals):
            raise ResultError(f"{key!r} contains non-numeric or non-finite values")
    if len(data["baseline"]) != len(data["treatment"]):
        raise ResultError("baseline and treatment must have the same length (paired seeds)")
    if len(data["baseline"]) < 2:
        raise ResultError("need at least 2 seeds")
    return data


def parse_result(stdout: str) -> dict:
    """Parse the LAST non-empty stdout line as JSON (earlier lines may be library chatter)."""
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise ResultError("experiment produced no output")
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ResultError(f"last output line is not JSON: {exc}") from exc
    return validate_result(data)
```

### Step 3: The experiment template (`src/experiments/templates.py`)
**Why a template first:** LLM-written experiment code fails often and is hard to audit. A vetted template gives you a working baseline immediately; free-form LLM code can come later, and must output the same JSON protocol.

Design choices, and why:
- **Split before fitting; scaling inside a `Pipeline`** → no data leakage.
- **Same train/test split for baseline and treatment per seed** → a *paired* design, which removes split-to-split noise.
- **`random_state=seed` everywhere** → identical results on re-run (reproducibility).
- **Train scores recorded too** → the Critic can flag overfitting.
- **One function, two uses:** `run_plan` runs it in-process, and `render_script` copies its source into a standalone script for the sandbox, so both paths run identical code (a test checks this).
```python
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
```

### Step 4: Statistics (`src/statistics/significance.py`)
**Why each function exists:**
- `check_normality` (Shapiro-Wilk) → don't assume a t-test is valid; check first.
- `corrected_resampled_ttest` → *the important one.* Repeated random splits overlap, so scores are not independent and an ordinary paired t-test gives p-values that are too small. The Nadeau-Bengio correction inflates the variance by (1/k + n_test/n_train). A test confirms it is never less conservative than the naive test.
- `wilcoxon` + `bootstrap_ci` → fallback when differences aren't normal.
- `hedges_gz` / `rank_biserial_paired` → effect size: "significant" is not the same as "matters".
- `holm_adjust` → multiple-comparison control across a batch of hypotheses.
- `is_constant` → floating-point tolerance. (A real bug caught by the tests: `[0.1, 0.1, 0.1]` has a standard deviation of ~1e-17, not 0, which produced an absurd effect size of 5.9e15 before the fix.)
```python
"""Statistics for paired experiments (same seeds/splits for baseline and treatment).

Every function is deterministic and unit-tested; the LLM never computes numbers.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


EPS = 1e-12  # floating-point tolerance: spreads below this count as "no variation"


def is_constant(x) -> bool:
    """True if all values are equal up to floating-point noise."""
    return float(np.ptp(np.asarray(x, dtype=float))) <= EPS


def _f(x) -> float | None:
    """JSON-safe float: NaN/inf become None."""
    x = float(x)
    return x if math.isfinite(x) else None


def describe(x) -> dict:
    a = np.asarray(x, dtype=float)
    n = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    return {"n": n, "mean": _f(a.mean()), "sd": _f(sd), "sem": _f(sd / math.sqrt(n)) if n else None}


def check_normality(x, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk. With very few points it has little power; the note says so."""
    a = np.asarray(x, dtype=float)
    if len(a) < 3 or is_constant(a):
        return {"test": "shapiro", "p": None, "normal": None, "note": "not testable (n<3 or constant values)"}
    p = float(stats.shapiro(a).pvalue)
    return {"test": "shapiro", "p": _f(p), "normal": bool(p > alpha),
            "note": "low power when n<20" if len(a) < 20 else ""}


def cohens_dz(diffs) -> float | None:
    """Paired effect size: mean(diff)/sd(diff). None if sd=0 and mean!=0 (undefined)."""
    d = np.asarray(diffs, dtype=float)
    if is_constant(d):
        return 0.0 if abs(d.mean()) <= EPS else None
    sd = d.std(ddof=1)
    return _f(d.mean() / sd)


def hedges_gz(diffs) -> float | None:
    """dz with the small-sample bias correction."""
    dz = cohens_dz(diffs)
    n = len(diffs)
    if dz is None or n < 3:
        return dz
    return _f(dz * (1 - 3 / (4 * (n - 1) - 1)))


def rank_biserial_paired(diffs) -> float:
    """Effect size for Wilcoxon: (sum of positive ranks - negative ranks) / total, in [-1, 1]."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.abs(d) > EPS]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    rp, rm = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((rp - rm) / (rp + rm))


def bootstrap_ci(diffs, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean difference (no normality assumption)."""
    a = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def corrected_resampled_ttest(diffs, n_train: int, n_test: int, alpha: float = 0.05) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test.

    Repeated random train/test splits overlap, so scores are NOT independent and an
    ordinary paired t-test is too optimistic. The correction inflates the variance by
    (1/k + n_test/n_train).
    """
    d = np.asarray(diffs, dtype=float)
    k, m, v = len(d), float(d.mean()), float(d.var(ddof=1))
    if is_constant(d):
        return {"t": None, "df": k - 1, "p": 1.0 if abs(m) <= EPS else 0.0, "ci": (m, m), "se": 0.0}
    se = math.sqrt((1.0 / k + n_test / n_train) * v)
    t = m / se
    p = float(2 * stats.t.sf(abs(t), k - 1))
    half = float(stats.t.ppf(1 - alpha / 2, k - 1) * se)
    return {"t": _f(t), "df": k - 1, "p": p, "ci": (m - half, m + half), "se": se}


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (controls family-wise error)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def paired_analysis(baseline, treatment, n_train: int | None = None, n_test: int | None = None,
                    alpha: float = 0.05, seed: int = 0, min_n: int = 5) -> dict:
    """Pick and run a paired test after checking assumptions; return everything needed to audit it.

    Selection (on the per-seed differences):
      all differences zero          -> no test needed
      normal (or untestable) + split sizes known -> corrected resampled t-test (primary)
      normal, split sizes unknown   -> ordinary paired t-test
      not normal                    -> Wilcoxon signed-rank + bootstrap CI
    The other tests are reported in `alternatives` for transparency.
    """
    b, t = np.asarray(baseline, dtype=float), np.asarray(treatment, dtype=float)
    if len(b) != len(t):
        raise ValueError("baseline and treatment must be paired (same length)")
    d = t - b
    out: dict = {"n": len(d), "alpha": alpha, "baseline": describe(b), "treatment": describe(t),
                 "mean_diff": _f(d.mean()), "diff_sd": describe(d)["sd"]}
    if len(d) < min_n:
        return {**out, "test": "insufficient_data", "p": None, "ci": None, "effect_size": None,
                "effect_size_name": None, "normality": None, "alternatives": {}}

    norm = check_normality(d, alpha)
    alt: dict = {}
    with np.errstate(all="ignore"):
        naive_p = _f(stats.ttest_rel(t, b).pvalue) if not is_constant(d) else None
    alt["paired_t_naive_p"] = naive_p
    all_zero = bool(np.all(np.abs(d) <= EPS))
    if not all_zero:
        alt["wilcoxon_p"] = _f(stats.wilcoxon(d).pvalue)

    if all_zero:
        test, p, ci, es, es_name = "none_identical_scores", 1.0, (0.0, 0.0), 0.0, "cohens_dz"
    elif norm["normal"] is not False and n_train and n_test:
        r = corrected_resampled_ttest(d, n_train, n_test, alpha)
        test, p, ci, es, es_name = "corrected_resampled_t", r["p"], r["ci"], hedges_gz(d), "hedges_gz"
    elif norm["normal"] is not False:
        tt = stats.ttest_rel(t, b)
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        half = float(stats.t.ppf(1 - alpha / 2, len(d) - 1) * se)
        test, p, ci = "paired_t", float(tt.pvalue), (float(d.mean()) - half, float(d.mean()) + half)
        es, es_name = hedges_gz(d), "hedges_gz"
    else:
        test, p = "wilcoxon_signed_rank", alt["wilcoxon_p"]
        ci, es, es_name = bootstrap_ci(d, alpha=alpha, seed=seed), rank_biserial_paired(d), "rank_biserial"
    return {**out, "test": test, "p": _f(p), "ci": [_f(ci[0]), _f(ci[1])], "effect_size": es,
            "effect_size_name": es_name, "normality": norm, "alternatives": alt}
```

### Step 5: The Critic (`src/agents/statistical_critic.py`)
**Why "code first, LLM second":** LLMs are unreliable at arithmetic and can be talked into a verdict. The verdict comes only from deterministic rules; the LLM may only *explain* the numbers (its prompt forbids recomputing them or claiming discovery).

Decision rule for `accept` (all must hold; thresholds are fixed *before* seeing results to limit p-hacking):
1. Bonferroni-adjusted p < 0.05
2. 95% CI excludes 0
3. |mean difference| ≥ 0.005 (practical size)
4. |effect size| ≥ 0.5
5. Direction matches `expected_direction`

`inconclusive` if fewer than 10 paired runs. `reject` otherwise. **"accept" means the pre-set thresholds were met, not that the hypothesis is true.**

The `statistical_quality` score (0 to 10) only reflects how reliable the *analysis* was (deductions for small n, non-normality, overfit gap and so on). It's a heuristic.

The agent refuses to analyse anything not labelled `executed`, which is the guard that stops the Week-1 stub's output being analysed as if real.
```python
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
    "You explain statistical results to a researcher in 3-4 plain sentences. Use ONLY the numbers "
    "provided; never recompute, round differently, or invent values. Do not claim a discovery or "
    "novelty. 'accept' means the pre-set thresholds were met, not that the hypothesis is true."
)


class StatisticalCritic:
    """Pure logic (no LLM, no state), easy to unit test.

    Thresholds are conventions chosen BEFORE seeing results, to limit p-hacking.
    """

    def __init__(self, alpha: float = 0.05, min_seeds: int = 10, min_abs_diff: float = 0.005,
                 min_effect: float = 0.5, overfit_gap: float = 0.15):
        self.alpha, self.min_seeds = alpha, min_seeds
        self.min_abs_diff, self.min_effect, self.overfit_gap = min_abs_diff, min_effect, overfit_gap

    def evaluate(self, plan: dict, result: dict, n_tests: int = 1) -> dict:
        """Return {"analysis", "verdict", "issues", "statistical_quality", "reasons"}.

        n_tests: how many hypotheses were tested in total (Bonferroni: p_adj = min(1, p * n_tests)).
        """
        validate_result(result)
        if result["result_label"] != "executed":
            raise ResultError(f"cannot analyse a result labelled {result['result_label']!r}")
        plan = ExperimentPlan(**plan).model_dump()
        a = paired_analysis(result["baseline"], result["treatment"], result.get("n_train"),
                            result.get("n_test"), alpha=self.alpha)
        issues: list[dict] = []

        def flag(severity: str, msg: str) -> None:
            issues.append({"severity": severity, "message": msg})

        n = a["n"]
        if n < self.min_seeds:
            flag("critical", f"only {n} paired runs; need at least {self.min_seeds}")
        elif n < 20:
            flag("minor", f"{n} paired runs is small; 20+ is preferable")
        if a["test"] == "insufficient_data":
            return self._pack(a, "inconclusive", issues, ["not enough paired runs for any test"], n_tests)
        norm = a.get("normality") or {}
        if norm.get("normal") is False:
            flag("minor", "differences are not normal; used Wilcoxon + bootstrap CI")
        if a["test"] == "paired_t":
            flag("minor", "split sizes unknown: ordinary paired t-test is optimistic for repeated splits")
        if a["baseline"]["sd"] <= 1e-12 and a["treatment"]["sd"] <= 1e-12:
            flag("minor", "zero variance across seeds (deterministic scores); check the seeds actually vary the split")
        if a["baseline"]["mean"] > 0.99 and a["treatment"]["mean"] > 0.99:
            flag("minor", "ceiling effect: both arms are near-perfect, so differences are hard to detect")
        if n_tests > 1:
            flag("info", f"multiple comparisons: Bonferroni correction applied over {n_tests} tests")
        if result.get("treatment_train") and result.get("treatment"):
            gap = float(np.mean(result["treatment_train"]) - np.mean(result["treatment"]))
            if gap > self.overfit_gap:
                flag("minor", f"possible overfitting: train-test gap {gap:.3f} in the treatment arm")

        p = a["p"]
        p_adj = None if p is None else min(1.0, p * n_tests)
        a["p_adjusted"] = p_adj
        reasons: list[str] = []
        if any(i["severity"] == "critical" for i in issues):
            return self._pack(a, "inconclusive", issues, ["critical issue (see issues)"], n_tests)

        diff, ci, es = a["mean_diff"], a["ci"], a["effect_size"]
        sig = p_adj is not None and p_adj < self.alpha
        ci_excludes_0 = ci is not None and (ci[0] > 0 or ci[1] < 0)
        big_enough = abs(diff) >= self.min_abs_diff
        effect_ok = es is None or abs(es) >= self.min_effect
        right_dir = (diff > 0) == (plan["expected_direction"] == "higher")
        if not sig:
            reasons.append(f"adjusted p={p_adj:.4g} is not below alpha={self.alpha}")
        if not ci_excludes_0:
            reasons.append("confidence interval includes 0")
        if not big_enough:
            reasons.append(f"|mean difference| {abs(diff):.4f} is below the practical threshold {self.min_abs_diff}")
        if not effect_ok:
            reasons.append(f"|effect size| {abs(es):.2f} is below {self.min_effect}")
        if sig and ci_excludes_0 and not right_dir:
            reasons.append("significant effect in the OPPOSITE direction to the hypothesis")
        verdict = "accept" if (sig and ci_excludes_0 and big_enough and effect_ok and right_dir) else "reject"
        if verdict == "accept":
            reasons.append("all pre-set criteria met")
        return self._pack(a, verdict, issues, reasons, n_tests)

    @staticmethod
    def _pack(analysis: dict, verdict: str, issues: list[dict], reasons: list[str], n_tests: int) -> dict:
        """Statistical-quality score 0-10: a HEURISTIC of analysis reliability, not scientific truth."""
        weights = {"critical": 4, "minor": 1, "info": 0}
        score = max(0, 10 - sum(weights[i["severity"]] for i in issues))
        return {"analysis": analysis, "verdict": verdict, "issues": issues, "reasons": reasons,
                "statistical_quality": score, "n_tests": n_tests}


def check_reproducibility(plan: dict, result: dict, atol: float = 1e-9) -> dict:
    """Re-run the vetted template and compare scores. Only valid for template experiments."""
    again = run_plan(ExperimentPlan(**plan))
    diffs = [float(np.max(np.abs(np.asarray(result[k]) - np.asarray(again[k])))) for k in ("baseline", "treatment")]
    return {"reproduced": bool(max(diffs) <= atol), "max_abs_diff": max(diffs)}


class StatisticalCriticAgent(BaseAgent):
    """Workflow node: reads experiment_plan + experiment_results, writes statistical_results,
    critic_feedback and verdict. Refuses anything that was not actually executed."""

    name = "statistical_critic"
    failure_on_error = FailureState.STATISTICAL_ANALYSIS_FAILED

    def __init__(self, critic: StatisticalCritic | None = None, llm: LLMClient | None = None,
                 n_tests: int = 1, check_repro: bool = False):
        super().__init__()
        self.critic, self.llm = critic or StatisticalCritic(), llm
        self.n_tests, self.check_repro = n_tests, check_repro

    def run(self, state: ResearchState) -> dict:
        result = state.get("experiment_results", {})
        if result.get("result_label") != "executed":
            return {"failure_state": FailureState.STATISTICAL_ANALYSIS_FAILED.value,
                    "error": f"result is labelled {result.get('result_label')!r}, not 'executed'; nothing to analyse"}
        ev = self.critic.evaluate(state["experiment_plan"], result, self.n_tests)
        feedback = {k: ev[k] for k in ("issues", "reasons", "statistical_quality", "verdict")}
        if self.check_repro:
            feedback["reproducibility"] = check_reproducibility(state["experiment_plan"], result)
        if self.llm is not None:
            import json
            feedback["interpretation"] = self.llm.complete(EXPLAIN_SYSTEM, json.dumps(ev, default=str))
        return {"statistical_results": ev["analysis"], "critic_feedback": feedback, "verdict": ev["verdict"]}
```

### Step 6: Experiment Designer (`src/agents/experiment_designer.py`)
**Why:** the hypothesis from M4 is prose ("stronger regularization lowers variance"). Someone has to turn it into a plan the template can run. The LLM proposes; `ExperimentPlan` validates; validation errors go back to the LLM (max 2 retries). This is a feedback loop, not reinforcement learning. `n_seeds` is forced to 20 so the LLM can't pick a flattering small number.
```python
"""Experiment Designer: turns the selected hypothesis into a validated, machine-runnable plan.

The LLM proposes design choices; pydantic validation (ExperimentPlan) is the gatekeeper.
Validation errors are fed back to the LLM (a feedback loop, not reinforcement learning).
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from src.agents.base import BaseAgent
from src.experiments.protocol import ALLOWED_PARAMS, ExperimentPlan
from src.llm.client import LLMClient, complete_json
from src.schemas.state import FailureState, ResearchState

SYSTEM = (
    "You design small, reproducible scikit-learn experiments that test ONE hypothesis by changing ONE "
    "hyperparameter. Reply with ONLY JSON with keys: null_hypothesis, dataset (iris|wine|breast_cancer|digits), "
    "model (logreg|svm|knn|tree|random_forest), metric (accuracy|f1_macro), independent_variable, "
    "baseline_value, treatment_value, expected_direction (higher|lower), fixed_params (object, may be empty), "
    f"falsification_criteria. Allowed independent_variable per model: {json.dumps({k: sorted(v) for k, v in ALLOWED_PARAMS.items()})}. "
    "Baseline and treatment values must differ."
)


class ExperimentDesigner(BaseAgent):
    name = "experiment_designer"
    failure_on_error = FailureState.CODE_GENERATION_FAILED  # closest existing failure state

    def __init__(self, llm: LLMClient, max_retries: int = 2, n_seeds: int = 20):
        super().__init__()
        self.llm, self.max_retries, self.n_seeds = llm, max_retries, n_seeds

    def run(self, state: ResearchState) -> dict:
        h = state["selected_hypothesis"]
        base = (f"Hypothesis: {h['hypothesis']}\nProposed experiment: {h.get('experiment', '')}\n"
                f"Success metric: {h.get('success_metric', '')}\nFalsification: {h.get('falsification_condition', '')}")
        prompt, last = base, ""
        for _ in range(self.max_retries + 1):
            data = complete_json(self.llm, SYSTEM, prompt)
            try:
                plan = ExperimentPlan(**{**data, "research_question": state["research_question"],
                                         "hypothesis": h["hypothesis"], "n_seeds": self.n_seeds})
                return {"experiment_plan": plan.model_dump()}
            except (ValidationError, TypeError) as exc:
                last = str(exc)
                prompt = base + f"\n\nYour previous plan was invalid:\n{last}\nFix it and reply with ONLY JSON."
        return {"failure_state": self.failure_on_error.value, "error": f"no valid experiment plan: {last[:300]}"}
```

### Step 7: Calibrate the Critic (`src/statistics/calibration.py`)
**Why:** a critic that was never tested might accept noise. This simulates many experiments with a known true difference and measures how often the Critic accepts: with true difference 0 that is the false-accept rate; with a real difference it is the power.
```python
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
```

### Step 8: Wire into the workflow (with M1/M5)
The graph file belongs to M5, so give them this change. **This snippet is not tested here**, so run their routing tests after adding it.
```python
# in build_graph(...): add the nodes
g.add_node("designer", designer)      # ExperimentDesigner(llm)
g.add_node("critic", critic)          # StatisticalCriticAgent(llm=llm)
# ... hypothesis -> designer -> executor -> critic -> END
g.add_conditional_edges("critic", lambda s: "end" if s.get("failure_state") else "synthesis_or_end", ...)
```
`StatisticalCriticAgent` returns `failure_state="STATISTICAL_ANALYSIS_FAILED"` for a `not_executed` result, so the existing failure routing stops the run safely.

### Step 9: Tests and Git
Test files (all in the zip): `tests/statistics/test_significance.py` (known-answer tests, e.g. Holm `[0.01,0.04,0.03] → [0.03,0.06,0.06]`), `tests/experiments/test_experiments.py` (determinism, script-equals-in-process), `tests/agents/test_statistical_critic.py`, `test_experiment_designer.py`, `tests/statistics/test_calibration.py`.

Branches: `feature/experiments-core` (Week 1), `feature/statistics` (Week 2), `feature/agent-statistical-critic`, `feature/agent-experiment-designer`, `feature/critic-calibration` (Week 3). PRs into `develop` only, one reviewer (suggest M4, since you share the result protocol).

## 6. What M4 needs from you (debug loop)
For the "RL feedback loop", the reward must mean more than "did not crash". Agreed definition:
- attempt succeeded ⇔ exit code 0 **and** `parse_result(stdout)` passes
- reward = 1.0 − 0.25 × (attempts − 1) on success, 0 on failure
- the error text fed back to the LLM includes the `ResultError` message, e.g. "baseline and treatment must have the same length"

(This is a feedback-driven loop; it does not train the model, so call it that in the report.)

## 7. Honest limits (put these in your report)
- **Seed-based repeated splits are not independent.** The corrected t-test helps, but it's still an approximation. Say so.
- **Thresholds are conventions** (alpha 0.05, 10 seeds, 0.005 difference, 0.5 effect). Report them and don't tune them after seeing results.
- **`accept` is not "discovery."** The experiments are small and use public toy datasets. Never write "novel finding" unless you can show reproducibility and novelty independently.
- **The calibration uses simulated Gaussian scores**, so it tests the rule's behaviour, not real models.
- **Statistical quality score and LLM interpretation are heuristics.**
- **Reproducibility check** only works for template experiments, and results can differ across library versions or CPUs.
- **No Docker was used in this environment.** The sandbox run path is tested only as a local subprocess; M4 must confirm the same script in the container.

## 8. Viva questions for your part
1. *What is a p-value?* Probability of data at least this extreme if the null hypothesis were true. It is not the probability the hypothesis is true.
2. *Why a paired test?* Both arms use the same splits per seed, so the differences remove split noise.
3. *Why not an ordinary paired t-test?* Repeated random splits overlap, so scores are correlated and p-values come out too small. The Nadeau-Bengio correction compensates.
4. *Why check normality first?* The t-test assumes roughly normal differences; otherwise use Wilcoxon plus a bootstrap CI.
5. *Why report effect size?* With enough runs, a tiny difference is "significant". Effect size and a minimum practical difference say whether it matters.
6. *Why Bonferroni/Holm?* Testing many hypotheses makes some "significant" by chance. Holm controls the family-wise error rate and is never worse than Bonferroni.
7. *Why not let the LLM decide the verdict?* LLMs make arithmetic mistakes and can be persuaded. The deterministic rules are testable.
8. *How do you know the Critic works?* Unit tests with known answers plus the simulation (Section 9).
9. *How do you prevent data leakage?* Split first, and scaling lives inside a Pipeline fitted only on training data.
10. *Is the debug loop reinforcement learning?* No: no policy is learned and the model's weights don't change. It is iterative refinement with execution feedback.

## 9. Real output from running this code (measured here, not invented)

Test suite: `36 passed`.

`python -m src.experiments.demo` (20 seeds each, Bonferroni over 3 tests):

| Experiment | Baseline | Treatment | Diff | Test | p | Bonferroni p | 95% CI | Effect | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer, logreg C 1.0 → 0.01 | 0.9743 | 0.9480 | −0.0263 | corrected t | 0.0036 | 0.0109 | [−0.0429, −0.0097] | −2.20 | reject (significant but *opposite* to the default `expected_direction="higher"`) |
| wine, knn k 5 → 1 | 0.9602 | 0.9583 | −0.0019 | corrected t | 0.924 | 1 | [−0.0419, +0.0382] | −0.06 | reject |
| iris, tree depth 3 → 2 | 0.9511 | 0.9411 | −0.0100 | Wilcoxon | 0.180 | 0.540 | [−0.0211, 0.0000] | −0.42 | reject |

All three re-ran to identical scores (`reproduced=True`). Note the first row: the Critic caught that a real, significant effect went the wrong way for the stated hypothesis. In the real pipeline the Designer sets `expected_direction`; make sure the hypothesis states a direction.

Critic calibration (`python -m src.statistics.calibration`, 1000 simulated experiments each, n=20, noise sd 0.01):

| True difference | Accept rate |
|---|---|
| 0.000 (no effect) | 0.0% |
| 0.005 | 2.7% |
| 0.010 | 13.9% |
| 0.020 | 95.5% |

Interpretation: the rule almost never accepts when there is no effect, but it is conservative: a true difference equal to one noise standard deviation is accepted only 14% of the time. That is a deliberate trade-off (few false accepts, lower power). If your report needs more power, raise `n_seeds` before loosening thresholds.

## 10. Final checklist
- [ ] Contracts agreed with M1, M4, M5 (Section 1)
- [ ] `experiments/` + `statistics/` + two agents merged via PR, tests green
- [ ] Sandbox runs `render_script` output and prints one JSON line (confirmed by M4 in Docker)
- [ ] Critic node wired and routing tested by M5
- [ ] 10 to 20 real experiments run; table of verdicts in the report
- [ ] Calibration results and limits documented
- [ ] Viva answers rehearsed
