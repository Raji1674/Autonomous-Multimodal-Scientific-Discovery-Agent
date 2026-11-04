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
        return {
             "t": None,
             "df": k - 1,
             "p": 1.0 if abs(m) <= EPS else None,
             "ci": (m, m),
              "se": 0.0,
    }
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
