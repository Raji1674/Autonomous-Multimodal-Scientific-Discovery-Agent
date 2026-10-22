import numpy as np
import pytest
from scipy import stats

from src.statistics.significance import (bootstrap_ci, check_normality, cohens_dz, corrected_resampled_ttest,
                                         hedges_gz, holm_adjust, paired_analysis, rank_biserial_paired)


def test_cohens_dz_known_value():
    assert cohens_dz([1, 2, 3]) == pytest.approx(2.0)  # mean 2, sd 1


def test_hedges_correction_known_value():
    assert hedges_gz([1, 2, 3]) == pytest.approx(2.0 * (1 - 3 / 7))


def test_dz_zero_variance_cases():
    assert cohens_dz([0.0, 0.0, 0.0]) == 0.0
    assert cohens_dz([0.1, 0.1, 0.1]) is None  # undefined, not infinity


def test_rank_biserial_known_value():
    # |d| ranks = [1.5, 3, 1.5]; positive ranks 4.5, negative 1.5 -> (4.5-1.5)/6
    assert rank_biserial_paired([1, 2, -1]) == pytest.approx(0.5)


def test_holm_known_values():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    d = np.random.default_rng(1).normal(0.02, 0.01, 30)
    lo, hi = bootstrap_ci(d, seed=3)
    assert lo < d.mean() < hi
    assert (lo, hi) == bootstrap_ci(d, seed=3)


def test_corrected_test_is_more_conservative_than_naive():
    rng = np.random.default_rng(0)
    base = rng.normal(0.9, 0.01, 20)
    treat = base + rng.normal(0.004, 0.004, 20)
    naive = stats.ttest_rel(treat, base).pvalue
    corrected = corrected_resampled_ttest(treat - base, n_train=105, n_test=45)["p"]
    assert corrected >= naive


def test_paired_analysis_matches_scipy_naive_reference():
    rng = np.random.default_rng(2)
    base = rng.normal(0.9, 0.01, 20)
    treat = base + rng.normal(0.01, 0.005, 20)
    a = paired_analysis(base, treat)  # no split sizes -> ordinary paired t
    assert a["test"] == "paired_t"
    assert a["p"] == pytest.approx(stats.ttest_rel(treat, base).pvalue)


def test_paired_analysis_picks_wilcoxon_for_non_normal_differences():
    d = np.array([0.001] * 17 + [0.2, 0.25, 0.3])  # heavily skewed
    base = np.full(20, 0.8)
    a = paired_analysis(base, base + d, n_train=100, n_test=50)
    assert a["normality"]["normal"] is False
    assert a["test"] == "wilcoxon_signed_rank"
    assert a["effect_size_name"] == "rank_biserial"


def test_identical_scores_need_no_test():
    a = paired_analysis([0.9] * 10, [0.9] * 10, n_train=100, n_test=50)
    assert a["test"] == "none_identical_scores" and a["p"] == 1.0


def test_too_few_runs_is_insufficient():
    assert paired_analysis([0.9, 0.8], [0.95, 0.85])["test"] == "insufficient_data"


def test_unpaired_lengths_rejected():
    with pytest.raises(ValueError):
        paired_analysis([0.9, 0.8, 0.7], [0.9, 0.8])


def test_normality_not_testable_for_constant_values():
    assert check_normality([1.0, 1.0, 1.0])["normal"] is None


def test_corrected_resampled_ttest_constant_nonzero_differences():
    result = corrected_resampled_ttest(
        [0.1, 0.1, 0.1, 0.1, 0.1],
        n_train=70,
        n_test=30,
    )

    assert result["se"] == 0.0
    assert result["ci"] == (0.1, 0.1)
    assert result["p"] is None  # not 0.0, because the t-test is undefined for constant values
