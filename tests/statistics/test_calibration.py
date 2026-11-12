from src.statistics.calibration import simulate_accept_rate


def test_false_accept_rate_is_low_under_the_null():
    r = simulate_accept_rate(0.0, n_sims=300, seed=1)
    assert r["accept_rate"] <= 0.05  # one-directional rule at alpha=0.05 should be well below this


def test_large_effect_is_usually_accepted_and_power_grows_with_effect():
    small = simulate_accept_rate(0.005, n_sims=200, seed=2)["accept_rate"]
    large = simulate_accept_rate(0.03, n_sims=200, seed=2)["accept_rate"]
    assert large > 0.9 and large >= small
