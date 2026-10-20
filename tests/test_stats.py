import numpy as np
import pytest

from qe.stats import holm, stratified_paired_bootstrap, wilcoxon


def test_holm_matches_hand_worked_example():
    p = [0.01, 0.04, 0.03]
    # terurut: 0.01*3=0.03 ; 0.03*2=0.06 ; 0.04*1=0.04 -> monoton: 0.06
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_monotone_and_capped_at_one():
    adj = holm([0.5, 0.6, 0.7, 0.8, 0.9])
    assert all(a == 1.0 for a in adj)
    adj2 = holm([0.001, 0.2])
    assert adj2[0] <= adj2[1]


def test_holm_preserves_input_order():
    # terkecil (0.01) dikali m=2 -> 0.02; terbesar (0.04) dikali 1 -> 0.04
    assert holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])


def test_bootstrap_is_deterministic_for_fixed_seed():
    rng = np.random.default_rng(0)
    a, b = rng.random(200), rng.random(200)
    strata = ["s%d" % (i % 4) for i in range(200)]
    r1 = stratified_paired_bootstrap(a, b, strata, n_resamples=500)
    r2 = stratified_paired_bootstrap(a, b, strata, n_resamples=500)
    assert r1 == r2


def test_bootstrap_point_estimate_is_the_paired_mean_difference():
    a = np.array([1.0, 0.5, 0.25, 1.0])
    b = np.array([0.5, 0.5, 1.0, 0.2])
    r = stratified_paired_bootstrap(a, b, ["x", "x", "y", "y"], n_resamples=200)
    assert r["delta"] == pytest.approx(float((a - b).mean()))
    assert r["n"] == 4 and r["n_strata"] == 2


def test_bootstrap_ci_brackets_a_clear_effect():
    a = np.full(300, 0.8)
    b = np.full(300, 0.3)
    r = stratified_paired_bootstrap(a, b, ["s"] * 300, n_resamples=500)
    assert r["ci_low"] == pytest.approx(0.5) and r["ci_high"] == pytest.approx(0.5)


def test_bootstrap_ci_contains_zero_for_null_effect():
    rng = np.random.default_rng(7)
    a = rng.random(400)
    r = stratified_paired_bootstrap(a, a.copy(), ["s"] * 400, n_resamples=500)
    assert r["ci_low"] <= 0 <= r["ci_high"]


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        stratified_paired_bootstrap([1.0, 2.0], [1.0], ["a", "b"])


def test_wilcoxon_on_identical_inputs_is_not_significant():
    a = [1.0, 0.5, 0.25]
    r = wilcoxon(a, a)
    assert r["n_nonzero"] == 0 and r["p_value"] == 1.0


def test_wilcoxon_detects_a_consistent_shift():
    a = [0.9] * 30
    b = [0.2] * 30
    assert wilcoxon(a, b)["p_value"] < 0.01
