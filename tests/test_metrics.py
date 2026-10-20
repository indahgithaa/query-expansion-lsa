import math

import pytest

from qe.metrics import ndcg_at_10, reciprocal_rank, robustness, summarize


def test_reciprocal_rank():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25


def test_ndcg_at_10_single_gold():
    assert ndcg_at_10(1) == 1.0
    assert ndcg_at_10(3) == pytest.approx(1 / math.log2(4))
    assert ndcg_at_10(10) == pytest.approx(1 / math.log2(11))
    assert ndcg_at_10(11) == 0.0


def test_summarize_recall_cutoffs_and_fail_rate():
    s = summarize([1, 2, 5, 11, 40])
    assert s["n"] == 5
    assert s["MRR"] == pytest.approx((1 + 0.5 + 0.2 + 1 / 11 + 0.025) / 5)
    assert s["Recall@1"] == pytest.approx(0.2)
    assert s["Recall@3"] == pytest.approx(0.4)
    assert s["Recall@5"] == pytest.approx(0.6)
    assert s["Recall@10"] == pytest.approx(0.6)
    assert s["Recall@20"] == pytest.approx(0.8)
    assert s["median_rank"] == 5.0
    assert s["fail_rate"] == pytest.approx(0.4)


def test_summarize_ndcg_ignores_ranks_beyond_10():
    assert summarize([11, 12])["nDCG@10"] == 0.0


def test_summarize_handles_empty():
    assert summarize([]) == {"n": 0}


def test_robustness_counts_and_index():
    method = [1, 5, 3, 20]
    base = [2, 2, 3, 9]
    r = robustness(method, base)
    assert (r["improved"], r["unchanged"], r["worsened"]) == (1, 1, 2)
    assert r["robustness_index"] == pytest.approx((1 - 2) / 4)
    assert r["pct_improved"] == pytest.approx(0.25)


def test_robustness_tracks_top10_crossings():
    r = robustness(method := [3, 30], base := [30, 3])
    assert r["entered_top10"] == 1
    assert r["left_top10"] == 1
    assert len(method) == len(base)


def test_robustness_counts_big_drops():
    # rank 1 -> 3 menurunkan RR sebesar 1 - 1/3 = 0.667 < -0.5
    assert robustness([3], [1])["big_drops"] == 1
    assert robustness([2], [1])["big_drops"] == 0


def test_robustness_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        robustness([1, 2], [1])
