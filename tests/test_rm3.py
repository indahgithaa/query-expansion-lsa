import collections

import numpy as np
import pytest

from qe.bm25 import BM25Index, rank
from qe.rm3 import baseline_weights, relevance_model, rm3_weights, softmax

DOCS = [
    "status gizi pasien anak buruk gizi".split(),
    "data biokimia hemoglobin rendah pasien".split(),
    "tujuan diet pasien mencapai status gizi normal".split(),
    "skrining gizi strongkids pada anak".split(),
    "prinsip diet rendah garam".split(),
]
FB = [list(d) for d in DOCS]  # tanpa stopword removal di tes ini


def test_softmax_sums_to_one_and_is_shift_invariant():
    x = np.array([10.0, 12.0, 11.0])
    s = softmax(x)
    assert s.sum() == pytest.approx(1.0)
    assert np.allclose(s, softmax(x + 1000.0))


def test_baseline_weights_are_query_term_counts():
    assert baseline_weights(["gizi", "status", "gizi"]) == {"gizi": 2, "status": 1}


def test_relevance_model_truncates_and_renormalizes():
    idx = BM25Index(DOCS)
    rm, fb_docs = relevance_model(idx, FB, baseline_weights(["gizi"]), n_feedback=3, n_terms=4)
    assert len(rm) == 4
    assert sum(rm.values()) == pytest.approx(1.0)
    assert len(fb_docs) == 3
    assert fb_docs == list(rank(idx.score({"gizi": 1}))[:3])


def test_relevance_model_without_renormalization_keeps_raw_mass():
    idx = BM25Index(DOCS)
    raw, _ = relevance_model(idx, FB, baseline_weights(["gizi"]), n_terms=4, renormalize=False)
    assert sum(raw.values()) < 1.0


def test_relevance_model_matches_manual_computation():
    idx = BM25Index(DOCS)
    base = baseline_weights(["gizi"])
    scores = idx.score(base)
    order = list(rank(scores)[:2])
    w = softmax(scores[order])
    manual = collections.defaultdict(float)
    for wd, di in zip(w, order):
        for t, c in collections.Counter(FB[di]).items():
            manual[t] += wd * c / len(FB[di])
    rm, _ = relevance_model(idx, FB, base, n_feedback=2, n_terms=50, renormalize=False)
    for t, v in rm.items():
        assert v == pytest.approx(manual[t])


def test_ties_broken_alphabetically():
    # Dua dokumen teratas identik, tiap term punya P(t|R) sama besar.
    docs = [["aa", "zz", "qq"], ["aa", "zz", "qq"], ["bb", "cc"], ["bb", "cc"], ["bb", "cc"]]
    idx = BM25Index(docs)
    rm, fb_docs = relevance_model(idx, docs, {"qq": 1}, n_feedback=2, n_terms=1, renormalize=False)
    assert fb_docs == [0, 1]
    assert list(rm) == ["aa"]  # aa, qq, zz seri -> alfabetis


def test_rm3_interpolation_formula():
    idx = BM25Index(DOCS)
    q = ["gizi", "status"]
    w, rm, _ = rm3_weights(idx, FB, q, lam=0.5, n_terms=5)
    for t in set(w) | set(rm):
        expected = 0.5 * (q.count(t) / len(q)) + 0.5 * rm.get(t, 0.0)
        assert w[t] == pytest.approx(expected)


def test_rm3_weights_sum_to_one_when_renormalized():
    idx = BM25Index(DOCS)
    w, _, _ = rm3_weights(idx, FB, ["gizi", "status"], lam=0.5)
    assert sum(w.values()) == pytest.approx(1.0)


def test_lambda_one_reduces_to_normalized_baseline():
    idx = BM25Index(DOCS)
    q = ["gizi", "status"]
    w, _, _ = rm3_weights(idx, FB, q, lam=1.0)
    assert w == pytest.approx({"gizi": 0.5, "status": 0.5})
