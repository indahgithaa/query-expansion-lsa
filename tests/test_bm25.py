import collections

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from qe.bm25 import BM25Index, rank, rank_of

DOCS = [
    "status gizi pasien anak buruk".split(),
    "data biokimia hemoglobin rendah pasien".split(),
    "tujuan diet pasien adalah mencapai status gizi normal".split(),
    "skrining gizi strongkids pada anak".split(),
    "prinsip diet rendah garam".split(),
]


@pytest.mark.parametrize("query", [
    ["status", "gizi"],
    ["diet"],
    ["pasien"],            # df tinggi -> idf bisa negatif, kena koreksi epsilon
    ["hemoglobin", "anak"],
    ["tidak", "ada"],      # term di luar vocab
])
def test_matches_rank_bm25_reference(query):
    ours = BM25Index(DOCS)
    ref = BM25Okapi(DOCS)
    expected = np.asarray(ref.get_scores(query), dtype=np.float64)
    got = ours.score(collections.Counter(query))
    assert np.allclose(got, expected, atol=1e-12)


def test_idf_formula_and_epsilon_floor():
    idx = BM25Index(DOCS)
    n = len(DOCS)
    df = idx.df["diet"]
    assert idx.idf["diet"] == pytest.approx(np.log(n - df + 0.5) - np.log(df + 0.5))
    # "pasien" ada di 3 dari 5 dokumen -> idf mentah negatif -> diganti 0.25 * rata-rata idf
    assert idx.df["pasien"] == 3
    assert idx.idf["pasien"] == pytest.approx(0.25 * idx.average_idf)


def test_weighted_query_is_linear_in_weights():
    idx = BM25Index(DOCS)
    a = idx.score({"status": 1.0})
    b = idx.score({"gizi": 1.0})
    both = idx.score({"status": 2.0, "gizi": 3.0})
    assert np.allclose(both, 2.0 * a + 3.0 * b)


def test_term_scores_agrees_with_score():
    idx = BM25Index(DOCS)
    assert np.allclose(idx.term_scores("gizi"), idx.score({"gizi": 1.0}))
    assert np.allclose(idx.term_scores("tidakada"), np.zeros(len(DOCS)))


def test_ranking_is_stable_on_ties():
    scores = np.array([1.0, 3.0, 3.0, 0.5, 3.0])
    assert list(rank(scores)) == [1, 2, 4, 0, 3]
    assert rank_of(scores, 4) == 3
    # semua seri -> urutan dokumen dipertahankan
    assert list(rank(np.zeros(4))) == [0, 1, 2, 3]
