import math

import numpy as np
import pytest

from qe.lsa import LSAModel, build_matrix, build_vocab, normalize_candidates

DOCS = [
    "gizi buruk anak marasmus".split(),
    "gizi buruk anak kwashiorkor".split(),
    "anak marasmus kwashiorkor gizi".split(),
    "diet rendah garam hipertensi".split(),
    "diet rendah garam jantung".split(),
    "hipertensi jantung diet garam".split(),
    "sekali 123 456".split(),
]


def test_vocab_drops_hapax_and_pure_numbers():
    vocab, df = build_vocab(DOCS)
    assert "sekali" not in vocab          # df = 1
    assert "123" not in vocab             # angka murni
    assert vocab == sorted(vocab)
    assert df["gizi"] == 3


def test_matrix_cell_weight_formula():
    vocab, df = build_vocab(DOCS)
    X = build_matrix(DOCS, vocab, df).toarray()
    j = vocab.index("gizi")
    n = len(DOCS)
    # tf("gizi", doc0) = 1 -> (1 + ln 1) * ln(N/df) = ln(N/df)
    assert X[0, j] == pytest.approx(math.log(n / df["gizi"]))
    assert X[3, j] == 0.0


def test_matrix_uses_log_damped_tf():
    docs = [["a", "a", "b"], ["a", "b"]]
    vocab, df = build_vocab(docs)
    X = build_matrix(docs, vocab, df).toarray()
    j = vocab.index("a")
    assert X[0, j] == pytest.approx((1 + math.log(2)) * math.log(2 / df["a"]))


def test_term_vectors_are_l2_normalized():
    m = LSAModel.fit(DOCS, k=3)
    norms = np.linalg.norm(m.term_vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert m.term_vectors.shape == (len(m.vocab), 3)


def test_fit_is_deterministic():
    a = LSAModel.fit(DOCS, k=3)
    b = LSAModel.fit(DOCS, k=3)
    assert a.vocab == b.vocab
    assert np.array_equal(a.term_vectors, b.term_vectors)


def test_expansion_excludes_query_terms_and_sums_to_one():
    m = LSAModel.fit(DOCS, k=3)
    e = m.expand(["gizi"], n=3)
    assert "gizi" not in e
    assert sum(e.values()) == pytest.approx(1.0)
    assert len(e) <= 3


def test_expansion_finds_co_occurring_terms():
    m = LSAModel.fit(DOCS, k=3)
    e = m.expand(["marasmus"], n=3)
    assert {"anak", "gizi"} <= set(e)     # tetangga laten di klaster gizi buruk
    assert "hipertensi" not in e


def test_expansion_empty_for_out_of_vocab_query():
    m = LSAModel.fit(DOCS, k=3)
    assert m.expand(["tidakadadivokab"], n=5) == {}
    assert m.expand(["gizi"], n=0) == {}


def test_candidates_are_a_prefix_chain_across_n():
    m = LSAModel.fit(DOCS, k=3)
    cands = m.expand_candidates(["gizi"], 5)
    assert [t for t, _ in cands[:2]] == list(m.expand(["gizi"], 2))
    assert normalize_candidates(cands, 2) == m.expand(["gizi"], 2)


def test_candidate_cosines_are_positive_and_descending():
    m = LSAModel.fit(DOCS, k=3)
    vals = [v for _, v in m.expand_candidates(["diet"], 10)]
    assert all(v > 0 for v in vals)
    assert vals == sorted(vals, reverse=True)


def test_normalize_candidates_breaks_nothing_on_empty():
    assert normalize_candidates([], 5) == {}
    assert normalize_candidates([("a", 1.0), ("b", 3.0)], 2) == pytest.approx({"a": 0.25, "b": 0.75})
