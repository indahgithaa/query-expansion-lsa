import numpy as np
import pytest

from qe.engine import Engine, combine, expansion_for_n
from qe.splits import make_split


@pytest.fixture(scope="module")
def engine(corpus, paraphrases):
    return Engine(corpus, paraphrases)


@pytest.fixture(scope="module")
def units(engine, corpus):
    split = make_split(corpus, 1)
    return engine.prepare_units(1, split.dev[:3], "dev")


def test_combine_adds_scaled_expansion():
    w = combine({"a": 1.0, "b": 2.0}, {"b": 0.25, "c": 0.75}, scale=4.0)
    assert w == pytest.approx({"a": 1.0, "b": 3.0, "c": 3.0})


def test_combine_does_not_mutate_input():
    base = {"a": 1.0}
    combine(base, {"b": 1.0}, 2.0)
    assert base == {"a": 1.0}


def test_expansion_for_n_truncates_and_renormalizes():
    cands = [("a", 0.6), ("b", 0.3), ("c", 0.1)]
    assert expansion_for_n(cands, 2) == pytest.approx({"a": 2 / 3, "b": 1 / 3})


def test_units_cover_both_scenarios(engine, corpus, units):
    assert {u.scenario for u in units} == {"TPL", "PARA"}
    tpl = {u.qid for u in units if u.scenario == "TPL"}
    para = {u.qid for u in units if u.scenario == "PARA"}
    assert tpl == para


def test_gold_index_points_at_gold_chunk(engine, corpus, units):
    for u in units:
        qa = corpus.qa_by_id[u.qid]
        assert corpus.documents[u.doc_id].chunks[u.gold_index].chunk_id == qa.gold_chunk_id


def test_para_text_differs_from_template(engine, corpus, units):
    for u in units:
        qa = corpus.qa_by_id[u.qid]
        if u.scenario == "PARA":
            assert u.text != qa.question
        else:
            assert u.text == qa.question


def test_ranks_are_within_document_bounds(engine, corpus, units):
    for u in units:
        n = len(corpus.documents[u.doc_id].chunks)
        for r in (engine.rank_b1(u), engine.rank_rm3(u)):
            assert 1 <= r <= n


def test_gamma_zero_reduces_each_method_to_its_base(engine, units):
    model = type("M", (), {})()
    for u in units:
        exp = {"zzzterm": 1.0}
        assert engine.rank_lsa_qe(u, exp, 0.0) == engine.rank_b1(u)
        assert engine.rank_rm3_lsa(u, exp, 0.0) == engine.rank_rm3(u)


def test_empty_expansion_reduces_each_method_to_its_base(engine, units):
    for u in units:
        assert engine.rank_lsa_qe(u, {}, 2.0) == engine.rank_b1(u)
        assert engine.rank_rm3_lsa(u, {}, 2.0) == engine.rank_rm3(u)


def test_lsa_qe_scaling_matches_the_specification(engine, units):
    u = units[0]
    exp = engine.fit_lsa([u.doc_id], k=5).expand(u.tokens, 5)
    expected = combine(u.base_weights, exp, 2.0 * len(u.tokens))
    scores = engine.index(u.doc_id).score(expected)
    assert engine.rank_lsa_qe(u, exp, 2.0) == int(np.nonzero(np.argsort(-scores, kind="stable") == u.gold_index)[0][0]) + 1


def test_rm3_lsa_scaling_matches_the_specification(engine, units):
    u = units[0]
    exp = engine.fit_lsa([u.doc_id], k=5).expand(u.tokens, 5)
    expected = combine(u.rm3_weights, exp, 2.0 * engine.lam)
    scores = engine.index(u.doc_id).score(expected)
    assert engine.rank_rm3_lsa(u, exp, 2.0) == int(np.nonzero(np.argsort(-scores, kind="stable") == u.gold_index)[0][0]) + 1


def test_index_is_cached_per_document(engine, units):
    a = engine.index(units[0].doc_id)
    assert engine.index(units[0].doc_id) is a


def test_lsa_trained_only_on_given_documents(engine, corpus):
    split = make_split(corpus, 1)
    model = engine.fit_lsa(split.collection[:4], k=5)
    held_out_terms = set()
    for c in corpus.chunks_of(split.collection[:4]):
        held_out_terms |= set(c.tokens)
    assert set(model.vocab) <= held_out_terms
