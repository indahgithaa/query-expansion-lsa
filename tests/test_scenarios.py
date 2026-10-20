import collections
import hashlib

import pytest

from qe.scenarios import N_PARAPHRASES, paraphrase_index, query_text


def test_paraphrase_index_matches_specification():
    seed, qid = 4, 123
    digest = hashlib.sha256(f"{seed}|{qid}".encode("utf-8")).digest()
    assert paraphrase_index(seed, qid) == int.from_bytes(digest[:8], "big") % 6


def test_paraphrase_index_in_range_and_deterministic():
    for seed in range(1, 11):
        for qid in range(0, 200):
            i = paraphrase_index(seed, qid)
            assert 0 <= i < N_PARAPHRASES
            assert i == paraphrase_index(seed, qid)


def test_paraphrase_index_uses_all_six_slots():
    counts = collections.Counter(paraphrase_index(1, q) for q in range(1200))
    assert set(counts) == set(range(6))
    assert min(counts.values()) > 120  # kira-kira seragam


def test_seed_changes_the_assignment():
    assert any(paraphrase_index(1, q) != paraphrase_index(2, q) for q in range(50))


def test_query_text_tpl_returns_template(corpus, paraphrases):
    qa = corpus.qas[0]
    assert query_text(qa, "TPL", 1, paraphrases, "dev") == qa.question


def test_query_text_para_uses_partition_list(corpus, paraphrases):
    qa = corpus.qas[0]
    i = paraphrase_index(1, qa.qid)
    assert query_text(qa, "PARA", 1, paraphrases, "dev") == paraphrases[qa.qtype]["dev"][i]
    assert query_text(qa, "PARA", 1, paraphrases, "test") == paraphrases[qa.qtype]["test"][i]


def test_dev_and_test_paraphrases_are_disjoint(paraphrases):
    for qtype, parts in paraphrases.items():
        assert not set(parts["dev"]) & set(parts["test"]), qtype


def test_query_text_rejects_unknown_scenario(corpus, paraphrases):
    with pytest.raises(ValueError):
        query_text(corpus.qas[0], "XXX", 1, paraphrases, "dev")
    with pytest.raises(ValueError):
        query_text(corpus.qas[0], "PARA", 1, paraphrases, "collection")
