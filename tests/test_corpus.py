import collections

import pytest

from qe.corpus import (EXPECTED_CHUNKS, EXPECTED_DOCS, EXPECTED_QA, QUESTION_TYPES,
                       TYPE_ORDER, make_chunk_id)


def test_tahap0_sanity_counts(corpus):
    assert len(corpus.documents) == EXPECTED_DOCS
    assert corpus.n_chunks == EXPECTED_CHUNKS
    assert len(corpus.qas) == EXPECTED_QA


def test_doc_ids_follow_sorted_normalized_filenames(corpus):
    names = [corpus.documents[d].filename for d in corpus.doc_ids]
    assert names == sorted(names)
    assert corpus.doc_ids[0] == "D001"
    assert corpus.doc_ids[-1] == f"D{EXPECTED_DOCS:03d}"


def test_chunk_id_format():
    assert make_chunk_id("D007", 12, 3) == "D007:P0012:C003"


def test_chunk_ids_are_unique_and_sorted_within_document(corpus):
    seen = set()
    for doc in corpus.documents.values():
        ids = [c.chunk_id for c in doc.chunks]
        assert ids == sorted(ids)
        assert not seen & set(ids)
        seen |= set(ids)
    assert len(seen) == EXPECTED_CHUNKS


def test_every_qa_has_exactly_one_gold_chunk(corpus):
    for qa in corpus.qas:
        doc = corpus.documents[qa.doc_id]
        matches = [c for c in doc.chunks if c.chunk_id == qa.gold_chunk_id]
        assert len(matches) == 1, qa.qid


def test_dedup_keeps_smallest_id_per_file_question(corpus):
    keys = [(qa.filename, qa.question) for qa in corpus.qas]
    assert len(keys) == len(set(keys))


def test_question_types_cover_the_eight_templates(corpus):
    assert set(QUESTION_TYPES.values()) == set(TYPE_ORDER)
    assert len(TYPE_ORDER) == 8
    counts = collections.Counter(qa.qtype for qa in corpus.qas)
    assert set(counts) == set(TYPE_ORDER)
    assert sum(counts.values()) == EXPECTED_QA


def test_qa_types_match_their_question_text(corpus):
    for qa in corpus.qas:
        assert QUESTION_TYPES[qa.question] == qa.qtype


def test_chunk_text_is_normalized(corpus):
    for doc in corpus.documents.values():
        for chunk in doc.chunks:
            assert "\u2060" not in chunk.text
            assert "#U" not in chunk.text


def test_qas_in_filters_by_document(corpus):
    subset = corpus.doc_ids[:3]
    qas = corpus.qas_in(subset)
    assert qas and all(qa.doc_id in set(subset) for qa in qas)
