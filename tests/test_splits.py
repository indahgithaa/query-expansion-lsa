import random

import pytest

from qe.corpus import TYPE_ORDER
from qe.splits import SEED_MULTIPLIER, make_split, types_by_doc


def test_split_sizes_and_disjointness(corpus):
    for seed in range(1, 11):
        s = make_split(corpus, seed)
        assert (len(s.collection), len(s.dev), len(s.test)) == (72, 24, 24)
        parts = [set(s.collection), set(s.dev), set(s.test)]
        assert set().union(*parts) == set(corpus.doc_ids)
        assert sum(len(p) for p in parts) == 120  # tidak ada tumpang tindih


def test_every_part_covers_all_eight_types(corpus):
    tbd = types_by_doc(corpus)
    for seed in range(1, 11):
        s = make_split(corpus, seed)
        for part in (s.collection, s.dev, s.test):
            seen = set().union(*(tbd[d] for d in part))
            assert seen == set(TYPE_ORDER)


def test_split_is_deterministic(corpus):
    assert make_split(corpus, 7) == make_split(corpus, 7)
    assert make_split(corpus, 7).dev != make_split(corpus, 8).dev


def test_split_follows_documented_shuffle(corpus):
    s = make_split(corpus, 3)
    shuffled = sorted(corpus.doc_ids)
    random.Random(3 * SEED_MULTIPLIER + s.attempt).shuffle(shuffled)
    assert list(s.collection) == shuffled[:72]
    assert list(s.dev) == shuffled[72:96]
    assert list(s.test) == shuffled[96:120]


def test_partition_of(corpus):
    s = make_split(corpus, 1)
    assert s.partition_of(s.dev[0]) == "dev"
    assert s.partition_of(s.test[0]) == "test"
    assert s.partition_of(s.collection[0]) == "collection"
