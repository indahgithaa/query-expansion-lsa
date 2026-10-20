"""Split per dokumen 72 koleksi / 24 dev / 24 test (Tahap 1)."""

import dataclasses
import random

from .corpus import TYPE_ORDER

N_COLLECTION = 72
N_DEV = 24
N_TEST = 24
SEED_MULTIPLIER = 1_000_003
MAX_ATTEMPTS = 10_000


@dataclasses.dataclass(frozen=True)
class Split:
    seed: int
    attempt: int
    collection: tuple
    dev: tuple
    test: tuple

    def partition_of(self, doc_id: str) -> str:
        if doc_id in set(self.dev):
            return "dev"
        if doc_id in set(self.test):
            return "test"
        return "collection"


def types_by_doc(corpus):
    out = {d: set() for d in corpus.doc_ids}
    for qa in corpus.qas:
        out[qa.doc_id].add(qa.qtype)
    return out


def _covers_all_types(doc_ids, tbd) -> bool:
    seen = set()
    for d in doc_ids:
        seen |= tbd[d]
    return seen >= set(TYPE_ORDER)


def make_split(corpus, seed: int, tbd=None,
               n_collection: int = N_COLLECTION, n_dev: int = N_DEV, n_test: int = N_TEST):
    """Split pertama (attempt terkecil) yang ketiga bagiannya memuat kedelapan tipe."""
    tbd = tbd if tbd is not None else types_by_doc(corpus)
    ordered = sorted(corpus.doc_ids)
    for attempt in range(MAX_ATTEMPTS):
        shuffled = list(ordered)
        random.Random(seed * SEED_MULTIPLIER + attempt).shuffle(shuffled)
        collection = shuffled[:n_collection]
        dev = shuffled[n_collection:n_collection + n_dev]
        test = shuffled[n_collection + n_dev:n_collection + n_dev + n_test]
        if all(_covers_all_types(p, tbd) for p in (collection, dev, test)):
            return Split(seed=seed, attempt=attempt,
                         collection=tuple(collection), dev=tuple(dev), test=tuple(test))
    raise RuntimeError(f"seed {seed}: tidak ada split valid dalam {MAX_ATTEMPTS} percobaan")
