"""Mesin retrieval: menyatukan korpus, skenario, dan keempat metode (Tahap 2-3).

Semua metode adalah query expansion; ekspansi hanya mengubah bobot query.
    B1        w = count_q
    RM3       w = lam * count_q/|q| + (1-lam) * P(t|R)
    LSA-QE    w = B1  + gamma * |q| * E(q)
    RM3+LSA   w = RM3 + gamma * lam * E(q)
"""

import dataclasses

import numpy as np

from .bm25 import BM25Index, rank
from .lsa import LSAModel, normalize_candidates
from .rm3 import LAMBDA, N_FEEDBACK, N_TERMS, baseline_weights, rm3_weights
from .scenarios import query_text
from .tokenize import strip_stopwords, tokenize_query

METHODS = ("B1", "RM3", "LSA-QE", "RM3+LSA")


@dataclasses.dataclass
class Unit:
    """Satu unit evaluasi (seed, skenario, qid)."""

    seed: int
    scenario: str
    qid: int
    doc_id: str
    qtype: str
    text: str
    tokens: tuple
    gold_index: int
    base_weights: dict
    rm3_weights: dict
    rm3_terms: dict
    feedback_docs: tuple


class Engine:
    def __init__(self, corpus, paraphrases, lam: float = LAMBDA,
                 n_feedback: int = N_FEEDBACK, n_terms: int = N_TERMS,
                 rm3_renormalize: bool = True):
        self.corpus = corpus
        self.paraphrases = paraphrases
        self.lam = lam
        self.n_feedback = n_feedback
        self.n_terms = n_terms
        self.rm3_renormalize = rm3_renormalize
        self._index_cache = {}
        self._fb_cache = {}

    # --- indeks per laporan (tidak bergantung seed, aman untuk di-cache) ---

    def index(self, doc_id: str) -> BM25Index:
        idx = self._index_cache.get(doc_id)
        if idx is None:
            chunks = self.corpus.documents[doc_id].chunks
            idx = BM25Index([c.tokens for c in chunks])
            self._index_cache[doc_id] = idx
        return idx

    def feedback_tokens(self, doc_id: str):
        toks = self._fb_cache.get(doc_id)
        if toks is None:
            chunks = self.corpus.documents[doc_id].chunks
            toks = [strip_stopwords(c.tokens) for c in chunks]
            self._fb_cache[doc_id] = toks
        return toks

    # --- penyiapan unit ---

    def make_unit(self, qa, seed: int, scenario: str, partition: str) -> Unit:
        text = query_text(qa, scenario, seed, self.paraphrases, partition)
        tokens = tokenize_query(text)
        doc = self.corpus.documents[qa.doc_id]
        gold_index = doc.index_of(qa.gold_chunk_id)
        idx = self.index(qa.doc_id)
        fb = self.feedback_tokens(qa.doc_id)
        base = baseline_weights(tokens)
        w_rm3, rm_terms, fb_docs = rm3_weights(
            idx, fb, tokens, lam=self.lam, n_feedback=self.n_feedback,
            n_terms=self.n_terms, renormalize=self.rm3_renormalize)
        return Unit(seed=seed, scenario=scenario, qid=qa.qid, doc_id=qa.doc_id,
                    qtype=qa.qtype, text=text, tokens=tuple(tokens),
                    gold_index=gold_index, base_weights=base, rm3_weights=w_rm3,
                    rm3_terms=rm_terms, feedback_docs=tuple(fb_docs))

    def prepare_units(self, seed: int, doc_ids, partition: str, scenarios=("TPL", "PARA")):
        qas = self.corpus.qas_in(doc_ids)
        return [self.make_unit(qa, seed, sc, partition) for sc in scenarios for qa in qas]

    # --- pemeringkatan ---

    def _rank(self, unit: Unit, weights) -> int:
        scores = self.index(unit.doc_id).score(weights)
        order = rank(scores)
        return int(np.nonzero(order == unit.gold_index)[0][0]) + 1

    def rank_b1(self, unit: Unit) -> int:
        return self._rank(unit, unit.base_weights)

    def rank_rm3(self, unit: Unit) -> int:
        return self._rank(unit, unit.rm3_weights)

    def rank_lsa_qe(self, unit: Unit, expansion, gamma: float) -> int:
        return self._rank(unit, combine(unit.base_weights, expansion,
                                        gamma * len(unit.tokens)))

    def rank_rm3_lsa(self, unit: Unit, expansion, gamma: float) -> int:
        return self._rank(unit, combine(unit.rm3_weights, expansion, gamma * self.lam))

    # --- LSA per seed ---

    def fit_lsa(self, collection_doc_ids, k: int) -> LSAModel:
        docs = [strip_stopwords(c.tokens) for c in self.corpus.chunks_of(collection_doc_ids)]
        return LSAModel.fit(docs, k=k)

    def expansion_candidates(self, model: LSAModel, units, max_n: int):
        return [model.expand_candidates(u.tokens, max_n) for u in units]


def combine(weights, expansion, scale: float) -> dict:
    """w + scale * E, dengan E sudah dinormalkan berjumlah 1."""
    out = dict(weights)
    for t, v in expansion.items():
        out[t] = out.get(t, 0.0) + scale * v
    return out


def expansion_for_n(candidates, n: int) -> dict:
    return normalize_candidates(candidates, n)
