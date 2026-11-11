"""Perakitan jalur retrieval untuk satu pengulangan (seed). Tidak memakai LLM."""
from dataclasses import dataclass

from . import config
from .bm25 import BM25Index, baseline_weights
from .data import Corpus
from .lsa import LSAModel, build_lsa, lsaqe_weights
from .rm3 import rm3_weights
from .split import split_documents
from .text import remove_stopwords, tokenize

METHODS = ("Baseline", "RM3", "LSA-QE")


@dataclass
class SeedContext:
    seed: int
    parts: dict
    lsa: LSAModel
    indexes: dict[str, BM25Index]
    nosw: dict[str, list[str]]


def prepare_seed(corpus: Corpus, seed: int, k: int = config.LSA_K) -> SeedContext:
    parts = split_documents(corpus, seed)
    lsa_docs = [remove_stopwords(c.tokens) for d in parts["collection"] for c in corpus.chunks[d]]
    lsa = build_lsa(lsa_docs, k=k)
    indexes, nosw = {}, {}
    for d in parts["dev"] + parts["test"]:
        cs = corpus.chunks[d]
        indexes[d] = BM25Index([c.cid for c in cs], [c.tokens for c in cs])
        for c in cs:
            nosw[c.cid] = remove_stopwords(c.tokens)
    return SeedContext(seed, parts, lsa, indexes, nosw)


def query_weights(ctx: SeedContext, doc_id: str, question: str, method: str, params: dict | None = None):
    params = params or {}
    q = remove_stopwords(tokenize(question))
    if method == "Baseline":
        return baseline_weights(q)
    if method == "RM3":
        return rm3_weights(ctx.indexes[doc_id], ctx.nosw, q,
                           params.get("fb_docs", config.RM3_FB_DOCS), params.get("fb_terms", config.RM3_FB_TERMS),
                           params.get("lam", config.RM3_LAMBDA))[0]
    if method == "LSA-QE":
        return lsaqe_weights(ctx.lsa, q, params.get("n_terms", config.LSAQE_N_TERMS),
                             params.get("gamma", config.LSAQE_GAMMA))[0]
    raise ValueError(method)


def retrieve(ctx: SeedContext, doc_id: str, question: str, method: str, params: dict | None = None):
    return ctx.indexes[doc_id].rank(query_weights(ctx, doc_id, question, method, params))
