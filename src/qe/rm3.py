"""RM3 — ekspansi lokal / pseudo-relevance feedback (Lavrenko & Croft, 2001).

Parameter ditetapkan a priori (tidak di-tuning): 5 chunk feedback, 30 term, lambda = 0.5.
"""

import collections

import numpy as np

from .bm25 import rank

N_FEEDBACK = 5
N_TERMS = 30
LAMBDA = 0.5


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    e = np.exp(x - x.max())
    return e / e.sum()


def relevance_model(index, fb_tokens, base_weights, n_feedback: int = N_FEEDBACK,
                    n_terms: int = N_TERMS, renormalize: bool = True):
    """P(t|R) terpotong n_terms teratas.

    `fb_tokens[i]` = token chunk ke-i tanpa stopword. Bobot dokumen feedback =
    softmax(skor BM25). Seri pada pemotongan diurutkan alfabetis.
    """
    scores = index.score(base_weights)
    order = rank(scores)[:n_feedback]
    weights = softmax(scores[order])

    probs = collections.defaultdict(float)
    for di, wd in zip(order, weights):
        toks = fb_tokens[int(di)]
        if not toks:
            continue
        inv_len = 1.0 / len(toks)
        for t, c in collections.Counter(toks).items():
            probs[t] += float(wd) * c * inv_len

    top = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[:n_terms]
    if renormalize:
        total = sum(v for _, v in top)
        if total > 0:
            top = [(t, v / total) for t, v in top]
    return dict(top), [int(i) for i in order]


def rm3_weights(index, fb_tokens, query_tokens, lam: float = LAMBDA,
                n_feedback: int = N_FEEDBACK, n_terms: int = N_TERMS,
                renormalize: bool = True):
    """w_t = lam * count_q(t)/|q| + (1 - lam) * P(t|R)."""
    base = baseline_weights(query_tokens)
    rm, fb_docs = relevance_model(index, fb_tokens, base, n_feedback, n_terms, renormalize)
    nq = len(query_tokens)
    out = collections.defaultdict(float)
    if nq and lam:
        for t, c in collections.Counter(query_tokens).items():
            out[t] += lam * c / nq
    if lam != 1.0:
        for t, p in rm.items():
            out[t] += (1.0 - lam) * p
    return dict(out), rm, fb_docs


def baseline_weights(query_tokens):
    """B1: w_t = jumlah kemunculan t di query."""
    return dict(collections.Counter(query_tokens))
