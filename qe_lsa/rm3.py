"""Metode pembanding RM3 sesuai Subbab 4.9 (Persamaan 2.4–2.5)."""
import math
from collections import Counter

import numpy as np

from . import config
from .bm25 import BM25Index, baseline_weights


def rm3_weights(index: BM25Index, chunk_tokens_nosw: dict[str, list[str]], query_tokens: list[str],
                fb_docs: int = config.RM3_FB_DOCS, fb_terms: int = config.RM3_FB_TERMS,
                lam: float = config.RM3_LAMBDA) -> tuple[dict[str, float], dict]:
    """chunk_tokens_nosw: cid -> token chunk setelah penghapusan stopword."""
    # 4.9.1 umpan balik: chunk teratas Baseline, bobot softmax atas skor Baseline
    ranked, scores = index.rank(baseline_weights(query_tokens))
    fb, fb_scores = ranked[:fb_docs], scores[:fb_docs]
    ex = np.exp(fb_scores - fb_scores.max())
    alpha = ex / ex.sum()

    # 4.9.2 model relevansi P(t|R) = sum_d alpha_d * tf(t,d)/|d|
    p_r: dict[str, float] = {}
    for cid, a in zip(fb, alpha):
        toks = chunk_tokens_nosw[cid]
        if not toks:
            continue
        for term, c in Counter(toks).items():
            p_r[term] = p_r.get(term, 0.0) + a * c / len(toks)

    # 4.9.3 pemotongan fbTerms teratas (tie-break alfabetis), normalisasi ulang, interpolasi
    top = sorted(p_r.items(), key=lambda x: (-x[1], x[0]))[:fb_terms]
    mass = sum(p for _, p in top)
    p_top = {t: p / mass for t, p in top} if mass > 0 else {}
    counts = Counter(query_tokens)
    qlen = len(query_tokens)
    w = {}
    for term in set(counts) | set(p_top):
        q_part = lam * counts[term] / qlen if qlen else 0.0
        w[term] = q_part + (1 - lam) * p_top.get(term, 0.0)
    debug = {"fb": fb, "fb_scores": fb_scores, "alpha": alpha, "p_r_raw": dict(top), "mass": mass, "p_top": p_top}
    return w, debug
