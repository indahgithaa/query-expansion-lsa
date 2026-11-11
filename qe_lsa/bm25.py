"""Retrieval BM25 Okapi berbobot sesuai Subbab 4.8 (Persamaan 2.1–2.3)."""
import math
from collections import Counter

import numpy as np

from . import config


class BM25Index:
    """Indeks satu laporan. Teks diindeks utuh (tanpa penghapusan stopword, Tabel 4.7)."""

    def __init__(self, chunk_ids: list[str], chunk_tokens: list[list[str]],
                 k1: float = config.BM25_K1, b: float = config.BM25_B, epsilon: float = config.BM25_EPSILON):
        self.chunk_ids = list(chunk_ids)
        self.n = len(chunk_ids)
        self.doc_len = np.array([len(t) for t in chunk_tokens], dtype=float)
        self.avgdl = float(self.doc_len.sum() / self.n)
        self.denom = k1 * (1 - b + b * self.doc_len / self.avgdl)  # penyebut normalisasi panjang
        self.k1 = k1

        tf: dict[str, dict[int, int]] = {}
        for i, toks in enumerate(chunk_tokens):
            for term, c in Counter(toks).items():
                tf.setdefault(term, {})[i] = c
        self.df = {t: len(p) for t, p in tf.items()}

        raw_idf = {t: math.log((self.n - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}
        self.mean_idf = sum(raw_idf.values()) / len(raw_idf)  # termasuk nilai negatif
        self.idf = {t: (epsilon * self.mean_idf if v < 0 else v) for t, v in raw_idf.items()}

        # daftar posting: term -> (indeks chunk, kontribusi BM25)
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, p in tf.items():
            idx = np.fromiter(p.keys(), dtype=np.int64)
            f = np.fromiter(p.values(), dtype=float)
            self.postings[term] = (idx, self.idf[term] * f * (k1 + 1) / (f + self.denom[idx]))

    def bm25(self, term: str, i: int) -> float:
        if term not in self.postings:
            return 0.0
        idx, val = self.postings[term]
        hit = np.nonzero(idx == i)[0]
        return float(val[hit[0]]) if hit.size else 0.0

    def score(self, weights: dict[str, float]) -> np.ndarray:
        """Persamaan 2.3: skor(d, w) = sum_t w_t * BM25(t, d)."""
        s = np.zeros(self.n)
        for term in sorted(weights):
            w = weights[term]
            if w > 0 and term in self.postings:
                idx, val = self.postings[term]
                s[idx] += w * val
        return s

    def rank(self, weights: dict[str, float]) -> tuple[list[str], np.ndarray]:
        """Urutan menurun dengan pengurutan stabil; skor sama mengikuti urutan pengidentifikasi."""
        s = self.score(weights)
        order = np.argsort(-s, kind="stable")
        return [self.chunk_ids[i] for i in order], s[order]


def baseline_weights(query_tokens: list[str]) -> dict[str, float]:
    """Bobot Baseline: jumlah kemunculan term pada pertanyaan."""
    return {t: float(c) for t, c in Counter(query_tokens).items()}
