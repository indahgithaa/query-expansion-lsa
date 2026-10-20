"""BM25 Okapi per laporan (Tahap 2).

Setara `rank_bm25.BM25Okapi` dengan k1=1.5, b=0.75, epsilon=0.25:
    idf(t)  = ln((N - df + 0.5) / (df + 0.5)), idf negatif -> 0.25 * rata-rata idf
    bm25(t,c) = idf(t) * tf(t,c) * (k1 + 1) / (tf(t,c) + k1 * (1 - b + b * |c| / avgdl))
Skor query berbobot: score(c) = sum_t w_t * bm25(t, c).
"""

import math

import numpy as np

K1 = 1.5
B = 0.75
EPSILON = 0.25


class BM25Index:
    """Indeks BM25 untuk satu laporan. Urutan dokumen = urutan `docs` yang diberikan."""

    def __init__(self, docs, k1: float = K1, b: float = B, epsilon: float = EPSILON):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.docs = [list(d) for d in docs]
        self.n = len(self.docs)
        self.doc_len = np.array([len(d) for d in self.docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.sum() / self.n) if self.n else 0.0

        freqs = []
        df = {}
        for d in self.docs:
            f = {}
            for t in d:
                f[t] = f.get(t, 0) + 1
            freqs.append(f)
            for t in f:
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.freqs = freqs

        idf = {}
        idf_sum = 0.0
        negatives = []
        for t, freq in df.items():
            v = math.log(self.n - freq + 0.5) - math.log(freq + 0.5)
            idf[t] = v
            idf_sum += v
            if v < 0:
                negatives.append(t)
        self.average_idf = idf_sum / len(idf) if idf else 0.0
        eps = self.epsilon * self.average_idf
        for t in negatives:
            idf[t] = eps
        self.idf = idf

        # Denominator per dokumen tidak bergantung pada term kecuali lewat tf.
        self._len_norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl) if self.n else None

        # Kontribusi BM25 per term, disimpan jarang: term -> (indeks dokumen, nilai).
        self._postings = {}
        for t in df:
            idx = np.array([i for i, f in enumerate(freqs) if t in f], dtype=np.int64)
            tf = np.array([freqs[i][t] for i in idx], dtype=np.float64)
            val = idf[t] * tf * (self.k1 + 1) / (tf + self._len_norm[idx])
            self._postings[t] = (idx, val)

    def term_scores(self, term: str) -> np.ndarray:
        """Vektor BM25 term terhadap seluruh dokumen (nol bila term tidak ada)."""
        out = np.zeros(self.n, dtype=np.float64)
        p = self._postings.get(term)
        if p is not None:
            out[p[0]] = p[1]
        return out

    def score(self, weights) -> np.ndarray:
        """score(c) = sum_t w_t * bm25(t, c) untuk mapping term -> bobot."""
        out = np.zeros(self.n, dtype=np.float64)
        for t, w in weights.items():
            p = self._postings.get(t)
            if p is not None and w:
                out[p[0]] += w * p[1]
        return out


def rank(scores: np.ndarray) -> np.ndarray:
    """Urut menurun dengan sort stabil; seri mengikuti urutan dokumen (chunk_id)."""
    return np.argsort(-scores, kind="stable")


def rank_of(scores: np.ndarray, index: int) -> int:
    """Peringkat 1-based dokumen `index` pada ranking stabil."""
    order = rank(scores)
    return int(np.nonzero(order == index)[0][0]) + 1
