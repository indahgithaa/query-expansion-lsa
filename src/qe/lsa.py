"""LSA — ekspansi global (Deerwester et al., 1990).

Dilatih per seed HANYA pada chunk dokumen koleksi (72 dokumen).
Bobot sel = (1 + ln tf) * ln(N/df); TruncatedSVD(n_components=k, random_state=0);
vektor term = baris V_k Sigma_k yang dinormalisasi L2.
"""

import math

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

MIN_DF = 2


def build_vocab(docs_tokens, min_df: int = MIN_DF):
    """Vocab dari token tanpa stopword: df >= min_df, buang angka murni."""
    df = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    vocab = sorted(t for t, c in df.items() if c >= min_df and not t.isdigit())
    return vocab, {t: df[t] for t in vocab}


def build_matrix(docs_tokens, vocab, df):
    """Matriks (n_dokumen x n_term) dengan bobot (1 + ln tf) * ln(N/df)."""
    index = {t: j for j, t in enumerate(vocab)}
    n = len(docs_tokens)
    idf = np.array([math.log(n / df[t]) for t in vocab], dtype=np.float64)
    rows, cols, vals = [], [], []
    for i, toks in enumerate(docs_tokens):
        tf = {}
        for t in toks:
            j = index.get(t)
            if j is not None:
                tf[j] = tf.get(j, 0) + 1
        for j, c in tf.items():
            rows.append(i)
            cols.append(j)
            vals.append((1.0 + math.log(c)) * idf[j])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float64)


class LSAModel:
    """Ruang laten LSA beserta vektor term yang sudah dinormalisasi L2."""

    def __init__(self, vocab, term_vectors):
        self.vocab = list(vocab)
        self.term_index = {t: j for j, t in enumerate(self.vocab)}
        self.term_vectors = term_vectors

    @classmethod
    def fit(cls, docs_tokens, k: int, min_df: int = MIN_DF, random_state: int = 0):
        vocab, df = build_vocab(docs_tokens, min_df)
        X = build_matrix(docs_tokens, vocab, df)
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        svd.fit(X)
        # X ~ U_k S_k V_k^T, sklearn components_ = V_k^T -> baris V_k S_k:
        tv = svd.components_.T * svd.singular_values_
        norms = np.linalg.norm(tv, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return cls(vocab, (tv / norms).astype(np.float32))

    def expand_candidates(self, query_tokens, max_n: int):
        """Kandidat ekspansi terurut: (term, cosine) menurun, seri alfabetis.

        Dipisah dari `expand` supaya satu perhitungan dipakai ulang untuk beberapa
        nilai n, karena daftar untuk n kecil adalah prefix dari daftar n besar.
        """
        idxs = [self.term_index[t] for t in query_tokens if t in self.term_index]
        if not idxs or max_n <= 0:
            return []
        centroid = self.term_vectors[idxs].mean(axis=0).astype(np.float64)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return []
        cos = np.asarray(self.term_vectors @ (centroid / norm).astype(np.float32), dtype=np.float64)
        qset = {t for t in query_tokens if t in self.term_index}
        cand = [(self.vocab[j], float(cos[j])) for j in np.nonzero(cos > 0)[0]
                if self.vocab[j] not in qset]
        cand.sort(key=lambda kv: (-kv[1], kv[0]))
        return cand[:max_n]

    def expand(self, query_tokens, n: int):
        """E(q): centroid vektor term query -> cosine ke vocab -> n term teratas.

        Term query dibuang; hanya cosine > 0 yang diambil; bobot = cosine yang
        dinormalkan sehingga jumlahnya 1.
        """
        return normalize_candidates(self.expand_candidates(query_tokens, n), n)


def normalize_candidates(candidates, n: int):
    """Ambil n kandidat teratas dan normalkan bobotnya sehingga berjumlah 1."""
    top = candidates[:n]
    total = sum(v for _, v in top)
    if total <= 0:
        return {}
    return {t: v / total for t, v in top}
