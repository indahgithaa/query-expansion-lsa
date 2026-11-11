"""Model LSA (Subbab 4.6) dan query expansion LSA (Subbab 4.7)."""
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from . import config
from .bm25 import baseline_weights
from .text import is_pure_number


@dataclass
class LSAModel:
    vocab: list[str]
    term_index: dict[str, int]
    vectors: np.ndarray  # |V| x k, baris ternormalisasi L2
    singular_values: np.ndarray
    stats: dict


def build_vocabulary(docs_tokens: list[list[str]], min_df: int = config.LSA_MIN_DF):
    """Subbab 4.6.1: df >= 2, bukan angka murni, urut alfabetis."""
    df = Counter()
    for toks in docs_tokens:
        df.update(set(toks))
    vocab = sorted(t for t, d in df.items() if d >= min_df and not is_pure_number(t))
    stats = {
        "n_tokens": sum(len(t) for t in docs_tokens),
        "n_unique": len(df),
        "n_df1": sum(1 for d in df.values() if d < min_df),
        "n_numeric_removed": sum(1 for t, d in df.items() if d >= min_df and is_pure_number(t)),
        "n_vocab": len(vocab),
    }
    return vocab, df, stats


def weight_matrix(docs_tokens: list[list[str]], vocab: list[str], df: Counter) -> sp.csr_matrix:
    """Subbab 4.6.2, Persamaan 2.6: a_ij = (1 + ln tf_ij) * ln(N / df_j)."""
    n = len(docs_tokens)
    index = {t: j for j, t in enumerate(vocab)}
    idf = np.array([math.log(n / df[t]) for t in vocab])
    rows, cols, vals = [], [], []
    for i, toks in enumerate(docs_tokens):
        for term, c in Counter(toks).items():
            j = index.get(term)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append((1.0 + math.log(c)) * idf[j])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float64)


def build_lsa(docs_tokens: list[list[str]], k: int = config.LSA_K) -> LSAModel:
    """docs_tokens: token setiap chunk data koleksi setelah penghapusan stopword."""
    vocab, df, stats = build_vocabulary(docs_tokens)
    A = weight_matrix(docs_tokens, vocab, df)
    svd = TruncatedSVD(n_components=k, random_state=config.LSA_RANDOM_STATE)
    svd.fit(A)
    vs = svd.components_.T * svd.singular_values_  # V_k Sigma_k
    norms = np.linalg.norm(vs, axis=1, keepdims=True)
    vs = np.divide(vs, norms, out=np.zeros_like(vs), where=norms > 0)
    stats.update({
        "shape": A.shape, "nnz": int(A.nnz), "density": A.nnz / (A.shape[0] * A.shape[1]),
        "frobenius_sq": float(A.multiply(A).sum()),
    })
    stats["energy"] = float((svd.singular_values_ ** 2).sum() / stats["frobenius_sq"])
    return LSAModel(vocab, {t: j for j, t in enumerate(vocab)}, vs, svd.singular_values_, stats)


def expand(model: LSAModel, query_tokens: list[str], n_terms: int = config.LSAQE_N_TERMS):
    """Subbab 4.7.1–4.7.2. Mengembalikan daftar (term, cosine) terpilih."""
    # Gambar 4.18: setiap token pertanyaan yang ada di kosakata dijumlahkan (token ganda dihitung ganda),
    # lalu dibagi banyaknya token tersebut. Lihat DECISIONS.md (A1).
    q_terms = [t for t in query_tokens if t in model.term_index]
    if not q_terms:
        return []
    centroid = model.vectors[[model.term_index[t] for t in q_terms]].mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return []
    cos = model.vectors @ (centroid / norm)
    exclude = set(query_tokens)
    cand = [(model.vocab[j], float(cos[j])) for j in np.nonzero(cos > 0)[0] if model.vocab[j] not in exclude]
    cand.sort(key=lambda x: (-x[1], x[0]))
    return cand[:n_terms]


def lsaqe_weights(model: LSAModel, query_tokens: list[str], n_terms: int = config.LSAQE_N_TERMS,
                  gamma: float = config.LSAQE_GAMMA) -> tuple[dict[str, float], list]:
    """Subbab 4.7.3, Persamaan 2.10–2.11: w_t = count_q(t) + gamma*|q|*e_t."""
    w = baseline_weights(query_tokens)
    selected = expand(model, query_tokens, n_terms)
    total = sum(c for _, c in selected)
    scale = gamma * len(query_tokens)
    for term, c in selected:
        w[term] = w.get(term, 0.0) + scale * c / total
    return w, selected
