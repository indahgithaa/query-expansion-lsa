"""Penelusuran manual untuk Bab 4 (Perancangan).

Menjalankan seluruh tahapan pipeline pada korpus ilustratif berukuran kecil dan
menuliskan setiap nilai antara ke `results/manual_trace_bab4.json`, sehingga
angka pada subbab Perhitungan Manual berasal dari kode yang sama dengan sistem
(aturan tetap no. 4) dan tidak mengutip teks klinis pasien (aturan tetap no. 5).

Korpus di bawah ini disusun sendiri untuk keperluan ilustrasi. Ukuran k, n, dan
parameter RM3 diperkecil agar tabel muat di halaman; nilai yang dikunci untuk
sistem sebenarnya ada di `configs/locked_config.json`.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from qe.bm25 import BM25Index, rank  # noqa: E402
from qe.engine import combine  # noqa: E402
from qe.lsa import LSAModel, build_matrix, build_vocab  # noqa: E402
from qe.metrics import ndcg_at_10, reciprocal_rank  # noqa: E402
from qe.normalize import normalize_text  # noqa: E402
from qe.rm3 import baseline_weights, rm3_weights  # noqa: E402
from qe.tokenize import strip_stopwords, tokenize_doc, tokenize_query  # noqa: E402

# Parameter versi ilustrasi (sistem memakai k=300, n=10, gamma=2.0; RM3 5/30/0,5).
K_LATENT = 3
N_EXPANSION = 3
GAMMA = 2.0
RM3_FEEDBACK = 2
RM3_TERMS = 3
RM3_LAMBDA = 0.5

QUERY = "Apa tujuan diet pada pasien ini?"

# Potongan laporan koleksi: hanya dipakai untuk melatih LSA.
COLLECTION = [
    ("K1", "Tujuan diet adalah memberikan energi dan protein sesuai kebutuhan."),
    ("K2", "Tujuan diet adalah memenuhi kebutuhan energi pasien."),
    ("K3", "Diet protein tinggi diberikan dengan protein nabati dan energi cukup."),
    ("K4", "Tujuan pemberian diet adalah memperbaiki asupan protein."),
    ("K5", "Prinsip diet dilakukan dalam bentuk makanan lunak."),
    ("K6", "Prinsip pemberian makanan lunak dengan porsi kecil."),
    ("K7", "Makanan diberikan dalam porsi kecil dan sering."),
    ("K8", "Status gizi ditentukan berdasarkan indeks massa tubuh."),
    ("K9", "Status gizi dinilai dengan indeks massa tubuh dan antropometri."),
    ("K10", "Penilaian status gizi memakai pengukuran antropometri."),
]

# Potongan laporan target: korpus yang diperingkat BM25.
TARGET = [
    ("C1", "Prinsip pemberian makanan dalam bentuk lunak dengan porsi kecil."),
    ("C2", "Tujuan diet pasien ditetapkan berdasarkan hasil pengkajian."),
    ("C3", "Diet memberikan energi dan protein sesuai kebutuhan harian."),
    ("C4", "Syarat pemberian cairan disesuaikan dengan kondisi saluran cerna."),
    ("C5", "Pemantauan asupan energi dilakukan setiap hari dan energi dicatat."),
    ("C6", "Status gizi ditentukan berdasarkan indeks massa tubuh."),
]
GOLD = "C3"


def r4(x):
    return round(float(x), 4)


def main():
    out = {
        "catatan": "Korpus ilustratif, bukan teks laporan pasien.",
        "parameter": {
            "k_latent": K_LATENT, "n_expansion": N_EXPANSION, "gamma": GAMMA,
            "rm3_n_feedback": RM3_FEEDBACK, "rm3_n_terms": RM3_TERMS,
            "rm3_lambda": RM3_LAMBDA,
            "bm25": {"k1": 1.5, "b": 0.75, "epsilon": 0.25},
        },
    }

    # --- Tahap 1: praproses ---------------------------------------------------
    q_norm = normalize_text(QUERY)
    q_tokens = tokenize_query(q_norm)
    out["praproses"] = {
        "query_asli": QUERY,
        "query_ternormalisasi": q_norm,
        "query_token_mentah": tokenize_doc(q_norm),
        "query_token_akhir": q_tokens,
        "target": [
            {"id": cid, "teks": txt, "token": tokenize_doc(normalize_text(txt)),
             "panjang": len(tokenize_doc(normalize_text(txt)))}
            for cid, txt in TARGET
        ],
        "koleksi": [
            {"id": cid, "teks": txt,
             "token_tanpa_stopword": strip_stopwords(tokenize_doc(normalize_text(txt)))}
            for cid, txt in COLLECTION
        ],
    }

    # --- Tahap 2: BM25 dan B1 -------------------------------------------------
    target_ids = [cid for cid, _ in TARGET]
    target_tokens = [tokenize_doc(normalize_text(t)) for _, t in TARGET]
    index = BM25Index(target_tokens)
    gold_index = target_ids.index(GOLD)

    tf_table = []
    for t in q_tokens:
        row = {"term": t, "df": index.df.get(t, 0), "idf": r4(index.idf.get(t, 0.0))}
        row["tf"] = {target_ids[i]: int(index.freqs[i].get(t, 0)) for i in range(index.n)}
        row["bm25"] = {target_ids[i]: r4(v) for i, v in enumerate(index.term_scores(t))}
        tf_table.append(row)

    base_w = baseline_weights(q_tokens)
    scores_b1 = index.score(base_w)
    order_b1 = rank(scores_b1)
    rank_b1 = int(np.nonzero(order_b1 == gold_index)[0][0]) + 1

    out["bm25"] = {
        "N": index.n,
        "panjang_dokumen": {target_ids[i]: int(index.doc_len[i]) for i in range(index.n)},
        "avgdl": r4(index.avgdl),
        "average_idf": r4(index.average_idf),
        "penyebut_normalisasi": {target_ids[i]: r4(index._len_norm[i]) for i in range(index.n)},
        "per_term": tf_table,
    }
    out["b1"] = {
        "bobot": {t: r4(v) for t, v in base_w.items()},
        "skor": {target_ids[i]: r4(v) for i, v in enumerate(scores_b1)},
        "peringkat": [target_ids[i] for i in order_b1],
        "peringkat_gold": rank_b1,
    }

    # --- Tahap 3: LSA ---------------------------------------------------------
    coll_tokens = [strip_stopwords(tokenize_doc(normalize_text(t))) for _, t in COLLECTION]
    vocab, df = build_vocab(coll_tokens)
    X = build_matrix(coll_tokens, vocab, df)
    n_coll = len(coll_tokens)

    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=K_LATENT, random_state=0)
    svd.fit(X)
    tv_raw = svd.components_.T * svd.singular_values_

    model = LSAModel.fit(coll_tokens, k=K_LATENT)

    idxs = [model.term_index[t] for t in q_tokens if t in model.term_index]
    centroid = model.term_vectors[idxs].mean(axis=0).astype(np.float64)
    c_norm = float(np.linalg.norm(centroid))
    centroid_unit = centroid / c_norm
    cos = np.asarray(model.term_vectors @ centroid_unit.astype(np.float32), dtype=np.float64)

    cand = model.expand_candidates(q_tokens, N_EXPANSION)
    expansion = model.expand(q_tokens, N_EXPANSION)

    out["lsa"] = {
        "n_dokumen_koleksi": n_coll,
        "vocab": vocab,
        "df": {t: int(df[t]) for t in vocab},
        "idf": {t: r4(math.log(n_coll / df[t])) for t in vocab},
        "matriks": [
            {"dokumen": COLLECTION[i][0],
             "sel": {vocab[j]: r4(X[i, j]) for j in range(len(vocab)) if X[i, j] != 0}}
            for i in range(n_coll)
        ],
        "nilai_singular": [r4(v) for v in svd.singular_values_],
        "rasio_varians": [r4(v) for v in svd.explained_variance_ratio_],
        "vektor_term_mentah": {vocab[j]: [r4(v) for v in tv_raw[j]] for j in range(len(vocab))},
        "vektor_term_ternormalisasi": {
            vocab[j]: [r4(v) for v in model.term_vectors[j]] for j in range(len(vocab))},
        "term_query_di_vocab": [t for t in q_tokens if t in model.term_index],
        "centroid_mentah": [r4(v) for v in centroid],
        "norma_centroid": r4(c_norm),
        "centroid_ternormalisasi": [r4(v) for v in centroid_unit],
        "cosine": {vocab[j]: r4(cos[j]) for j in range(len(vocab))},
        "kandidat_terpilih": [[t, r4(v)] for t, v in cand],
        "bobot_ekspansi": {t: r4(v) for t, v in expansion.items()},
    }

    # --- Tahap 4: LSA-QE ------------------------------------------------------
    scale_lsa = GAMMA * len(q_tokens)
    w_lsa = combine(base_w, expansion, scale_lsa)
    scores_lsa = index.score(w_lsa)
    order_lsa = rank(scores_lsa)
    rank_lsa = int(np.nonzero(order_lsa == gold_index)[0][0]) + 1

    extra = {}
    for t in expansion:
        extra[t] = {
            "df": index.df.get(t, 0), "idf": r4(index.idf.get(t, 0.0)),
            "bm25": {target_ids[i]: r4(v) for i, v in enumerate(index.term_scores(t))},
        }

    out["lsa_qe"] = {
        "skala": r4(scale_lsa),
        "bobot": {t: r4(v) for t, v in sorted(w_lsa.items())},
        "bm25_term_ekspansi": extra,
        "skor": {target_ids[i]: r4(v) for i, v in enumerate(scores_lsa)},
        "peringkat": [target_ids[i] for i in order_lsa],
        "peringkat_gold": rank_lsa,
    }

    # --- Tahap 5: pembanding RM3 dan RM3+LSA ----------------------------------
    fb_tokens = [strip_stopwords(t) for t in target_tokens]
    w_rm3, rm_terms, fb_docs = rm3_weights(
        index, fb_tokens, q_tokens, lam=RM3_LAMBDA,
        n_feedback=RM3_FEEDBACK, n_terms=RM3_TERMS, renormalize=True)
    scores_rm3 = index.score(w_rm3)
    order_rm3 = rank(scores_rm3)
    rank_rm3 = int(np.nonzero(order_rm3 == gold_index)[0][0]) + 1

    fb_scores = scores_b1[fb_docs]
    e = np.exp(fb_scores - fb_scores.max())
    fb_w = e / e.sum()

    w_both = combine(w_rm3, expansion, GAMMA * RM3_LAMBDA)
    scores_both = index.score(w_both)
    order_both = rank(scores_both)
    rank_both = int(np.nonzero(order_both == gold_index)[0][0]) + 1

    out["rm3"] = {
        "dokumen_feedback": [target_ids[i] for i in fb_docs],
        "skor_feedback": [r4(v) for v in fb_scores],
        "bobot_softmax": [r4(v) for v in fb_w],
        "model_relevansi": {t: r4(v) for t, v in sorted(rm_terms.items(), key=lambda kv: -kv[1])},
        "bobot": {t: r4(v) for t, v in sorted(w_rm3.items())},
        "skor": {target_ids[i]: r4(v) for i, v in enumerate(scores_rm3)},
        "peringkat": [target_ids[i] for i in order_rm3],
        "peringkat_gold": rank_rm3,
    }
    out["rm3_lsa"] = {
        "skala": r4(GAMMA * RM3_LAMBDA),
        "bobot": {t: r4(v) for t, v in sorted(w_both.items())},
        "skor": {target_ids[i]: r4(v) for i, v in enumerate(scores_both)},
        "peringkat": [target_ids[i] for i in order_both],
        "peringkat_gold": rank_both,
    }

    # --- Tahap 6: metrik ------------------------------------------------------
    out["metrik"] = {}
    for name, r in (("B1", rank_b1), ("RM3", rank_rm3),
                    ("LSA-QE", rank_lsa), ("RM3+LSA", rank_both)):
        out["metrik"][name] = {
            "peringkat_gold": r,
            "RR": r4(reciprocal_rank(r)),
            "Recall@1": 1 if r <= 1 else 0,
            "Recall@3": 1 if r <= 3 else 0,
            "Recall@5": 1 if r <= 5 else 0,
            "nDCG@10": r4(ndcg_at_10(r)),
        }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "results", "manual_trace_bab4.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("ditulis:", os.path.normpath(path))
    for m, v in out["metrik"].items():
        print(f"  {m:9s} peringkat gold = {v['peringkat_gold']}  RR = {v['RR']}")


if __name__ == "__main__":
    main()
