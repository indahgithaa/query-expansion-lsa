"""Penelusuran manual untuk Bab 4 (Perancangan).

Menjalankan pipeline sebenarnya pada SATU unit evaluasi nyata dari korpus, lalu
menuliskan setiap nilai antara ke `results/manual_trace_bab4.json`. Seluruh angka
pada subbab Perhitungan Manual berasal dari berkas itu (aturan tetap no. 4), dan
dihitung oleh modul `src/qe/` yang sama dengan yang dipakai Tahap 2 sampai 5.

Unit yang dipakai dipilih dari partisi **dev seed 1** supaya partisi test tidak
disentuh (aturan tetap no. 3). Unit ini dipilih karena memperlihatkan mekanisme
metode dengan jelas: kata kunci pertanyaan tidak muncul sama sekali pada chunk
acuan, sehingga Baseline gagal, sedangkan ekspansi LSA menemukan istilah
jawabannya.
Contoh dipilih untuk keperluan penjelasan, bukan sebagai bukti kinerja; bukti
kinerja ada pada hasil pengujian seluruh unit.
"""

import collections
import glob
import hashlib
import json
import math
import os
import re
import sys
import unicodedata

import numpy as np
from scipy import stats as sps
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io  # noqa: E402
from qe.bm25 import rank  # noqa: E402
from qe.corpus import TYPE_ORDER, load_corpus, load_paraphrases  # noqa: E402
from qe.engine import Engine, combine, expansion_for_n  # noqa: E402
from qe.lsa import build_matrix, build_vocab  # noqa: E402
from qe.metrics import ndcg_at_10, reciprocal_rank, robustness, summarize  # noqa: E402
from qe.scenarios import paraphrase_index  # noqa: E402
from qe.splits import SEED_MULTIPLIER, make_split, types_by_doc  # noqa: E402
from qe.stats import (BOOTSTRAP_SEED, N_RESAMPLES, holm,  # noqa: E402
                      stratified_paired_bootstrap, wilcoxon)
from qe.normalize import normalize_text  # noqa: E402
from qe.tokenize import stopwords, strip_stopwords, tokenize_doc  # noqa: E402

QID = 63
SEED = 1
PARTITION = "dev"
SCENARIO = "TPL"
N_DIM_TAMPIL = 3    # dimensi laten yang dicuplik untuk tabel
N_PESAING = 3       # chunk pesaing teratas yang ditampilkan
N_TOP = 5           # panjang daftar peringkat yang ditampilkan
N_KONTEKS = 5       # chunk teratas yang disusun menjadi konteks jawaban

# Template prompt sesuai rancangan Subbab Pembentukan Jawaban: instruksi,
# konteks, pertanyaan. Lapisan generasi belum dijalankan; yang ditelusuri di sini
# adalah masukan yang dibentuk pipeline retrieval untuknya.
PROMPT_TEMPLATE = (
    "Jawab pertanyaan hanya berdasarkan konteks di bawah ini. "
    "Apabila konteks tidak memuat informasi yang ditanyakan, "
    "nyatakan bahwa informasi tersebut tidak tersedia.\n\n"
    "Konteks:\n{konteks}\n\n"
    "Pertanyaan: {pertanyaan}\n"
    "Jawaban:"
)


def r4(x):
    return round(float(x), 4)


def r6(x):
    return round(float(x), 6)


N_TOP_B1_LENGKAP = 20   # chunk teratas Baseline yang ditabelkan penuh
N_KANDIDAT_LENGKAP = 20  # kandidat ekspansi sebelum dipotong menjadi n
N_SUBMATRIKS = 4        # chunk koleksi pada cuplikan matriks LSA
N_UNIT_METRIK = 10      # unit dev yang dipakai memperagakan agregasi dan uji
N_BOOT_TAMPIL = 3       # ulangan bootstrap yang ditampilkan


def _raw_records(doc_filename):
    """Rekaman mentah (sebelum normalisasi) milik satu laporan, kunci (page, chunk)."""
    for p in glob.glob(os.path.join(io.DATA, "chunks", "*.chunks.json")):
        if normalize_text(os.path.basename(p)) == doc_filename:
            with open(p, encoding="utf-8") as fh:
                return {(int(r["page_number"]), int(r["chunk_index"])): r
                        for r in json.load(fh)}
    raise KeyError(doc_filename)


def _contoh_normalisasi(corpus):
    """Cari contoh nyata tiap langkah normalisasi pada korpus mentah."""
    hitung = {"nfkc": 0, "hash_u": 0, "word_joiner": 0}
    contoh = {}
    for d in corpus.doc_ids:
        doc = corpus.documents[d]
        raw = _raw_records(doc.filename)
        for c in doc.chunks:
            t = raw[(c.page_number, c.chunk_index)]["text"]
            if "⁠" in t:
                hitung["word_joiner"] += t.count("⁠")
                for m in re.finditer("⁠", t):
                    a, b = max(0, m.start() - 30), m.start() + 30
                    sebelum = re.findall(r"\w+", t[a:b].lower())
                    sesudah = re.findall(r"\w+", normalize_text(t[a:b]).lower())
                    lama = contoh.get("word_joiner")
                    # utamakan kemunculan di tengah kata, karena di situlah token pecah
                    if lama is None or (sebelum != sesudah and not lama["pecah"]):
                        contoh["word_joiner"] = {
                            "chunk_id": c.chunk_id,
                            "sebelum": t[a:b].replace("⁠", "<U+2060>"),
                            "sesudah": normalize_text(t[a:b]),
                            "token_sebelum": sebelum, "token_sesudah": sesudah,
                            "pecah": sebelum != sesudah,
                        }
            if re.search(r"#U[0-9a-fA-F]{4}", t):
                hitung["hash_u"] += 1
            n = unicodedata.normalize("NFKC", t)
            if n != t:
                hitung["nfkc"] += 1
                if "nfkc" not in contoh:
                    pos = [i for i, ch in enumerate(t)
                           if unicodedata.normalize("NFKC", ch) != ch]
                    a, b = max(0, pos[0] - 20), pos[0] + 40
                    contoh["nfkc"] = {
                        "chunk_id": c.chunk_id,
                        "sebelum": t[a:b],
                        "sesudah": unicodedata.normalize("NFKC", t[a:b]),
                        "token_sebelum": re.findall(r"\w+", t[a:b].lower()),
                        "token_sesudah": re.findall(
                            r"\w+", unicodedata.normalize("NFKC", t[a:b]).lower()),
                        "karakter": [{"karakter": ch, "kode": f"U+{ord(ch):04X}",
                                      "nama": unicodedata.name(ch, "?"),
                                      "menjadi": unicodedata.normalize("NFKC", ch)}
                                     for ch in dict.fromkeys(t[a:b])
                                     if unicodedata.normalize("NFKC", ch) != ch][:6],
                    }
    return hitung, contoh


def trace_lengkap(*, corpus, engine, split, u, doc, gold, index, scores_b1, order_b1,
                  coll_tokens, vocab, df, X, svd, tv, tvn, vi, model, cand, expansion,
                  w, scores, order, fb, alpha, scores_rm3, cfg, k, n, gamma):
    """Nilai antara tambahan untuk tabel-tabel rinci Perhitungan Manual."""
    out = {}

    # --- dataset ------------------------------------------------------------
    with open(os.path.join(io.DATA, "dataset", "dataset.json"), encoding="utf-8") as fh:
        raw_qa = json.load(fh)
    rec = next(r for r in raw_qa if int(r["id"]) == u.qid)
    per_tipe = collections.Counter(q.qtype for q in corpus.qas)
    panjang = sorted(len(d.chunks) for d in corpus.documents.values())
    out["dataset"] = {
        "dokumen": len(corpus.doc_ids), "chunk": corpus.n_chunks,
        "qa_mentah": len(raw_qa), "qa": len(corpus.qas),
        "qa_duplikat": len(raw_qa) - len(corpus.qas),
        "chunk_per_dokumen": {"min": panjang[0], "maks": panjang[-1],
                              "median": float(np.median(panjang)),
                              "rata": r4(np.mean(panjang))},
        "qa_per_tipe": {t: per_tipe[t] for t in TYPE_ORDER},
        "kolom_qa": list(rec.keys()),
        "contoh_qa": {"id": rec["id"], "page_number": rec["page_number"],
                      "chunk_index": rec["chunk_index"], "question": rec["question"],
                      "answer": rec["answer"]},
        "kolom_chunk": ["pdf_name", "page_number", "chunk_index", "text"],
        "contoh_chunk": {"page_number": gold.page_number,
                         "chunk_index": gold.chunk_index,
                         "chunk_id": gold.chunk_id},
    }

    # --- normalisasi --------------------------------------------------------
    hitung, contoh = _contoh_normalisasi(corpus)
    out["normalisasi"] = {"jumlah": hitung, "contoh": contoh}

    # --- pembagian data -----------------------------------------------------
    tbd = types_by_doc(corpus)
    cak = {}
    for nama, ids in (("koleksi", split.collection), ("validasi", split.dev),
                      ("pengujian", split.test)):
        tipe = set()
        for d in ids:
            tipe |= tbd[d]
        cak[nama] = {"dokumen": len(ids),
                     "chunk": sum(len(corpus.documents[d].chunks) for d in ids),
                     "qa": len(corpus.qas_in(ids)),
                     "tipe_tercakup": len(tipe),
                     "dokumen_pedoman": sum(1 for d in ids if "pedoman" in tbd[d])}
    out["pembagian"] = {
        "seed": split.seed, "attempt": split.attempt,
        "seed_acak": split.seed * SEED_MULTIPLIER + split.attempt,
        "bagian": cak,
        "validasi": list(split.dev), "pengujian": list(split.test),
        "d029_di": split.partition_of(u.doc_id),
        "parafrasa_indeks": paraphrase_index(split.seed, u.qid),
        "parafrasa_teks": engine.paraphrases[u.qtype]["dev"][paraphrase_index(split.seed, u.qid)],
        "sha256_awal": hashlib.sha256(f"{split.seed}|{u.qid}".encode()).hexdigest()[:16],
    }

    # --- praproses chunk acuan ---------------------------------------------
    raw = _raw_records(doc.filename)[(gold.page_number, gold.chunk_index)]["text"]
    tok_gold = list(gold.tokens)
    sw = stopwords()
    tf_gold = collections.Counter(tok_gold)
    tanpa_sw = strip_stopwords(tok_gold)
    out["praproses_acuan"] = {
        "teks_mentah_berubah": raw != gold.text,
        "panjang_token": len(tok_gold),
        "token_unik": len(tf_gold),
        "token_awal": tok_gold[:25],
        "tf": sorted(([t, c, t in sw] for t, c in tf_gold.items()),
                     key=lambda r: (-r[1], r[0])),
        "panjang_tanpa_stopword": len(tanpa_sw),
        "unik_tanpa_stopword": len(set(tanpa_sw)),
        "stopword_terbuang": len(tok_gold) - len(tanpa_sw),
    }

    # --- BM25 lengkap -------------------------------------------------------
    lens = index.doc_len
    neg = [t for t in index.df
           if math.log(index.n - index.df[t] + 0.5) - math.log(index.df[t] + 0.5) < 0]
    t_max = max(index.df, key=lambda t: (index.df[t], t))
    eps = index.epsilon * index.average_idf
    top20 = list(order_b1[:N_TOP_B1_LENGKAP])
    out["bm25"] = {
        "jumlah_panjang": int(lens.sum()), "N": index.n, "avgdl": r4(index.avgdl),
        "panjang_min": int(lens.min()), "panjang_maks": int(lens.max()),
        "jumlah_term_unik": len(index.df),
        "rata_idf": r4(index.average_idf), "epsilon": index.epsilon,
        "nilai_pengganti": r4(eps),
        "jumlah_idf_negatif": len(neg),
        "contoh_negatif": {"term": t_max, "df": index.df[t_max],
                           "idf_mentah": r4(math.log(index.n - index.df[t_max] + 0.5)
                                            - math.log(index.df[t_max] + 0.5)),
                           "idf_pakai": r4(index.idf[t_max])},
        "term": [{"term": t, "df": int(index.df.get(t, 0)),
                  "idf": r4(index.idf[t]) if t in index.idf else None}
                 for t in list(u.tokens) + [t for t, _ in cand]],
        "teratas": [{"peringkat": p, "chunk_id": doc.chunks[i].chunk_id,
                     "panjang": int(lens[i]), "penyebut": r4(index._len_norm[i]),
                     "acuan": bool(i == u.gold_index),
                     "tf": [int(index.freqs[i].get(t, 0)) for t in u.tokens],
                     "bm25": [r4(index.term_scores(t)[i]) for t in u.tokens],
                     "skor": r4(scores_b1[i])}
                    for p, i in enumerate(top20, 1)],
        "seri": int(len(scores_b1) - len(set(np.round(scores_b1, 10)))),
        "chunk_skor_nol": int((scores_b1 == 0).sum()),
    }

    # --- LSA lengkap --------------------------------------------------------
    n_dok = len(coll_tokens)
    df_all = collections.Counter()
    n_tok = 0
    for toks in coll_tokens:
        n_tok += len(toks)
        df_all.update(set(toks))
    satu = sum(1 for t, c in df_all.items() if c < 2)
    digit = sum(1 for t, c in df_all.items() if c >= 2 and t.isdigit())
    tampil = list(u.tokens) + [t for t, _ in cand]
    idf_l = {t: math.log(n_dok / df[t]) for t in tampil if t in df}

    # cuplikan matriks: chunk koleksi yang memuat paling banyak term tampil
    coll_ids = [c.chunk_id for c in corpus.chunks_of(split.collection)]
    skor_sub = [(len(set(toks) & set(tampil)), -i) for i, toks in enumerate(coll_tokens)]
    pilih = [-s[1] for s in sorted(skor_sub, reverse=True)[:N_SUBMATRIKS]]
    sub = []
    for i in pilih:
        tf_i = collections.Counter(coll_tokens[i])
        sub.append({"chunk_id": coll_ids[i], "panjang": len(coll_tokens[i]),
                    "tf": {t: tf_i.get(t, 0) for t in tampil},
                    "bobot": {t: (r4((1 + math.log(tf_i[t])) * idf_l[t])
                                  if tf_i.get(t, 0) else 0.0) for t in tampil if t in idf_l}})
    sv = svd.singular_values_
    frob = float(X.multiply(X).sum())
    t_raw = "skrining"
    out["lsa"] = {
        "dokumen": n_dok, "token": n_tok, "term_unik": len(df_all),
        "term_df1": satu, "term_digit": digit, "kosakata": len(vocab),
        "term": [{"term": t, "df": int(df[t]), "idf": r4(idf_l[t])}
                 for t in tampil if t in df],
        "submatriks": sub,
        "singular_awal": [r4(x) for x in sv[:10]],
        "singular_akhir": [r4(x) for x in sv[-5:]],
        "energi_k": r4(float((sv ** 2).sum()) / frob),
        "frobenius_kuadrat": r4(frob),
        "vektor_mentah": {"term": t_raw,
                          "tiga_dimensi": [r4(x) for x in tv[vi[t_raw]][:3]],
                          "norma": r4(np.linalg.norm(tv[vi[t_raw]])),
                          "ternormalisasi": [r4(x) for x in tvn[vi[t_raw]][:3]]},
    }

    # --- QE lengkap ---------------------------------------------------------
    qv = [tvn[vi[t]] for t in u.tokens if t in vi]
    jumlah = np.sum(qv, axis=0)
    centroid = np.mean(qv, axis=0)
    cu = centroid / np.linalg.norm(centroid)
    t_cos = cand[0][0]
    v = tvn[vi[t_cos]]
    cand20 = model.expand_candidates(u.tokens, N_KANDIDAT_LENGKAP)
    skala = gamma * len(u.tokens)
    out["qe"] = {
        "vektor_query": {t: [r4(x) for x in tvn[vi[t]][:3]] for t in u.tokens if t in vi},
        "jumlah": [r4(x) for x in jumlah[:3]],
        "centroid": [r4(x) for x in centroid[:3]],
        "norma": r4(np.linalg.norm(centroid)),
        "jumlah_kuadrat_norma": r4(float((centroid ** 2).sum())),
        "centroid_unit": [r4(x) for x in cu[:3]],
        "cosine_contoh": {"term": t_cos,
                          "vektor": [r4(x) for x in v[:3]],
                          "hasil_kali": [r6(a * b) for a, b in zip(v[:3], cu[:3])],
                          "jumlah_tiga": r6(float(np.dot(v[:3], cu[:3]))),
                          "jumlah_300": r4(float(np.dot(v, cu)))},
        "kandidat20": [{"peringkat": p, "term": t, "cosine": r4(c), "terpilih": p <= n}
                       for p, (t, c) in enumerate(cand20, 1)],
        "skala": r4(skala),
        "bobot_akhir": [{"term": t,
                         "count": int(u.base_weights.get(t, 0)),
                         "e": r4(expansion.get(t, 0.0)),
                         "tambahan": r4(skala * expansion.get(t, 0.0)),
                         "w": r4(w[t])}
                        for t in tampil if t in w],
        "jumlah_w": r4(sum(w.values())),
    }

    # --- LSA-QE: pesaing dan pergeseran ------------------------------------
    i_top1 = int(order_b1[0])
    rank_b1_of = {int(i): p for p, i in enumerate(order_b1, 1)}
    rinci_top1 = []
    for t, wt in sorted(w.items(), key=lambda kv: -kv[1]):
        b = float(index.term_scores(t)[i_top1])
        if b:
            rinci_top1.append({"term": t, "bobot": r4(wt),
                               "tf": int(index.freqs[i_top1].get(t, 0)),
                               "bm25": r4(b), "kontribusi": r4(wt * b)})
    rank_lsa_of = {int(i): p for p, i in enumerate(order, 1)}
    out["lsa_qe"] = {
        "pesaing": {"chunk_id": doc.chunks[i_top1].chunk_id,
                    "panjang": int(lens[i_top1]),
                    "rincian": rinci_top1, "skor": r4(scores[i_top1]),
                    "peringkat": rank_lsa_of[i_top1]},
        "teratas10": [{"peringkat": p, "chunk_id": doc.chunks[i].chunk_id,
                       "skor": r4(scores[i]), "peringkat_b1": rank_b1_of[int(i)],
                       "acuan": bool(i == u.gold_index)}
                      for p, i in enumerate(order[:10], 1)],
    }

    # --- RM3 lengkap --------------------------------------------------------
    fb_tok = engine.feedback_tokens(u.doc_id)
    s_fb = scores_b1[fb]
    geser = s_fb - s_fb.max()
    e = np.exp(geser)
    probs = collections.defaultdict(float)
    for di, a in zip(fb, alpha):
        toks = fb_tok[di]
        for t, c in collections.Counter(toks).items():
            probs[t] += float(a) * c / len(toks)
    top30 = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[:cfg["rm3"]["n_terms"]]
    tot30 = sum(v for _, v in top30)
    lam = cfg["rm3"]["lambda"]
    t_rm = "skrining"
    contoh_rm = []
    for di, a in zip(fb, alpha):
        toks = fb_tok[di]
        c = toks.count(t_rm)
        contoh_rm.append({"chunk_id": doc.chunks[di].chunk_id, "alpha": r4(a),
                          "tf": c, "panjang": len(toks), "rasio": r6(c / len(toks)),
                          "kontribusi": r6(float(a) * c / len(toks))})
    rinci_rm3 = []
    for t, wt in sorted(u.rm3_weights.items(), key=lambda kv: -kv[1]):
        b = float(index.term_scores(t)[u.gold_index])
        if b:
            rinci_rm3.append({"term": t, "bobot": r4(wt),
                              "tf": int(index.freqs[u.gold_index].get(t, 0)),
                              "bm25": r4(b), "kontribusi": r4(wt * b)})
    order_rm3 = rank(scores_rm3)
    out["rm3"] = {
        "softmax": [{"chunk_id": doc.chunks[i].chunk_id, "skor": r4(s),
                     "geser": r4(g), "exp": r6(x), "alpha": r4(a),
                     "panjang_tanpa_sw": len(fb_tok[i])}
                    for i, s, g, x, a in zip(fb, s_fb, geser, e, alpha)],
        "jumlah_exp": r6(e.sum()),
        "contoh_term": t_rm, "contoh": contoh_rm,
        "contoh_jumlah": r6(probs[t_rm]),
        "jumlah_term_kandidat": len(probs),
        "massa_30": r4(tot30),
        "model30": [{"peringkat": p, "term": t, "p": r6(v), "p_norm": r4(v / tot30),
                     "w": r4(u.rm3_weights[t])}
                    for p, (t, v) in enumerate(top30, 1)],
        "bobot_query": [{"term": t, "bagian_query": r4(lam / len(u.tokens)),
                         "bagian_model": r4((1 - lam) * u.rm3_terms.get(t, 0.0)),
                         "w": r4(u.rm3_weights[t])} for t in u.tokens],
        "jumlah_w": r4(sum(u.rm3_weights.values())),
        "rincian_gold": rinci_rm3,
        "skor_gold": r4(scores_rm3[u.gold_index]),
        "teratas": [{"peringkat": p, "chunk_id": doc.chunks[i].chunk_id,
                     "skor": r4(scores_rm3[i]), "acuan": bool(i == u.gold_index)}
                    for p, i in enumerate(order_rm3[:5], 1)],
    }

    # --- metrik dan uji atas beberapa unit ---------------------------------
    units = engine.prepare_units(split.seed, split.dev, "dev", scenarios=("TPL",))
    units = sorted(units, key=lambda x: x.qid)[:N_UNIT_METRIK]
    baris = []
    for x in units:
        e_x = expansion_for_n(model.expand_candidates(x.tokens, n), n)
        r_b = engine.rank_b1(x)
        r_r = engine.rank_rm3(x)
        r_l = engine.rank_lsa_qe(x, e_x, gamma)
        baris.append({"qid": x.qid, "doc_id": x.doc_id, "tipe": x.qtype,
                      "chunk": len(corpus.documents[x.doc_id].chunks),
                      "rank": {"Baseline": r_b, "RM3": r_r, "LSA-QE": r_l}})
    met = {}
    for m in ("Baseline", "RM3", "LSA-QE"):
        rk = [b["rank"][m] for b in baris]
        s = summarize(rk)
        met[m] = {"RR": [r4(reciprocal_rank(r)) for r in rk],
                  "nDCG": [r4(ndcg_at_10(r)) for r in rk],
                  "MRR": r4(s["MRR"]), "nDCG@10": r4(s["nDCG@10"]),
                  "Recall@1": r4(s["Recall@1"]), "Recall@5": r4(s["Recall@5"]),
                  "Recall@10": r4(s["Recall@10"]), "median": s["median_rank"],
                  "gagal": r4(s["fail_rate"]),
                  "jumlah_RR": r4(sum(reciprocal_rank(r) for r in rk)),
                  "jumlah_nDCG": r4(sum(ndcg_at_10(r) for r in rk))}
    rob = {}
    for m in ("RM3", "LSA-QE"):
        rob[m] = {kk: (r4(vv) if isinstance(vv, float) else vv) for kk, vv in
                  robustness([b["rank"][m] for b in baris],
                             [b["rank"]["Baseline"] for b in baris]).items()}

    def rr(m):
        return np.array([reciprocal_rank(b["rank"][m]) for b in baris])

    pasangan = (("RM3", "Baseline"), ("LSA-QE", "Baseline"), ("LSA-QE", "RM3"))
    uji = []
    for a, b in pasangan:
        d = rr(a) - rr(b)
        nz = np.nonzero(d)[0]
        absd = np.abs(d[nz])
        peringkat = sps.rankdata(absd)
        wplus = float(peringkat[d[nz] > 0].sum())
        wmin = float(peringkat[d[nz] < 0].sum())
        wx = wilcoxon(rr(a), rr(b))
        bs = stratified_paired_bootstrap(rr(a), rr(b), ["dev-1-TPL"] * len(baris))
        uji.append({"a": a, "b": b, "selisih": [r4(x) for x in d],
                    "delta": r4(d.mean()),
                    "tak_nol": [{"qid": baris[i]["qid"], "d": r4(d[i]),
                                 "abs": r4(abs(d[i])), "peringkat": float(pr),
                                 "tanda": int(np.sign(d[i]))}
                                for i, pr in zip(nz, peringkat)],
                    "W_plus": wplus, "W_minus": wmin,
                    "statistik": wx["statistic"], "p": r6(wx["p_value"]),
                    "n_tak_nol": wx["n_nonzero"],
                    "ci": [r4(bs["ci_low"]), r4(bs["ci_high"])]})
    adj = holm([x["p"] for x in uji])
    for x, pa in zip(uji, adj):
        x["p_holm"] = r6(pa)
    urutan_holm = sorted(range(len(uji)), key=lambda i: uji[i]["p"])
    holm_rinci = []
    berjalan = 0.0
    for j, i in enumerate(urutan_holm):
        kali = (len(uji) - j) * uji[i]["p"]
        berjalan = max(berjalan, kali)
        holm_rinci.append({"urutan": j + 1, "pasangan": f"{uji[i]['a']} - {uji[i]['b']}",
                           "p": uji[i]["p"], "pengali": len(uji) - j,
                           "hasil_kali": r6(kali), "p_holm": r6(min(1.0, berjalan))})

    # tiga ulangan bootstrap pertama, memakai generator yang sama dengan stats.py
    d_lb = rr("LSA-QE") - rr("Baseline")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(baris), size=(N_RESAMPLES, len(baris)))
    boot = [{"ulangan": j + 1, "indeks": [int(i) + 1 for i in idx[j]],
             "rata": r4(d_lb[idx[j]].mean())} for j in range(N_BOOT_TAMPIL)]
    out["agregasi"] = {
        "aturan": f"{N_UNIT_METRIK} unit pertama menurut qid pada dev seed "
                  f"{split.seed} skenario TPL",
        "unit": baris, "metrik": met, "ketahanan": rob,
        "uji": uji, "holm": holm_rinci,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "ulangan": N_RESAMPLES,
                      "contoh": boot,
                      "ci": uji[1]["ci"]},
    }
    return out


def main():
    with open(os.path.join(io.CONFIGS, "locked_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    k = cfg["methods"]["LSA-QE"]["k"]
    n = cfg["methods"]["LSA-QE"]["n"]
    gamma = cfg["methods"]["LSA-QE"]["gamma"]

    corpus = load_corpus(io.DATA)
    paraphrases = load_paraphrases(os.path.join(io.DATA, "paraphrases.json"))
    engine = Engine(corpus, paraphrases,
                    n_feedback=cfg["rm3"]["n_feedback"], n_terms=cfg["rm3"]["n_terms"],
                    lam=cfg["rm3"]["lambda"], rm3_renormalize=cfg["rm3"]["renormalize"])
    split = make_split(corpus, SEED, types_by_doc(corpus))
    units = engine.prepare_units(SEED, getattr(split, PARTITION), PARTITION,
                                 scenarios=(SCENARIO,))
    u = next(x for x in units if x.qid == QID)

    doc = corpus.documents[u.doc_id]
    gold = doc.chunks[u.gold_index]
    index = engine.index(u.doc_id)
    n_coll_chunks = sum(len(corpus.documents[d].chunks) for d in split.collection)

    out = {
        "unit": {
            "qid": u.qid, "seed": SEED, "partisi": PARTITION, "skenario": SCENARIO,
            "tipe_pertanyaan": u.qtype, "doc_id": u.doc_id,
            "chunk_acuan": gold.chunk_id, "indeks_chunk_acuan": u.gold_index,
        },
        "parameter": {
            "k": k, "n": n, "gamma": gamma,
            "bm25": cfg["retrieval"]["bm25"],
            "rm3": {kk: cfg["rm3"][kk] for kk in ("n_feedback", "n_terms", "lambda")},
        },
        "korpus": {
            "dokumen_koleksi": len(split.collection),
            "chunk_koleksi": n_coll_chunks,
            "chunk_laporan_sasaran": len(doc.chunks),
            "panjang_chunk_acuan": len(gold.tokens),
            "avgdl": r4(index.avgdl),
        },
    }

    # --- 1. praproses -------------------------------------------------------
    norm = normalize_text(u.text)
    kecil = norm.lower()
    tok = tokenize_doc(u.text)
    sw = stopwords()
    out["praproses"] = {
        "pertanyaan": u.text,
        "token_sebelum_stopword": tok,
        "token_query": list(u.tokens),
        "panjang_query": len(u.tokens),
        "teks_chunk_acuan": gold.text,
        "stopword_terbuang": [t for t in tok if t in sw],
        "jumlah_stopword": len(sw),
        "langkah": [
            {"no": 1, "langkah": "Teks pertanyaan asli", "hasil": u.text},
            {"no": 2, "langkah": "Normalisasi NFKC, dekode #Uxxxx, buang U+2060",
             "hasil": norm, "berubah": norm != u.text},
            {"no": 3, "langkah": "Konversi ke huruf kecil",
             "hasil": kecil, "berubah": kecil != norm},
            {"no": 4, "langkah": "Tokenisasi dengan pola \\w+",
             "hasil": " | ".join(tok), "berubah": True},
            {"no": 5, "langkah": "Penghapusan stopword",
             "hasil": " | ".join(u.tokens), "berubah": len(tok) != len(u.tokens)},
        ],
    }

    # --- 2. BM25 dan Baseline -----------------------------------------------
    scores_b1 = index.score(u.base_weights)
    order_b1 = rank(scores_b1)
    rank_b1 = int(np.nonzero(order_b1 == u.gold_index)[0][0]) + 1

    out["bm25"] = {
        "N": index.n,
        "average_idf": r4(index.average_idf),
        "penyebut_normalisasi_gold": r4(index._len_norm[u.gold_index]),
        "per_term": [
            {"term": t,
             "df": int(index.df.get(t, 0)),
             "idf": r4(index.idf.get(t, 0.0)),
             "tf_gold": int(index.freqs[u.gold_index].get(t, 0)),
             "bm25_gold": r4(index.term_scores(t)[u.gold_index])}
            for t in u.tokens
        ],
    }
    out["b1"] = {
        "skor_gold": r4(scores_b1[u.gold_index]),
        "peringkat_gold": rank_b1,
        "teratas": [
            {"peringkat": p, "chunk_id": doc.chunks[i].chunk_id,
             "panjang": len(doc.chunks[i].tokens), "skor": r4(scores_b1[i]),
             "teks": doc.chunks[i].text,
             "tf": {t: int(index.freqs[i].get(t, 0)) for t in u.tokens}}
            for p, i in enumerate(order_b1[:N_PESAING], 1)
        ],
    }

    # --- 3. model LSA -------------------------------------------------------
    coll_tokens = [strip_stopwords(c.tokens) for c in corpus.chunks_of(split.collection)]
    vocab, df = build_vocab(coll_tokens)
    X = build_matrix(coll_tokens, vocab, df)
    n_dok = len(coll_tokens)
    vi = {t: j for j, t in enumerate(vocab)}

    svd = TruncatedSVD(n_components=k, random_state=0)
    svd.fit(X)
    tv = svd.components_.T * svd.singular_values_
    norms = np.linalg.norm(tv, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tvn = tv / norms

    model = engine.fit_lsa(split.collection, k)
    cand = model.expand_candidates(u.tokens, n)
    expansion = expansion_for_n(cand, n)

    # satu sel dengan tf > 1 untuk memperagakan faktor (1 + ln tf)
    term_sel = cand[0][0] if cand else vocab[0]
    i_sel, tf_sel = max(
        ((i, toks.count(term_sel)) for i, toks in enumerate(coll_tokens)),
        key=lambda it: it[1])
    idf_sel = math.log(n_dok / df[term_sel])

    centroid = tvn[[vi[t] for t in u.tokens if t in vi]].mean(axis=0)
    c_norm = float(np.linalg.norm(centroid))
    centroid_unit = centroid / c_norm
    cos = tvn @ centroid_unit
    n_kandidat = int(sum(1 for j, t in enumerate(vocab)
                         if cos[j] > 0 and t not in set(u.tokens)))

    tampil = list(u.tokens) + [t for t, _ in cand]
    out["lsa"] = {
        "dokumen": n_dok, "term": len(vocab),
        "nonzero": int(X.nnz),
        "kepadatan": round(X.nnz / (X.shape[0] * X.shape[1]), 6),
        "contoh_sel": {"term": term_sel, "dokumen_ke": int(i_sel), "tf": int(tf_sel),
                       "df": int(df[term_sel]), "idf": r4(idf_sel),
                       "bobot": r4((1 + math.log(tf_sel)) * idf_sel)},
        "nilai_singular": {"sigma_1": r4(svd.singular_values_[0]),
                           "sigma_2": r4(svd.singular_values_[1]),
                           "sigma_3": r4(svd.singular_values_[2]),
                           f"sigma_{k}": r4(svd.singular_values_[-1])},
        "vektor_term": {
            t: [r4(x) for x in tvn[vi[t]][:N_DIM_TAMPIL]] for t in tampil if t in vi
        },
        "centroid": [r4(x) for x in centroid[:N_DIM_TAMPIL]],
        "norma_centroid": r4(c_norm),
        "centroid_ternormalisasi": [r4(x) for x in centroid_unit[:N_DIM_TAMPIL]],
        "cosine_term_query": {t: r4(cos[vi[t]]) for t in u.tokens if t in vi},
        "jumlah_kandidat_positif": n_kandidat,
        "kandidat": [{"peringkat": p, "term": t, "cosine": r4(v),
                      "df_laporan": int(index.df.get(t, 0)),
                      "tf_gold": int(index.freqs[u.gold_index].get(t, 0))}
                     for p, (t, v) in enumerate(cand, 1)],
        "jumlah_cosine": r4(sum(v for _, v in cand)),
        "bobot_ekspansi": {t: r4(v) for t, v in
                           sorted(expansion.items(), key=lambda kv: -kv[1])},
    }

    # --- 4. LSA-QE ----------------------------------------------------------
    skala = gamma * len(u.tokens)
    w = combine(u.base_weights, expansion, skala)
    scores = index.score(w)
    order = rank(scores)
    rank_lsa = int(np.nonzero(order == u.gold_index)[0][0]) + 1

    rincian = []
    total = 0.0
    for t, wt in sorted(w.items(), key=lambda kv: -kv[1]):
        b = float(index.term_scores(t)[u.gold_index])
        total += wt * b
        rincian.append({"term": t, "bobot": r4(wt),
                        "tf_gold": int(index.freqs[u.gold_index].get(t, 0)),
                        "df": int(index.df.get(t, 0)),
                        "bm25_gold": r4(b), "kontribusi": r4(wt * b)})
    out["lsa_qe"] = {
        "skala": r4(skala),
        "rincian_skor_gold": rincian,
        "skor_gold": r4(total),
        "peringkat_gold": rank_lsa,
        "teratas": [{"peringkat": p, "chunk_id": doc.chunks[i].chunk_id,
                     "skor": r4(scores[i])}
                    for p, i in enumerate(order[:N_TOP], 1)],
    }

    # --- 5. pembanding ------------------------------------------------------
    fb = list(u.feedback_docs)
    s_fb = scores_b1[fb]
    e = np.exp(s_fb - s_fb.max())
    alpha = e / e.sum()
    scores_rm3 = index.score(u.rm3_weights)
    rank_rm3 = int(np.nonzero(rank(scores_rm3) == u.gold_index)[0][0]) + 1
    w_both = combine(u.rm3_weights, expansion, gamma * cfg["rm3"]["lambda"])
    scores_both = index.score(w_both)
    rank_both = int(np.nonzero(rank(scores_both) == u.gold_index)[0][0]) + 1

    out["rm3"] = {
        "feedback": [{"chunk_id": doc.chunks[i].chunk_id, "skor_b1": r4(scores_b1[i]),
                      "alpha": r4(a)} for i, a in zip(fb, alpha)],
        "model_relevansi": [{"term": t, "p": r4(v)} for t, v in
                            sorted(u.rm3_terms.items(), key=lambda kv: -kv[1])[:10]],
        "skor_gold": r4(scores_rm3[u.gold_index]),
        "peringkat_gold": rank_rm3,
    }

    # --- 6. metrik ----------------------------------------------------------
    out["metrik"] = {
        m: {"peringkat_gold": r, "RR": r4(reciprocal_rank(r)),
            "Recall@1": int(r <= 1), "Recall@5": int(r <= 5),
            "Recall@10": int(r <= 10), "nDCG@10": r4(ndcg_at_10(r))}
        for m, r in (("Baseline", rank_b1), ("RM3", rank_rm3),
                     ("LSA-QE", rank_lsa))
    }

    # --- 7. pembentukan jawaban ---------------------------------------------
    konteks_idx = list(order[:N_KONTEKS])
    potongan = [f"[{p}] {doc.chunks[i].text}" for p, i in enumerate(konteks_idx, 1)]
    prompt = PROMPT_TEMPLATE.format(konteks="\n\n".join(potongan),
                                    pertanyaan=u.text)
    out["jawaban"] = {
        "n_konteks": N_KONTEKS,
        "template_prompt": PROMPT_TEMPLATE,
        "konteks": [{"urutan": p, "chunk_id": doc.chunks[i].chunk_id,
                     "skor": r4(scores[i]), "acuan": bool(i == u.gold_index),
                     "panjang": len(doc.chunks[i].tokens),
                     "teks": doc.chunks[i].text}
                    for p, i in enumerate(konteks_idx, 1)],
        "konteks_memuat_acuan": bool(u.gold_index in konteks_idx),
        "prompt_terisi": prompt,
        "panjang_prompt_karakter": len(prompt),
        "catatan": ("Lapisan generasi belum dijalankan pada penelitian ini; "
                    "yang ditelusuri adalah masukan yang dibentuk untuknya."),
    }

    out["lengkap"] = trace_lengkap(
        corpus=corpus, engine=engine, split=split, u=u, doc=doc, gold=gold,
        index=index, scores_b1=scores_b1, order_b1=order_b1, coll_tokens=coll_tokens,
        vocab=vocab, df=df, X=X, svd=svd, tv=tv, tvn=tvn, vi=vi, model=model,
        cand=cand, expansion=expansion, w=w, scores=scores, order=order,
        fb=fb, alpha=alpha, scores_rm3=scores_rm3, cfg=cfg, k=k, n=n, gamma=gamma)

    path = os.path.join(io.RESULTS, "manual_trace_bab4.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("ditulis:", io.rel(path))
    print(f"unit qid={QID} ({u.qtype}) pada {u.doc_id}, {len(doc.chunks)} chunk")
    for m, v in out["metrik"].items():
        print(f"  {m:9s} peringkat gold = {v['peringkat_gold']:3d}  RR = {v['RR']}")


if __name__ == "__main__":
    main()
