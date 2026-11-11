"""Uji penerimaan: kode harus mereproduksi angka perhitungan manual Subbab 4.12.

Unit utama: qid 63, data validasi pengulangan 1, skenario TPL. Toleransi 1e-4 kecuali disebut lain.
Angka di tes ini disalin dari Bab 4 dan tidak boleh diubah agar tes lulus.
"""
import hashlib

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from qe_lsa import config
from qe_lsa.bm25 import baseline_weights
from qe_lsa.data import load_corpus
from qe_lsa.evaluate import aggregate, holm, stratified_bootstrap, wilcoxon_test
from qe_lsa.lsa import lsaqe_weights
from qe_lsa.pipeline import METHODS, prepare_seed, query_weights
from qe_lsa.rm3 import rm3_weights
from qe_lsa.split import load_paraphrases, paraphrase_index
from qe_lsa.text import remove_stopwords, stopwords, tokenize

TOL = 1e-4


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def ctx(corpus):
    return prepare_seed(corpus, 1)


@pytest.fixture(scope="module")
def unit(corpus, ctx):
    qa = next(q for q in corpus.qas if q.qid == 63)
    q = remove_stopwords(tokenize(qa.question))
    return qa, ctx.indexes[qa.doc_id], q


def rank_of(index, weights, ref):
    ranked, scores = index.rank(weights)
    r = ranked.index(ref)
    return r + 1, float(scores[r])


# ---------- 4.3–4.5 data, preprocessing, pembagian ----------
def test_corpus_summary(corpus):  # Tabel 4.6
    sizes = [len(c) for c in corpus.chunks.values()]
    assert len(corpus.doc_ids) == 120 and sum(sizes) == 13737
    assert (min(sizes), max(sizes), int(np.median(sizes))) == (32, 256, 109)
    assert sum(sizes) / 120 == pytest.approx(114.475)
    assert (corpus.n_qa_raw, corpus.n_qa_raw - len(corpus.qas), len(corpus.qas)) == (911, 53, 858)


def test_stopwords():  # Subbab 4.12.3
    assert len(stopwords()) == 810


def test_split_seed1(corpus, ctx):  # Tabel 4.13–4.14
    p = ctx.parts
    assert p["attempt"] == 0
    assert p["dev"] == ["D062", "D065", "D071", "D026", "D083", "D012", "D014", "D105", "D068", "D051", "D100",
                        "D015", "D036", "D049", "D072", "D104", "D029", "D004", "D035", "D045", "D059", "D009",
                        "D110", "D099"]
    assert p["test"] == ["D097", "D063", "D003", "D064", "D030", "D042", "D081", "D102", "D095", "D008", "D109",
                         "D056", "D019", "D073", "D060", "D116", "D106", "D108", "D066", "D023", "D120", "D041",
                         "D085", "D067"]
    for part, n_chunk, n_qa in (("collection", 8128, 511), ("dev", 2848, 174), ("test", 2761, 173)):
        assert sum(len(corpus.chunks[d]) for d in p[part]) == n_chunk
        assert sum(q.doc_id in p[part] for q in corpus.qas) == n_qa


def test_paraphrase_index():  # Subbab 4.12.2
    assert hashlib.sha256(b"1|63").hexdigest().startswith("3ba88302e3808a05")
    assert paraphrase_index(1, 63) == 3
    assert load_paraphrases()["dev"]["Skrining"][3] == "Instrumen apa yang dipakai untuk menapis status gizi?"


def test_preprocessing_question_and_ref(corpus, unit):  # Tabel 4.18–4.19
    qa, index, q = unit
    assert q == ["metode", "skrining", "status", "gizi", "direkomendasikan"]
    ref = next(c for c in corpus.chunks["D029"] if c.cid == "D029:P0008:C000")
    assert (len(ref.tokens), len(set(ref.tokens))) == (120, 66)
    nosw = remove_stopwords(ref.tokens)
    assert (len(nosw), len(set(nosw))) == (78, 42)


# ---------- 4.12.4–4.12.5 BM25 dan Baseline ----------
def test_bm25_index(unit):
    qa, index, q = unit
    assert qa.ref_cid == "D029:P0008:C000" and index.n == 115
    assert index.doc_len.sum() == 12791
    assert index.avgdl == pytest.approx(111.2261, abs=TOL)
    assert index.idf["skrining"] == pytest.approx(2.5374, abs=TOL)
    assert index.idf["gizi"] == pytest.approx(0.7388, abs=TOL)
    assert index.mean_idf == pytest.approx(3.5494, abs=TOL)
    assert index.df["1"] == 79 and index.idf["1"] == pytest.approx(0.8874, abs=TOL)
    assert len(index.df) == 1743 and sum(v < 0 for v in
                                         (np.log((115 - d + .5) / (d + .5)) for d in index.df.values())) == 7


def test_bm25_matches_rank_bm25(corpus, unit):
    """Silang-periksa dengan pustaka acuan rank_bm25 untuk bobot Baseline."""
    qa, index, q = unit
    ref = BM25Okapi([c.tokens for c in corpus.chunks[qa.doc_id]], k1=1.5, b=0.75, epsilon=0.25)
    assert np.allclose(index.score(baseline_weights(q)), ref.get_scores(q))


def test_baseline(unit):  # Tabel 4.21–4.24
    qa, index, q = unit
    assert rank_of(index, baseline_weights(q), qa.ref_cid) == (20, pytest.approx(2.4504, abs=TOL))
    ranked, scores = index.rank(baseline_weights(q))
    assert ranked[:3] == ["D029:P0007:C000", "D029:P0031:C002", "D029:P0032:C000"]
    assert np.allclose(scores[:3], [6.0936, 6.0301, 5.4696], atol=TOL)
    assert int((scores == 0).sum()) == 73


# ---------- 4.12.6–4.12.8 LSA dan LSA-QE ----------
def test_lsa_model(ctx):  # Tabel 4.25, 4.29
    s = ctx.lsa.stats
    assert (s["n_tokens"], s["n_unique"], s["n_df1"], s["n_numeric_removed"], s["n_vocab"]) == \
           (729195, 22753, 8564, 1464, 12725)
    assert s["shape"] == (8128, 12725) and s["nnz"] == 350401
    assert s["frobenius_sq"] == pytest.approx(8689125.9783, abs=1e-3)
    assert s["energy"] == pytest.approx(0.4156, abs=TOL)
    assert ctx.lsa.singular_values[0] == pytest.approx(476.0010, abs=TOL)
    assert ctx.lsa.singular_values[-1] == pytest.approx(71.0811, abs=TOL)
    v = ctx.lsa.vectors[ctx.lsa.term_index["skrining"]][:3]
    assert np.allclose(v, [0.2922, -0.4126, 0.2357], atol=TOL)


EXPANSION = [("malnutrisi", 0.5798, 0.1144), ("alat", 0.5586, 0.1103), ("anak", 0.5335, 0.1053),
             ("strongkids", 0.5024, 0.0992), ("pyms", 0.4945, 0.0976), ("berdasarkan", 0.4926, 0.0972),
             ("kondisi", 0.4869, 0.0961), ("pembanding", 0.4837, 0.0955), ("hasil", 0.4785, 0.0945),
             ("divalidasi", 0.4551, 0.0898)]


def test_lsaqe_expansion(ctx, unit):  # Tabel 4.32–4.34
    qa, index, q = unit
    w, sel = lsaqe_weights(ctx.lsa, q)
    assert [t for t, _ in sel] == [t for t, _, _ in EXPANSION]
    total = sum(c for _, c in sel)
    assert total == pytest.approx(5.0657, abs=TOL)
    for (t, c), (_, c_ref, e_ref) in zip(sel, EXPANSION):
        assert c == pytest.approx(c_ref, abs=TOL)
        assert c / total == pytest.approx(e_ref, abs=TOL)
    assert w["malnutrisi"] == pytest.approx(1.1445, abs=TOL)
    assert sum(w.values()) == pytest.approx(15.0, abs=1e-9)


def test_lsaqe_ranking(ctx, unit):  # Tabel 4.35–4.37
    qa, index, q = unit
    w, _ = lsaqe_weights(ctx.lsa, q)
    assert rank_of(index, w, qa.ref_cid) == (1, pytest.approx(20.8881, abs=TOL))
    ranked, scores = index.rank(w)
    assert [c.split(":", 1)[1] for c in ranked[:10]] == [
        "P0008:C000", "P0031:C002", "P0031:C001", "P0055:C000", "P0031:C000",
        "P0007:C000", "P0032:C000", "P0063:C000", "P0017:C000", "P0040:C000"]
    assert scores[5] == pytest.approx(8.9778, abs=TOL)


# ---------- 4.12.9 RM3 ----------
def test_rm3(ctx, unit):  # Tabel 4.38–4.43
    qa, index, q = unit
    w, dbg = rm3_weights(index, ctx.nosw, q)
    assert np.allclose(dbg["alpha"], [0.4042, 0.3793, 0.2165], atol=TOL)
    assert [len(ctx.nosw[c]) for c in dbg["fb"]] == [18, 73, 80]
    assert dbg["mass"] == pytest.approx(0.660935, abs=1e-6)
    assert list(dbg["p_r_raw"])[:5] == ["2", "skrining", "strongkidz", "gizi", "1"]
    assert dbg["p_r_raw"]["skrining"] == pytest.approx(0.070886, abs=1e-6)
    assert w["skrining"] == pytest.approx(0.1536, abs=TOL)
    assert w["gizi"] == pytest.approx(0.1450, abs=TOL)
    assert w["metode"] == pytest.approx(0.1000, abs=TOL)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert rank_of(index, w, qa.ref_cid) == (3, pytest.approx(1.2045, abs=TOL))


# ---------- 4.12.10–4.12.11 sepuluh unit, metrik, uji ----------
RANKS_4_45 = {  # qid: (Baseline, RM3, LSA-QE)
    14: (6, 4, 2), 15: (5, 4, 3), 16: (73, 30, 26), 17: (11, 11, 13), 18: (1, 1, 1),
    19: (136, 56, 100), 20: (45, 16, 66), 21: (76, 106, 93), 22: (7, 10, 7), 23: (47, 19, 25)}


@pytest.fixture(scope="module")
def ten_units(corpus, ctx):
    units = sorted((q for q in corpus.qas if q.doc_id in ctx.parts["dev"]), key=lambda q: q.qid)[:10]
    ranks = {m: [] for m in METHODS}
    for u in units:
        for m in METHODS:
            ranked, _ = ctx.indexes[u.doc_id].rank(query_weights(ctx, u.doc_id, u.question, m))
            ranks[m].append(ranked.index(u.ref_cid) + 1)
    return [u.qid for u in units], ranks


def test_ten_unit_ranks(ten_units):  # Tabel 4.45
    qids, ranks = ten_units
    assert qids == list(RANKS_4_45)
    for i, m in enumerate(METHODS):
        assert ranks[m] == [RANKS_4_45[q][i] for q in qids], m


def test_ten_unit_metrics(ten_units):  # Tabel 4.47
    _, ranks = ten_units
    expected = {"Baseline": (0.1678, 0.1, 0.2, 0.4, 0.2076, 28, 0.6),
                "RM3": (0.1867, 0.1, 0.3, 0.4, 0.2150, 13.5, 0.6),
                "LSA-QE": (0.2167, 0.1, 0.3, 0.4, 0.2464, 19, 0.6)}
    for m, (mrr, r1, r5, r10, ndcg, med, fail) in expected.items():
        a = aggregate(ranks[m])
        got = (a["mrr"], a["r@1"], a["r@5"], a["r@10"], a["ndcg10"], a["median_rank"], a["fail_rate"])
        assert np.allclose(got, (mrr, r1, r5, r10, ndcg, med, fail), atol=TOL), m


def test_wilcoxon_holm(ten_units):  # Tabel 4.48–4.50
    _, ranks = ten_units
    rr = {m: 1 / np.array(ranks[m], dtype=float) for m in METHODS}
    expected = [(8, 29, 7, 7, 0.148438), (8, 28, 8, 8, 0.195312), (9, 26, 19, 19, 0.734375)]
    tests = [wilcoxon_test(rr[a] - rr[b]) for a, b in config.PAIRS]
    for t, (n, wp, wm, stat, p) in zip(tests, expected):
        assert (t["n_nonzero"], t["w_plus"], t["w_minus"], t["statistic"]) == (n, wp, wm, stat)
        assert t["p"] == pytest.approx(p, abs=1e-6)
    assert np.allclose(holm([t["p"] for t in tests]), [0.445313, 0.445313, 0.734375], atol=1e-6)


def test_bootstrap(ten_units):  # Tabel 4.51
    _, ranks = ten_units
    d = 1 / np.array(ranks["LSA-QE"], dtype=float) - 1 / np.array(ranks["Baseline"], dtype=float)
    lo, hi, draws = stratified_bootstrap(d, [1] * 10, return_samples=True)
    assert [list(x + 1) for x in draws] == [[4, 3, 7, 8, 5, 2, 1, 6, 3, 5], [2, 10, 7, 10, 4, 3, 2, 8, 9, 4],
                                            [10, 5, 1, 2, 9, 4, 10, 4, 7, 6]]
    assert np.allclose([d[x].mean() for x in draws], [0.0495, 0.0291, 0.0472], atol=TOL)
    assert (lo, hi) == (pytest.approx(0.0001, abs=TOL), pytest.approx(0.1176, abs=TOL))
