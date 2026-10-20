"""Eksplorasi arah lanjutan - SEMUA dihitung di dev (seed 1-3), test tidak disentuh.

Menguji apakah ada cara mengalahkan LSA-QE, khususnya dalam menurunkan angka
kegagalan. Empat arah diuji; hasilnya dipakai untuk membenarkan pilihan metode
akhir di `reports/06_ringkasan_akhir.md`.
"""

import collections
import json
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io
from qe.engine import Engine, combine, expansion_for_n
from qe.corpus import load_corpus, load_paraphrases
from qe.splits import make_split, types_by_doc

DEV_SEEDS = (1, 2, 3)
RRF_K0 = 60
LM_MU = (500, 2000)
FEATURES = ("centroid_norm", "vocab_cov", "cos_mean", "cos_max",
            "nqc", "wig", "gap12", "clarity", "qlen", "mean_idf")


class LMDirichlet:
    """Language model dengan smoothing Dirichlet (Zhai & Lafferty, 2001).

    Dipakai hanya sebagai sistem pembanding yang benar-benar berbeda dari BM25,
    untuk menguji apakah kegagalan fusi disebabkan kurangnya keragaman sistem.
    """

    def __init__(self, docs, mu):
        self.mu = mu
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype=np.float64)
        cf = collections.Counter()
        for d in docs:
            cf.update(d)
        self.cf = cf
        self.total = max(sum(cf.values()), 1)
        self.tf = [collections.Counter(d) for d in docs]

    def score(self, weights):
        out = np.zeros(self.n, dtype=np.float64)
        for t, w in weights.items():
            pc = self.cf.get(t, 0) / self.total
            if pc <= 0:
                continue
            tf = np.array([f.get(t, 0) for f in self.tf], dtype=np.float64)
            out += w * np.log((tf + self.mu * pc) / (self.len + self.mu))
        return out


def ranks_of(scores):
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def gold_rank(scores, gold):
    return int(np.nonzero(np.argsort(-scores, kind="stable") == gold)[0][0]) + 1


def spearman(a, b):
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                out[order[m]] = (i + j) / 2.0
            i = j + 1
        return out
    ra, rb = rk(list(a)), rk(list(b))
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def summarise(ranks):
    n = len(ranks)
    r10 = sum(1 for r in ranks if r <= 10) / n
    return {"MRR": sum(1.0 / r for r in ranks) / n,
            "Recall@5": sum(1 for r in ranks if r <= 5) / n,
            "Recall@10": r10, "fail_rate": 1 - r10}


def collect(engine, corpus, config):
    """Satu lintasan atas dev: peringkat tiap metode, prediktor, dan jangkauan morfologi."""
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from qe.bm25 import rank as bm25_rank

    stemmer = StemmerFactory().create_stemmer()
    stem_cache = {}

    def stem(t):
        if t not in stem_cache:
            stem_cache[t] = stemmer.stem(t)
        return stem_cache[t]

    p = config["methods"]
    k, n, gamma = p["LSA-QE"]["k"], p["LSA-QE"]["n"], p["LSA-QE"]["gamma"]
    tbd = types_by_doc(corpus)
    rows = []
    for seed in DEV_SEEDS:
        split = make_split(corpus, seed, tbd)
        units = engine.prepare_units(seed, split.dev, "dev")
        model = engine.fit_lsa(split.collection, k)
        cands = engine.expansion_candidates(model, units, n)
        lm_cache = {}
        for u, cd in zip(units, cands):
            idx = engine.index(u.doc_id)
            exp = expansion_for_n(cd, n)
            w = {"B1": u.base_weights, "RM3": u.rm3_weights,
                 "LSA-QE": combine(u.base_weights, exp, gamma * len(u.tokens)),
                 "RM3+LSA": combine(u.rm3_weights, exp, gamma * engine.lam)}
            scores = {m: idx.score(v) for m, v in w.items()}
            for mu in LM_MU:
                key = (u.doc_id, mu)
                if key not in lm_cache:
                    lm_cache[key] = LMDirichlet(
                        [c.tokens for c in corpus.documents[u.doc_id].chunks], mu)
                scores[f"LM{mu}"] = lm_cache[key].score(w["LSA-QE"])

            s = scores["B1"]
            srt = np.sort(s)[::-1]
            mean, sd = float(s.mean()), float(s.std()) + 1e-9
            top5 = srt[:5]
            inv = [model.term_index[t] for t in u.tokens if t in model.term_index]
            cos = [v for _, v in cd]

            fb = engine.feedback_tokens(u.doc_id)
            order = bm25_rank(s)[:engine.n_feedback]
            pr, bg = collections.Counter(), collections.Counter()
            for di in order:
                pr.update(fb[int(di)])
            for toks in fb:
                bg.update(toks)
            zp, zb = max(sum(pr.values()), 1), max(sum(bg.values()), 1)
            clarity = sum((v / zp) * math.log2((v / zp) / max(bg[t] / zb, 1e-12))
                          for t, v in pr.items())

            gold = corpus.documents[u.doc_id].chunks[u.gold_index]
            gold_tokens = set(gold.tokens)
            gold_stems = {stem(x) for x in gold_tokens}
            qset = set(u.tokens)
            rows.append({
                "seed": seed, "scenario": u.scenario, "qtype": u.qtype,
                "gold_rank": {m: gold_rank(v, u.gold_index) for m, v in scores.items()},
                "all_ranks": {m: ranks_of(v) for m, v in scores.items()},
                "exact_overlap": len(qset & gold_tokens),
                "morph_reaches": bool({t for t in qset - gold_tokens if stem(t) in gold_stems}),
                "lsa_reaches": bool({t for t, _ in cd if t in gold_tokens}),
                "centroid_norm": float(np.linalg.norm(model.term_vectors[inv].mean(axis=0))) if inv else 0.0,
                "vocab_cov": len(inv) / max(len(u.tokens), 1),
                "cos_mean": float(np.mean(cos)) if cos else 0.0,
                "cos_max": float(cos[0]) if cos else 0.0,
                "nqc": float(top5.std() / mean) if mean else 0.0,
                "wig": float((top5.mean() - mean) / sd),
                "gap12": float((srt[0] - srt[1]) / (srt[0] + 1e-9)) if len(srt) > 1 else 0.0,
                "clarity": clarity,
                "qlen": float(len(u.tokens)),
                "mean_idf": float(np.mean([idx.idf.get(t, 0.0) for t in u.tokens])) if u.tokens else 0.0,
            })
        del model, cands
        print(f"  [seed {seed}] {len(units)} unit dev", flush=True)
    return rows


def explore_baselines(rows):
    methods = ["B1", "RM3", "LSA-QE", "RM3+LSA", "LM500", "LM2000"]
    out = {m: summarise([r["gold_rank"][m] for r in rows]) for m in methods}
    qe = ["B1", "RM3", "LSA-QE", "RM3+LSA"]
    out["oracle_QE"] = summarise([min(r["gold_rank"][m] for m in qe) for r in rows])
    out["oracle_all"] = summarise([min(r["gold_rank"][m] for m in methods) for r in rows])
    return out


def explore_morphology(rows):
    def tab(subset):
        c = collections.Counter((r["morph_reaches"], r["lsa_reaches"]) for r in subset)
        n = max(len(subset), 1)
        return {"n": len(subset),
                "both": c[(True, True)] / n, "morph_only": c[(True, False)] / n,
                "lsa_only": c[(False, True)] / n, "neither": c[(False, False)] / n,
                "morph_only_count": c[(True, False)]}
    return {"all": tab(rows), "hard": tab([r for r in rows if r["exact_overlap"] == 0])}


def explore_selective(rows):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    n = len(rows)
    rb = np.array([1.0 / r["gold_rank"]["B1"] for r in rows])
    rl = np.array([1.0 / r["gold_rank"]["LSA-QE"] for r in rows])
    delta = rl - rb
    seeds = np.array([r["seed"] for r in rows])
    always, never = float(rl.mean()), float(rb.mean())
    oracle = float(np.maximum(rb, rl).mean())

    per_feature = {}
    for f in FEATURES:
        v = np.array([r[f] for r in rows])
        best, thr, direction = -1.0, None, None
        for t in np.quantile(v, np.linspace(0.02, 0.98, 60)):
            for d in (1, -1):
                mask = v >= t if d > 0 else v <= t
                m = float(np.where(mask, rl, rb).mean())
                if m > best:
                    best, thr, direction = m, float(t), d
        per_feature[f] = {"spearman_with_delta": spearman(v, delta),
                          "best_in_sample_MRR": best, "threshold": thr,
                          "direction": ">=" if direction > 0 else "<="}

    X = np.array([[r[f] for f in FEATURES] for r in rows])
    models = {}
    for name in ("logistic", "ridge"):
        ins, held = [], []
        for h in DEV_SEEDS:
            tr, te = seeds != h, seeds == h
            sc = StandardScaler().fit(X[tr])
            if name == "logistic":
                m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), (delta[tr] > 0).astype(int))
                p_tr, p_te = m.predict_proba(sc.transform(X[tr]))[:, 1], m.predict_proba(sc.transform(X[te]))[:, 1]
            else:
                m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), delta[tr])
                p_tr, p_te = m.predict(sc.transform(X[tr])), m.predict(sc.transform(X[te]))
            best, thr = -1.0, None
            for t in np.quantile(p_tr, np.linspace(0.02, 0.98, 60)):
                v = float(np.where(p_tr >= t, rl[tr], rb[tr]).mean())
                if v > best:
                    best, thr = v, t
            ins.append(best)
            held.append(float(np.where(p_te >= thr, rl[te], rb[te]).mean()))
        models[name] = {"in_sample_MRR": float(np.mean(ins)),
                        "held_out_MRR": float(np.mean(held)),
                        "always_expand_MRR": float(np.mean([rl[seeds == h].mean() for h in DEV_SEEDS])),
                        "gain_vs_always": float(np.mean(held) - np.mean([rl[seeds == h].mean() for h in DEV_SEEDS]))}
    return {"n": n, "never_expand_MRR": never, "always_expand_MRR": always,
            "oracle_MRR": oracle, "oracle_headroom": oracle - always,
            "helped": int((delta > 0).sum()), "hurt": int((delta < 0).sum()),
            "per_feature": per_feature, "models": models}


def explore_fusion(rows):
    # Indeks chunk gold dipulihkan dari peringkatnya di bawah B1.
    gold_idx = [int(np.nonzero(r["all_ranks"]["B1"] == r["gold_rank"]["B1"])[0][0]) for r in rows]

    def rrf(weights, k0=RRF_K0):
        ranks = []
        for r, g in zip(rows, gold_idx):
            f = sum(w / (k0 + r["all_ranks"][m]) for m, w in weights.items())
            ranks.append(gold_rank(f, g))
        return summarise(ranks)

    out = {
        "RRF(LSA-QE+RM3+LSA)": rrf({"LSA-QE": 1, "RM3+LSA": 1}),
        "RRF(B1+LSA-QE)": rrf({"B1": 1, "LSA-QE": 1}),
        "RRF(all four)": rrf({"B1": 1, "RM3": 1, "LSA-QE": 1, "RM3+LSA": 1}),
        "RRF(weighted 3:3:1:1)": rrf({"LSA-QE": 3, "RM3+LSA": 3, "RM3": 1, "B1": 1}),
        "RRF(LSA-QE:3,RM3+LSA:1)": rrf({"LSA-QE": 3, "RM3+LSA": 1}),
        "RRF(LSA-QE+LM500)": rrf({"LSA-QE": 1, "LM500": 1}),
        "RRF(LSA-QE+RM3+LSA+LM500)": rrf({"LSA-QE": 1, "RM3+LSA": 1, "LM500": 1}),
    }
    for k0 in (10, 20, 100):
        out[f"RRF(all four, k0={k0})"] = rrf({"B1": 1, "RM3": 1, "LSA-QE": 1, "RM3+LSA": 1}, k0)
    return out


def main():
    t0 = time.time()
    with open(os.path.join(io.CONFIGS, "locked_config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    corpus = load_corpus(io.DATA)
    paraphrases = load_paraphrases(os.path.join(io.DATA, "paraphrases.json"))
    engine = Engine(corpus, paraphrases,
                    n_feedback=config["rm3"]["n_feedback"], n_terms=config["rm3"]["n_terms"],
                    lam=config["rm3"]["lambda"], rm3_renormalize=config["rm3"]["renormalize"])

    print("Mengumpulkan peringkat dan prediktor di dev...")
    rows = collect(engine, corpus, config)

    payload = {
        "partition": "dev",
        "seeds": list(DEV_SEEDS),
        "n_units": len(rows),
        "note": ("Seluruh eksplorasi di berkas ini dihitung pada partisi dev. "
                 "Tidak ada kandidat yang mengalahkan acuan, sehingga tidak ada "
                 "yang dipromosikan ke test dan config terkunci tetap utuh."),
        "config_sha256": io.sha256_file(os.path.join(io.CONFIGS, "locked_config.json")),
        "baselines": explore_baselines(rows),
        "morphology": explore_morphology(rows),
        "selective_expansion": explore_selective(rows),
        "rank_fusion": explore_fusion(rows),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    io.write_json("06_explorations.json", payload)

    b = payload["baselines"]
    print(f"\nSelesai dalam {time.time() - t0:.0f}s\n")
    print("%-24s %8s %8s %8s" % ("", "MRR", "R@10", "gagal"))
    for m in ("B1", "RM3", "LSA-QE", "RM3+LSA", "LM500", "oracle_QE"):
        v = b[m]
        print("%-24s %8.4f %8.3f %8.3f" % (m, v["MRR"], v["Recall@10"], v["fail_rate"]))
    print("\nmorfologi, kontribusi unik : %.1f%% unit" % (100 * payload["morphology"]["all"]["morph_only"]))
    s = payload["selective_expansion"]
    print("selective expansion        : held-out %+.4f (logistic), %+.4f (ridge)" % (
        s["models"]["logistic"]["gain_vs_always"], s["models"]["ridge"]["gain_vs_always"]))
    best = min(payload["rank_fusion"].items(), key=lambda kv: kv[1]["fail_rate"])
    print("rank fusion terbaik        : %s gagal %.3f (vs RM3+LSA %.3f, LSA-QE %.3f)" % (
        best[0], best[1]["fail_rate"], b["RM3+LSA"]["fail_rate"], b["LSA-QE"]["fail_rate"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
