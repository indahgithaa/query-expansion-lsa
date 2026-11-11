"""Reproduksi Subbab 4.12 (data validasi, pengulangan 1, skenario TPL). Tidak menyentuh data pengujian."""
import time
import numpy as np
from qe_lsa import config
from qe_lsa.data import load_corpus
from qe_lsa.pipeline import prepare_seed, query_weights
from qe_lsa.bm25 import baseline_weights
from qe_lsa.lsa import lsaqe_weights
from qe_lsa.rm3 import rm3_weights
from qe_lsa.text import tokenize, remove_stopwords, stopwords
from qe_lsa.evaluate import aggregate, wilcoxon_test, holm, stratified_bootstrap

t0 = time.time()
C = load_corpus()
print("stopwords", len(stopwords()), "| QA raw", C.n_qa_raw, "| QA", len(C.qas),
      "| chunks", sum(len(v) for v in C.chunks.values()))
ctx = prepare_seed(C, 1)
p = ctx.parts
print("attempt", p["attempt"], "| dev", p["dev"][:5], "| test", p["test"][:5])
for part in ("collection", "dev", "test"):
    print(part, len(p[part]), "chunks", sum(len(C.chunks[d]) for d in p[part]),
          "QA", sum(1 for q in C.qas if q.doc_id in p[part]))
print("LSA stats", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ctx.lsa.stats.items()})
print("sv top5", np.round(ctx.lsa.singular_values[:5], 4), "last", np.round(ctx.lsa.singular_values[-1], 4))
v = ctx.lsa.vectors[ctx.lsa.term_index["skrining"]][:3]; print("v_skrining[:3]", np.round(v, 4))

qa = next(q for q in C.qas if q.qid == 63)
idx = ctx.indexes[qa.doc_id]
q = remove_stopwords(tokenize(qa.question))
print("\nqid63", qa.doc_id, qa.ref_cid, "q", q, "n_chunks", idx.n, "avgdl", round(idx.avgdl, 4),
      "mean_idf", round(idx.mean_idf, 4), "len_ref", int(idx.doc_len[idx.chunk_ids.index(qa.ref_cid)]))
print("idf skrining", round(idx.idf["skrining"], 4), "idf gizi", round(idx.idf["gizi"], 4), "idf '1'", round(idx.idf["1"], 4))
def show(name, w, top=10):
    ranked, s = idx.rank(w)
    r = ranked.index(qa.ref_cid) + 1
    print(f"{name}: ref score {s[r-1]:.4f} rank {r}; top", [(c.split(':',1)[1], round(float(x),4)) for c, x in zip(ranked[:top], s[:top])])
    return r
show("Baseline", baseline_weights(q), 3)
w, sel = lsaqe_weights(ctx.lsa, q)
print("LSA exp", [(t, round(c, 4), round(w[t], 4)) for t, c in sel])
show("LSA-QE", w, 10)
w, dbg = rm3_weights(idx, ctx.nosw, q)
print("RM3 alpha", np.round(dbg["alpha"], 4), "mass", round(dbg["mass"], 6), "w_skrining", round(w["skrining"], 4))
print("RM3 terms", [(t, round(x, 6)) for t, x in dbg["p_r_raw"].items()])
show("RM3", w, 5)

print("\nSepuluh unit dev seed 1 TPL")
units = sorted((q for q in C.qas if q.doc_id in p["dev"]), key=lambda q: q.qid)[:10]
ranks = {m: [] for m in ("Baseline", "RM3", "LSA-QE")}
for u in units:
    row = []
    for m in ranks:
        ranked, _ = ctx.indexes[u.doc_id].rank(query_weights(ctx, u.doc_id, u.question, m))
        ranks[m].append(ranked.index(u.ref_cid) + 1)
    print(u.qid, u.doc_id, u.qtype, [ranks[m][-1] for m in ranks])
for m in ranks: print(m, {k: round(v, 4) for k, v in aggregate(ranks[m]).items()})
rr = {m: np.array([1 / r for r in ranks[m]]) for m in ranks}
tests = [wilcoxon_test(rr[a] - rr[b]) for a, b in config.PAIRS]
for (a, b), t in zip(config.PAIRS, tests): print(a, "vs", b, t)
print("Holm", np.round(holm([t["p"] for t in tests]), 6))
lo, hi, draws = stratified_bootstrap(rr["LSA-QE"] - rr["Baseline"], [1] * 10, return_samples=True)
print("CI", round(lo, 4), round(hi, 4), "first draws (1-based)", [list(d + 1) for d in draws])
print(f"elapsed {time.time()-t0:.1f}s")
