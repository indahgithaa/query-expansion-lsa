"""Pencarian grid parameter LSA-QE (64) dan RM3 (75) pada DATA VALIDASI seed 1–3 (Subbab 4.1, Tabel 4.2).

Kriteria (ditetapkan sebelum dijalankan): rata-rata MRR(TPL) dan MRR(PARA) atas seluruh unit validasi seed 1–3.
Tie-break: nilai lebih kecil didahulukan menurut urutan grid (LSA-QE: k, n, gamma; RM3: fbDocs, fbTerms, lambda).
Pemeriksaan kestabilan: leave-one-seed-out (pilih pada dua seed, ukur pada seed ketiga).
Data pengujian TIDAK diindeks maupun dievaluasi oleh skrip ini.

Keluaran: results/02_dev_grid_units.csv, results/02_dev_grid.csv, results/02_dev_selection.json
"""
import itertools
import json
import time

import pandas as pd

from qe_lsa import config
from qe_lsa.data import load_corpus
from qe_lsa.lsa import build_lsa, lsaqe_weights
from qe_lsa.bm25 import baseline_weights
from qe_lsa.pipeline import prepare_seed
from qe_lsa.rm3 import rm3_weights
from qe_lsa.split import load_paraphrases, question_text
from qe_lsa.text import remove_stopwords, tokenize


def main():
    t0 = time.time()
    corpus = load_corpus()
    para = load_paraphrases()
    rows = []
    for seed in config.TUNING_SEEDS:
        ctx = prepare_seed(corpus, seed, k=config.LSA_K, index_parts=("dev",))
        lsa_docs = [remove_stopwords(c.tokens) for d in ctx.parts["collection"] for c in corpus.chunks[d]]
        lsa_models = {k: build_lsa(lsa_docs, k=k) for k in config.LSAQE_GRID["k"]}
        units = [q for q in corpus.qas if q.doc_id in ctx.parts["dev"]]
        for qa in units:
            for scen in ("TPL", "PARA"):
                q = remove_stopwords(tokenize(question_text(qa, seed, "dev", scen, para)))
                index = ctx.indexes[qa.doc_id]
                base = dict(seed=seed, scenario=scen, qid=qa.qid, qtype=qa.qtype)

                def rank(w):
                    return index.rank(w)[0].index(qa.ref_cid) + 1

                rows.append(dict(base, method="Baseline", config="-", rank=rank(baseline_weights(q))))
                for k, n, g in itertools.product(*config.LSAQE_GRID.values()):
                    w, _ = lsaqe_weights(lsa_models[k], q, n, g)
                    rows.append(dict(base, method="LSA-QE", config=f"k={k},n={n},gamma={g}", rank=rank(w)))
                for fd, ft, lam in itertools.product(*config.RM3_GRID.values()):
                    w, _ = rm3_weights(index, ctx.nosw, q, fd, ft, lam)
                    rows.append(dict(base, method="RM3", config=f"fbDocs={fd},fbTerms={ft},lambda={lam}",
                                     rank=rank(w)))
        print(f"seed {seed}: {len(units)} QA x 2 skenario, {time.time() - t0:.0f}s", flush=True)

    units = pd.DataFrame(rows)
    units["rr"] = 1.0 / units["rank"]
    units.to_csv(config.RESULTS_DIR / "02_dev_grid_units.csv", index=False)

    order = {m: {c: i for i, c in enumerate(units.loc[units.method == m, "config"].drop_duplicates())}
             for m in ("LSA-QE", "RM3")}

    def summarize(df):
        g = df.groupby(["method", "config", "scenario"])["rr"].mean().unstack("scenario")
        g["criterion"] = (g["TPL"] + g["PARA"]) / 2
        return g.reset_index()

    def select(summary, method):
        s = summary[summary.method == method].copy()
        s["grid_order"] = s["config"].map(order[method])
        return s.sort_values(["criterion", "grid_order"], ascending=[False, True]).iloc[0]

    summary = summarize(units)
    summary.to_csv(config.RESULTS_DIR / "02_dev_grid.csv", index=False)
    n_units = int((units.method == "Baseline").sum())
    selection = {"n_units": n_units, "criterion": "mean(MRR_TPL, MRR_PARA), dev seed 1-3",
                 "baseline": summary[summary.method == "Baseline"].iloc[0][["TPL", "PARA", "criterion"]].to_dict(),
                 "selected": {}, "bab4_table_4_2": {"LSA-QE": "k=300,n=10,gamma=2.0",
                                                     "RM3": "fbDocs=3,fbTerms=20,lambda=0.5"},
                 "leave_one_seed_out": {}}
    for m in ("LSA-QE", "RM3"):
        best = select(summary, m)
        selection["selected"][m] = {"config": best["config"], "TPL": best["TPL"], "PARA": best["PARA"],
                                    "criterion": best["criterion"]}
        loso = []
        for held in config.TUNING_SEEDS:
            tr = summarize(units[(units.seed != held)])
            pick = select(tr, m)["config"]
            te = summarize(units[(units.seed == held)])
            te = te[te.method == m].set_index("config")
            loso.append({"held_out_seed": held, "selected_on_other_two": pick,
                         "heldout_criterion_selected": te.loc[pick, "criterion"],
                         "heldout_criterion_bab4": te.loc[selection["bab4_table_4_2"][m], "criterion"],
                         "heldout_best_possible": te["criterion"].max()})
        selection["leave_one_seed_out"][m] = loso
    (config.RESULTS_DIR / "02_dev_selection.json").write_text(json.dumps(selection, indent=2, default=float),
                                                              encoding="utf-8")
    print(json.dumps(selection, indent=2, default=float))
    print(f"selesai {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
