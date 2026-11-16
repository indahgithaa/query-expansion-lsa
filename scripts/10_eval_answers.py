"""Agregasi kualitas jawaban per metode (Tabel 4.57) dan analisis tambahan.

Masukan : results/05_prompt_map.csv, results/09_answer_scores_local.csv,
          results/generation/faithfulness.jsonl (opsional; bila belum ada, Faithfulness dilewati),
          results/generation/answers.jsonl (untuk laju jawaban "tidak tersedia")
Keluaran: results/10_answer_metrics.json

Terencana (Tabel 4.57): rata-rata Faithfulness, BERTScore F1, BLEU, ROUGE-L per metode atas 3.444 unit uji.
Eksploratif (tidak direncanakan pada Subbab 4.11): per skenario, per tipe, uji berpasangan (bootstrap berstrata,
Wilcoxon, Holm atas 3 pasangan per metrik), dan laju jawaban "tidak tersedia".
Faithfulness bernilai kosong (jawaban tanpa pernyataan atau galat juri) dikeluarkan dari rata-rata; jumlahnya
dilaporkan. Uji berpasangan Faithfulness hanya memakai unit yang bernilai pada kedua metode.
"""
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from qe_lsa import config
from qe_lsa.data import QTYPES
from qe_lsa.evaluate import holm, stratified_bootstrap, verdict, wilcoxon_test
from qe_lsa.pipeline import METHODS

METRICS = ("faithfulness", "bertscore_f1", "bleu", "rouge_l")
ABSTAIN = re.compile(r"tidak tersedia", re.IGNORECASE)


def load() -> pd.DataFrame:
    m = pd.read_csv(config.RESULTS_DIR / "05_prompt_map.csv", dtype={"pid": str})
    s = pd.read_csv(config.RESULTS_DIR / "09_answer_scores_local.csv", dtype={"pid": str})
    df = m.merge(s, on="pid", how="left", validate="many_to_one")
    faith_path = config.RESULTS_DIR / "generation" / "faithfulness.jsonl"
    if faith_path.exists():
        f = pd.read_json(faith_path, lines=True, dtype={"pid": str})[["pid", "faithfulness"]]
        df = df.merge(f, on="pid", how="left", validate="many_to_one")
    else:
        df["faithfulness"] = np.nan
    ans = pd.read_json(config.RESULTS_DIR / "generation" / "answers.jsonl", lines=True, dtype={"pid": str})
    ans["abstain"] = ans["answer"].str.contains(ABSTAIN)
    df = df.merge(ans[["pid", "abstain"]], on="pid", how="left", validate="many_to_one")
    assert df[["bleu", "rouge_l", "bertscore_f1"]].notna().all().all()
    return df


def summary(sub: pd.DataFrame) -> dict:
    out = {"n_units": int(len(sub))}
    for k in METRICS:
        v = sub[k].dropna()
        out[k] = float(v.mean()) if len(v) else None
        out[f"{k}_n"] = int(len(v))
    out["abstain_rate"] = float(sub["abstain"].mean())
    return out


def main():
    df = load()
    out = {"overall": {}, "per_scenario": defaultdict(dict), "per_qtype": defaultdict(dict), "significance": {}}
    for meth in METHODS:
        sm = df[df.method == meth]
        out["overall"][meth] = summary(sm)
        for scen in ("TPL", "PARA"):
            out["per_scenario"][scen][meth] = summary(sm[sm.scenario == scen])
        for t in QTYPES:
            out["per_qtype"][t][meth] = summary(sm[sm.qtype == t])

    for k in METRICS:
        wide = df.pivot_table(index=["seed", "scenario", "qid"], columns="method", values=k,
                              dropna=False).reset_index().sort_values(["seed", "scenario", "qid"])
        tests = []
        for a, b in config.PAIRS:
            ok = wide[a].notna() & wide[b].notna()
            if ok.sum() == 0:
                continue
            sub = wide[ok]
            d = (sub[a] - sub[b]).to_numpy(dtype=float)
            strata = (sub["seed"].astype(str) + "|" + sub["scenario"]).to_numpy()
            lo, hi = stratified_bootstrap(d, strata)
            tests.append({"pair": f"{a} vs {b}", "n": int(ok.sum()), "delta": float(d.mean()), "ci95": [lo, hi],
                          "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                          **wilcoxon_test(d)})
        for t, p_adj in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = p_adj
            t["significant"] = verdict(p_adj, tuple(t["ci95"]))
        out["significance"][k] = tests

    path = config.RESULTS_DIR / "10_answer_metrics.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"overall": out["overall"], "significance": out["significance"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
