"""Metrik performa retrieval dan uji signifikansi pada data uji (Subbab 4.11, Tabel 4.53–4.56).

Masukan : results/04_test_runs.jsonl
Keluaran: results/07_retrieval_metrics.json
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from qe_lsa import config
from qe_lsa.data import QTYPES
from qe_lsa.evaluate import aggregate, holm, stratified_bootstrap, verdict, wilcoxon_test
from qe_lsa.pipeline import METHODS


def main():
    df = pd.read_json(config.RESULTS_DIR / "04_test_runs.jsonl", lines=True)
    wide = df.pivot_table(index=["seed", "scenario", "qid", "qtype"], columns="method", values="rank").reset_index()
    wide = wide.sort_values(["seed", "scenario", "qid"]).reset_index(drop=True)
    out = {"n_units": len(wide), "overall": {}, "per_scenario": defaultdict(dict), "per_qtype": defaultdict(dict),
           "robustness": {}, "significance": []}

    for m in METHODS:
        out["overall"][m] = aggregate(wide[m])
        for scen in ("TPL", "PARA"):
            out["per_scenario"][scen][m] = aggregate(wide.loc[wide.scenario == scen, m])
        for t in QTYPES:
            sub = wide[wide.qtype == t]
            out["per_qtype"][t][m] = {"n": len(sub), "mrr": aggregate(sub[m])["mrr"]}
        tpl, para = out["per_scenario"]["TPL"][m], out["per_scenario"]["PARA"][m]
        out["robustness"][m] = {"mrr_tpl": tpl["mrr"], "mrr_para": para["mrr"], "gap": tpl["mrr"] - para["mrr"],
                                "fail_tpl": tpl["fail_rate"], "fail_para": para["fail_rate"]}

    rr = {m: 1.0 / wide[m].to_numpy(dtype=float) for m in METHODS}
    strata = (wide["seed"].astype(str) + "|" + wide["scenario"]).to_numpy()
    tests = []
    for a, b in config.PAIRS:
        d = rr[a] - rr[b]
        lo, hi = stratified_bootstrap(d, strata)
        tests.append({"pair": f"{a} vs {b}", "delta_mrr": float(d.mean()), "ci95": [lo, hi],
                      "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                      **wilcoxon_test(d)})
    for t, p_adj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p_adj
        t["significant"] = verdict(p_adj, tuple(t["ci95"]))
    out["significance"] = tests

    path = config.RESULTS_DIR / "07_retrieval_metrics.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"overall": out["overall"], "robustness": out["robustness"], "significance": tests},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
