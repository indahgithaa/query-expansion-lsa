"""Tahap 5 - evaluasi akhir pada seed 1-10 dengan config terkunci.

Dijalankan SEKALI pada partisi test. Opsi `--partition dev --out-dir ...` disediakan
untuk memvalidasi skrip tanpa menyentuh test (dev sudah terbuka sejak Tahap 4).
"""

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io
from qe.corpus import TYPE_ORDER, load_corpus, load_paraphrases
from qe.engine import METHODS, Engine, expansion_for_n
from qe.metrics import reciprocal_rank, robustness, summarize
from qe.scenarios import SCENARIOS
from qe.splits import make_split, types_by_doc
from qe.stats import holm, stratified_paired_bootstrap, wilcoxon

LOCKED_PATH = os.path.join(io.CONFIGS, "locked_config.json")
COMPARISONS = (("RM3", "B1"), ("LSA-QE", "B1"), ("RM3+LSA", "B1"),
               ("RM3+LSA", "RM3"), ("RM3+LSA", "LSA-QE"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition", choices=("test", "dev"), default="test")
    p.add_argument("--seeds", default="1-10", help="mis. 1-10 atau 1,2,3")
    p.add_argument("--out-dir", default=None,
                   help="direktori keluaran; default results/ (wajib diisi untuk --partition dev)")
    p.add_argument("--config", default=LOCKED_PATH)
    return p.parse_args()


def parse_seeds(spec):
    if "-" in spec:
        lo, hi = spec.split("-")
        return tuple(range(int(lo), int(hi) + 1))
    return tuple(int(x) for x in spec.split(","))


def main():
    args = parse_args()
    if args.partition == "dev" and not args.out_dir:
        print("BERHENTI: --partition dev wajib memakai --out-dir supaya hasilnya tidak "
              "tercampur dengan hasil test di results/.", file=sys.stderr)
        return 1
    if not os.path.exists(args.config):
        print(f"BERHENTI: {io.rel(args.config)} tidak ada. Jalankan scripts/03_lock_config.py "
              f"dulu setelah konfigurasi disetujui.", file=sys.stderr)
        return 1

    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)
    params = config["methods"]
    seeds = parse_seeds(args.seeds)
    out_dir = args.out_dir or io.RESULTS
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    corpus = load_corpus(io.DATA)
    paraphrases = load_paraphrases(os.path.join(io.DATA, "paraphrases.json"))
    engine = Engine(corpus, paraphrases,
                    n_feedback=config["rm3"]["n_feedback"],
                    n_terms=config["rm3"]["n_terms"],
                    lam=config["rm3"]["lambda"],
                    rm3_renormalize=config["rm3"]["renormalize"])
    tbd = types_by_doc(corpus)

    records = []  # satu baris per (seed, skenario, qid) per metode
    for seed in seeds:
        split = make_split(corpus, seed, tbd)
        docs = getattr(split, args.partition)
        units = engine.prepare_units(seed, docs, args.partition, scenarios=SCENARIOS)

        ks = sorted({params[m]["k"] for m in ("LSA-QE", "RM3+LSA")})
        expansions = {}
        for k in ks:
            model = engine.fit_lsa(split.collection, k)
            max_n = max(params[m]["n"] for m in ("LSA-QE", "RM3+LSA") if params[m]["k"] == k)
            cands = engine.expansion_candidates(model, units, max_n)
            for m in ("LSA-QE", "RM3+LSA"):
                if params[m]["k"] == k:
                    expansions[m] = [expansion_for_n(c, params[m]["n"]) for c in cands]
            del model, cands

        for i, u in enumerate(units):
            ranks = {
                "B1": engine.rank_b1(u),
                "RM3": engine.rank_rm3(u),
                "LSA-QE": engine.rank_lsa_qe(u, expansions["LSA-QE"][i], params["LSA-QE"]["gamma"]),
                "RM3+LSA": engine.rank_rm3_lsa(u, expansions["RM3+LSA"][i], params["RM3+LSA"]["gamma"]),
            }
            for method, r in ranks.items():
                records.append({"seed": seed, "scenario": u.scenario, "qid": u.qid,
                                "qtype": u.qtype, "doc_id": u.doc_id, "method": method,
                                "rank": r, "rr": reciprocal_rank(r),
                                "n_chunks": len(corpus.documents[u.doc_id].chunks)})
        print(f"[seed {seed}] {len(units)} unit {args.partition} ({time.time() - t0:.0f}s)", flush=True)

    # --- indeks unit agar keempat metode sejajar ---
    order = []
    seen = set()
    for r in records:
        key = (r["seed"], r["scenario"], r["qid"])
        if key not in seen:
            seen.add(key)
            order.append(key)
    pos = {key: i for i, key in enumerate(order)}
    rr = {m: [0.0] * len(order) for m in METHODS}
    rank_by = {m: [0] * len(order) for m in METHODS}
    meta = [None] * len(order)
    for r in records:
        i = pos[(r["seed"], r["scenario"], r["qid"])]
        rr[r["method"]][i] = r["rr"]
        rank_by[r["method"]][i] = r["rank"]
        meta[i] = (r["seed"], r["scenario"], r["qtype"])

    def subset(method, pred):
        return [rank_by[method][i] for i in range(len(order)) if pred(meta[i])]

    overall, by_scenario, by_type, by_seed = {}, {}, {}, {}
    for m in METHODS:
        by_scenario[m] = {sc: summarize(subset(m, lambda x, sc=sc: x[1] == sc)) for sc in SCENARIOS}
        overall[m] = summarize(rank_by[m])
        overall[m]["macro_MRR"] = sum(by_scenario[m][sc]["MRR"] for sc in SCENARIOS) / len(SCENARIOS)
        by_type[m] = {t: summarize(subset(m, lambda x, t=t: x[2] == t)) for t in TYPE_ORDER}
        by_seed[m] = {str(s): summarize(subset(m, lambda x, s=s: x[0] == s)) for s in seeds}

    robust = {m: {"overall": robustness(rank_by[m], rank_by["B1"]),
                  **{sc: robustness(subset(m, lambda x, sc=sc: x[1] == sc),
                                    subset("B1", lambda x, sc=sc: x[1] == sc)) for sc in SCENARIOS},
                  "by_type": {t: robustness(subset(m, lambda x, t=t: x[2] == t),
                                            subset("B1", lambda x, t=t: x[2] == t))
                              for t in TYPE_ORDER}}
              for m in METHODS if m != "B1"}

    strata = [(x[0], x[1]) for x in meta]
    boot_cfg = config["test"]["bootstrap"]
    tests = []
    for a, b in COMPARISONS:
        bs = stratified_paired_bootstrap(rr[a], rr[b], strata,
                                         n_resamples=boot_cfg["n_resamples"],
                                         seed=boot_cfg["rng_seed"])
        wx = wilcoxon(rr[a], rr[b])
        tests.append({"a": a, "b": b, **bs, "wilcoxon_statistic": wx["statistic"],
                      "wilcoxon_p": wx["p_value"], "wilcoxon_n_nonzero": wx["n_nonzero"]})
    for t, p_adj in zip(tests, holm([t["wilcoxon_p"] for t in tests])):
        t["wilcoxon_p_holm"] = p_adj
        t["significant_holm_0.05"] = p_adj < 0.05
        t["ci_excludes_zero"] = (t["ci_low"] > 0) or (t["ci_high"] < 0)

    payload = {
        "partition": args.partition,
        "seeds": list(seeds),
        "scenarios": list(SCENARIOS),
        "n_units": len(order),
        "config": {"path": io.rel(args.config), "sha256": io.sha256_file(args.config),
                   "methods": params},
        "overall": overall,
        "by_scenario": by_scenario,
        "by_type": by_type,
        "by_seed": by_seed,
        "robustness_vs_B1": robust,
        "significance": tests,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    prefix = "04_test" if args.partition == "test" else "04_dryrun_dev"
    jpath = os.path.join(out_dir, f"{prefix}.json")
    with open(jpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    import csv as _csv
    cpath = os.path.join(out_dir, f"{prefix}_units.csv")
    with open(cpath, "w", encoding="utf-8", newline="") as fh:
        wtr = _csv.DictWriter(fh, fieldnames=list(records[0]))
        wtr.writeheader()
        wtr.writerows(records)

    print(f"\nSelesai dalam {time.time() - t0:.0f}s, {len(order)} unit x {len(METHODS)} metode")
    for m in METHODS:
        o, s = overall[m], by_scenario[m]
        print(f"  {m:<8} MRR {o['MRR']:.4f} (TPL {s['TPL']['MRR']:.4f} / PARA {s['PARA']['MRR']:.4f}) "
              f"R@10 {o['Recall@10']:.3f} gagal {o['fail_rate']:.3f}")
    print()
    for t in tests:
        star = "*" if t["significant_holm_0.05"] else " "
        print(f"  {t['a']:>8} vs {t['b']:<8} dMRR {t['delta']:+.4f} "
              f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] p_Holm {t['wilcoxon_p_holm']:.4g} {star}")
    print(f"\n  -> {jpath}\n  -> {cpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
