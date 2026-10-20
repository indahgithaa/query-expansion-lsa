"""Tahap 6 - ablasi, komplementaritas, dan analisis kegagalan.

Menghitung ke `results/05_analysis.json`. Laporannya ditulis oleh
`scripts/05_analysis_report.py` yang hanya membaca dari `results/`.
"""

import collections
import csv
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io
from qe.corpus import TYPE_ORDER, load_corpus, load_paraphrases
from qe.engine import Engine, expansion_for_n
from qe.metrics import reciprocal_rank
from qe.scenarios import SCENARIOS
from qe.splits import make_split, types_by_doc

LOCKED_PATH = os.path.join(io.CONFIGS, "locked_config.json")
TOP_TERMS = 10

# Pemilihan contoh ditetapkan sebelum melihat hasil dan tidak bergantung pada
# metrik: tipe pada indeks genap TYPE_ORDER (0, 2, 4, 6) ditambah tipe terakhir
# (indeks 7). Tiap tipe diwakili unit seed 1 dengan qid terkecil. Skenario PARA
# dipakai untuk contoh ke-2 dan ke-4 (memenuhi syarat minimal dua parafrasa),
# TPL untuk sisanya.
EXAMPLE_TYPE_INDICES = (0, 2, 4, 6, 7)
EXAMPLE_PARA_SLOTS = (1, 3)
EXAMPLE_SEED = 1

# Penanda format pernyataan PES (problem "berkaitan dengan" etiologi "ditandai oleh" sign).
PES_MARKERS = ("berkaitan", "ditandai", "disebabkan")


def main():
    t0 = time.time()
    with open(LOCKED_PATH, encoding="utf-8") as fh:
        config = json.load(fh)
    params = config["methods"]
    seeds = config["test"]["seeds"]

    corpus = load_corpus(io.DATA)
    paraphrases = load_paraphrases(os.path.join(io.DATA, "paraphrases.json"))
    engine = Engine(corpus, paraphrases,
                    n_feedback=config["rm3"]["n_feedback"], n_terms=config["rm3"]["n_terms"],
                    lam=config["rm3"]["lambda"], rm3_renormalize=config["rm3"]["renormalize"])
    tbd = types_by_doc(corpus)

    with open(os.path.join(io.RESULTS, "04_test_units.csv"), encoding="utf-8") as fh:
        rank_of = {(int(r["seed"]), r["scenario"], int(r["qid"]), r["method"]): int(r["rank"])
                   for r in csv.DictReader(fh)}

    jaccard_rows = []
    overlap_terms = collections.Counter()
    rescued, harmed, lost_any = [], [], []
    examples = []

    for seed in seeds:
        split = make_split(corpus, seed, tbd)
        units = engine.prepare_units(seed, split.test, "test", scenarios=SCENARIOS)
        model = engine.fit_lsa(split.collection, params["RM3+LSA"]["k"])
        cands = engine.expansion_candidates(model, units, max(TOP_TERMS, params["RM3+LSA"]["n"]))

        for u, cand in zip(units, cands):
            lsa_exp = expansion_for_n(cand, params["RM3+LSA"]["n"])
            rm3_set, lsa_set = set(u.rm3_terms), set(lsa_exp)
            inter = rm3_set & lsa_set
            union = rm3_set | lsa_set
            jac = len(inter) / len(union) if union else 0.0
            overlap_terms.update(inter)

            r = {m: rank_of[(seed, u.scenario, u.qid, m)]
                 for m in ("B1", "RM3", "LSA-QE", "RM3+LSA")}
            # LSA membuang angka murni saat membangun vocab; RM3 tidak menyaring apa pun,
            # jadi porsi ini mengukur berapa banyak kapasitas ekspansi RM3 terpakai angka.
            digit_terms = [t for t in rm3_set if t.isdigit()]
            digit_mass = sum(v for t, v in u.rm3_terms.items() if t.isdigit())

            jaccard_rows.append({
                "seed": seed, "scenario": u.scenario, "qid": u.qid, "qtype": u.qtype,
                "n_rm3": len(rm3_set), "n_lsa": len(lsa_set), "n_shared": len(inter),
                "jaccard": jac, "query_len": len(u.tokens),
                "rm3_digit_terms": len(digit_terms),
                "rm3_digit_share": len(digit_terms) / len(rm3_set) if rm3_set else 0.0,
                "rm3_digit_mass": digit_mass,
                **{f"rank_{m}": r[m] for m in r},
            })

            tag = {"seed": seed, "scenario": u.scenario, "qid": u.qid, "qtype": u.qtype,
                   **{f"rank_{m}": r[m] for m in r}}
            # Sinergi: kedua komponen gagal, gabungan berhasil.
            if r["RM3+LSA"] <= 10 < min(r["RM3"], r["LSA-QE"]):
                rescued.append(tag)
            # Pasangan simetrisnya: kedua komponen berhasil, gabungan gagal.
            if max(r["RM3"], r["LSA-QE"]) <= 10 < r["RM3+LSA"]:
                harmed.append(tag)
            # Pembacaan yang lebih longgar: salah satu komponen berhasil, gabungan gagal.
            if min(r["RM3"], r["LSA-QE"]) <= 10 < r["RM3+LSA"]:
                lost_any.append(tag)

            if seed == EXAMPLE_SEED:
                slot = _example_slot(u)
                if slot is not None and not any(e["slot"] == slot for e in examples):
                    rm3_top = sorted(u.rm3_terms.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_TERMS]
                    lsa_top = cand[:TOP_TERMS]
                    examples.append({
                        "slot": slot, "qid": u.qid, "qtype": u.qtype, "scenario": u.scenario,
                        "query": u.text, "query_tokens": list(u.tokens),
                        "rm3_top": [{"term": t, "weight": v} for t, v in rm3_top],
                        "lsa_top": [{"term": t, "cosine": v} for t, v in lsa_top],
                        "shared": sorted({t for t, _ in rm3_top} & {t for t, _ in lsa_top}),
                        "ranks": r,
                    })
        del model, cands
        print(f"[seed {seed}] {len(units)} unit ({time.time() - t0:.0f}s)", flush=True)

    examples.sort(key=lambda e: e["slot"])

    # --- agregasi Jaccard ---
    def jstats(rows):
        vals = [r["jaccard"] for r in rows]
        if not vals:
            return {"n": 0}
        return {"n": len(vals), "mean": sum(vals) / len(vals),
                "median": statistics.median(vals),
                "pct_zero": sum(1 for v in vals if v == 0) / len(vals),
                "mean_shared": sum(r["n_shared"] for r in rows) / len(rows),
                "mean_rm3_digit_share": sum(r["rm3_digit_share"] for r in rows) / len(rows),
                "mean_rm3_digit_mass": sum(r["rm3_digit_mass"] for r in rows) / len(rows)}

    jaccard = {
        "overall": jstats(jaccard_rows),
        "by_scenario": {sc: jstats([r for r in jaccard_rows if r["scenario"] == sc])
                        for sc in SCENARIOS},
        "by_type": {t: jstats([r for r in jaccard_rows if r["qtype"] == t]) for t in TYPE_ORDER},
        "max_possible": min(config["rm3"]["n_terms"], params["RM3+LSA"]["n"])
                        / max(config["rm3"]["n_terms"], params["RM3+LSA"]["n"]),
        "most_shared_terms": [{"term": t, "count": c} for t, c in overlap_terms.most_common(20)],
    }

    # --- ablasi: kontribusi komponen dan komplementaritas ---
    d_test = io.read_json("04_test.json")
    o = d_test["overall"]
    ablation = {
        "mrr": {m: o[m]["MRR"] for m in o},
        "lokal_saja_atas_B1": o["RM3"]["MRR"] - o["B1"]["MRR"],
        "global_saja_atas_B1": o["LSA-QE"]["MRR"] - o["B1"]["MRR"],
        "gabungan_atas_B1": o["RM3+LSA"]["MRR"] - o["B1"]["MRR"],
        "jumlah_kontribusi_terpisah": (o["RM3"]["MRR"] - o["B1"]["MRR"])
                                      + (o["LSA-QE"]["MRR"] - o["B1"]["MRR"]),
        "lsa_ditambahkan_ke_rm3": o["RM3+LSA"]["MRR"] - o["RM3"]["MRR"],
        "rm3_ditambahkan_ke_lsa": o["RM3+LSA"]["MRR"] - o["LSA-QE"]["MRR"],
        "recall10": {m: o[m]["Recall@10"] for m in o},
        "rm3_ditambahkan_ke_lsa_recall10": o["RM3+LSA"]["Recall@10"] - o["LSA-QE"]["Recall@10"],
        "fail_rate": {m: o[m]["fail_rate"] for m in o},
    }

    def bucket(items, note):
        return {"count": len(items), "pct_of_units": len(items) / len(jaccard_rows),
                "definition": note,
                "by_type": dict(collections.Counter(x["qtype"] for x in items)),
                "by_scenario": dict(collections.Counter(x["scenario"] for x in items))}

    in_top10 = {m: sum(1 for r in jaccard_rows if r[f"rank_{m}"] <= 10)
                for m in ("B1", "RM3", "LSA-QE", "RM3+LSA")}
    rescue = {
        "rescued_by_combination_only": bucket(
            rescued, "RM3+LSA <= 10 sementara RM3 > 10 DAN LSA-QE > 10 (sinergi)"),
        "harmed_by_combination_only": bucket(
            harmed, "RM3 <= 10 DAN LSA-QE <= 10 sementara RM3+LSA > 10 (pasangan simetris)"),
        "lost_if_either_component_succeeded": bucket(
            lost_any, "RM3 <= 10 ATAU LSA-QE <= 10 sementara RM3+LSA > 10 (bukan pasangan "
                      "simetris dari sinergi; dicantumkan agar tidak disalahbaca)"),
        "net_symmetric": len(rescued) - len(harmed),
        "units_in_top10": in_top10,
    }

    # --- analisis kegagalan per tipe ---
    bt = d_test["by_type"]
    rb = d_test["robustness_vs_B1"]
    failure = {}
    for t in TYPE_ORDER:
        rows_t = [r for r in jaccard_rows if r["qtype"] == t]
        failure[t] = {
            "n": bt["B1"][t]["n"],
            "mrr": {m: bt[m][t]["MRR"] for m in bt},
            "delta_vs_B1": {m: bt[m][t]["MRR"] - bt["B1"][t]["MRR"] for m in bt if m != "B1"},
            "fail_rate": {m: bt[m][t]["fail_rate"] for m in bt},
            "robustness_index": {m: rb[m]["by_type"][t]["robustness_index"] for m in rb},
            "big_drops": {m: rb[m]["by_type"][t]["big_drops"] for m in rb},
            "mean_jaccard": jstats(rows_t).get("mean", 0.0),
            "mean_query_len": sum(r["query_len"] for r in rows_t) / len(rows_t) if rows_t else 0,
        }
    worse = sorted((t for t in TYPE_ORDER if failure[t]["delta_vs_B1"]["RM3+LSA"] < 0),
                   key=lambda t: failure[t]["delta_vs_B1"]["RM3+LSA"])

    # Diagnostik penanda format PES: seberapa sering penanda muncul di chunk gold,
    # dibanding seberapa sering ia muncul di chunk mana pun pada laporan yang sama.
    for qtype in TYPE_ORDER:
        qas = [q for q in corpus.qas if q.qtype == qtype]
        in_gold, shares = 0, []
        for q in qas:
            doc = corpus.documents[q.doc_id]
            gold = next(ch for ch in doc.chunks if ch.chunk_id == q.gold_chunk_id)
            if any(m in gold.tokens for m in PES_MARKERS):
                in_gold += 1
            shares.append(sum(1 for ch in doc.chunks
                              if any(m in ch.tokens for m in PES_MARKERS)) / len(doc.chunks))
        failure[qtype]["pes_markers"] = {
            "terms": list(PES_MARKERS),
            "share_of_gold_containing": in_gold / len(qas),
            "mean_share_of_chunks_containing": sum(shares) / len(shares),
        }

    payload = {
        "config_sha256": io.sha256_file(LOCKED_PATH),
        "n_units": len(jaccard_rows),
        "ablation": ablation,
        "jaccard": jaccard,
        "rescue": rescue,
        "failure_by_type": failure,
        "types_worse_than_B1": worse,
        "example_selection_rule": (
            "Tipe pada indeks genap TYPE_ORDER (0,2,4,6) ditambah indeks 7; tiap tipe "
            "diwakili unit seed 1 dengan qid terkecil; skenario PARA untuk contoh ke-2 "
            "dan ke-4, TPL untuk sisanya. Aturan ditetapkan tanpa melihat metrik."),
        "examples": examples,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    io.write_json("05_analysis.json", payload)
    io.write_csv("05_expansion_overlap.csv", jaccard_rows, fieldnames=list(jaccard_rows[0]))

    print(f"\nSelesai dalam {time.time() - t0:.0f}s")
    print(f"  Jaccard rata-rata {jaccard['overall']['mean']:.4f} "
          f"(maksimum mungkin {jaccard['max_possible']:.4f}), "
          f"{jaccard['overall']['pct_zero']:.1%} query tanpa irisan sama sekali")
    print(f"  Sinergi (kedua komponen gagal, gabungan berhasil): {len(rescued)}")
    print(f"  Kebalikannya (kedua komponen berhasil, gabungan gagal): {len(harmed)}")
    print(f"  Unit di top-10: " + ", ".join(f"{m} {v}" for m, v in rescue["units_in_top10"].items()))
    print(f"  Tipe yang turun dibanding B1 (RM3+LSA): {', '.join(worse) if worse else '-'}")
    return 0


def _example_slot(u):
    for slot, ti in enumerate(EXAMPLE_TYPE_INDICES):
        if u.qtype != TYPE_ORDER[ti]:
            continue
        want = "PARA" if slot in EXAMPLE_PARA_SLOTS else "TPL"
        if u.scenario == want:
            return slot
    return None


if __name__ == "__main__":
    raise SystemExit(main())
