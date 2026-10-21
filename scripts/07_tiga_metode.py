"""Agregasi ulang hasil test untuk tiga metode: Baseline, RM3, LSA-QE.

Skrip ini TIDAK menjalankan ulang retrieval. Seluruh angka dihitung ulang dari
`results/04_test_units.csv`, yaitu keluaran per unit dari test yang sudah
dikunci. Yang berubah hanya cakupan pelaporan: metode RM3+LSA tidak lagi
dilaporkan, sehingga keluarga uji Holm menyusut dari lima pasangan menjadi tiga
dan satu pasangan baru masuk, yaitu LSA-QE terhadap RM3.

Label "B1" pada berkas keluaran pipeline dipetakan menjadi "Baseline" pada
laporan, karena metode tersebut adalah retrieval BM25 tanpa query expansion.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.qe import metrics, stats  # noqa: E402

AKAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
UNITS = os.path.join(AKAR, "results", "04_test_units.csv")
KELUAR = os.path.join(AKAR, "results", "07_tiga_metode.json")

# nama pipeline -> nama laporan
NAMA = {"B1": "Baseline", "RM3": "RM3", "LSA-QE": "LSA-QE"}
URUT = ["Baseline", "RM3", "LSA-QE"]
DASAR = "Baseline"

# pasangan yang diuji, (A, B) dengan delta = A - B
PASANGAN = [
    ("RM3", "Baseline"),
    ("LSA-QE", "Baseline"),
    ("LSA-QE", "RM3"),
]


def baca_unit():
    """key (seed, scenario, qid) -> {metode -> rank}, plus metadata unit."""
    unit = {}
    with open(UNITS, newline="", encoding="utf-8") as f:
        for b in csv.DictReader(f):
            if b["method"] not in NAMA:
                continue
            k = (int(b["seed"]), b["scenario"], int(b["qid"]))
            d = unit.setdefault(k, {"qtype": b["qtype"], "doc_id": b["doc_id"],
                                    "rank": {}})
            d["rank"][NAMA[b["method"]]] = int(b["rank"])
    lengkap = {k: v for k, v in unit.items() if len(v["rank"]) == len(URUT)}
    if len(lengkap) != len(unit):
        raise SystemExit(f"ada unit tak lengkap: {len(unit) - len(lengkap)}")
    return lengkap


def agregat(unit, kunci=None):
    """Ringkasan metrik tiap metode atas subhimpunan unit."""
    pilih = [v for k, v in unit.items() if kunci is None or kunci(k, v)]
    return {m: metrics.summarize([u["rank"][m] for u in pilih]) for m in URUT}


def main():
    unit = baca_unit()
    kunci = sorted(unit)
    seeds = sorted({k[0] for k in kunci})
    skenario = sorted({k[1] for k in kunci})
    tipe = sorted({v["qtype"] for v in unit.values()})
    print(f"unit: {len(kunci)} | seed: {seeds} | skenario: {skenario}")

    hasil = {
        "catatan": "Agregasi ulang dari results/04_test_units.csv untuk tiga "
                   "metode. Retrieval tidak dijalankan ulang.",
        "sumber": "results/04_test_units.csv",
        "metode": URUT,
        "metode_dasar": DASAR,
        "pemetaan_nama": NAMA,
        "n_units": len(kunci),
        "seeds": seeds,
        "scenarios": skenario,
        "overall": agregat(unit),
        "by_scenario": {s: agregat(unit, lambda k, v, s=s: k[1] == s)
                        for s in skenario},
        "by_type": {t: agregat(unit, lambda k, v, t=t: v["qtype"] == t)
                    for t in tipe},
        "by_seed": {str(s): agregat(unit, lambda k, v, s=s: k[0] == s)
                    for s in seeds},
    }

    # indeks ketahanan terhadap Baseline
    dasar = [unit[k]["rank"][DASAR] for k in kunci]
    ket = {}
    for m in URUT:
        if m == DASAR:
            continue
        ranks = [unit[k]["rank"][m] for k in kunci]
        blok = {"overall": metrics.robustness(ranks, dasar)}
        for s in skenario:
            sub = [k for k in kunci if k[1] == s]
            blok[s] = metrics.robustness([unit[k]["rank"][m] for k in sub],
                                         [unit[k]["rank"][DASAR] for k in sub])
        blok["by_type"] = {}
        for t in tipe:
            sub = [k for k in kunci if unit[k]["qtype"] == t]
            blok["by_type"][t] = metrics.robustness(
                [unit[k]["rank"][m] for k in sub],
                [unit[k]["rank"][DASAR] for k in sub])
        ket[m] = blok
    hasil["robustness_vs_baseline"] = ket

    # uji signifikansi: bootstrap berstrata + Wilcoxon + Holm atas tiga pasangan
    strata = [(k[0], k[1]) for k in kunci]
    rr = {m: [metrics.reciprocal_rank(unit[k]["rank"][m]) for k in kunci]
          for m in URUT}

    uji = []
    for a, b in PASANGAN:
        bs = stats.stratified_paired_bootstrap(rr[a], rr[b], strata)
        w = stats.wilcoxon(rr[a], rr[b])
        wx = {"wilcoxon_statistic": w["statistic"],
              "wilcoxon_p": w["p_value"],
              "wilcoxon_n_nonzero": w["n_nonzero"]}
        uji.append({"a": a, "b": b, **bs, **wx})
        print(f"{a} vs {b}: delta={bs['delta']:+.6f} "
              f"CI=[{bs['ci_low']:+.6f}; {bs['ci_high']:+.6f}] "
              f"p={wx['wilcoxon_p']:.6g}")

    p_holm = stats.holm([u["wilcoxon_p"] for u in uji])
    for u, p in zip(uji, p_holm):
        u["wilcoxon_p_holm"] = float(p)
        u["significant_holm_0.05"] = bool(p < 0.05)
        u["ci_excludes_zero"] = bool(u["ci_low"] > 0 or u["ci_high"] < 0)
        u["signifikan"] = bool(u["significant_holm_0.05"] and u["ci_excludes_zero"])
    hasil["significance"] = uji
    hasil["n_perbandingan_holm"] = len(uji)

    with open(KELUAR, "w", encoding="utf-8", newline="\n") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print()
    print(f"{'pasangan':<22}{'delta':>10}{'p Holm':>12}{'CI != 0':>9}{'signifikan':>12}")
    for u in uji:
        print(f"{u['a'] + ' vs ' + u['b']:<22}{u['delta']:>+10.4f}"
              f"{u['wilcoxon_p_holm']:>12.4g}{str(u['ci_excludes_zero']):>9}"
              f"{str(u['signifikan']):>12}")
    print()
    print(f"ditulis: {KELUAR}")


if __name__ == "__main__":
    main()
