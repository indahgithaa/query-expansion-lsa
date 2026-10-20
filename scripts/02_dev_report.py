"""Tahap 4 - laporan seleksi dev.

Hanya membaca dari `results/` sehingga setiap angka di laporan dapat ditelusuri
ke berkas hasil (aturan tetap no. 4).
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io

PARAM_METHODS = ("LSA-QE", "RM3+LSA")


def fmt(x, nd=4):
    return f"{float(x):.{nd}f}"


def main():
    sel = io.read_json("02_dev_selection.json")
    with open(os.path.join(io.RESULTS, "02_dev_grid.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    grid = sel["grid"]
    base = sel["baselines"]
    n_configs = len(grid["k"]) * len(grid["n"]) * len(grid["gamma"])

    L = []
    w = L.append
    w("# Tahap 4 - Seleksi parameter di dev")
    w("")
    w("Sumber angka: `results/02_dev_selection.json` dan `results/02_dev_grid.csv`.")
    w("Test belum dibuka. Seluruh isi laporan ini berasal dari seed 1-3 pada bagian **dev**")
    w("(24 dokumen per seed); dokumen test tidak pernah disentuh, dan LSA hanya dilatih")
    w("pada 72 dokumen koleksi tiap seed.")
    w("")

    w("## Ringkasan")
    w("")
    w(f"- Grid {n_configs} konfigurasi per metode berparameter: "
      f"k in {{{', '.join(map(str, grid['k']))}}}, "
      f"n in {{{', '.join(map(str, grid['n']))}}}, "
      f"gamma in {{{', '.join(fmt(g, 1) for g in grid['gamma'])}}}.")
    w(f"- Kriteria: {sel['criterion']}. Seri dipecah dengan {sel['tie_break']}.")
    w(f"- Unit evaluasi: {base['B1']['n_units']} (3 seed x 2 skenario x ~171 QA dev).")
    w("- RM3 tidak di-tuning: 5 chunk feedback, 30 term, lambda = 0,5, ditetapkan a priori.")
    w(f"- Waktu komputasi grid: {sel['runtime_seconds']:.0f} detik.")
    w("")

    w("## Baseline tanpa tuning")
    w("")
    w("| Metode | MRR TPL | MRR PARA | Kriteria | Recall@1 | Recall@10 | nDCG@10 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in ("B1", "RM3"):
        b = base[m]
        w(f"| {m} | {fmt(b['MRR_TPL'])} | {fmt(b['MRR_PARA'])} | **{fmt(b['selection_score'])}** | "
          f"{fmt(b['Recall@1'], 3)} | {fmt(b['Recall@10'], 3)} | {fmt(b['nDCG@10'])} |")
    w("")
    gap_b1 = float(base["B1"]["MRR_TPL"]) - float(base["B1"]["MRR_PARA"])
    w(f"Parafrasa jauh lebih sulit daripada template: pada B1 selisih MRR antara TPL dan PARA")
    w(f"mencapai {fmt(gap_b1)}. Inilah celah yang ingin ditutup oleh ekspansi query.")
    w("")

    w("## Konfigurasi terpilih")
    w("")
    w("| Metode | k | n | gamma | Kriteria (dev) | MRR TPL | MRR PARA |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in PARAM_METHODS:
        s = sel["selection"][m]
        b = s["best"]
        row = next(r for r in rows if r["method"] == m and int(r["k"]) == b["k"]
                   and int(r["n"]) == b["n"] and float(r["gamma"]) == b["gamma"])
        w(f"| {m} | {b['k']} | {b['n']} | {fmt(b['gamma'], 1)} | **{fmt(s['best_score'])}** | "
          f"{fmt(row['MRR_TPL'])} | {fmt(row['MRR_PARA'])} |")
    w("")
    w("Parameter dipilih terpisah untuk tiap metode berparameter supaya perbandingan")
    w("RM3+LSA vs LSA-QE pada Tahap 5 tidak berat sebelah (lihat `DECISIONS.md`).")
    w("")

    w("## Delapan konfigurasi teratas per metode")
    w("")
    for m in PARAM_METHODS:
        sub = sorted((r for r in rows if r["method"] == m),
                     key=lambda r: -float(r["selection_score"]))
        w(f"### {m}")
        w("")
        w("| # | k | n | gamma | Kriteria | MRR TPL | MRR PARA | Recall@1 | Recall@10 |")
        w("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for i, r in enumerate(sub[:8], 1):
            w(f"| {i} | {r['k']} | {r['n']} | {fmt(r['gamma'], 1)} | {fmt(r['selection_score'])} | "
              f"{fmt(r['MRR_TPL'])} | {fmt(r['MRR_PARA'])} | {fmt(r['Recall@1'], 3)} | "
              f"{fmt(r['Recall@10'], 3)} |")
        worst = min(float(r["selection_score"]) for r in sub)
        w("")
        w(f"Rentang seluruh {len(sub)} konfigurasi: {fmt(worst)} sampai "
          f"{fmt(max(float(r['selection_score']) for r in sub))}.")
        w("")

    w("## Sensitivitas terhadap tiap parameter")
    w("")
    w("Tiap sel adalah kriteria terbaik yang dicapai saat parameter itu dikunci")
    w("(dua parameter lain dibebaskan).")
    w("")
    for m in PARAM_METHODS:
        sens = sel["sensitivity"][m]
        w(f"**{m}**")
        w("")
        for name, label in (("k", "k"), ("n", "n"), ("gamma", "gamma")):
            keys = list(sens[name])
            w("| " + label + " | " + " | ".join(keys) + " |")
            w("| --- | " + " | ".join("---:" for _ in keys) + " |")
            w("| kriteria | " + " | ".join(fmt(sens[name][x]) for x in keys) + " |")
            w("")

    w("## Optimum di batas grid")
    w("")
    any_boundary = False
    for m in PARAM_METHODS:
        s = sel["selection"][m]
        at = s["at_grid_boundary"]
        hit = [p for p, v in at.items() if v]
        if hit:
            any_boundary = True
            w(f"- **{m}**: parameter {', '.join(hit)} jatuh di batas grid "
              f"(k={s['best']['k']}, n={s['best']['n']}, gamma={fmt(s['best']['gamma'], 1)}).")
        else:
            w(f"- **{m}**: tidak ada parameter di batas grid "
              f"(k={s['best']['k']}, n={s['best']['n']}, gamma={fmt(s['best']['gamma'], 1)}).")
    w("")
    if any_boundary:
        w("Sesuai aturan tetap, hal ini **dilaporkan saja dan grid tidak diperluas**.")
        w("Konsekuensinya perlu dicatat saat menafsirkan hasil: kriteria RM3+LSA naik monoton")
        w("terhadap gamma di seluruh grid, artinya metode gabungan masih ingin memberi bobot")
        w("lebih besar pada komponen LSA. Nilai gamma optimum yang sebenarnya mungkin > 4,")
        w("dan angka RM3+LSA pada Tahap 5 karenanya adalah batas bawah.")
    w("")

    w("## Bias seleksi (leave-one-seed-out)")
    w("")
    w("Untuk tiap seed yang ditahan, config dipilih ulang hanya dari dua seed sisanya,")
    w("lalu dinilai pada seed yang ditahan. Selisihnya terhadap nilai in-sample")
    w("memperkirakan seberapa optimistis angka dev.")
    w("")
    w("| Metode | Kriteria in-sample | LOSO | Bias | Config terpilih per lipatan |")
    w("| --- | ---: | ---: | ---: | --- |")
    for m in PARAM_METHODS:
        s = sel["selection"][m]
        picks = "; ".join(
            f"tahan seed {x['held_out_seed']} -> k={x['config']['k']}, n={x['config']['n']}, "
            f"gamma={fmt(x['config']['gamma'], 1)}" for x in s["loso"])
        w(f"| {m} | {fmt(s['in_sample_mean_per_seed'])} | {fmt(s['loso_mean'])} | "
          f"{s['selection_bias']:+.4f} | {picks} |")
    w("")
    w("Bias kecil untuk kedua metode, dan konfigurasi yang dipilih tiap lipatan berdekatan")
    w("(k 300-500, n 5-10), jadi seleksi tidak terlihat menempel pada satu seed tertentu.")
    w("")

    w("## Temuan yang perlu diperhatikan sebelum test dibuka")
    w("")
    lsa_s = sel["selection"]["LSA-QE"]["best_score"]
    comb_s = sel["selection"]["RM3+LSA"]["best_score"]
    b1_s = float(base["B1"]["selection_score"])
    rm3_s = float(base["RM3"]["selection_score"])
    w(f"1. Kedua metode berbasis LSA unggul jelas atas B1 ({fmt(b1_s)}) dan RM3 ({fmt(rm3_s)}).")
    w(f"   RM3 sendiri hanya menambah {fmt(rm3_s - b1_s)} di atas B1.")
    w(f"2. **Pada dev, LSA-QE ({fmt(lsa_s)}) sedikit mengungguli RM3+LSA ({fmt(comb_s)}),")
    w(f"   selisih {fmt(lsa_s - comb_s)}.** Dengan kata lain, menambahkan RM3 di atas LSA")
    w("   belum terlihat memberi keuntungan pada dev. Ini langsung menyentuh pertanyaan")
    w("   penelitian dan harus dilaporkan apa adanya, bukan dikompensasi dengan mengubah grid.")
    w("3. Arah gamma pada RM3+LSA (monoton naik sampai batas grid) mendukung pembacaan yang")
    w("   sama: semakin besar porsi LSA, semakin baik metode gabungan.")
    w("4. Ekspansi menutup sebagian celah TPL-PARA: pada B1 selisihnya "
      f"{fmt(gap_b1)}, pada LSA-QE terpilih "
      f"{fmt(float(next(r for r in rows if r['method'] == 'LSA-QE' and int(r['k']) == sel['selection']['LSA-QE']['best']['k'] and int(r['n']) == sel['selection']['LSA-QE']['best']['n'] and float(r['gamma']) == sel['selection']['LSA-QE']['best']['gamma'])['MRR_TPL']) - float(next(r for r in rows if r['method'] == 'LSA-QE' and int(r['k']) == sel['selection']['LSA-QE']['best']['k'] and int(r['n']) == sel['selection']['LSA-QE']['best']['n'] and float(r['gamma']) == sel['selection']['LSA-QE']['best']['gamma'])['MRR_PARA']))}.")
    w("")

    w("## Status")
    w("")
    w("Config **belum dikunci** dan test **belum dibuka**. Langkah berikutnya menunggu")
    w("keputusan pengguna sesuai Tahap 4:")
    w("")
    w("1. Setujui konfigurasi di atas (atau tentukan yang lain dari `results/02_dev_grid.csv`).")
    w("2. `scripts/03_lock_config.py` menulis `configs/locked_config.json` dan mencatat")
    w("   SHA-256-nya di `DECISIONS.md`.")
    w("3. Baru setelah itu `scripts/04_test.py` dijalankan satu kali pada seed 1-10.")

    path = io.write_report("02_dev_selection.md", "\n".join(L))
    print(f"-> {io.rel(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
