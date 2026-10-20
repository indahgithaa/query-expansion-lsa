"""Tahap 5 - laporan hasil test. Hanya membaca dari `results/`."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io
from qe.corpus import TYPE_ORDER
from qe.engine import METHODS

OTHERS = ("RM3", "LSA-QE", "RM3+LSA")


def f(x, nd=4):
    return f"{float(x):.{nd}f}"


def main():
    d = io.read_json("04_test.json")
    overall, by_sc, by_type, by_seed = d["overall"], d["by_scenario"], d["by_type"], d["by_seed"]
    robust, sig = d["robustness_vs_B1"], d["significance"]
    cfg = d["config"]

    L = []
    w = L.append
    w("# Tahap 5 - Hasil test")
    w("")
    w("Sumber angka: `results/04_test.json` dan `results/04_test_units.csv`.")
    w("")
    w(f"Test dijalankan **satu kali** pada seed {d['seeds'][0]}-{d['seeds'][-1]} dengan config")
    w(f"terkunci `{cfg['path']}` (SHA-256 `{cfg['sha256']}`).")
    w(f"Total {d['n_units']} unit evaluasi = 10 seed x 2 skenario x ~172 QA test.")
    w("Parameter terpilih: " + "; ".join(
        f"{m} k={p['k']}, n={p['n']}, gamma={f(p['gamma'], 1)}" for m, p in cfg["methods"].items())
      + ". RM3 tidak di-tuning.")
    w("")

    w("## Hasil utama")
    w("")
    w("| Metode | MRR | R@1 | R@3 | R@5 | R@10 | R@20 | nDCG@10 | Median peringkat | Gagal (>10) |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in METHODS:
        o = overall[m]
        w(f"| {m} | {f(o['MRR'])} | {f(o['Recall@1'], 3)} | {f(o['Recall@3'], 3)} | "
          f"{f(o['Recall@5'], 3)} | {f(o['Recall@10'], 3)} | {f(o['Recall@20'], 3)} | "
          f"{f(o['nDCG@10'])} | {o['median_rank']:.0f} | {f(o['fail_rate'], 3)} |")
    w("")
    w("Urutan berbeda tergantung metrik, dan perbedaan itu adalah temuan utama laporan ini:")
    w("")
    w(f"- **MRR dan presisi puncak** dimenangkan LSA-QE ({f(overall['LSA-QE']['MRR'])}; "
      f"R@1 {f(overall['LSA-QE']['Recall@1'], 3)}).")
    w(f"- **Recall dalam dan tingkat kegagalan** dimenangkan RM3+LSA "
      f"(R@10 {f(overall['RM3+LSA']['Recall@10'], 3)}, R@20 {f(overall['RM3+LSA']['Recall@20'], 3)}, "
      f"gagal {f(overall['RM3+LSA']['fail_rate'], 3)} vs {f(overall['LSA-QE']['fail_rate'], 3)}).")
    w(f"- **nDCG@10 praktis seri** ({f(overall['LSA-QE']['nDCG@10'])} vs "
      f"{f(overall['RM3+LSA']['nDCG@10'])}).")
    w("")

    w("## Per skenario")
    w("")
    w("| Metode | MRR TPL | MRR PARA | Selisih | R@10 TPL | R@10 PARA |")
    w("| --- | ---: | ---: | ---: | ---: | ---: |")
    for m in METHODS:
        t, p = by_sc[m]["TPL"], by_sc[m]["PARA"]
        w(f"| {m} | {f(t['MRR'])} | {f(p['MRR'])} | {f(t['MRR'] - p['MRR'])} | "
          f"{f(t['Recall@10'], 3)} | {f(p['Recall@10'], 3)} |")
    w("")
    gap_b1 = by_sc["B1"]["TPL"]["MRR"] - by_sc["B1"]["PARA"]["MRR"]
    gap_best = by_sc["LSA-QE"]["TPL"]["MRR"] - by_sc["LSA-QE"]["PARA"]["MRR"]
    w(f"Parafrasa tetap lebih sulit daripada template untuk semua metode, tetapi ekspansi")
    w(f"mempersempit celahnya dari {f(gap_b1)} (B1) menjadi {f(gap_best)} (LSA-QE).")
    w(f"Keuntungan terbesar LSA-QE atas B1 justru ada di PARA "
      f"({f(by_sc['LSA-QE']['PARA']['MRR'] - by_sc['B1']['PARA']['MRR'])}) dibanding TPL "
      f"({f(by_sc['LSA-QE']['TPL']['MRR'] - by_sc['B1']['TPL']['MRR'])}): ekspansi memang")
    w("menolong justru ketika kata-kata query tidak lagi cocok persis dengan dokumen.")
    w("")

    w("## Per tipe pertanyaan")
    w("")
    w("Label tipe tidak dipakai saat inferensi; pemecahan ini hanya untuk analisis.")
    w("")
    w("| Tipe | n | B1 | RM3 | LSA-QE | RM3+LSA | d(LSA-QE) | d(RM3+LSA) |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for t in TYPE_ORDER:
        b = by_type["B1"][t]["MRR"]
        row = " | ".join(f(by_type[m][t]["MRR"]) for m in METHODS)
        w(f"| {t} | {by_type['B1'][t]['n']} | {row} | "
          f"{by_type['LSA-QE'][t]['MRR'] - b:+.4f} | {by_type['RM3+LSA'][t]['MRR'] - b:+.4f} |")
    w("")
    worse = [t for t in TYPE_ORDER if by_type["RM3+LSA"][t]["MRR"] < by_type["B1"][t]["MRR"]]
    w(f"Tipe yang **turun** dibanding B1 pada metode utama: {', '.join(worse) if worse else 'tidak ada'}.")
    w("Analisis kegagalannya ada di `reports/05_analisis.md`.")
    w("")

    w("## Per seed")
    w("")
    w("| Seed | " + " | ".join(METHODS) + " | Pemenang |")
    w("| ---: | " + " | ".join("---:" for _ in METHODS) + " | --- |")
    wins = {m: 0 for m in METHODS}
    for s in d["seeds"]:
        vals = {m: by_seed[m][str(s)]["MRR"] for m in METHODS}
        best = max(vals, key=lambda m: vals[m])
        wins[best] += 1
        w(f"| {s} | " + " | ".join(f(vals[m]) for m in METHODS) + f" | {best} |")
    w("")
    w("Jumlah seed dimenangkan: " + ", ".join(f"{m} {wins[m]}" for m in METHODS if wins[m]) + ".")
    w("Keunggulan LSA-QE atas RM3+LSA tidak konsisten antar seed, dan keduanya selalu")
    w("mengungguli B1.")
    w("")

    w("## Robustness terhadap B1")
    w("")
    w("| Metode | Naik | Sama | Turun | RI | Masuk top-10 | Keluar top-10 | Penurunan besar (dRR < -0,5) |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in OTHERS:
        r = robust[m]["overall"]
        w(f"| {m} | {r['improved']} | {r['unchanged']} | {r['worsened']} | "
          f"{r['robustness_index']:+.4f} | {r['entered_top10']} | {r['left_top10']} | {r['big_drops']} |")
    w("")
    rc, rl = robust["RM3+LSA"]["overall"], robust["LSA-QE"]["overall"]
    w(f"Di sinilah metode gabungan unggul paling jelas. Robustness Index RM3+LSA")
    w(f"({rc['robustness_index']:+.4f}) jauh di atas LSA-QE ({rl['robustness_index']:+.4f}),")
    w(f"dan jumlah penurunan besar hanya {rc['big_drops']} dibanding {rl['big_drops']} pada")
    w(f"LSA-QE - kurang dari separuhnya. RM3+LSA juga memasukkan lebih banyak query ke")
    w(f"top-10 ({rc['entered_top10']} vs {rl['entered_top10']}) sambil mengeluarkan lebih")
    w(f"sedikit ({rc['left_top10']} vs {rl['left_top10']}).")
    w("")
    w("Dengan kata lain: LSA sendiri lebih agresif - lebih sering menang besar, tetapi juga")
    w("lebih sering merusak query yang semula sudah baik. Komponen RM3 meredam kerusakan itu.")
    w("")
    w("### Robustness per skenario")
    w("")
    w("| Metode | RI TPL | RI PARA | Penurunan besar TPL | Penurunan besar PARA |")
    w("| --- | ---: | ---: | ---: | ---: |")
    for m in OTHERS:
        w(f"| {m} | {robust[m]['TPL']['robustness_index']:+.4f} | "
          f"{robust[m]['PARA']['robustness_index']:+.4f} | {robust[m]['TPL']['big_drops']} | "
          f"{robust[m]['PARA']['big_drops']} |")
    w("")

    w("## Signifikansi")
    w("")
    w("Keluarga perbandingan ditetapkan sebelum test dibuka (lima perbandingan).")
    import json as _json
    with open(os.path.join(io.ROOT, cfg["path"]), encoding="utf-8") as fh:
        rng_seed = _json.load(fh)["test"]["bootstrap"]["rng_seed"]
    w(f"CI 95% dari paired bootstrap berstrata (seed x skenario, "
      f"{sig[0]['n_resamples']:,} resample, rng {rng_seed}),".replace(",", "."))
    w("ditambah Wilcoxon signed-rank dengan koreksi Holm atas kelima p-value.")
    w("")
    w("| Perbandingan | dMRR | CI 95% | CI tidak memuat 0 | p Wilcoxon | p Holm | Signifikan |")
    w("| --- | ---: | --- | :---: | ---: | ---: | :---: |")
    for t in sig:
        w(f"| {t['a']} vs {t['b']} | {t['delta']:+.4f} | "
          f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | {'ya' if t['ci_excludes_zero'] else 'tidak'} | "
          f"{t['wilcoxon_p']:.3g} | {t['wilcoxon_p_holm']:.3g} | "
          f"{'**ya**' if t['significant_holm_0.05'] else 'tidak'} |")
    w("")
    last = sig[-1]
    w("Pembacaan:")
    w("")
    w("1. Ketiga metode QE mengungguli B1 secara signifikan, dan **RM3+LSA mengungguli RM3**")
    w(f"   secara signifikan ({sig[3]['delta']:+.4f}). Jadi menambahkan LSA di atas PRF jelas berguna.")
    w(f"2. RM3 sendiri hanya menambah {sig[0]['delta']:+.4f} di atas B1 - signifikan secara")
    w("   statistik karena n besar, tetapi kecil secara praktis.")
    w(f"3. **RM3+LSA vs LSA-QE** ({last['delta']:+.4f}) memberi dua sinyal yang berbeda:")
    w(f"   CI bootstrap [{last['ci_low']:+.4f}, {last['ci_high']:+.4f}] tidak memuat nol,")
    w(f"   tetapi Wilcoxon tidak signifikan setelah Holm (p = {last['wilcoxon_p_holm']:.3g}).")
    w("   Keduanya tidak bertentangan: bootstrap menguji selisih **rata-rata** RR dan peka")
    w("   pada sedikit query dengan perubahan besar, sedangkan Wilcoxon menguji **arah**")
    w(f"   perubahan pada {last['wilcoxon_n_nonzero']} dari {last['n']} unit yang peringkatnya")
    w("   memang berubah. Arah perubahan itu ternyata berimbang, sehingga selisih MRR yang")
    w("   kecil tadi tidak didukung bukti arah yang konsisten. Kesimpulan yang dapat")
    w("   dipertahankan: **LSA-QE dan RM3+LSA setara pada MRR**, bukan bahwa LSA-QE lebih baik.")
    w("")

    w("## Jawaban atas pertanyaan penelitian")
    w("")
    w("> Apakah menggabungkan ekspansi global (LSA) dengan ekspansi lokal (PRF/RM3) lebih")
    w("> baik daripada tanpa QE, RM3 saja, dan LSA saja?")
    w("")
    w(f"- **Lebih baik daripada tanpa QE**: ya, jelas dan signifikan "
      f"({sig[2]['delta']:+.4f} MRR, CI [{sig[2]['ci_low']:+.4f}, {sig[2]['ci_high']:+.4f}]).")
    w(f"- **Lebih baik daripada RM3 saja**: ya, jelas dan signifikan "
      f"({sig[3]['delta']:+.4f} MRR).")
    w("- **Lebih baik daripada LSA saja**: **tidak pada MRR.** Keduanya setara pada MRR dan")
    w("  nDCG@10. Metode gabungan unggul pada recall dalam (R@10, R@20), tingkat kegagalan,")
    w("  dan terutama pada robustness (RI +0,164 vs +0,106; penurunan besar 53 vs 107).")
    w("")
    w("Dalam kerangka Xu & Croft (1996), hasil ini menunjukkan analisis **global** adalah")
    w("sumber perbaikan utama pada korpus ini, sedangkan analisis **lokal** terutama")
    w("berperan sebagai penstabil: ia tidak menaikkan MRR rata-rata di atas LSA, tetapi")
    w("memangkas separuh kasus di mana ekspansi merusak query yang semula baik.")
    w("")
    w("Catatan pembatas: `gamma = 2` dipakai untuk kedua metode atas keputusan pengguna,")
    w("sedangkan di dev RM3+LSA mencapai nilai tertinggi pada `gamma = 4` (batas grid).")
    w("Angka RM3+LSA di sini karena itu adalah batas bawah yang konservatif.")

    path = io.write_report("04_test.md", "\n".join(L))
    print(f"-> {io.rel(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
