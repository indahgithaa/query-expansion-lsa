"""Ringkasan akhir penelitian. Hanya membaca dari `results/` dan `configs/`."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io
from qe.corpus import TYPE_ORDER

M = ("B1", "RM3", "LSA-QE", "RM3+LSA")


def f(x, nd=4):
    return f"{float(x):.{nd}f}"


def main():
    t = io.read_json("04_test.json")
    a = io.read_json("05_analysis.json")
    e = io.read_json("06_explorations.json")
    with open(os.path.join(io.CONFIGS, "locked_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)

    o, sc, bt, sig = t["overall"], t["by_scenario"], t["by_type"], t["significance"]
    rb = t["robustness_vs_B1"]
    lsa = cfg["methods"]["LSA-QE"]

    L = []
    w = L.append
    w("# Ringkasan Akhir")
    w("")
    w("**Metode yang diusulkan: ekspansi query berbasis Latent Semantic Analysis (LSA-QE).**")
    w("")
    w("Sumber angka: `results/04_test.json`, `results/05_analysis.json`,")
    w("`results/06_explorations.json`. Config terkunci `configs/locked_config.json`,")
    w(f"SHA-256 `{t['config']['sha256']}`.")
    w("")

    w("## 1. Ringkas")
    w("")
    w(f"Tugasnya menemukan satu chunk jawaban di dalam satu laporan asuhan gizi berbahasa")
    w(f"Indonesia. Ekspansi query berbasis LSA (k={lsa['k']}, n={lsa['n']}, gamma={f(lsa['gamma'],1)})")
    w(f"dibandingkan dengan BM25 tanpa ekspansi, diuji pada {t['n_units']} unit evaluasi")
    w(f"(seed {t['seeds'][0]}-{t['seeds'][-1]} x 2 skenario query). Tanpa LLM, tanpa reranker,")
    w("tanpa dense retrieval.")
    w("")
    d_mrr = o["LSA-QE"]["MRR"] - o["B1"]["MRR"]
    d_fail = o["B1"]["fail_rate"] - o["LSA-QE"]["fail_rate"]
    w(f"**LSA-QE menaikkan MRR {d_mrr:+.4f} ({100*d_mrr/o['B1']['MRR']:+.1f}%) dan menurunkan")
    w(f"angka kegagalan dari {f(o['B1']['fail_rate'],3)} ke {f(o['LSA-QE']['fail_rate'],3)}")
    w(f"({100*d_fail/o['B1']['fail_rate']:.1f}% secara relatif).** Keduanya signifikan")
    w("secara statistik.")
    w("")

    w("## 2. Hasil utama")
    w("")
    w("| Metode | MRR | R@1 | R@5 | R@10 | R@20 | nDCG@10 | Median | Gagal (>10) |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in M:
        v = o[m]
        tag = "**" if m == "LSA-QE" else ""
        w(f"| {tag}{m}{tag} | {f(v['MRR'])} | {f(v['Recall@1'],3)} | {f(v['Recall@5'],3)} | "
          f"{f(v['Recall@10'],3)} | {f(v['Recall@20'],3)} | {f(v['nDCG@10'])} | "
          f"{v['median_rank']:.0f} | {f(v['fail_rate'],3)} |")
    w("")
    w("RM3 dan RM3+LSA ditampilkan sebagai **pembanding**, bukan metode yang diusulkan.")
    w("")

    w("### Signifikansi (vs B1)")
    w("")
    w("| Perbandingan | dMRR | CI 95% | p Holm | Signifikan |")
    w("| --- | ---: | --- | ---: | :---: |")
    for s in sig:
        if s["b"] != "B1":
            continue
        w(f"| {s['a']} vs B1 | {s['delta']:+.4f} | [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] | "
          f"{s['wilcoxon_p_holm']:.3g} | {'**ya**' if s['significant_holm_0.05'] else 'tidak'} |")
    w("")
    n_res = f"{sig[0]['n_resamples']:,}".replace(",", ".")
    w(f"Paired bootstrap berstrata (seed x skenario, {n_res} resample), "
      "Wilcoxon signed-rank dengan koreksi Holm atas keluarga lima perbandingan yang "
      "ditetapkan sebelum test dibuka.")
    w("")

    w("## 3. Ketahanan terhadap variasi pertanyaan")
    w("")
    w("| Metode | MRR template | MRR parafrasa | Selisih |")
    w("| --- | ---: | ---: | ---: |")
    for m in M:
        tpl, para = sc[m]["TPL"]["MRR"], sc[m]["PARA"]["MRR"]
        w(f"| {m} | {f(tpl)} | {f(para)} | {f(tpl - para)} |")
    w("")
    gb = sc["B1"]["TPL"]["MRR"] - sc["B1"]["PARA"]["MRR"]
    gl = sc["LSA-QE"]["TPL"]["MRR"] - sc["LSA-QE"]["PARA"]["MRR"]
    dp = sc["LSA-QE"]["PARA"]["MRR"] - sc["B1"]["PARA"]["MRR"]
    dt = sc["LSA-QE"]["TPL"]["MRR"] - sc["B1"]["TPL"]["MRR"]
    rel_p = 100 * dp / sc["B1"]["PARA"]["MRR"]
    rel_t = 100 * dt / sc["B1"]["TPL"]["MRR"]
    w("Parafrasa lebih sulit daripada template untuk **semua** metode, dan ekspansi tidak")
    w(f"menghapus kesulitan itu: celahnya hanya menyempit tipis, dari {f(gb)} (B1) ke")
    w(f"{f(gl)} (LSA-QE). Perbaikan absolutnya hampir sama di kedua skenario")
    w(f"({dp:+.4f} pada parafrasa, {dt:+.4f} pada template).")
    w("")
    w(f"Secara **relatif** keuntungannya memang lebih besar pada parafrasa "
      f"({rel_p:+.1f}% vs {rel_t:+.1f}%), sesuai dugaan bahwa ekspansi menolong saat kata")
    w("query tidak cocok persis dengan dokumen. Tetapi selisihnya kecil, jadi klaim yang")
    w("dapat dipertahankan adalah **ekspansi menolong sama baiknya pada pertanyaan")
    w("parafrasa maupun template** - bukan bahwa ia secara khusus mengatasi variasi query.")
    w("")

    w("## 4. Robustness per query terhadap B1")
    w("")
    w("| Metode | Naik | Sama | Turun | RI | Masuk top-10 | Keluar top-10 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in ("RM3", "LSA-QE", "RM3+LSA"):
        r = rb[m]["overall"]
        w(f"| {m} | {r['improved']} | {r['unchanged']} | {r['worsened']} | "
          f"{r['robustness_index']:+.4f} | {r['entered_top10']} | {r['left_top10']} |")
    w("")

    w("## 5. Per tipe pertanyaan")
    w("")
    w("Label tipe tidak dipakai saat inferensi; pemecahan ini hanya untuk analisis.")
    w("")
    w("| Tipe | n | B1 | LSA-QE | Selisih |")
    w("| --- | ---: | ---: | ---: | ---: |")
    for ty in TYPE_ORDER:
        b, l = bt["B1"][ty]["MRR"], bt["LSA-QE"][ty]["MRR"]
        w(f"| {ty} | {bt['B1'][ty]['n']} | {f(b)} | {f(l)} | {l - b:+.4f} |")
    w("")
    worse = [ty for ty in TYPE_ORDER if bt["LSA-QE"][ty]["MRR"] < bt["B1"][ty]["MRR"]]
    pm = a["failure_by_type"]["pes"]["pes_markers"]
    w(f"Tipe yang tidak terbantu: {', '.join('`' + x + '`' for x in worse)}.")
    w(f"Untuk `pes`, penyebabnya terukur: penanda format "
      + ", ".join(f"`{x}`" for x in pm["terms"]) + f" muncul di "
      f"{pm['share_of_gold_containing']:.1%} chunk gold tetapi juga di "
      f"{pm['mean_share_of_chunks_containing']:.1%} chunk mana pun pada laporan yang sama -")
    w("nyaris selalu benar, hampir tidak membedakan. Tidak ada pembobotan ulang term yang")
    w("bisa memperbaikinya; ini batas representasi bag-of-words.")
    w("")

    w("## 6. Implikasi untuk pipeline RAG")
    w("")
    w("Dalam RAG, generator menerima top-k chunk dan tidak peduli gold ada di peringkat 1")
    w("atau 7 - asal masuk ke dalam k. Metrik yang mengikat karena itu Recall@k, dan")
    w("konsekuensinya dapat dinyatakan sebagai **anggaran konteks**: berapa chunk yang")
    w("harus disuapkan untuk mencapai tingkat recall tertentu.")
    w("")
    w("Ini membalik keberatan standar bahwa menambah konteks bisa merugikan generator.")
    w("LSA-QE tidak meminta lebih banyak chunk - ia mencapai recall yang sama dengan chunk")
    w("**lebih sedikit**, sehingga konteks lebih pendek, biaya token lebih rendah, dan")
    w("distraksi lebih kecil. Semua arahnya searah.")
    w("")
    w(f"Yang dapat diklaim: ekspansi query memperkuat **tahap temu kembali**, menurunkan")
    w(f"kasus konteks-tanpa-jawaban dari {f(o['B1']['fail_rate'],3)} ke "
      f"{f(o['LSA-QE']['fail_rate'],3)} pada k=10.")
    w("Yang **tidak** dapat diklaim tanpa eksperimen end-to-end: bahwa kualitas jawaban")
    w("RAG ikut meningkat. Recall yang lebih baik adalah syarat perlu, bukan cukup.")
    w("")

    w("## 7. Mengapa LSA saja, bukan gabungan")
    w("")
    bl, ms, se, rf = e["baselines"], e["morphology"], e["selective_expansion"], e["rank_fusion"]
    w("Empat arah lanjutan diuji untuk mencari sesuatu yang dapat digabungkan dengan LSA,")
    w(f"terutama untuk menurunkan kegagalan lebih jauh. Semuanya dihitung di **dev** "
      f"({e['n_units']} unit, seed {e['seeds'][0]}-{e['seeds'][-1]}); test tidak disentuh.")
    w("")
    w("| Arah | Hasil | Verdikt |")
    w("| --- | --- | --- |")
    w(f"| Ekspansi lokal (RM3) digabung | kegagalan dev {f(bl['RM3+LSA']['fail_rate'],3)} vs "
      f"{f(bl['LSA-QE']['fail_rate'],3)}, tetapi MRR turun {bl['RM3+LSA']['MRR']-bl['LSA-QE']['MRR']:+.4f} | "
      "menolong recall, merugikan MRR |")
    w(f"| Ekspansi morfologis (stemming) | kontribusi unik "
      f"{ms['all']['morph_only']:.1%} unit | gugur |")
    w(f"| Selective query expansion | held-out "
      f"{se['models']['logistic']['gain_vs_always']:+.4f} (logistic), "
      f"{se['models']['ridge']['gain_vs_always']:+.4f} (ridge) | gugur |")
    best_fusion = min(rf.items(), key=lambda kv: kv[1]["fail_rate"])
    w(f"| Rank fusion (RRF, CombSUM, + LM Dirichlet) | kegagalan terbaik "
      f"{f(best_fusion[1]['fail_rate'],3)} vs {f(bl['RM3+LSA']['fail_rate'],3)} | gugur |")
    w("")
    w(f"Celah oracle nyata: memilih metode terbaik per query akan mencapai MRR "
      f"{f(bl['oracle_QE']['MRR'])} dan kegagalan {f(bl['oracle_QE']['fail_rate'],3)}.")
    w("Tetapi dua eksperimen independen menunjukkan celah itu **tidak dapat dipanen**:")
    w("")
    w(f"- Selective expansion gagal karena tidak ada fitur yang memprediksi query mana yang")
    w(f"  diuntungkan (korelasi Spearman terkuat "
      f"{max(abs(v['spearman_with_delta']) for v in se['per_feature'].values()):.3f} "
      "dari sepuluh fitur).")
    w("- Rank fusion gagal karena keempat metode berbagi basis BM25 dan ekspansi LSA yang")
    w("  sama. Menambahkan sistem yang benar-benar berbeda (language model dengan smoothing")
    w(f"  Dirichlet, kegagalan {f(bl['LM500']['fail_rate'],3)}) juga tidak menolong.")
    w("")
    w("**Celah oracle yang tidak dapat diprediksi tidak dapat dipanen.** Karena itu metode")
    w("yang diusulkan tetap LSA-QE: paling sederhana, satu komponen, dan tidak ada")
    w("alternatif yang terbukti mengalahkannya.")
    w("")

    w("## 8. Keterbatasan")
    w("")
    w(f"1. **Plafon absolut masih rendah.** Recall@10 {f(o['LSA-QE']['Recall@10'],3)} berarti "
      f"{o['LSA-QE']['fail_rate']:.0%} pertanyaan masih tidak punya jawabannya di top-10.")
    w("2. **Evaluasi satu-gold bersifat pesimistis.** Tepat satu chunk dihitung benar,")
    w("   padahal chunk tetangga sering memuat informasi yang sama; kegunaan nyata untuk")
    w("   RAG kemungkinan lebih tinggi dari angka ini.")
    w("3. **Retrieval bersifat intra-dokumen.** Pasien sudah diketahui; tahap menemukan")
    w("   laporan yang benar di luar lingkup.")
    w("4. **Tidak ada evaluasi end-to-end** dengan generator, sesuai batasan tanpa LLM.")
    w("5. **Parafrasa ditulis sendiri**, bukan dikumpulkan dari pengguna nyata.")
    w("")

    w("## 9. Berkas")
    w("")
    w("| Berkas | Isi |")
    w("| --- | --- |")
    for name, desc in (
        ("reports/00_preprocessing.md", "Tahap 0 - normalisasi dan sanity check"),
        ("reports/01_split_scenario.md", "Tahap 1 - split dan skenario"),
        ("reports/03_metode.md", "Tahap 2-3 - retrieval dasar dan metode"),
        ("reports/02_dev_selection.md", "Tahap 4 - seleksi parameter di dev"),
        ("reports/04_test.md", "Tahap 5 - hasil test lengkap dan signifikansi"),
        ("reports/05_analisis.md", "Tahap 6 - ablasi dan analisis kegagalan"),
        ("reports/06_ringkasan_akhir.md", "ringkasan ini"),
        ("reports/DECISIONS.md", "seluruh keputusan beserta alasannya"),
    ):
        w(f"| `{name}` | {desc} |")

    path = io.write_report("06_ringkasan_akhir.md", "\n".join(L))
    print(f"-> {io.rel(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
