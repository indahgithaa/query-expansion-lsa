"""Tahap 6 - laporan ablasi dan analisis kegagalan. Hanya membaca dari `results/`."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io
from qe.corpus import TYPE_ORDER

METHODS = ("B1", "RM3", "LSA-QE", "RM3+LSA")


def f(x, nd=4):
    return f"{float(x):.{nd}f}"


def main():
    d = io.read_json("05_analysis.json")
    t = io.read_json("04_test.json")
    ab, j, rsc, fail = d["ablation"], d["jaccard"], d["rescue"], d["failure_by_type"]

    L = []
    w = L.append
    w("# Tahap 6 - Ablasi dan analisis kegagalan")
    w("")
    w("Sumber angka: `results/05_analysis.json`, `results/05_expansion_overlap.csv`,")
    w("`results/04_test.json`. Config terkunci SHA-256 `" + d["config_sha256"] + "`.")
    w(f"Dihitung atas {d['n_units']} unit test (seed 1-10, skenario TPL dan PARA).")
    w("")

    w("## 1. Kontribusi tiap komponen")
    w("")
    w("| Langkah | dMRR |")
    w("| --- | ---: |")
    w(f"| Lokal saja (RM3 - B1) | {ab['lokal_saja_atas_B1']:+.4f} |")
    w(f"| Global saja (LSA-QE - B1) | {ab['global_saja_atas_B1']:+.4f} |")
    w(f"| Jumlah kedua kontribusi bila aditif | {ab['jumlah_kontribusi_terpisah']:+.4f} |")
    w(f"| Gabungan sebenarnya (RM3+LSA - B1) | {ab['gabungan_atas_B1']:+.4f} |")
    w(f"| LSA ditambahkan di atas RM3 | {ab['lsa_ditambahkan_ke_rm3']:+.4f} |")
    w(f"| RM3 ditambahkan di atas LSA | {ab['rm3_ditambahkan_ke_lsa']:+.4f} |")
    w("")
    w(f"Gabungan ({ab['gabungan_atas_B1']:+.4f}) **lebih kecil** daripada jumlah kontribusi")
    w(f"terpisah ({ab['jumlah_kontribusi_terpisah']:+.4f}). Pada MRR, kedua komponen karena itu")
    w("**sub-aditif**: keduanya sebagian besar memperbaiki query yang sama, dan menambahkan")
    w("RM3 di atas LSA justru menggeser sedikit bobot dari term LSA yang lebih berguna.")
    w("")
    w("Gambarannya berbalik pada recall dalam:")
    w("")
    w("| Metrik | B1 | RM3 | LSA-QE | RM3+LSA |")
    w("| --- | ---: | ---: | ---: | ---: |")
    w("| Recall@10 | " + " | ".join(f(ab["recall10"][m], 3) for m in METHODS) + " |")
    w("| Tingkat kegagalan | " + " | ".join(f(ab["fail_rate"][m], 3) for m in METHODS) + " |")
    w("| Unit di top-10 | " + " | ".join(str(rsc["units_in_top10"][m]) for m in METHODS) + " |")
    w("")
    w(f"RM3 yang ditambahkan di atas LSA menaikkan Recall@10 sebesar "
      f"{ab['rm3_ditambahkan_ke_lsa_recall10']:+.4f} meskipun menurunkan MRR sebesar "
      f"{ab['rm3_ditambahkan_ke_lsa']:+.4f}.")
    w("Jumlah unit yang masuk top-10 naik monoton di seluruh rangkaian metode")
    w(" -> ".join(f"{m} {rsc['units_in_top10'][m]}" for m in METHODS) + ".")
    w("Artinya komponen lokal memindahkan gold dari luar top-10 ke dalam top-10, tetapi")
    w("jarang memindahkannya ke peringkat satu atau dua - persis di mana MRR memberi bobot.")
    w("")

    w("## 2. Seberapa berbeda kedua himpunan term ekspansi")
    w("")
    o = j["overall"]
    w(f"Jaccard antara 30 term RM3 dan {t['config']['methods']['RM3+LSA']['n']} term LSA")
    w(f"per query: rata-rata **{f(o['mean'])}**, median {f(o['median'])}, rata-rata")
    w(f"{f(o['mean_shared'], 2)} term berimpit. Nilai maksimum yang mungkin adalah")
    w(f"{f(j['max_possible'])} (karena himpunan LSA hanya berisi "
      f"{t['config']['methods']['RM3+LSA']['n']} term), jadi yang tercapai sekitar "
      f"{o['mean'] / j['max_possible']:.0%} dari maksimum.")
    w(f"Sebanyak {o['pct_zero']:.1%} query tidak punya satu pun term bersama.")
    w("")
    w("| Pecahan | Jaccard rata-rata |")
    w("| --- | ---: |")
    for sc, v in j["by_scenario"].items():
        w(f"| Skenario {sc} | {f(v['mean'])} |")
    for ty in TYPE_ORDER:
        w(f"| Tipe {ty} | {f(j['by_type'][ty]['mean'])} |")
    w("")
    w("Term yang paling sering muncul di kedua himpunan sekaligus: "
      + ", ".join(f"`{x['term']}`" for x in j["most_shared_terms"][:10]) + ".")
    w("Semuanya kosakata gizi umum - bukan istilah spesifik pasien.")
    w("")
    w("**Temuan penting:** kedua ekspansi hampir tidak beririsan, tetapi itu tidak")
    w("membuatnya saling melengkapi. Berbeda belum tentu berguna; bagian 4 menunjukkan")
    w("sebagian besar term khas RM3 adalah angka.")
    w("")

    w("## 3. Query yang hanya terselamatkan oleh gabungan")
    w("")
    r1 = rsc["rescued_by_combination_only"]
    r2 = rsc["harmed_by_combination_only"]
    r3 = rsc["lost_if_either_component_succeeded"]
    w("| Kategori | Definisi | Jumlah | % unit |")
    w("| --- | --- | ---: | ---: |")
    w(f"| Sinergi | {r1['definition']} | **{r1['count']}** | {r1['pct_of_units']:.2%} |")
    w(f"| Kebalikannya | {r2['definition']} | {r2['count']} | {r2['pct_of_units']:.2%} |")
    w("")
    w(f"Neto sinergi **{rsc['net_symmetric']:+d} query**. Ada efek gabungan yang nyata, tetapi")
    w(f"kecil: {r1['count']} dari {d['n_units']} unit.")
    w("")
    w(f"Sebarannya per tipe: " + ", ".join(f"{k} {v}" for k, v in sorted(
        r1["by_type"].items(), key=lambda kv: -kv[1])) + ".")
    w(f"Per skenario: " + ", ".join(f"{k} {v}" for k, v in r1["by_scenario"].items())
      + " - sinergi muncul seimbang pada template maupun parafrasa.")
    w("")
    w(f"Sebagai pembanding yang **tidak** simetris: ada {r3['count']} unit di mana salah satu")
    w("komponen berhasil masuk top-10 tetapi gabungan gagal. Angka itu tidak boleh")
    w("dibandingkan langsung dengan 24, karena syaratnya jauh lebih longgar (cukup satu")
    w("komponen berhasil, bukan keduanya). Pembandingnya yang benar adalah baris")
    w(f"\"Kebalikannya\" di tabel atas ({r2['count']}).")
    w("")

    w("## 4. Contoh term ekspansi")
    w("")
    w("Cara pemilihan contoh (ditetapkan tanpa melihat metrik): " + d["example_selection_rule"])
    w("")
    for e in d["examples"]:
        w(f"### Contoh {e['slot'] + 1} - tipe `{e['qtype']}`, skenario {e['scenario']}")
        w("")
        w(f"> {e['query']}")
        w("")
        w(f"Token query setelah stopword dibuang: "
          + ", ".join(f"`{x}`" for x in e["query_tokens"]) + ".")
        w("Peringkat gold: " + ", ".join(f"{m} {e['ranks'][m]}" for m in METHODS) + ".")
        w("")
        w("| # | Term RM3 | Bobot | Term LSA | Cosine |")
        w("| ---: | --- | ---: | --- | ---: |")
        for i in range(max(len(e["rm3_top"]), len(e["lsa_top"]))):
            a = e["rm3_top"][i] if i < len(e["rm3_top"]) else None
            b = e["lsa_top"][i] if i < len(e["lsa_top"]) else None
            w(f"| {i + 1} | " + (f"`{a['term']}` | {f(a['weight'])}" if a else " | ") + " | "
              + (f"`{b['term']}` | {f(b['cosine'])}" if b else " | ") + " |")
        w("")
        w("Term bersama di sepuluh besar: "
          + (", ".join(f"`{x}`" for x in e["shared"]) if e["shared"] else "tidak ada") + ".")
        w("")
    w("Pola yang terlihat konsisten di kelima contoh: RM3 mengambil kata yang benar-benar")
    w("muncul di chunk peringkat atas laporan itu - termasuk banyak angka dan satuan -")
    w("sedangkan LSA mengambil kosakata domain yang bertetangga secara laten dan tidak")
    w("harus muncul di laporan tersebut (`malnutrisi`, `strongkids`, `pyms`, `preskripsi`).")
    w("")

    w("## 5. Angka yang memakan kapasitas ekspansi RM3")
    w("")
    w(f"Rata-rata **{o['mean_rm3_digit_share']:.1%}** dari 30 term ekspansi RM3 adalah angka")
    w(f"murni, membawa **{o['mean_rm3_digit_mass']:.1%}** massa probabilitas model relevansi.")
    w("LSA membuang angka murni saat membangun vocab, jadi angkanya nol di sana.")
    w("")
    w("| Tipe | Porsi term RM3 yang berupa angka |")
    w("| --- | ---: |")
    for ty in TYPE_ORDER:
        w(f"| {ty} | {j['by_type'][ty]['mean_rm3_digit_share']:.1%} |")
    w("")
    w("Inilah mekanisme paling konkret di balik lemahnya kontribusi RM3. Laporan asuhan gizi")
    w("padat angka (nilai lab, berat badan, kebutuhan energi), dan karena chunk indeks tidak")
    w("dibersihkan dari angka, model relevansi membuang sekitar seperenam kapasitasnya pada")
    w("token yang hampir tidak membedakan chunk. Pada tipe `biokimia` porsinya tertinggi")
    w(f"({j['by_type']['biokimia']['mean_rm3_digit_share']:.1%}), dan di sanalah RM3 justru")
    w(f"menolong (dMRR {fail['biokimia']['delta_vs_B1']['RM3']:+.4f}) - angka memang menjadi")
    w("isi jawabannya. Di tipe lain, angka hanyalah derau.")
    w("")

    w("## 6. Analisis kegagalan per tipe")
    w("")
    w("| Tipe | n | dMRR RM3 | dMRR LSA-QE | dMRR RM3+LSA | RI RM3+LSA | Penurunan besar | Gagal | Jaccard |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ty in TYPE_ORDER:
        v = fail[ty]
        w(f"| {ty} | {v['n']} | {v['delta_vs_B1']['RM3']:+.4f} | {v['delta_vs_B1']['LSA-QE']:+.4f} | "
          f"{v['delta_vs_B1']['RM3+LSA']:+.4f} | {v['robustness_index']['RM3+LSA']:+.4f} | "
          f"{v['big_drops']['RM3+LSA']} | {f(v['fail_rate']['RM3+LSA'], 3)} | {f(v['mean_jaccard'], 3)} |")
    w("")
    worse = d["types_worse_than_B1"]
    w(f"Dua tipe **turun** dibanding B1 pada metode utama: "
      + ", ".join(f"`{x}`" for x in worse) + ". Keduanya juga satu-satunya tipe dengan")
    w("Robustness Index negatif.")
    w("")
    for ty in worse:
        v = fail[ty]
        w(f"### `{ty}` ({v['delta_vs_B1']['RM3+LSA']:+.4f}, RI {v['robustness_index']['RM3+LSA']:+.4f})")
        w("")
        if ty == "pes":
            pm = v["pes_markers"]
            w(f"Jaccard terendah dari semua tipe ({f(v['mean_jaccard'], 3)}): RM3 dan LSA menunjuk")
            w("arah yang hampir sepenuhnya berbeda, dan tidak satu pun mengenai sasaran.")
            w("Pertanyaan PES meminta **struktur** jawaban (problem-etiologi-sign), bukan topik.")
            w("Yang menandai chunk jawaban adalah penanda format "
              + ", ".join(f"`{x}`" for x in pm["terms"]) + ": penanda itu muncul di")
            w(f"**{pm['share_of_gold_containing']:.1%}** chunk gold tipe `pes`, tetapi juga di")
            w(f"**{pm['mean_share_of_chunks_containing']:.1%}** chunk mana pun pada laporan yang")
            w("sama. Jadi penanda itu nyaris selalu benar tetapi hampir tidak membedakan -")
            w("sekitar satu dari enam chunk memuatnya. Tidak ada satu pun term topik yang bisa")
            w("ditambahkan untuk memperbaikinya, dan ekspansi justru menambah kosakata gizi")
            w("umum yang menarik ranking ke chunk asesmen.")
            w(f"Tingkat kegagalan karena itu tetap tinggi untuk semua metode "
              f"({f(v['fail_rate']['B1'], 3)} pada B1, {f(v['fail_rate']['RM3+LSA'], 3)} pada RM3+LSA).")
            w("Ini keterbatasan representasi bag-of-words, bukan keterbatasan ekspansi: tidak")
            w("ada pembobotan ulang term yang bisa menyandikan pola sintaksis.")
        else:
            w(f"Tipe paling jarang ({v['n']} unit, satu-satunya yang di bawah 400), sehingga")
            w("estimasinya paling tidak stabil dari semua tipe dan selisih sekecil ini masih")
            w("dalam jangkauan derau. Jaccard juga rendah "
              f"({f(v['mean_jaccard'], 3)}). Pertanyaan pedoman meminta nama rujukan")
            w("(mis. penuntun diet atau standar acuan) yang biasanya muncul satu kali saja di")
            w("laporan, sehingga umpan balik semu dari lima chunk teratas hampir tidak pernah")
            w(f"memuatnya; RM3 sendiri bahkan merugikan di sini "
              f"({v['delta_vs_B1']['RM3']:+.4f}).")
        w("")
    w("Sebaliknya, tipe dengan keuntungan terbesar adalah `skrining` "
      f"({fail['skrining']['delta_vs_B1']['RM3+LSA']:+.4f}) dan `prinsip_diet` "
      f"({fail['prinsip_diet']['delta_vs_B1']['RM3+LSA']:+.4f}): keduanya menanyakan topik")
    w("berkosakata khas yang punya tetangga laten jelas, persis situasi yang dilayani LSA.")
    w(f"`syarat_diet` punya Robustness Index tertinggi "
      f"({fail['syarat_diet']['robustness_index']['RM3+LSA']:+.4f}) dan menyumbang "
      f"{rsc['rescued_by_combination_only']['by_type'].get('syarat_diet', 0)} dari "
      f"{rsc['rescued_by_combination_only']['count']} kasus sinergi.")
    w("")

    w("## 7. Kesimpulan")
    w("")
    w("1. **Komponen global adalah sumber perbaikan utama.** LSA menyumbang "
      f"{ab['global_saja_atas_B1']:+.4f} MRR di atas B1, hampir delapan kali kontribusi RM3 "
      f"({ab['lokal_saja_atas_B1']:+.4f}).")
    w("2. **Keduanya sub-aditif pada MRR, tetapi saling melengkapi pada kedalaman.** "
      f"Gabungan hanya mencapai {ab['gabungan_atas_B1']:+.4f} dari potensi "
      f"{ab['jumlah_kontribusi_terpisah']:+.4f} bila aditif, namun memberi Recall@10 dan")
    w("   tingkat kegagalan terbaik dari keempat metode.")
    rl = t["robustness_vs_B1"]["LSA-QE"]["overall"]
    rc = t["robustness_vs_B1"]["RM3+LSA"]["overall"]
    w("3. **Nilai sebenarnya dari komponen lokal adalah kestabilan, bukan rata-rata.** "
      f"Penurunan besar berkurang dari {rl['big_drops']} (LSA-QE) menjadi "
      f"{rc['big_drops']} (RM3+LSA), dan Robustness Index naik dari "
      f"{rl['robustness_index']:+.4f} menjadi {rc['robustness_index']:+.4f}.")
    w("4. **Himpunan term yang berbeda tidak otomatis berarti informasi yang melengkapi.** "
      f"Jaccard hanya {f(o['mean'])}, tetapi sekitar seperenam term RM3 adalah angka, "
      "sehingga sebagian besar perbedaan itu tidak membawa daya pembeda.")
    w("5. **Dua tipe pertanyaan tidak terbantu ekspansi apa pun** (`pes`, `pedoman`). "
      "Keduanya meminta struktur atau rujukan bernama, bukan topik - di luar jangkauan")
    w("   pembobotan ulang term.")
    w("")
    w("Dalam kerangka Xu & Croft (1996): pada korpus ini analisis global membawa sinyal,")
    w("analisis lokal membawa rem. Menggabungkan keduanya layak dilakukan bila yang dinilai")
    w("adalah keandalan dan recall; bila yang dinilai hanya MRR, LSA saja sudah setara.")

    path = io.write_report("05_analisis.md", "\n".join(L))
    print(f"-> {io.rel(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
