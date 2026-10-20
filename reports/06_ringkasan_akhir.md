# Ringkasan Akhir

**Metode yang diusulkan: ekspansi query berbasis Latent Semantic Analysis (LSA-QE).**

Sumber angka: `results/04_test.json`, `results/05_analysis.json`,
`results/06_explorations.json`. Config terkunci `configs/locked_config.json`,
SHA-256 `cf1e14b2d9c967cffc4283b30b63eea67139aaf9957c507d224e021087667ab3`.

## 1. Ringkas

Tugasnya menemukan satu chunk jawaban di dalam satu laporan asuhan gizi berbahasa
Indonesia. Ekspansi query berbasis LSA (k=300, n=10, gamma=2.0)
dibandingkan dengan BM25 tanpa ekspansi, diuji pada 3444 unit evaluasi
(seed 1-10 x 2 skenario query). Tanpa LLM, tanpa reranker,
tanpa dense retrieval.

**LSA-QE menaikkan MRR +0.0343 (+12.4%) dan menurunkan
angka kegagalan dari 0.425 ke 0.369
(13.2% secara relatif).** Keduanya signifikan
secara statistik.

## 2. Hasil utama

| Metode | MRR | R@1 | R@5 | R@10 | R@20 | nDCG@10 | Median | Gagal (>10) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 0.2763 | 0.139 | 0.427 | 0.575 | 0.724 | 0.3334 | 7 | 0.425 |
| RM3 | 0.2807 | 0.135 | 0.446 | 0.615 | 0.759 | 0.3468 | 7 | 0.385 |
| **LSA-QE** | 0.3106 | 0.163 | 0.474 | 0.631 | 0.777 | 0.3739 | 6 | 0.369 |
| RM3+LSA | 0.3031 | 0.148 | 0.481 | 0.650 | 0.790 | 0.3729 | 6 | 0.350 |

RM3 dan RM3+LSA ditampilkan sebagai **pembanding**, bukan metode yang diusulkan.

### Signifikansi (vs B1)

| Perbandingan | dMRR | CI 95% | p Holm | Signifikan |
| --- | ---: | --- | ---: | :---: |
| RM3 vs B1 | +0.0044 | [+0.0002, +0.0085] | 8.72e-11 | **ya** |
| LSA-QE vs B1 | +0.0343 | [+0.0255, +0.0430] | 6.63e-24 | **ya** |
| RM3+LSA vs B1 | +0.0268 | [+0.0199, +0.0336] | 2.5e-31 | **ya** |

Paired bootstrap berstrata (seed x skenario, 10.000 resample), Wilcoxon signed-rank dengan koreksi Holm atas keluarga lima perbandingan yang ditetapkan sebelum test dibuka.

## 3. Ketahanan terhadap variasi pertanyaan

| Metode | MRR template | MRR parafrasa | Selisih |
| --- | ---: | ---: | ---: |
| B1 | 0.3084 | 0.2442 | 0.0642 |
| RM3 | 0.3131 | 0.2482 | 0.0650 |
| LSA-QE | 0.3412 | 0.2800 | 0.0612 |
| RM3+LSA | 0.3350 | 0.2711 | 0.0639 |

Parafrasa lebih sulit daripada template untuk **semua** metode, dan ekspansi tidak
menghapus kesulitan itu: celahnya hanya menyempit tipis, dari 0.0642 (B1) ke
0.0612 (LSA-QE). Perbaikan absolutnya hampir sama di kedua skenario
(+0.0358 pada parafrasa, +0.0328 pada template).

Secara **relatif** keuntungannya memang lebih besar pada parafrasa (+14.7% vs +10.6%), sesuai dugaan bahwa ekspansi menolong saat kata
query tidak cocok persis dengan dokumen. Tetapi selisihnya kecil, jadi klaim yang
dapat dipertahankan adalah **ekspansi menolong sama baiknya pada pertanyaan
parafrasa maupun template** - bukan bahwa ia secara khusus mengatasi variasi query.

## 4. Robustness per query terhadap B1

| Metode | Naik | Sama | Turun | RI | Masuk top-10 | Keluar top-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RM3 | 1359 | 988 | 1097 | +0.0761 | 244 | 107 |
| LSA-QE | 1540 | 729 | 1175 | +0.1060 | 344 | 151 |
| RM3+LSA | 1654 | 701 | 1089 | +0.1641 | 394 | 137 |

## 5. Per tipe pertanyaan

Label tipe tidak dipakai saat inferensi; pemecahan ini hanya untuk analisis.

| Tipe | n | B1 | LSA-QE | Selisih |
| --- | ---: | ---: | ---: | ---: |
| skrining | 480 | 0.3137 | 0.4191 | +0.1053 |
| status_gizi | 480 | 0.3533 | 0.3505 | -0.0027 |
| biokimia | 474 | 0.1189 | 0.1433 | +0.0245 |
| pes | 480 | 0.1270 | 0.1145 | -0.0125 |
| tujuan_diet | 480 | 0.3586 | 0.3857 | +0.0271 |
| prinsip_diet | 476 | 0.4436 | 0.5234 | +0.0798 |
| syarat_diet | 480 | 0.2435 | 0.2703 | +0.0268 |
| pedoman | 94 | 0.1476 | 0.1416 | -0.0060 |

Tipe yang tidak terbantu: `status_gizi`, `pes`, `pedoman`.
Untuk `pes`, penyebabnya terukur: penanda format `berkaitan`, `ditandai`, `disebabkan` muncul di 98.3% chunk gold tetapi juga di 16.6% chunk mana pun pada laporan yang sama -
nyaris selalu benar, hampir tidak membedakan. Tidak ada pembobotan ulang term yang
bisa memperbaikinya; ini batas representasi bag-of-words.

## 6. Implikasi untuk pipeline RAG

Dalam RAG, generator menerima top-k chunk dan tidak peduli gold ada di peringkat 1
atau 7 - asal masuk ke dalam k. Metrik yang mengikat karena itu Recall@k, dan
konsekuensinya dapat dinyatakan sebagai **anggaran konteks**: berapa chunk yang
harus disuapkan untuk mencapai tingkat recall tertentu.

Ini membalik keberatan standar bahwa menambah konteks bisa merugikan generator.
LSA-QE tidak meminta lebih banyak chunk - ia mencapai recall yang sama dengan chunk
**lebih sedikit**, sehingga konteks lebih pendek, biaya token lebih rendah, dan
distraksi lebih kecil. Semua arahnya searah.

Yang dapat diklaim: ekspansi query memperkuat **tahap temu kembali**, menurunkan
kasus konteks-tanpa-jawaban dari 0.425 ke 0.369 pada k=10.
Yang **tidak** dapat diklaim tanpa eksperimen end-to-end: bahwa kualitas jawaban
RAG ikut meningkat. Recall yang lebih baik adalah syarat perlu, bukan cukup.

## 7. Mengapa LSA saja, bukan gabungan

Empat arah lanjutan diuji untuk mencari sesuatu yang dapat digabungkan dengan LSA,
terutama untuk menurunkan kegagalan lebih jauh. Semuanya dihitung di **dev** (1026 unit, seed 1-3); test tidak disentuh.

| Arah | Hasil | Verdikt |
| --- | --- | --- |
| Ekspansi lokal (RM3) digabung | kegagalan dev 0.339 vs 0.365, tetapi MRR turun -0.0168 | menolong recall, merugikan MRR |
| Ekspansi morfologis (stemming) | kontribusi unik 1.3% unit | gugur |
| Selective query expansion | held-out -0.0006 (logistic), -0.0052 (ridge) | gugur |
| Rank fusion (RRF, CombSUM, + LM Dirichlet) | kegagalan terbaik 0.347 vs 0.339 | gugur |

Celah oracle nyata: memilih metode terbaik per query akan mencapai MRR 0.3973 dan kegagalan 0.266.
Tetapi dua eksperimen independen menunjukkan celah itu **tidak dapat dipanen**:

- Selective expansion gagal karena tidak ada fitur yang memprediksi query mana yang
  diuntungkan (korelasi Spearman terkuat 0.142 dari sepuluh fitur).
- Rank fusion gagal karena keempat metode berbagi basis BM25 dan ekspansi LSA yang
  sama. Menambahkan sistem yang benar-benar berbeda (language model dengan smoothing
  Dirichlet, kegagalan 0.362) juga tidak menolong.

**Celah oracle yang tidak dapat diprediksi tidak dapat dipanen.** Karena itu metode
yang diusulkan tetap LSA-QE: paling sederhana, satu komponen, dan tidak ada
alternatif yang terbukti mengalahkannya.

## 8. Keterbatasan

1. **Plafon absolut masih rendah.** Recall@10 0.631 berarti 37% pertanyaan masih tidak punya jawabannya di top-10.
2. **Evaluasi satu-gold bersifat pesimistis.** Tepat satu chunk dihitung benar,
   padahal chunk tetangga sering memuat informasi yang sama; kegunaan nyata untuk
   RAG kemungkinan lebih tinggi dari angka ini.
3. **Retrieval bersifat intra-dokumen.** Pasien sudah diketahui; tahap menemukan
   laporan yang benar di luar lingkup.
4. **Tidak ada evaluasi end-to-end** dengan generator, sesuai batasan tanpa LLM.
5. **Parafrasa ditulis sendiri**, bukan dikumpulkan dari pengguna nyata.

## 9. Berkas

| Berkas | Isi |
| --- | --- |
| `reports/00_preprocessing.md` | Tahap 0 - normalisasi dan sanity check |
| `reports/01_split_scenario.md` | Tahap 1 - split dan skenario |
| `reports/03_metode.md` | Tahap 2-3 - retrieval dasar dan metode |
| `reports/02_dev_selection.md` | Tahap 4 - seleksi parameter di dev |
| `reports/04_test.md` | Tahap 5 - hasil test lengkap dan signifikansi |
| `reports/05_analisis.md` | Tahap 6 - ablasi dan analisis kegagalan |
| `reports/06_ringkasan_akhir.md` | ringkasan ini |
| `reports/DECISIONS.md` | seluruh keputusan beserta alasannya |
