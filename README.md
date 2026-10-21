# Ekspansi query berbasis LSA untuk retrieval chunk laporan asuhan gizi

**Metode yang diusulkan: ekspansi query berbasis Latent Semantic Analysis**
(LSA; Deerwester et al., 1990). Ekspansi lokal (PRF/RM3; Lavrenko & Croft, 2001)
dan gabungannya diukur sebagai **pembanding**. Kerangka teori: analisis lokal vs
global (Xu & Croft, 1996).

Hasil akhir dan alasan memilih LSA saja ada di
[`reports/06_ringkasan_akhir.md`](reports/06_ringkasan_akhir.md).

Tugasnya adalah menemukan satu chunk jawaban di dalam satu laporan asuhan gizi
berbahasa Indonesia. Tanpa LLM, tanpa reranker, tanpa dense retrieval.

> Repo ini memuat data klinis pasien yang sudah dianonimkan. **Jangan di-push ke
> remote publik.** Laporan di `reports/` merujuk dokumen dengan `doc_id`, bukan
> nama berkas. Bab 4 skripsi mengutip satu chunk nyata sebagai contoh perhitungan
> manual, atas keputusan pengguna (lihat `reports/DECISIONS.md`).

## Pemasangan

```
pip install -r requirements.txt
```

Diuji pada Python 3.14.7 (Windows). Semua versi dipin demi determinisme.

## Menjalankan

```
python scripts/00_preprocess.py      # Tahap 0: normalisasi, dedup, sanity check
python scripts/01_splits.py          # Tahap 1: split per dokumen + skenario
python scripts/02_dev_grid.py        # Tahap 2-4: grid 64 konfigurasi di dev (~2,5 menit)
python scripts/02_dev_report.py      # Tahap 4: laporan seleksi dev
python scripts/03_lock_config.py     # kunci config (hanya setelah disetujui)
python scripts/04_test.py            # Tahap 5: test seed 1-10, SEKALI (~2 menit)
python scripts/04_test_report.py     # Tahap 5: laporan hasil test
python scripts/05_analysis.py        # Tahap 6: ablasi dan analisis kegagalan (~1,5 menit)
python scripts/05_analysis_report.py # Tahap 6: laporan analisis
python scripts/06_explorations.py    # eksplorasi arah lanjutan, dev-only (~30 detik)
python scripts/06_final_report.py    # ringkasan akhir
python scripts/07_tiga_metode.py     # agregasi ulang tiga metode (Baseline, RM3, LSA-QE)
python scripts/manual_trace_bab4.py  # penelusuran manual untuk Bab 4 (<1 detik)
python -m pytest                     # 94 tes
```

`scripts/04_test.py --partition dev --out-dir <dir>` menjalankan pipeline test
pada dev untuk memvalidasi skrip tanpa menyentuh partisi test.

## Struktur

```
src/qe/        modul: normalisasi, korpus, tokenisasi, BM25, RM3, LSA, split,
               skenario, metrik, statistik, mesin retrieval
scripts/       satu skrip per tahap; menulis ke results/ dan reports/
tests/         pytest untuk BM25 (dibandingkan dengan rank_bm25), RM3, LSA,
               metrik, split, penugasan parafrasa, dan mesin
configs/       config yang dikunci sebelum test dibuka (+ SHA-256 di DECISIONS.md)
results/       semua angka mentah; setiap angka di laporan berasal dari sini
reports/       satu laporan markdown per tahap, dalam bahasa Indonesia
data/          masukan: dataset QA, chunk per laporan, parafrasa
```

## Protokol

- **Dev (seed 1-3)** dipakai untuk memilih `k`, `n`, `gamma`. **Test (seed 1-10)**
  dibuka sekali, setelah config dikunci dan SHA-256-nya dicatat.
- Split dilakukan per dokumen: 72 koleksi / 24 dev / 24 test. LSA hanya dilatih
  pada bagian koleksi, jadi tidak pernah melihat laporan yang dievaluasi.
- Dua skenario query: **TPL** (template asli) dan **PARA** (satu dari enam
  parafrasa, dipilih deterministik dari `sha256(f"{seed}|{qid}")`).
- Label tipe pertanyaan tidak pernah dipakai saat inferensi.
- RM3 tidak di-tuning: 5 chunk feedback, 30 term, `lambda = 0,5`.

Setiap keputusan yang tidak ditentukan spesifikasi dicatat di
[`reports/DECISIONS.md`](reports/DECISIONS.md) beserta alasannya.

## Laporan

| Berkas | Isi |
| --- | --- |
| [`reports/00_preprocessing.md`](reports/00_preprocessing.md) | Tahap 0 - normalisasi dan sanity check |
| [`reports/01_split_scenario.md`](reports/01_split_scenario.md) | Tahap 1 - split dan skenario |
| [`reports/03_metode.md`](reports/03_metode.md) | Tahap 2-3 - retrieval dasar dan keempat metode |
| [`reports/02_dev_selection.md`](reports/02_dev_selection.md) | Tahap 4 - seleksi parameter di dev |
| [`reports/04_test.md`](reports/04_test.md) | Tahap 5 - hasil test dan signifikansi |
| [`reports/05_analisis.md`](reports/05_analisis.md) | Tahap 6 - ablasi dan analisis kegagalan |
| [`reports/DECISIONS.md`](reports/DECISIONS.md) | Semua keputusan beserta alasannya |

`results/manual_trace_bab4.json` memuat seluruh nilai antara yang dipakai pada
subbab Perhitungan Manual di Bab 4 skripsi. Unit yang ditelusuri adalah data
nyata, yaitu qid 63 pada dokumen D029 (partisi dev, seed 1), dengan parameter
terkunci `k=300`, `n=10`, `gamma=2`. Perhitungannya memakai modul `src/qe/` yang
sama dengan pipeline, sehingga rumus di bab itu dijamin rumus yang dijalankan.

## Hasil ringkas (test, seed 1-10, 3.444 unit)

| Metode | MRR | Recall@10 | nDCG@10 | Gagal (>10) | Robustness Index |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 (tanpa QE) | 0,2763 | 0,575 | 0,3334 | 0,425 | - |
| **LSA-QE (diusulkan)** | **0,3106** | 0,631 | **0,3739** | 0,369 | +0,106 |
| RM3 (pembanding) | 0,2807 | 0,615 | 0,3468 | 0,385 | +0,076 |
| RM3+LSA (pembanding) | 0,3031 | **0,650** | 0,3729 | **0,350** | **+0,164** |

LSA-QE menaikkan MRR +0,0343 (+12,4%) dan menurunkan angka kegagalan 13,2% secara
relatif, keduanya signifikan. Empat arah lanjutan diuji di dev untuk mencari
sesuatu yang dapat digabungkan dengan LSA - ekspansi morfologis, selective query
expansion, rank fusion, dan ekspansi lokal - **tidak satu pun terbukti layak
dipromosikan**; alasannya di [`reports/06_ringkasan_akhir.md`](reports/06_ringkasan_akhir.md).

Penomoran berkas laporan mengikuti urutan penulisan, bukan nomor tahap, karena
nama `02_dev_selection.md` untuk Tahap 4 sudah ditetapkan spesifikasi.
