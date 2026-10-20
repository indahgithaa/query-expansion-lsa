# Query expansion LSA + PRF (RM3) untuk retrieval chunk laporan asuhan gizi

Apakah menggabungkan ekspansi **global** (LSA; Deerwester et al., 1990) dengan
ekspansi **lokal** (PRF/RM3; Lavrenko & Croft, 2001) lebih baik daripada tanpa QE,
RM3 saja, dan LSA saja? Kerangka teori: analisis lokal vs global (Xu & Croft, 1996).

Tugasnya adalah menemukan satu chunk jawaban di dalam satu laporan asuhan gizi
berbahasa Indonesia. Tanpa LLM, tanpa reranker, tanpa dense retrieval.

> Repo ini memuat data klinis pasien yang sudah dianonimkan. **Jangan di-push ke
> remote publik.** Laporan di `reports/` sengaja tidak mengutip teks klinis dan
> merujuk dokumen dengan `doc_id`, bukan nama berkas.

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

## Hasil ringkas (test, seed 1-10, 3.444 unit)

| Metode | MRR | Recall@10 | nDCG@10 | Gagal (>10) | Robustness Index |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 (tanpa QE) | 0,2763 | 0,575 | 0,3334 | 0,425 | - |
| RM3 (lokal) | 0,2807 | 0,615 | 0,3468 | 0,385 | +0,076 |
| LSA-QE (global) | **0,3106** | 0,631 | **0,3739** | 0,369 | +0,106 |
| RM3+LSA (gabungan) | 0,3031 | **0,650** | 0,3729 | **0,350** | **+0,164** |

Metode gabungan mengungguli tanpa-QE dan RM3 saja secara signifikan, tetapi
**setara dengan LSA saja pada MRR**. Keunggulannya ada pada recall dalam, tingkat
kegagalan, dan kestabilan (penurunan besar 53 vs 107). Rinciannya di
[`reports/04_test.md`](reports/04_test.md) dan [`reports/05_analisis.md`](reports/05_analisis.md).

Penomoran berkas laporan mengikuti urutan penulisan, bukan nomor tahap, karena
nama `02_dev_selection.md` untuk Tahap 4 sudah ditetapkan spesifikasi.
