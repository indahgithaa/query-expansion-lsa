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
python scripts/04_test.py            # Tahap 5: test seed 1-10, SEKALI
python scripts/05_analysis.py        # Tahap 6: ablasi dan analisis kegagalan
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
| [`reports/DECISIONS.md`](reports/DECISIONS.md) | Semua keputusan beserta alasannya |

Penomoran berkas laporan mengikuti urutan penulisan, bukan nomor tahap, karena
nama `02_dev_selection.md` untuk Tahap 4 sudah ditetapkan spesifikasi.
