# Catatan Revisi Skripsi Berdasarkan Implementasi

Tanggal: 2026-11-11. Daftar ini membandingkan isi Bab 1–4 dengan kode dan hasil yang benar-benar dijalankan. Bukti setiap poin ada di `reports/07_audit_bab4.md` dan `reports/DECISIONS.md`.

## A. Wajib direvisi (isi tidak sesuai dengan yang dijalankan)

| No | Lokasi | Tertulis sekarang | Yang benar |
|---|---|---|---|
| 1 | Persamaan 2.9 dan keterangannya | Penjumlahan atas himpunan *t* ∈ *q* ∩ *V*; "*q* = himpunan *term query*" | Penjumlahan atas setiap **token** pertanyaan yang terdapat pada *V*, dibagi banyaknya token tersebut. Token ganda (misalnya "diet" pada tipe Syarat diet dan Pedoman) dihitung ganda. Hanya cara ini yang menghasilkan angka Tabel 4.45–4.51 |
| 2 | Tabel 4.1, baris RM3 | "lima *chunk* teratas" | "tiga *chunk* teratas" (sesuai Tabel 4.2 dan Subbab 4.12.9) |
| 3 | Tabel 4.2, Gambar 4.2, Gambar 4.30, Subbab 4.10 | "Qwen3.8-27B" | Qwen3.8-27B **terkuantisasi INT4** (`RedHatAI/Qwen3.8-27B-INT4`, W4A16) yang dijalankan dengan vLLM pada Kaggle GPU T4×2. Model BF16 (±54 GB) tidak muat di GPU yang tersedia |
| 4 | 3.8.1 Perangkat lunak | Python 3.12.14, PyMuPDF, Sentence Transformers, Qdrant, Qdrant Client, PyTorch 2.13.0 | Lokal: Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, PySastrawi 1.2.1, pandas 3.0.6, tqdm 4.70.1, pytest 8.4.2, rank-bm25 0.2.2 (hanya pengujian). Kaggle: Python 3.13.16, vLLM 0.33.0, PyTorch 2.14.0. PyMuPDF, Sentence Transformers, dan Qdrant **tidak dipakai** (sisa topik lama). Daftar final disusun di Fase 5 |
| 5 | 3.8.2 Perangkat keras | GT 710 sebagai kartu grafis | Retrieval dan evaluasi berjalan di CPU lokal (GT 710 tidak dipakai). Generasi memakai Kaggle dengan 2× NVIDIA Tesla T4 16 GB |
| 6 | 3.7.2 | Faithfulness dan BERTScore saja; "Semakin tinggi nilai **ketiga** metrik" | Empat metrik: Faithfulness, BERTScore, BLEU, ROUGE-L (sesuai 4.10 dan Tabel 4.57) |
| 7 | 3.6 | "matriks istilah-dokumen berbobot TF-IDF"; "pertanyaan diproyeksikan ke ruang laten"; parameter "bobot *query* asli" | Bobot (1 + ln tf)·ln(N/df); pertanyaan direpresentasikan sebagai *centroid* vektor *term*; parameter γ = bobot komponen ekspansi |

## B. Perlu ditambahkan (benar, tetapi belum tertulis)

| No | Lokasi yang disarankan | Isi |
|---|---|---|
| 1 | 4.10 atau Bab 5 | Prompt identik antarmetode hanya dibangkitkan sekali: 10.332 unit×metode menjadi 7.343 prompt unik. Adil karena konteks identik dengan *greedy decoding* menghasilkan jawaban identik |
| 2 | 4.10 | Pengaturan teknis generasi: fp16, *KV cache* fp16, mode berpikir dimatikan lewat `enable_thinking=False`, teks instruksi *prompt* (lihat Kode Program 5.11) |
| 3 | 4.11.2 | Uji Wilcoxon memakai sebaran eksak untuk n kecil dan aproksimasi normal dengan koreksi nilai kembar untuk n besar |
| 4 | 4.13 atau Bab 6 | Dua analisis **eksploratif** (bukan bagian rancangan utama): (a) hanya unit uji dari laporan yang tidak pernah menjadi data validasi seed 1–3 (45% unit uji berasal dari laporan yang ikut penyetelan); (b) uji pada tingkat laporan karena satu laporan muncul berulang antarpengulangan |
| 5 | 4.11 / 4.13.6 | Model juri untuk Faithfulness (RAGAS) belum disebut. Rencana: model Qwen yang sama, dijalankan di Kaggle |
| 6 | Bab 5 | Pembangkit bilangan acak: `random.Random` untuk pembagian data dan `numpy.random.default_rng` untuk *bootstrap*. Keduanya dipilih karena mereproduksi Tabel 4.14 dan 4.51 secara persis |

## C. Perbaikan rujukan silang (redaksional)

| Lokasi | Tertulis | Seharusnya |
|---|---|---|
| 4.6.2 | Persamaan (2.7) | 2.6 |
| 4.6.3 | Persamaan (2.8) untuk SVD; (2.9) untuk normalisasi | 2.7; uraian setelah 2.7 |
| 4.10 | Faithfulness 2.18, BERTScore 2.19–2.21, BLEU 2.16, ROUGE-L 2.17 | 2.19, 2.21–2.23, 2.17, 2.18 |
| 4.11.2, 4.12.11 | Holm Persamaan 2.22 | 2.24 |
| 4.12.2 | Gambar 4.10 (parafrasa) | Gambar 4.11 |
| 4.12.9 | Gambar 4.23, 4.24 | Gambar 4.27, 4.28 |
| 4.12.10 | Gambar 4.28 | Gambar 4.32 |
| 4.12.11 | Gambar 4.31 | Gambar 4.36 |
| Bab 2 | Nomor persamaan melompati 2.16 dan 2.20 | Nomori ulang (lalu sesuaikan semua rujukan) |

## D. Yang sudah sesuai (tidak perlu diubah)

- Semua angka perhitungan manual Subbab 4.12 tereproduksi dengan toleransi 1e-4 oleh kode (16 tes otomatis di `tests/test_manual_bab4.py`).
- Konfigurasi Tabel 4.2 terbukti terpilih dari grid validasi: LSA-QE peringkat 1 dari 64 dan RM3 peringkat 1 dari 75 (`results/02_dev_selection.json`).
- Ringkasan korpus (Tabel 4.6), pembagian data (Tabel 4.13–4.14), statistik LSA (Tabel 4.25, 4.29), dan jumlah stopword 810.

## E. Bahan untuk Bab 6 (temuan yang perlu dibahas)

- Pemilihan parameter RM3 tidak stabil: selisih konfigurasi terbaik dan kedua hanya 0,00007, dan *leave-one-seed-out* memilih konfigurasi berbeda di ketiga lipatan. LSA-QE stabil di 2 dari 3 lipatan.
- Penurunan MRR dari TPL ke PARA hampir sama pada ketiga metode (±0,06). LSA-QE unggul di kedua skenario, tetapi tidak membuat sistem lebih tahan terhadap parafrasa (`results/07_retrieval_metrics.json`).
- RM3 memasukkan angka murni (misalnya penomoran subbab) ke model relevansi, sedangkan LSA membuangnya.
- Ejaan berbeda antarlaporan (strongkids vs StrongKidz) membuat sebagian *term* ekspansi LSA tidak berkontribusi.
