# Audit Kesesuaian Implementasi terhadap Bab 4

Tanggal: 2026-11-11 · Status: **menunggu keputusan (Titik Periksa 0)**

## Catatan awal

Repo awalnya hanya berisi `data/`. Belum ada implementasi, `results/`, `reports/`, atau `docs/skripsi_bab1-4.pdf`. Karena itu audit dilakukan dengan cara berikut:

1. Jalur retrieval ditulis ulang langsung dari Bab 4 (paket `qe_lsa/`).
2. Kode diverifikasi dengan mereproduksi angka Subbab 4.12. Reproduksi hanya memakai data validasi pengulangan 1, skenario TPL. Data pengujian tidak disentuh; yang dibaca hanya daftar ID dokumennya untuk analisis tumpang tindih di bagian C.

File bukti:

| File | Isi |
|---|---|
| `results/fase0_reproduce_4_12_eq29_set.txt` | Kode sesuai Persamaan 2.9 (centroid atas himpunan term) |
| `results/fase0_reproduce_4_12_token_centroid.txt` | Varian centroid atas barisan token (Gambar 4.18) |
| `results/fase0_checks.txt` | Normalisasi, pembagian data, tumpang tindih tuning–pengujian |

Skrip pembuat: `scripts/reproduce_4_12.py`, `scripts/fase0_token_centroid_variant.py`, `scripts/fase0_checks.py`.

**Ringkasan:** semua angka Subbab 4.12 tereproduksi dengan toleransi 1e-4, dengan satu pengecualian, yaitu peringkat LSA-QE pada qid 20 (A1). Pengecualian ini berasal dari ketidakkonsistenan internal di Bab 4, bukan dari bug.

## A. Tabel audit

| Subbab | Yang ditulis di Bab 4 | Yang ada di kode | Sesuai? | Usulan |
|---|---|---|---|---|
| 4.3 Pemuatan | Nama berkas dinormalisasi lalu diurutkan → D001–D120. ID `{doc}:P{4 digit}:C{3 digit}`. Deduplikasi (file, pertanyaan), id terkecil dipertahankan. Acuan harus tepat satu | `data.py`: sama. Hasil: 120 laporan, 13.737 chunk, 911→858 QA, 0 acuan gagal | Ya | – |
| 4.4.1 Normalisasi | NFKC → #Uxxxx → hapus U+2060 | `text.normalize`. 984 chunk berubah oleh NFKC, 0 #Uxxxx, 52 U+2060 (cocok Tabel 4.15) | Ya | – |
| 4.4.2 Tokenisasi | *Case folding*, lalu `\w+` | `text.tokenize`. Chunk acuan: 120 token / 66 term unik (cocok) | Ya | – |
| 4.4.3 Stopword | Sastrawi dinormalisasi + 5 kata = 810. Dihapus dari pertanyaan, umpan balik RM3, dan dokumen LSA; tidak dari indeks | 810. Tanpa stopword: 78 token / 42 unik (cocok) | Ya | – |
| 4.5.1 Pembagian | Urutkan ID, acak dengan nilai awal `seed×1.000.003+attempt`, 72/24/24, syarat 8 tipe per bagian | `random.Random(...).shuffle` mereproduksi Tabel 4.14 persis. 8.128/2.848/2.761 chunk, 511/174/173 QA. attempt = 0 untuk seed 1–10 | Ya | Sebutkan di Bab 5 bahwa pembangkitnya `random.Random` milik Python |
| 4.5.2 Parafrasa | SHA-256(`seed\|qid`), 8 byte pertama *big-endian* mod 6. Dua daftar parafrasa (validasi/pengujian) masing-masing 6 kalimat × 8 tipe | Fungsi indeks sesuai (`1\|63` → `3ba88302e3808a05` → 3) | **Data belum ada** | **B1**: daftar 96 kalimat parafrasa tidak ada di repo maupun di Bab 4 |
| 4.6.1 Kosakata | df ≥ 2, bukan angka murni, urut alfabetis | 729.195 token, 22.753 unik, 8.564 df=1, 1.464 angka, 12.725 term (cocok Tabel 4.25) | Ya | – |
| 4.6.2 Bobot | (1+ln tf)·ln(N/df), matriks jarang | nnz 350.401, ‖A‖²_F 8.689.125,9783 (cocok) | Ya | Rujukan "Persamaan (2.7)" seharusnya 2.6 (lihat D) |
| 4.6.3 SVD | *Truncated* SVD k=300, random_state=0, V_kΣ_k, normalisasi L2 | `TruncatedSVD` (randomized, bawaan). σ₁=476,0010, σ₃₀₀=71,0811, energi 0,4156, v_skrining[:3] cocok | Ya | – |
| 4.7.1 Centroid | Persamaan 2.9: jumlah atas **himpunan** q∩V. Gambar 4.18: iterasi **setiap term t pada q** | Kode mengikuti Persamaan 2.9 (himpunan) | **Tidak cocok dengan angka 4.12** | **A1** |
| 4.7.2 Seleksi | cos > 0, keluarkan term pertanyaan, urut menurun, tie-break alfabetis, n=10 | Sama. 10 term dan cosine Tabel 4.32–4.33 cocok | Ya | – |
| 4.7.3 Bobot | e_t = cos/Σcos; w_t = count + γ\|q\|e_t, γ=2 | Sama. Tabel 4.34 cocok; skor acuan 20,8881, peringkat 1 | Ya | – |
| 4.8.1 Indeks BM25 | Per laporan, teks utuh, idf Okapi; idf negatif → 0,25 × rata-rata idf (rata-rata termasuk nilai negatif) | `bm25.py`. avgdl 111,2261, idf(skrining) 2,5374, rata-rata idf 3,5494, idf(1) 0,8874 | Ya | – |
| 4.8.2 Skor | Σ w_t·BM25, hanya w_t > 0; pengurutan stabil; skor sama → urutan ID | `argsort(kind="stable")` atas chunk yang sudah urut ID. Baseline acuan 2,4504, peringkat 20; tiga teratas cocok | Ya | – |
| 4.9 RM3 (konfigurasi) | **Tabel 4.1: "lima chunk teratas"**; Tabel 4.2, 4.9 dan 4.12.9: fbDocs = 3, fbTerms = 20, λ = 0,5 | fbDocs = 3, fbTerms = 20, λ = 0,5 | Ya (terhadap Tabel 4.2) | **A2**: Tabel 4.1 perlu dikoreksi menjadi "tiga" |
| 4.9.1 Umpan balik | *Softmax* atas skor Baseline dikurangi skor maksimum | α = 0,4042 / 0,3793 / 0,2165 (cocok) | Ya | – |
| 4.9.2 Model relevansi | tf/\|d\| dengan \|d\| tanpa stopword; chunk kosong dilewati | Panjang 18/73/80, P(skrining) 0,070886 (cocok) | Ya | Lihat **B3** (angka murni ikut masuk) |
| 4.9.3 Potong & interpolasi | 20 teratas (tie-break alfabetis), normalisasi ulang, λ·count/\|q\| + (1−λ)P | Massa 0,660935, w_skrining 0,1536, 20 term Tabel 4.40 sama urutannya, skor acuan 1,2045, peringkat 3 | Ya | – |
| 4.11.1 Metrik | RR, R@{1,3,5,10,20}, NDCG@10 biner, median, gagal (> 10) | `evaluate.py`. Tabel 4.47 cocok (dengan varian A1) | Ya | – |
| 4.11.2 Bootstrap | Strata = pengulangan × skenario, 10.000 ulangan, seed 20261019, persentil 2,5/97,5 | `np.random.default_rng(20261019).integers` mereproduksi tiga *draw* Tabel 4.51 dan CI 0,0001–0,1176 | Ya | Sebutkan pembangkitnya di Bab 5 |
| 4.11.2 Wilcoxon | Nol dikeluarkan; semua nol → p = 1; dua sisi | `scipy.stats.wilcoxon(zero_method="wilcox")`. n kecil: eksak (Tabel 4.49 cocok) | Ya | **C3**: untuk n besar scipy memakai aproksimasi normal; perlu disebut |
| 4.11.2 Holm | 3 pasangan, maksimum berjalan, batas 1 | `holm()`. Tabel 4.50 cocok (0,445313 / 0,445313 / 0,734375) | Ya | – |
| 4.11.2 Simpulan | Signifikan ⇔ p_Holm < 0,05 **dan** CI tidak memuat 0 | `verdict()` | Ya | – |
| 4.2 / 4.13 Penyetelan | Grid 64 + 75 konfigurasi pada validasi seed 1–3 (1.026 unit) dengan kriteria MRR TPL+PARA; cek *leave-one-seed-out* | Jumlah unit 1.026 tereproduksi; grid belum dijalankan | Tertunda | Butuh B1. Lihat juga **C1** |
| 4.10 Generasi | Qwen3.8-27B, 5 chunk, *greedy*, 1.024 token | Belum (Fase 2) | – | **B2**: pastikan nama model persisnya di Hugging Face |

## B. Ketidakcocokan yang perlu diputuskan

**A1. Centroid: himpunan atau barisan token?** Pertanyaan Syarat diet dan Pedoman mengandung "diet" dua kali setelah stopword dihapus.

- Persamaan 2.9 (himpunan): qid 20 LSA-QE berperingkat **49**. Akibatnya MRR menjadi 0,2173, Wilcoxon LSA-QE vs Baseline W = 7 (p = 0,148), dan CI 0,0008–0,1179.
- Barisan token (Gambar 4.18): peringkat **66**, dan Tabel 4.45–4.51 cocok seluruhnya.
- Kedua varian memilih term ekspansi yang berbeda (`kebutuhan` vs `edukasi`).

Usulan: ikuti Gambar 4.18 (barisan token) dan ubah notasi Persamaan 2.9 menjadi penjumlahan atas token, misalnya "untuk setiap token t pada q yang terdapat pada V, dengan m = banyaknya token tersebut". Alasannya, dengan usulan ini semua angka 4.12 tetap berlaku, dan cara ini konsisten dengan bobot Baseline yang juga menghitung kemunculan ganda. Alternatifnya, pertahankan Persamaan 2.9 dan perbarui Tabel 4.45–4.51.

**A2. Tabel 4.1 menulis "lima chunk teratas" untuk RM3**, padahal semua bagian lain menulis fbDocs = 3. Usulan: koreksi Tabel 4.1. Kode tidak perlu diubah.

**B1. Daftar parafrasa belum tersedia (penghambat).** Skenario PARA, grid search, dan *leave-one-seed-out* semuanya membutuhkan daftar 6 parafrasa validasi dan 6 parafrasa pengujian untuk tiap tipe. Bab 4 hanya menampilkan satu kalimat ("Instrumen apa yang dipakai untuk menapis status gizi?", validasi Skrining indeks 3). Mohon kirimkan file daftarnya. Format yang dipakai kode: `config/paraphrases.json` = `{"dev": {tipe: [6]}, "test": {tipe: [6]}}`.

**B2. "Qwen3.8-27B"**: nama ini perlu dicocokkan dengan ID model yang benar-benar ada di Hugging Face sebelum Fase 2.

**B3. RM3 memasukkan angka murni** ("2" menjadi term teratas di Tabel 4.40), sedangkan kosakata LSA membuangnya. Bab 4 memang merancangnya begitu, dan 4.12.9 sudah membahasnya. Usulan: **pertahankan** sesuai Bab 4 dan bahas di Bab 6 sebagai salah satu kelemahan analisis lokal. Mengubahnya sekarang berarti mengubah rancangan.

## C. Saran perbaikan metodologi (opsional, perlu persetujuan)

**C1. Tumpang tindih penyetelan dan pengujian.** 775 dari 1.722 unit uji per skenario (45,0%) berasal dari laporan yang pernah menjadi data validasi pada seed 1–3, yaitu data yang dipakai memilih parameter (`results/fase0_checks.txt`). Konfigurasi memang dikunci sebelum data uji dipakai, tetapi laporan yang sama ikut menentukan parameter. Usulan: hasil utama tetap mengikuti Bab 4. Tambahkan analisis sensitivitas (**eksploratif**) yang hanya memakai unit uji dari laporan yang tidak pernah masuk data validasi seed 1–3.

**C2. Unit tidak independen.** 106 laporan muncul di data uji 1–5 kali sepanjang 10 pengulangan, dan tiap laporan muncul dua kali (TPL dan PARA). Wilcoxon atas ±3.444 unit memperlakukan semuanya sebagai independen, sehingga nilai p bisa terlalu kecil. Usulan: hasil utama tetap mengikuti Bab 4. Tambahkan pemeriksaan **eksploratif** di tingkat laporan, yaitu *bootstrap* per laporan atau Wilcoxon atas rata-rata selisih RR per laporan.

**C3. Metode nilai p Wilcoxon.** Untuk n besar dengan banyak *ties*, scipy memakai aproksimasi normal dengan koreksi *ties* dan tanpa koreksi kontinuitas. Usulan: tambahkan satu kalimat di 4.11.2 atau Bab 5.

## D. Kesalahan rujukan silang di Bab 4 (perbaikan redaksional)

| Lokasi | Tertulis | Seharusnya |
|---|---|---|
| 4.6.2 | Persamaan (2.7) untuk bobot sel | 2.6 |
| 4.6.3 | SVD Persamaan (2.8); normalisasi Persamaan (2.9) | 2.7; uraian setelah 2.7 (2.9 adalah centroid) |
| 4.10 | Faithfulness 2.18, BERTScore 2.19–2.21, BLEU 2.16, ROUGE-L 2.17 | 2.19, 2.21–2.23, 2.17, 2.18 |
| 4.10 | "empat metrik"; Bab 3.7.2 hanya menyebut Faithfulness + BERTScore dan menulis "ketiga metrik" | Samakan: empat metrik (Bab 3.7.2 perlu ditambah BLEU dan ROUGE-L) |
| 4.11.2, 4.12.11 | Holm Persamaan 2.22 | 2.24 |
| 4.12.2 | Parafrasa "sesuai Gambar 4.10" | Gambar 4.11 |
| 4.12.9 | *Softmax* "Gambar 4.23", model relevansi "Gambar 4.24" | Gambar 4.27, 4.28 |
| 4.12.10 | RR/NDCG "Gambar 4.28" | Gambar 4.32 |
| 4.12.11 | Holm "Gambar 4.31" | Gambar 4.36 |
| Bab 2 | Penomoran persamaan melompati 2.16 dan 2.20 | Nomori ulang |
| Brief Fase 1 | "peringkat sepuluh unit di Tabel 4.44" | Tabel 4.45 |
| 3.6 | "matriks berbobot TF-IDF", "diproyeksikan ke ruang laten", parameter "bobot query asli" | Samakan dengan 4.6–4.7: log-tf × ln(N/df), centroid, γ = bobot komponen ekspansi |
