# Tahap 4 - Seleksi parameter di dev

Sumber angka: `results/02_dev_selection.json` dan `results/02_dev_grid.csv`.
Test belum dibuka. Seluruh isi laporan ini berasal dari seed 1-3 pada bagian **dev**
(24 dokumen per seed); dokumen test tidak pernah disentuh, dan LSA hanya dilatih
pada 72 dokumen koleksi tiap seed.

## Ringkasan

- Grid 64 konfigurasi per metode berparameter: k in {100, 200, 300, 500}, n in {5, 10, 20, 30}, gamma in {0.5, 1.0, 2.0, 4.0}.
- Kriteria: mean(MRR(TPL), MRR(PARA)) pada seed 1-3 digabung. Seri dipecah dengan k terkecil, lalu n, lalu gamma.
- Unit evaluasi: 1026 (3 seed x 2 skenario x ~171 QA dev).
- RM3 tidak di-tuning: 5 chunk feedback, 30 term, lambda = 0,5, ditetapkan a priori.
- Waktu komputasi grid: 148 detik.

## Baseline tanpa tuning

| Metode | MRR TPL | MRR PARA | Kriteria | Recall@1 | Recall@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 0.2994 | 0.2276 | **0.2635** | 0.107 | 0.579 | 0.3253 |
| RM3 | 0.3127 | 0.2344 | **0.2736** | 0.119 | 0.605 | 0.3390 |

Parafrasa jauh lebih sulit daripada template: pada B1 selisih MRR antara TPL dan PARA
mencapai 0.0718. Inilah celah yang ingin ditutup oleh ekspansi query.

## Konfigurasi terpilih

| Metode | k | n | gamma | Kriteria (dev) | MRR TPL | MRR PARA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSA-QE | 300 | 10 | 2.0 | **0.3281** | 0.3414 | 0.3148 |
| RM3+LSA | 300 | 10 | 4.0 | **0.3207** | 0.3363 | 0.3051 |

Parameter dipilih terpisah untuk tiap metode berparameter supaya perbandingan
RM3+LSA vs LSA-QE pada Tahap 5 tidak berat sebelah (lihat `DECISIONS.md`).

## Delapan konfigurasi teratas per metode

### LSA-QE

| # | k | n | gamma | Kriteria | MRR TPL | MRR PARA | Recall@1 | Recall@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 300 | 10 | 2.0 | 0.3281 | 0.3414 | 0.3148 | 0.185 | 0.635 |
| 2 | 500 | 5 | 1.0 | 0.3245 | 0.3504 | 0.2986 | 0.183 | 0.636 |
| 3 | 300 | 10 | 1.0 | 0.3232 | 0.3422 | 0.3042 | 0.175 | 0.635 |
| 4 | 500 | 5 | 2.0 | 0.3230 | 0.3493 | 0.2968 | 0.182 | 0.637 |
| 5 | 300 | 10 | 4.0 | 0.3225 | 0.3329 | 0.3121 | 0.177 | 0.622 |
| 6 | 500 | 10 | 2.0 | 0.3207 | 0.3406 | 0.3008 | 0.180 | 0.640 |
| 7 | 500 | 5 | 4.0 | 0.3193 | 0.3463 | 0.2922 | 0.185 | 0.613 |
| 8 | 500 | 20 | 2.0 | 0.3191 | 0.3269 | 0.3112 | 0.161 | 0.657 |

Rentang seluruh 64 konfigurasi: 0.2791 sampai 0.3281.

### RM3+LSA

| # | k | n | gamma | Kriteria | MRR TPL | MRR PARA | Recall@1 | Recall@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 300 | 10 | 4.0 | 0.3207 | 0.3363 | 0.3051 | 0.181 | 0.647 |
| 2 | 500 | 10 | 4.0 | 0.3201 | 0.3416 | 0.2986 | 0.184 | 0.638 |
| 3 | 500 | 5 | 4.0 | 0.3198 | 0.3454 | 0.2943 | 0.181 | 0.637 |
| 4 | 500 | 20 | 4.0 | 0.3148 | 0.3281 | 0.3014 | 0.162 | 0.662 |
| 5 | 300 | 5 | 4.0 | 0.3137 | 0.3394 | 0.2880 | 0.173 | 0.628 |
| 6 | 200 | 5 | 4.0 | 0.3123 | 0.3394 | 0.2853 | 0.166 | 0.620 |
| 7 | 300 | 10 | 2.0 | 0.3113 | 0.3328 | 0.2899 | 0.161 | 0.661 |
| 8 | 500 | 30 | 4.0 | 0.3109 | 0.3157 | 0.3060 | 0.156 | 0.668 |

Rentang seluruh 64 konfigurasi: 0.2790 sampai 0.3207.

## Sensitivitas terhadap tiap parameter

Tiap sel adalah kriteria terbaik yang dicapai saat parameter itu dikunci
(dua parameter lain dibebaskan).

**LSA-QE**

| k | 100 | 200 | 300 | 500 |
| --- | ---: | ---: | ---: | ---: |
| kriteria | 0.3108 | 0.3181 | 0.3281 | 0.3245 |

| n | 5 | 10 | 20 | 30 |
| --- | ---: | ---: | ---: | ---: |
| kriteria | 0.3245 | 0.3281 | 0.3191 | 0.3136 |

| gamma | 0.5 | 1.0 | 2.0 | 4.0 |
| --- | ---: | ---: | ---: | ---: |
| kriteria | 0.3123 | 0.3245 | 0.3281 | 0.3225 |

**RM3+LSA**

| k | 100 | 200 | 300 | 500 |
| --- | ---: | ---: | ---: | ---: |
| kriteria | 0.3068 | 0.3123 | 0.3207 | 0.3201 |

| n | 5 | 10 | 20 | 30 |
| --- | ---: | ---: | ---: | ---: |
| kriteria | 0.3198 | 0.3207 | 0.3148 | 0.3109 |

| gamma | 0.5 | 1.0 | 2.0 | 4.0 |
| --- | ---: | ---: | ---: | ---: |
| kriteria | 0.2876 | 0.3016 | 0.3113 | 0.3207 |

## Optimum di batas grid

- **LSA-QE**: tidak ada parameter di batas grid (k=300, n=10, gamma=2.0).
- **RM3+LSA**: parameter gamma jatuh di batas grid (k=300, n=10, gamma=4.0).

Sesuai aturan tetap, hal ini **dilaporkan saja dan grid tidak diperluas**.
Konsekuensinya perlu dicatat saat menafsirkan hasil: kriteria RM3+LSA naik monoton
terhadap gamma di seluruh grid, artinya metode gabungan masih ingin memberi bobot
lebih besar pada komponen LSA. Nilai gamma optimum yang sebenarnya mungkin > 4,
dan angka RM3+LSA pada Tahap 5 karenanya adalah batas bawah.

## Bias seleksi (leave-one-seed-out)

Untuk tiap seed yang ditahan, config dipilih ulang hanya dari dua seed sisanya,
lalu dinilai pada seed yang ditahan. Selisihnya terhadap nilai in-sample
memperkirakan seberapa optimistis angka dev.

| Metode | Kriteria in-sample | LOSO | Bias | Config terpilih per lipatan |
| --- | ---: | ---: | ---: | --- |
| LSA-QE | 0.3283 | 0.3215 | +0.0068 | tahan seed 1 -> k=300, n=10, gamma=2.0; tahan seed 2 -> k=300, n=10, gamma=2.0; tahan seed 3 -> k=500, n=5, gamma=1.0 |
| RM3+LSA | 0.3209 | 0.3084 | +0.0125 | tahan seed 1 -> k=300, n=10, gamma=4.0; tahan seed 2 -> k=300, n=5, gamma=4.0; tahan seed 3 -> k=500, n=5, gamma=4.0 |

Bias kecil untuk kedua metode, dan konfigurasi yang dipilih tiap lipatan berdekatan
(k 300-500, n 5-10), jadi seleksi tidak terlihat menempel pada satu seed tertentu.

## Temuan yang perlu diperhatikan sebelum test dibuka

1. Kedua metode berbasis LSA unggul jelas atas B1 (0.2635) dan RM3 (0.2736).
   RM3 sendiri hanya menambah 0.0101 di atas B1.
2. **Pada dev, LSA-QE (0.3281) sedikit mengungguli RM3+LSA (0.3207),
   selisih 0.0074.** Dengan kata lain, menambahkan RM3 di atas LSA
   belum terlihat memberi keuntungan pada dev. Ini langsung menyentuh pertanyaan
   penelitian dan harus dilaporkan apa adanya, bukan dikompensasi dengan mengubah grid.
3. Arah gamma pada RM3+LSA (monoton naik sampai batas grid) mendukung pembacaan yang
   sama: semakin besar porsi LSA, semakin baik metode gabungan.
4. Ekspansi menutup sebagian celah TPL-PARA: pada B1 selisihnya 0.0718, pada LSA-QE terpilih 0.0267.

## Status

Config **belum dikunci** dan test **belum dibuka**. Langkah berikutnya menunggu
keputusan pengguna sesuai Tahap 4:

1. Setujui konfigurasi di atas (atau tentukan yang lain dari `results/02_dev_grid.csv`).
2. `scripts/03_lock_config.py` menulis `configs/locked_config.json` dan mencatat
   SHA-256-nya di `DECISIONS.md`.
3. Baru setelah itu `scripts/04_test.py` dijalankan satu kali pada seed 1-10.
