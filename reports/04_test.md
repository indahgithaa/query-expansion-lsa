# Tahap 5 - Hasil test

Sumber angka: `results/04_test.json` dan `results/04_test_units.csv`.

Test dijalankan **satu kali** pada seed 1-10 dengan config
terkunci `configs/locked_config.json` (SHA-256 `cf1e14b2d9c967cffc4283b30b63eea67139aaf9957c507d224e021087667ab3`).
Total 3444 unit evaluasi = 10 seed x 2 skenario x ~172 QA test.
Parameter terpilih: LSA-QE k=300, n=10, gamma=2.0; RM3+LSA k=300, n=10, gamma=2.0. RM3 tidak di-tuning.

## Hasil utama

| Metode | MRR | R@1 | R@3 | R@5 | R@10 | R@20 | nDCG@10 | Median peringkat | Gagal (>10) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 0.2763 | 0.139 | 0.311 | 0.427 | 0.575 | 0.724 | 0.3334 | 7 | 0.425 |
| RM3 | 0.2807 | 0.135 | 0.317 | 0.446 | 0.615 | 0.759 | 0.3468 | 7 | 0.385 |
| LSA-QE | 0.3106 | 0.163 | 0.361 | 0.474 | 0.631 | 0.777 | 0.3739 | 6 | 0.369 |
| RM3+LSA | 0.3031 | 0.148 | 0.352 | 0.481 | 0.650 | 0.790 | 0.3729 | 6 | 0.350 |

Urutan berbeda tergantung metrik, dan perbedaan itu adalah temuan utama laporan ini:

- **MRR dan presisi puncak** dimenangkan LSA-QE (0.3106; R@1 0.163).
- **Recall dalam dan tingkat kegagalan** dimenangkan RM3+LSA (R@10 0.650, R@20 0.790, gagal 0.350 vs 0.369).
- **nDCG@10 praktis seri** (0.3739 vs 0.3729).

## Per skenario

| Metode | MRR TPL | MRR PARA | Selisih | R@10 TPL | R@10 PARA |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 0.3084 | 0.2442 | 0.0642 | 0.611 | 0.539 |
| RM3 | 0.3131 | 0.2482 | 0.0650 | 0.650 | 0.579 |
| LSA-QE | 0.3412 | 0.2800 | 0.0612 | 0.681 | 0.581 |
| RM3+LSA | 0.3350 | 0.2711 | 0.0639 | 0.699 | 0.600 |

Parafrasa tetap lebih sulit daripada template untuk semua metode, tetapi ekspansi
mempersempit celahnya dari 0.0642 (B1) menjadi 0.0612 (LSA-QE).
Keuntungan terbesar LSA-QE atas B1 justru ada di PARA (0.0358) dibanding TPL (0.0328): ekspansi memang
menolong justru ketika kata-kata query tidak lagi cocok persis dengan dokumen.

## Per tipe pertanyaan

Label tipe tidak dipakai saat inferensi; pemecahan ini hanya untuk analisis.

| Tipe | n | B1 | RM3 | LSA-QE | RM3+LSA | d(LSA-QE) | d(RM3+LSA) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| skrining | 480 | 0.3137 | 0.3407 | 0.4191 | 0.4004 | +0.1053 | +0.0866 |
| status_gizi | 480 | 0.3533 | 0.3484 | 0.3505 | 0.3588 | -0.0027 | +0.0055 |
| biokimia | 474 | 0.1189 | 0.1366 | 0.1433 | 0.1407 | +0.0245 | +0.0219 |
| pes | 480 | 0.1270 | 0.1233 | 0.1145 | 0.1165 | -0.0125 | -0.0105 |
| tujuan_diet | 480 | 0.3586 | 0.3432 | 0.3857 | 0.3733 | +0.0271 | +0.0147 |
| prinsip_diet | 476 | 0.4436 | 0.4518 | 0.5234 | 0.4987 | +0.0798 | +0.0551 |
| syarat_diet | 480 | 0.2435 | 0.2538 | 0.2703 | 0.2666 | +0.0268 | +0.0230 |
| pedoman | 94 | 0.1476 | 0.1091 | 0.1416 | 0.1298 | -0.0060 | -0.0178 |

Tipe yang **turun** dibanding B1 pada metode utama: pes, pedoman.
Analisis kegagalannya ada di `reports/05_analisis.md`.

## Per seed

| Seed | B1 | RM3 | LSA-QE | RM3+LSA | Pemenang |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.2695 | 0.2772 | 0.2922 | 0.2961 | RM3+LSA |
| 2 | 0.2566 | 0.2734 | 0.3092 | 0.3043 | LSA-QE |
| 3 | 0.2683 | 0.2586 | 0.2960 | 0.2781 | LSA-QE |
| 4 | 0.2867 | 0.2947 | 0.3076 | 0.3147 | RM3+LSA |
| 5 | 0.2931 | 0.2947 | 0.3002 | 0.3019 | RM3+LSA |
| 6 | 0.3083 | 0.3103 | 0.3595 | 0.3459 | LSA-QE |
| 7 | 0.3042 | 0.3040 | 0.3462 | 0.3291 | LSA-QE |
| 8 | 0.2449 | 0.2621 | 0.2716 | 0.2739 | RM3+LSA |
| 9 | 0.2685 | 0.2706 | 0.3396 | 0.3118 | LSA-QE |
| 10 | 0.2636 | 0.2623 | 0.2855 | 0.2767 | LSA-QE |

Jumlah seed dimenangkan: LSA-QE 6, RM3+LSA 4.
Keunggulan LSA-QE atas RM3+LSA tidak konsisten antar seed, dan keduanya selalu
mengungguli B1.

## Robustness terhadap B1

| Metode | Naik | Sama | Turun | RI | Masuk top-10 | Keluar top-10 | Penurunan besar (dRR < -0,5) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RM3 | 1359 | 988 | 1097 | +0.0761 | 244 | 107 | 29 |
| LSA-QE | 1540 | 729 | 1175 | +0.1060 | 344 | 151 | 107 |
| RM3+LSA | 1654 | 701 | 1089 | +0.1641 | 394 | 137 | 53 |

Di sinilah metode gabungan unggul paling jelas. Robustness Index RM3+LSA
(+0.1641) jauh di atas LSA-QE (+0.1060),
dan jumlah penurunan besar hanya 53 dibanding 107 pada
LSA-QE - kurang dari separuhnya. RM3+LSA juga memasukkan lebih banyak query ke
top-10 (394 vs 344) sambil mengeluarkan lebih
sedikit (137 vs 151).

Dengan kata lain: LSA sendiri lebih agresif - lebih sering menang besar, tetapi juga
lebih sering merusak query yang semula sudah baik. Komponen RM3 meredam kerusakan itu.

### Robustness per skenario

| Metode | RI TPL | RI PARA | Penurunan besar TPL | Penurunan besar PARA |
| --- | ---: | ---: | ---: | ---: |
| RM3 | +0.0772 | +0.0749 | 16 | 13 |
| LSA-QE | +0.1341 | +0.0778 | 63 | 44 |
| RM3+LSA | +0.1882 | +0.1400 | 29 | 24 |

## Signifikansi

Keluarga perbandingan ditetapkan sebelum test dibuka (lima perbandingan).
CI 95% dari paired bootstrap berstrata (seed x skenario. 10.000 resample. rng 20261019).
ditambah Wilcoxon signed-rank dengan koreksi Holm atas kelima p-value.

| Perbandingan | dMRR | CI 95% | CI tidak memuat 0 | p Wilcoxon | p Holm | Signifikan |
| --- | ---: | --- | :---: | ---: | ---: | :---: |
| RM3 vs B1 | +0.0044 | [+0.0002, +0.0085] | ya | 4.36e-11 | 8.72e-11 | **ya** |
| LSA-QE vs B1 | +0.0343 | [+0.0255, +0.0430] | ya | 1.66e-24 | 6.63e-24 | **ya** |
| RM3+LSA vs B1 | +0.0268 | [+0.0199, +0.0336] | ya | 5.01e-32 | 2.5e-31 | **ya** |
| RM3+LSA vs RM3 | +0.0224 | [+0.0160, +0.0286] | ya | 1.07e-17 | 3.21e-17 | **ya** |
| RM3+LSA vs LSA-QE | -0.0075 | [-0.0128, -0.0023] | ya | 0.437 | 0.437 | tidak |

Pembacaan:

1. Ketiga metode QE mengungguli B1 secara signifikan, dan **RM3+LSA mengungguli RM3**
   secara signifikan (+0.0224). Jadi menambahkan LSA di atas PRF jelas berguna.
2. RM3 sendiri hanya menambah +0.0044 di atas B1 - signifikan secara
   statistik karena n besar, tetapi kecil secara praktis.
3. **RM3+LSA vs LSA-QE** (-0.0075) memberi dua sinyal yang berbeda:
   CI bootstrap [-0.0128, -0.0023] tidak memuat nol,
   tetapi Wilcoxon tidak signifikan setelah Holm (p = 0.437).
   Keduanya tidak bertentangan: bootstrap menguji selisih **rata-rata** RR dan peka
   pada sedikit query dengan perubahan besar, sedangkan Wilcoxon menguji **arah**
   perubahan pada 2276 dari 3444 unit yang peringkatnya
   memang berubah. Arah perubahan itu ternyata berimbang, sehingga selisih MRR yang
   kecil tadi tidak didukung bukti arah yang konsisten. Kesimpulan yang dapat
   dipertahankan: **LSA-QE dan RM3+LSA setara pada MRR**, bukan bahwa LSA-QE lebih baik.

## Jawaban atas pertanyaan penelitian

> Apakah menggabungkan ekspansi global (LSA) dengan ekspansi lokal (PRF/RM3) lebih
> baik daripada tanpa QE, RM3 saja, dan LSA saja?

- **Lebih baik daripada tanpa QE**: ya, jelas dan signifikan (+0.0268 MRR, CI [+0.0199, +0.0336]).
- **Lebih baik daripada RM3 saja**: ya, jelas dan signifikan (+0.0224 MRR).
- **Lebih baik daripada LSA saja**: **tidak pada MRR.** Keduanya setara pada MRR dan
  nDCG@10. Metode gabungan unggul pada recall dalam (R@10, R@20), tingkat kegagalan,
  dan terutama pada robustness (RI +0,164 vs +0,106; penurunan besar 53 vs 107).

Dalam kerangka Xu & Croft (1996), hasil ini menunjukkan analisis **global** adalah
sumber perbaikan utama pada korpus ini, sedangkan analisis **lokal** terutama
berperan sebagai penstabil: ia tidak menaikkan MRR rata-rata di atas LSA, tetapi
memangkas separuh kasus di mana ekspansi merusak query yang semula baik.

Catatan pembatas: `gamma = 2` dipakai untuk kedua metode atas keputusan pengguna,
sedangkan di dev RM3+LSA mencapai nilai tertinggi pada `gamma = 4` (batas grid).
Angka RM3+LSA di sini karena itu adalah batas bawah yang konservatif.
