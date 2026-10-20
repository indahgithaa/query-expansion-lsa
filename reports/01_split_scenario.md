# Tahap 1 - Split dan skenario

Sumber angka: `results/01_splits.json`, `results/01_splits_summary.csv`.

## Prosedur split

Split dilakukan **per dokumen**, bukan per QA, supaya tidak ada chunk dari satu
laporan yang muncul di dua bagian sekaligus. Untuk tiap seed, daftar `doc_id`
terurut diacak dengan `random.Random(seed * 1_000_003 + attempt).shuffle`, lalu
dipotong 72 koleksi / 24 dev / 24 test. Percobaan dinaikkan sampai ketiga bagian
memuat kedelapan tipe pertanyaan.

Bagian **koleksi** hanya dipakai untuk melatih LSA (Tahap 3). Bagian **dev**
dipakai untuk memilih parameter (seed 1-3), bagian **test** untuk pengujian akhir
(seed 1-10). Dokumen dev dan test tidak pernah ikut melatih LSA.

## Hasil per seed

| Seed | attempt | QA koleksi | QA dev | QA test | Chunk koleksi | Chunk dev | Chunk test |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 511 | 174 | 173 | 8128 | 2848 | 2761 |
| 2 | 0 | 513 | 171 | 174 | 8341 | 2585 | 2811 |
| 3 | 0 | 513 | 168 | 177 | 7950 | 2826 | 2961 |
| 4 | 0 | 514 | 172 | 172 | 8291 | 2571 | 2875 |
| 5 | 0 | 515 | 170 | 173 | 8067 | 2697 | 2973 |
| 6 | 0 | 517 | 172 | 169 | 7917 | 3261 | 2559 |
| 7 | 0 | 515 | 173 | 170 | 8235 | 2847 | 2655 |
| 8 | 0 | 515 | 173 | 170 | 8363 | 2687 | 2687 |
| 9 | 0 | 518 | 170 | 170 | 8349 | 2519 | 2869 |
| 10 | 0 | 513 | 171 | 174 | 7797 | 2948 | 2992 |

Semua seed menemukan split yang sah pada attempt = 0
atau lebih kecil, dan ketiga bagian setiap seed memuat kedelapan tipe
(`all_types_covered` bernilai true di seluruh entri).

## Skenario

- **TPL** - pertanyaan template asli, delapan kalimat tetap.
- **PARA** - satu dari enam parafrasa milik tipe pertanyaan tersebut. Indeksnya
  `int.from_bytes(sha256(f"{seed}|{qid}").digest()[:8], "big") % 6`, sehingga
  penugasan deterministik, bebas dari urutan pemrosesan, dan berubah antar seed.
  Partisi dev memakai daftar parafrasa `dev`, partisi test memakai daftar `test`.

Unit evaluasi adalah (seed, skenario, qid). Untuk seleksi dev terdapat
171 QA rata-rata per seed x 3 seed x 2 skenario ~= 1026 unit.

## Keseragaman indeks parafrasa

| Indeks | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev (seed 1-3) | 80 | 84 | 88 | 90 | 97 | 74 |
| test (seed 1-10) | 277 | 290 | 288 | 297 | 284 | 286 |

Sebaran mendekati seragam, jadi tidak ada parafrasa yang mendominasi evaluasi.
