# Tahap 0 - Pra-pemrosesan

Sumber angka: `results/00_preprocess.json`.

## Normalisasi

Teks dan nama file dinormalisasi dengan urutan: NFKC, lalu pola `#Uxxxx` diubah
menjadi karakter Unicode yang diwakilinya, lalu U+2060 (word joiner) dihapus.
`doc_id` D001-D120 diberikan menurut urutan nama file chunk ternormalisasi,
dan `chunk_id` berformat `{doc_id}:P{page:04d}:C{chunk_index:03d}`.

## Sanity check (wajib)

| Pemeriksaan | Harapan | Hasil | Status |
| --- | ---: | ---: | --- |
| QA mentah | 911 | 911 | OK |
| QA setelah dedup | 858 | 858 | OK |
| Dokumen | 120 | 120 | OK |
| Chunk | 13.737 | 13.737 | OK |
| Gold per QA | tepat 1 | tepat 1 untuk 858 QA | OK |

Deduplikasi memakai kunci (nama file ternormalisasi, teks pertanyaan) dan menyimpan
baris dengan `id` numerik terkecil; 53 baris duplikat dibuang (911 - 858).

## Ukuran korpus

| Besaran | Min | Median | Rata-rata | Maks |
| --- | ---: | ---: | ---: | ---: |
| Chunk per dokumen | 32 | 109.0 | 114.5 | 256 |

Setiap QA dijawab oleh tepat satu chunk dalam laporan pasien yang sama, sehingga
korpus target saat retrieval adalah seluruh chunk laporan tersebut.

## Sebaran tipe pertanyaan

Tipe ditentukan dari teks template persis (bukan model); label ini hanya dipakai
untuk analisis dan penjaminan cakupan split, tidak pernah dipakai saat inferensi.

| Tipe | Jumlah QA | Jumlah dokumen |
| --- | ---: | ---: |
| skrining | 120 | 120 |
| status_gizi | 120 | 120 |
| biokimia | 119 | 119 |
| pes | 120 | 120 |
| tujuan_diet | 120 | 120 |
| prinsip_diet | 119 | 119 |
| syarat_diet | 119 | 119 |
| pedoman | 21 | 21 |
| **Total** | **858** | **120** |

Tipe `pedoman` jauh lebih jarang (21 QA pada 21 dokumen). Karena itu syarat
"ketiga bagian split memuat kedelapan tipe" pada Tahap 1 praktis ditentukan oleh
tipe ini.

## Parafrasa

`data/paraphrases.json` memuat 8 tipe x 2 partisi x 6 kalimat.
Daftar `dev` dan `test` tidak beririsan, sehingga kalimat yang dipakai saat seleksi
parameter tidak muncul lagi saat pengujian.

## Catatan

Pola `#Uxxxx` tidak ditemukan sama sekali pada data yang diberikan; aturannya tetap
diterapkan agar pipeline tahan terhadap ekspor berikutnya. U+2060 memang ada
(112 kemunculan) dan dihapus.
