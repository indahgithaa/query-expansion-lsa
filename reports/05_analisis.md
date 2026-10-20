# Tahap 6 - Ablasi dan analisis kegagalan

Sumber angka: `results/05_analysis.json`, `results/05_expansion_overlap.csv`,
`results/04_test.json`. Config terkunci SHA-256 `cf1e14b2d9c967cffc4283b30b63eea67139aaf9957c507d224e021087667ab3`.
Dihitung atas 3444 unit test (seed 1-10, skenario TPL dan PARA).

## 1. Kontribusi tiap komponen

| Langkah | dMRR |
| --- | ---: |
| Lokal saja (RM3 - B1) | +0.0044 |
| Global saja (LSA-QE - B1) | +0.0343 |
| Jumlah kedua kontribusi bila aditif | +0.0387 |
| Gabungan sebenarnya (RM3+LSA - B1) | +0.0268 |
| LSA ditambahkan di atas RM3 | +0.0224 |
| RM3 ditambahkan di atas LSA | -0.0075 |

Gabungan (+0.0268) **lebih kecil** daripada jumlah kontribusi
terpisah (+0.0387). Pada MRR, kedua komponen karena itu
**sub-aditif**: keduanya sebagian besar memperbaiki query yang sama, dan menambahkan
RM3 di atas LSA justru menggeser sedikit bobot dari term LSA yang lebih berguna.

Gambarannya berbalik pada recall dalam:

| Metrik | B1 | RM3 | LSA-QE | RM3+LSA |
| --- | ---: | ---: | ---: | ---: |
| Recall@10 | 0.575 | 0.615 | 0.631 | 0.650 |
| Tingkat kegagalan | 0.425 | 0.385 | 0.369 | 0.350 |
| Unit di top-10 | 1980 | 2117 | 2173 | 2237 |

RM3 yang ditambahkan di atas LSA menaikkan Recall@10 sebesar +0.0186 meskipun menurunkan MRR sebesar -0.0075.
Jumlah unit yang masuk top-10 naik monoton di seluruh rangkaian metode
B1 1980 -> RM3 2117 -> LSA-QE 2173 -> RM3+LSA 2237.
Artinya komponen lokal memindahkan gold dari luar top-10 ke dalam top-10, tetapi
jarang memindahkannya ke peringkat satu atau dua - persis di mana MRR memberi bobot.

## 2. Seberapa berbeda kedua himpunan term ekspansi

Jaccard antara 30 term RM3 dan 10 term LSA
per query: rata-rata **0.0931**, median 0.0811, rata-rata
3.30 term berimpit. Nilai maksimum yang mungkin adalah
0.3333 (karena himpunan LSA hanya berisi 10 term), jadi yang tercapai sekitar 28% dari maksimum.
Sebanyak 11.6% query tidak punya satu pun term bersama.

| Pecahan | Jaccard rata-rata |
| --- | ---: |
| Skenario TPL | 0.0929 |
| Skenario PARA | 0.0932 |
| Tipe skrining | 0.0910 |
| Tipe status_gizi | 0.1174 |
| Tipe biokimia | 0.0842 |
| Tipe pes | 0.0191 |
| Tipe tujuan_diet | 0.0991 |
| Tipe prinsip_diet | 0.1299 |
| Tipe syarat_diet | 0.1177 |
| Tipe pedoman | 0.0597 |

Term yang paling sering muncul di kedua himpunan sekaligus: `makanan`, `gizi`, `energi`, `hasil`, `protein`, `anak`, `tujuan`, `pemberian`, `kebutuhan`, `terkait`.
Semuanya kosakata gizi umum - bukan istilah spesifik pasien.

**Temuan penting:** kedua ekspansi hampir tidak beririsan, tetapi itu tidak
membuatnya saling melengkapi. Berbeda belum tentu berguna; bagian 4 menunjukkan
sebagian besar term khas RM3 adalah angka.

## 3. Query yang hanya terselamatkan oleh gabungan

| Kategori | Definisi | Jumlah | % unit |
| --- | --- | ---: | ---: |
| Sinergi | RM3+LSA <= 10 sementara RM3 > 10 DAN LSA-QE > 10 (sinergi) | **24** | 0.70% |
| Kebalikannya | RM3 <= 10 DAN LSA-QE <= 10 sementara RM3+LSA > 10 (pasangan simetris) | 12 | 0.35% |

Neto sinergi **+12 query**. Ada efek gabungan yang nyata, tetapi
kecil: 24 dari 3444 unit.

Sebarannya per tipe: syarat_diet 14, status_gizi 5, biokimia 2, skrining 1, prinsip_diet 1, pes 1.
Per skenario: TPL 12, PARA 12 - sinergi muncul seimbang pada template maupun parafrasa.

Sebagai pembanding yang **tidak** simetris: ada 190 unit di mana salah satu
komponen berhasil masuk top-10 tetapi gabungan gagal. Angka itu tidak boleh
dibandingkan langsung dengan 24, karena syaratnya jauh lebih longgar (cukup satu
komponen berhasil, bukan keduanya). Pembandingnya yang benar adalah baris
"Kebalikannya" di tabel atas (12).

## 4. Contoh term ekspansi

Cara pemilihan contoh (ditetapkan tanpa melihat metrik): Tipe pada indeks genap TYPE_ORDER (0,2,4,6) ditambah indeks 7; tiap tipe diwakili unit seed 1 dengan qid terkecil; skenario PARA untuk contoh ke-2 dan ke-4, TPL untuk sisanya. Aturan ditetapkan tanpa melihat metrik.

### Contoh 1 - tipe `skrining`, skenario TPL

> Apa metode skrining status gizi yang direkomendasikan?

Token query setelah stopword dibuang: `metode`, `skrining`, `status`, `gizi`, `direkomendasikan`.
Peringkat gold: B1 3, RM3 3, LSA-QE 2, RM3+LSA 3.

| # | Term RM3 | Bobot | Term LSA | Cosine |
| ---: | --- | ---: | --- | ---: |
| 1 | `gizi` | 0.0921 | `malnutrisi` | 0.5798 |
| 2 | `tanggal` | 0.0892 | `alat` | 0.5586 |
| 3 | `skrining` | 0.0736 | `anak` | 0.5335 |
| 4 | `tgl` | 0.0728 | `strongkids` | 0.5024 |
| 5 | `kebutuhan` | 0.0712 | `pyms` | 0.4945 |
| 6 | `antropometri` | 0.0542 | `berdasarkan` | 0.4926 |
| 7 | `formulir` | 0.0536 | `kondisi` | 0.4869 |
| 8 | `strongkids` | 0.0369 | `pembanding` | 0.4837 |
| 9 | `ruang` | 0.0363 | `hasil` | 0.4785 |
| 10 | `rumah` | 0.0362 | `divalidasi` | 0.4551 |

Term bersama di sepuluh besar: `strongkids`.

### Contoh 2 - tipe `biokimia`, skenario PARA

> Data biokimia mana yang di luar nilai normal?

Token query setelah stopword dibuang: `data`, `biokimia`, `nilai`, `normal`.
Peringkat gold: B1 9, RM3 7, LSA-QE 7, RM3+LSA 8.

| # | Term RM3 | Bobot | Term LSA | Cosine |
| ---: | --- | ---: | --- | ---: |
| 1 | `normal` | 0.0885 | `pemeriksaan` | 0.6986 |
| 2 | `5` | 0.0826 | `hasil` | 0.6490 |
| 3 | `dl` | 0.0793 | `lab` | 0.6454 |
| 4 | `3` | 0.0698 | `laboratorium` | 0.5913 |
| 5 | `mg` | 0.0608 | `hematologi` | 0.5617 |
| 6 | `7` | 0.0565 | `terbaru` | 0.5441 |
| 7 | `0` | 0.0470 | `fisik` | 0.5180 |
| 8 | `2` | 0.0454 | `diambil` | 0.5152 |
| 9 | `4` | 0.0450 | `klinis` | 0.5060 |
| 10 | `mmol` | 0.0434 | `leukosit` | 0.4980 |

Term bersama di sepuluh besar: tidak ada.

### Contoh 3 - tipe `tujuan_diet`, skenario TPL

> Apa tujuan diet pada pasien ini?

Token query setelah stopword dibuang: `tujuan`, `diet`.
Peringkat gold: B1 1, RM3 1, LSA-QE 3, RM3+LSA 2.

| # | Term RM3 | Bobot | Term LSA | Cosine |
| ---: | --- | ---: | --- | ---: |
| 1 | `gizi` | 0.1092 | `prinsip` | 0.6796 |
| 2 | `makanan` | 0.1037 | `syarat` | 0.6699 |
| 3 | `1` | 0.0670 | `sasaran` | 0.5480 |
| 4 | `buruk` | 0.0612 | `makanan` | 0.5295 |
| 5 | `2` | 0.0539 | `edukasi` | 0.5169 |
| 6 | `pemberian` | 0.0536 | `materi` | 0.5147 |
| 7 | `bb` | 0.0455 | `dianjurkan` | 0.5129 |
| 8 | `terkait` | 0.0364 | `terkait` | 0.5129 |
| 9 | `anak` | 0.0344 | `konten` | 0.5105 |
| 10 | `diet` | 0.0318 | `pemberian` | 0.5083 |

Term bersama di sepuluh besar: `makanan`, `pemberian`, `terkait`.

### Contoh 4 - tipe `syarat_diet`, skenario PARA

> Syarat apa saja yang harus dipenuhi pada diet pasien ini?

Token query setelah stopword dibuang: `syarat`, `dipenuhi`, `diet`.
Peringkat gold: B1 7, RM3 2, LSA-QE 7, RM3+LSA 5.

| # | Term RM3 | Bobot | Term LSA | Cosine |
| ---: | --- | ---: | --- | ---: |
| 1 | `1` | 0.0846 | `prinsip` | 0.7150 |
| 2 | `gizi` | 0.0838 | `preskripsi` | 0.5902 |
| 3 | `2` | 0.0822 | `energi` | 0.5678 |
| 4 | `makanan` | 0.0506 | `tujuan` | 0.5628 |
| 5 | `berkaitan` | 0.0405 | `makanan` | 0.5372 |
| 6 | `ditandai` | 0.0405 | `sesuai` | 0.5271 |
| 7 | `kgbba` | 0.0405 | `kebutuhan` | 0.5164 |
| 8 | `diet` | 0.0348 | `protein` | 0.5092 |
| 9 | `terkait` | 0.0337 | `dihitung` | 0.4911 |
| 10 | `5` | 0.0319 | `pemberian` | 0.4771 |

Term bersama di sepuluh besar: `makanan`.

### Contoh 5 - tipe `pedoman`, skenario TPL

> Apa pedoman diet yang digunakan dalam perencanaan diet pasien ini?

Token query setelah stopword dibuang: `pedoman`, `diet`, `perencanaan`, `diet`.
Peringkat gold: B1 7, RM3 9, LSA-QE 2, RM3+LSA 5.

| # | Term RM3 | Bobot | Term LSA | Cosine |
| ---: | --- | ---: | --- | ---: |
| 1 | `lemak` | 0.0634 | `tujuan` | 0.5813 |
| 2 | `1` | 0.0621 | `prinsip` | 0.5410 |
| 3 | `4` | 0.0517 | `makanan` | 0.5017 |
| 4 | `diet` | 0.0495 | `gizi` | 0.4988 |
| 5 | `berat` | 0.0449 | `syarat` | 0.4916 |
| 6 | `protein` | 0.0445 | `edukasi` | 0.4707 |
| 7 | `kebutuhan` | 0.0440 | `dianjurkan` | 0.4628 |
| 8 | `badan` | 0.0437 | `motivasi` | 0.4626 |
| 9 | `kondisi` | 0.0429 | `mencoba` | 0.4572 |
| 10 | `tinggi` | 0.0411 | `rekomendasi` | 0.4470 |

Term bersama di sepuluh besar: tidak ada.

Pola yang terlihat konsisten di kelima contoh: RM3 mengambil kata yang benar-benar
muncul di chunk peringkat atas laporan itu - termasuk banyak angka dan satuan -
sedangkan LSA mengambil kosakata domain yang bertetangga secara laten dan tidak
harus muncul di laporan tersebut (`malnutrisi`, `strongkids`, `pyms`, `preskripsi`).

## 5. Angka yang memakan kapasitas ekspansi RM3

Rata-rata **16.7%** dari 30 term ekspansi RM3 adalah angka
murni, membawa **16.7%** massa probabilitas model relevansi.
LSA membuang angka murni saat membangun vocab, jadi angkanya nol di sana.

| Tipe | Porsi term RM3 yang berupa angka |
| --- | ---: |
| skrining | 12.5% |
| status_gizi | 17.3% |
| biokimia | 22.9% |
| pes | 16.5% |
| tujuan_diet | 14.8% |
| prinsip_diet | 15.8% |
| syarat_diet | 17.4% |
| pedoman | 13.9% |

Inilah mekanisme paling konkret di balik lemahnya kontribusi RM3. Laporan asuhan gizi
padat angka (nilai lab, berat badan, kebutuhan energi), dan karena chunk indeks tidak
dibersihkan dari angka, model relevansi membuang sekitar seperenam kapasitasnya pada
token yang hampir tidak membedakan chunk. Pada tipe `biokimia` porsinya tertinggi
(22.9%), dan di sanalah RM3 justru
menolong (dMRR +0.0177) - angka memang menjadi
isi jawabannya. Di tipe lain, angka hanyalah derau.

## 6. Analisis kegagalan per tipe

| Tipe | n | dMRR RM3 | dMRR LSA-QE | dMRR RM3+LSA | RI RM3+LSA | Penurunan besar | Gagal | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| skrining | 480 | +0.0269 | +0.1053 | +0.0866 | +0.2333 | 2 | 0.200 | 0.091 |
| status_gizi | 480 | -0.0049 | -0.0027 | +0.0055 | +0.1583 | 23 | 0.144 | 0.117 |
| biokimia | 474 | +0.0177 | +0.0245 | +0.0219 | +0.2321 | 2 | 0.599 | 0.084 |
| pes | 480 | -0.0037 | -0.0125 | -0.0105 | -0.1417 | 1 | 0.604 | 0.019 |
| tujuan_diet | 480 | -0.0153 | +0.0271 | +0.0147 | +0.1292 | 9 | 0.169 | 0.099 |
| prinsip_diet | 476 | +0.0082 | +0.0798 | +0.0551 | +0.2185 | 4 | 0.206 | 0.130 |
| syarat_diet | 480 | +0.0103 | +0.0268 | +0.0230 | +0.3792 | 6 | 0.492 | 0.118 |
| pedoman | 94 | -0.0385 | -0.0060 | -0.0178 | -0.1383 | 6 | 0.564 | 0.060 |

Dua tipe **turun** dibanding B1 pada metode utama: `pedoman`, `pes`. Keduanya juga satu-satunya tipe dengan
Robustness Index negatif.

### `pedoman` (-0.0178, RI -0.1383)

Tipe paling jarang (94 unit, satu-satunya yang di bawah 400), sehingga
estimasinya paling tidak stabil dari semua tipe dan selisih sekecil ini masih
dalam jangkauan derau. Jaccard juga rendah (0.060). Pertanyaan pedoman meminta nama rujukan
(mis. penuntun diet atau standar acuan) yang biasanya muncul satu kali saja di
laporan, sehingga umpan balik semu dari lima chunk teratas hampir tidak pernah
memuatnya; RM3 sendiri bahkan merugikan di sini (-0.0385).

### `pes` (-0.0105, RI -0.1417)

Jaccard terendah dari semua tipe (0.019): RM3 dan LSA menunjuk
arah yang hampir sepenuhnya berbeda, dan tidak satu pun mengenai sasaran.
Pertanyaan PES meminta **struktur** jawaban (problem-etiologi-sign), bukan topik.
Yang menandai chunk jawaban adalah penanda format `berkaitan`, `ditandai`, `disebabkan`: penanda itu muncul di
**98.3%** chunk gold tipe `pes`, tetapi juga di
**16.6%** chunk mana pun pada laporan yang
sama. Jadi penanda itu nyaris selalu benar tetapi hampir tidak membedakan -
sekitar satu dari enam chunk memuatnya. Tidak ada satu pun term topik yang bisa
ditambahkan untuk memperbaikinya, dan ekspansi justru menambah kosakata gizi
umum yang menarik ranking ke chunk asesmen.
Tingkat kegagalan karena itu tetap tinggi untuk semua metode (0.598 pada B1, 0.604 pada RM3+LSA).
Ini keterbatasan representasi bag-of-words, bukan keterbatasan ekspansi: tidak
ada pembobotan ulang term yang bisa menyandikan pola sintaksis.

Sebaliknya, tipe dengan keuntungan terbesar adalah `skrining` (+0.0866) dan `prinsip_diet` (+0.0551): keduanya menanyakan topik
berkosakata khas yang punya tetangga laten jelas, persis situasi yang dilayani LSA.
`syarat_diet` punya Robustness Index tertinggi (+0.3792) dan menyumbang 14 dari 24 kasus sinergi.

## 7. Kesimpulan

1. **Komponen global adalah sumber perbaikan utama.** LSA menyumbang +0.0343 MRR di atas B1, hampir delapan kali kontribusi RM3 (+0.0044).
2. **Keduanya sub-aditif pada MRR, tetapi saling melengkapi pada kedalaman.** Gabungan hanya mencapai +0.0268 dari potensi +0.0387 bila aditif, namun memberi Recall@10 dan
   tingkat kegagalan terbaik dari keempat metode.
3. **Nilai sebenarnya dari komponen lokal adalah kestabilan, bukan rata-rata.** Penurunan besar berkurang dari 107 (LSA-QE) menjadi 53 (RM3+LSA), dan Robustness Index naik dari +0.1060 menjadi +0.1641.
4. **Himpunan term yang berbeda tidak otomatis berarti informasi yang melengkapi.** Jaccard hanya 0.0931, tetapi sekitar seperenam term RM3 adalah angka, sehingga sebagian besar perbedaan itu tidak membawa daya pembeda.
5. **Dua tipe pertanyaan tidak terbantu ekspansi apa pun** (`pes`, `pedoman`). Keduanya meminta struktur atau rujukan bernama, bukan topik - di luar jangkauan
   pembobotan ulang term.

Dalam kerangka Xu & Croft (1996): pada korpus ini analisis global membawa sinyal,
analisis lokal membawa rem. Menggabungkan keduanya layak dilakukan bila yang dinilai
adalah keandalan dan recall; bila yang dinilai hanya MRR, LSA saja sudah setara.
