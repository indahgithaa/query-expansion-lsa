# Tahap 2-3 - Retrieval dasar dan metode ekspansi

Dokumen ini adalah spesifikasi metode yang diimplementasikan, bukan laporan hasil.
Angka empiris ada di `reports/02_dev_selection.md` dan berkas `results/`.
Nilai parameter yang disebut di sini bersifat definisional (ditetapkan sebelum
melihat hasil) dan tercermin di `configs/locked_config.json` setelah config dikunci.

## 1. Kerangka

Semua metode yang dibandingkan adalah **query expansion**. Keempatnya memakai
indeks dan fungsi skor yang sama persis; yang berbeda hanya vektor bobot query
`w`. Dengan begitu selisih antar metode tidak tercampur perbedaan implementasi
retrieval.

Kerangkanya mengikuti analisis lokal vs global (Xu & Croft, 1996):

- **lokal** - RM3 (Lavrenko & Croft, 2001), memakai dokumen peringkat atas dari
  query itu sendiri sebagai umpan balik semu;
- **global** - LSA (Deerwester et al., 1990), memakai struktur ko-okurensi term di
  seluruh koleksi, lepas dari query;
- **gabungan** - RM3+LSA, metode utama penelitian ini.

## 2. Tokenisasi

| | Query | Chunk yang diindeks | Token feedback RM3 | Dokumen pelatihan LSA |
| --- | --- | --- | --- | --- |
| Lowercase + `\w+` | ya | ya | ya | ya |
| Stopword dihapus | **ya** | tidak | **ya** | **ya** |

Stopword = daftar Sastrawi ditambah `{apa, bagaimana, yang, ini, pasien}`. Kelima
kata tambahan muncul di hampir semua template sehingga tidak membedakan apa pun.

Chunk yang diindeks sengaja mempertahankan stopword: panjang dokumen dan `avgdl`
BM25 dihitung atas teks penuh, sehingga normalisasi panjang tidak terdistorsi.

## 3. BM25

Korpus target satu query adalah **seluruh chunk laporan pasien yang ditanyakan**;
indeks BM25 dibangun per laporan. Pencarian karena itu bersifat intra-dokumen:
tugasnya menemukan chunk jawaban di dalam satu laporan, bukan menemukan laporannya.

BM25 Okapi dengan `k1 = 1,5`, `b = 0,75`:

```
idf(t)     = ln((N - df(t) + 0,5) / (df(t) + 0,5))        idf negatif -> 0,25 * rata-rata idf
bm25(t, c) = idf(t) * tf(t,c) * (k1 + 1)
             / (tf(t,c) + k1 * (1 - b + b * |c| / avgdl))
```

Rata-rata idf pada koreksi epsilon dihitung atas seluruh nilai idf termasuk yang
negatif, **sebelum** penggantian - persis seperti `rank_bm25.BM25Okapi`.
`tests/test_bm25.py` membandingkan implementasi ini dengan `rank_bm25` pada lima
query, termasuk query yang memicu idf negatif.

Query berbobot:

```
skor(c) = sum_t  w_t * bm25(t, c)
```

Ranking memakai sort stabil menurun, sehingga seri mengikuti urutan `chunk_id`.
Metrik dihitung dari peringkat gold.

## 4. B1 - tanpa ekspansi

```
w_t = jumlah kemunculan t di query (setelah stopword dihapus)
```

Ini pembanding dasar sekaligus sumber peringkat awal untuk umpan balik RM3.

## 5. RM3 - ekspansi lokal

Parameter **ditetapkan a priori dan tidak di-tuning**: 5 chunk feedback, 30 term,
`lambda = 0,5`.

1. Jalankan B1 pada laporan target, ambil 5 chunk teratas sebagai umpan balik semu.
2. Bobot tiap dokumen umpan balik = `softmax(skor BM25)` atas kelima skor itu.
3. Model relevansi atas token umpan balik tanpa stopword:
   `P(t|R) = sum_d bobot_d * tf(t,d) / |d|`
4. Ambil 30 term dengan `P(t|R)` tertinggi; seri diurutkan alfabetis.
5. Normalkan ulang ke-30 term itu sehingga berjumlah 1 (lihat `DECISIONS.md`:
   tanpa normalisasi ulang, bobot efektif antara bagian query dan bagian feedback
   berbeda tiap query padahal `lambda` ditetapkan tetap).
6. Interpolasi:
   `w_t = lambda * count_q(t) / |q| + (1 - lambda) * P(t|R)`

Karena kedua suku adalah distribusi, `w` berjumlah 1.

Catatan: softmax diterapkan pada skor BM25 mentah. Skor BM25 berskala belasan,
jadi softmax sangat tajam dan bobot umpan balik praktis didominasi chunk peringkat 1.
Ini perilaku yang diminta spesifikasi dan dicatat apa adanya.

## 6. LSA - ekspansi global

Dilatih **per seed, hanya pada chunk 72 dokumen koleksi**. Dokumen dev dan test
tidak pernah ikut melatih LSA, sehingga ruang laten tidak pernah melihat laporan
yang dievaluasi.

1. Vocab dari token tanpa stopword: `df >= 2`, angka murni dibuang.
2. Matriks (dokumen x term) dengan bobot sel `(1 + ln tf) * ln(N/df)`.
3. `TruncatedSVD(n_components = k, random_state = 0)`.
4. Vektor term = baris `V_k Sigma_k`, dinormalisasi L2.

Ekspansi `E(q)`:

1. Ambil centroid vektor term query yang ada di vocab, normalisasi L2.
2. Cosine centroid terhadap seluruh vocab.
3. Buang term yang sudah ada di query.
4. Ambil `n` teratas dengan cosine > 0; seri alfabetis.
5. Bobot = cosine, dinormalkan sehingga berjumlah 1.

Bila tidak ada satu pun term query yang masuk vocab, `E(q)` kosong dan metode
turun persis ke basisnya (diuji di `tests/test_engine.py`).

## 7. Penggabungan

```
LSA-QE    w = B1  + gamma * |q|    * E(q)
RM3+LSA   w = RM3 + gamma * lambda * E(q)
```

Kedua skala dipilih supaya `gamma` punya arti yang sama di kedua metode: bobot B1
berjumlah `|q|` dan bagian query RM3 berjumlah `lambda`, jadi pada keduanya

```
massa komponen LSA / massa komponen query = gamma
```

Tanpa penyetaraan ini, `gamma` yang sama akan berarti dua hal berbeda dan grid
tidak bisa dibandingkan antar metode.

## 8. Grid parameter

| Parameter | Nilai | Keterangan |
| --- | --- | --- |
| `k` | 100, 200, 300, 500 | dimensi ruang laten LSA |
| `n` | 5, 10, 20, 30 | jumlah term ekspansi LSA |
| `gamma` | 0,5; 1; 2; 4 | bobot komponen LSA |

Grid ditetapkan sebelum melihat hasil dan sengaja melampaui optimum penelitian
sebelumnya (`k = 300`, `n = 10`, `gamma = 2`) agar optimum tidak jatuh di batas.
Bila optimum tetap di batas, hal itu dilaporkan dan grid **tidak** diperluas.
RM3 tidak ikut grid.

## 9. Apa yang tidak dipakai

Sesuai aturan tetap: tanpa LLM, tanpa reranker, tanpa dense retrieval - hanya
metode matematis dari paper rujukan. Label tipe pertanyaan tidak pernah dibaca
jalur inferensi; label hanya dipakai untuk analisis dan untuk menjamin ketiga
bagian split memuat kedelapan tipe.
