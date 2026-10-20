# Catatan Keputusan

Format: `tanggal | keputusan | alasan`

## Lingkungan dan reproduktifitas

2026-10-20 | Python 3.14.7; versi paket dipin di `requirements.txt` (numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, PySastrawi 1.2.1, rank-bm25 0.2.2, pytest 8.4.2) | Aturan tetap no. 6 menuntut determinisme; versi yang dipin membuat hasil SVD dan BM25 bisa direproduksi persis.

2026-10-20 | Semua seed tetap: split `random.Random(seed * 1_000_003 + attempt)`, `TruncatedSVD(random_state=0)`, bootstrap `np.random.default_rng(20261019)` | Aturan tetap no. 6.

2026-10-20 | `rank-bm25` dipasang hanya sebagai rujukan uji, bukan dipakai di pipeline | `tests/test_bm25.py` membandingkan implementasi sendiri dengan `BM25Okapi` pada lima query; implementasi sendiri diperlukan karena BM25 harus menerima query berbobot (`sum_t w_t * BM25_t`), yang tidak didukung `rank_bm25`.

## Data masukan

2026-10-20 | `data/paraphrases.json` tidak ada di repo; 96 parafrasa (8 tipe x 2 partisi x 6 kalimat) ditulis sendiri lalu disetujui pengguna | File ini terdaftar sebagai masukan tetapi tidak tersedia. Parafrasa adalah data evaluasi yang ditulis sekali dan dipakai identik oleh keempat metode, jadi aturan "tanpa LLM" (no. 1) yang berlaku saat inferensi tetap terjaga. File dapat ditimpa pengguna tanpa mengubah kode.

2026-10-20 | Daftar parafrasa `dev` dan `test` dibuat tidak beririsan | Mencegah kalimat yang dipakai saat memilih parameter muncul lagi saat pengujian.

## Tahap 0 - pra-pemrosesan

2026-10-20 | Urutan normalisasi: NFKC, lalu dekode `#Uxxxx`, lalu hapus U+2060 | Mengikuti urutan yang tertulis di spesifikasi. `#Uxxxx` hanya ASCII sehingga urutannya terhadap NFKC tidak mengubah hasil.

2026-10-20 | Pola `#Uxxxx` tidak ditemukan sama sekali pada data (0 kemunculan di nama file maupun teks); aturannya tetap diimplementasikan | Spesifikasi memintanya, dan pipeline jadi tahan terhadap ekspor berikutnya. U+2060 memang ada (112 kemunculan) dan dihapus.

2026-10-20 | `doc_id` D001-D120 diberikan menurut `sorted()` Python (urutan titik kode) atas nama file chunk ternormalisasi, termasuk akhiran `.chunks.json` | Spesifikasi menyebut "urutan nama file ternormalisasi" tanpa menetapkan kolasi; urutan titik kode adalah default yang tidak bergantung locale, jadi deterministik lintas mesin.

2026-10-20 | Chunk di dalam dokumen diurutkan menaik menurut `chunk_id` | Membuat aturan "seri mengikuti urutan chunk_id" pada Tahap 2 terpenuhi hanya dengan sort stabil.

2026-10-20 | Deduplikasi memakai kunci (nama file ternormalisasi, teks pertanyaan), menyimpan `id` numerik terkecil | Spesifikasi. Hasilnya 911 -> 858, sesuai sanity check wajib.

2026-10-20 | Label tipe pertanyaan dihitung dari pencocokan teks template persis dan hanya dipakai untuk analisis dan penjaminan cakupan split | Aturan tetap no. 2. Tidak ada jalur kode inferensi yang membaca `qa.qtype`.

## Tahap 2 - retrieval dasar

2026-10-20 | Stopword = daftar Sastrawi (PySastrawi) + {apa, bagaimana, yang, ini, pasien}, dinormalisasi dan di-lowercase | Spesifikasi. Kelima kata tambahan muncul di hampir semua template sehingga tidak membawa daya pembeda.

2026-10-20 | Stopword dihapus dari query dan dari token feedback RM3, tetapi tidak dari chunk yang diindeks | Spesifikasi. Akibatnya statistik panjang dokumen dan `avgdl` BM25 dihitung atas teks penuh.

2026-10-20 | `idf` negatif diganti `0.25 * rata-rata idf`, dengan rata-rata dihitung atas seluruh nilai idf termasuk yang negatif, sebelum penggantian | Perilaku `rank_bm25.BM25Okapi` (epsilon = 0.25) yang dijadikan acuan kesetaraan; diuji di `tests/test_bm25.py`.

2026-10-20 | Indeks BM25 dibangun per laporan dan di-cache lintas seed dan skenario | Indeks hanya bergantung pada isi laporan, tidak pada seed, sehingga cache tidak membocorkan apa pun antar partisi.

## Tahap 3 - metode

2026-10-20 | `P(t|R)` dinormalkan ulang setelah dipotong 30 term teratas sehingga berjumlah 1 | **Menyimpang dari pembacaan harfiah spesifikasi.** `P(t|R)` sebagaimana didefinisikan adalah distribusi (jumlah 1 sebelum pemotongan). Tanpa normalisasi ulang, massa sisa setelah pemotongan berbeda-beda tiap query, sehingga bobot efektif antara bagian query dan bagian feedback tidak lagi `lambda : (1 - lambda)` melainkan bergantung query - padahal `lambda` ditetapkan 0,5. Normalisasi ulang adalah praktik standar RM3 (Lavrenko & Croft, 2001; implementasi Anserini/Galago) dan membuat skala `gamma * lambda` pada RM3+LSA koheren. Dapat dikembalikan ke perilaku harfiah lewat `rm3_renormalize = false` di config tanpa mengubah kode.

2026-10-20 | Softmax diterapkan langsung pada skor BM25 mentah (tanpa suhu/penskalaan) | Spesifikasi. Catatan: karena skor BM25 berskala belasan, softmax sangat tajam dan bobot dokumen feedback praktis didominasi chunk peringkat 1.

2026-10-20 | "Angka murni" yang dibuang dari vocab LSA didefinisikan sebagai `token.isdigit()` | Default paling lugas; mencakup digit Unicode. Token campuran seperti `2x` atau `g6` tetap dipertahankan karena membawa makna.

2026-10-20 | Vektor term LSA = `svd.components_.T * svd.singular_values_`, lalu dinormalisasi L2 | Dengan `X` berbentuk (dokumen x term), scikit-learn menyimpan `components_ = V_k^T`, sehingga ekspresi ini tepat sama dengan "baris `V_k Sigma_k`" pada spesifikasi.

2026-10-20 | Seri pada pemilihan term ekspansi (RM3 dan LSA) dipecah secara alfabetis | Spesifikasi menyebutnya untuk RM3; aturan yang sama dipakai untuk LSA demi determinisme.

2026-10-20 | Kandidat ekspansi LSA dihitung sekali per (query, k) sampai `n = 30`, lalu dipotong untuk tiap `n` | Daftar untuk `n` kecil adalah prefix daftar `n` besar, jadi hasilnya identik dan grid jauh lebih murah. Diuji di `tests/test_lsa.py::test_candidates_are_a_prefix_chain_across_n`.

2026-10-20 | Model LSA dilatih ulang per (seed, k) tanpa cache ke disk | 12 fit untuk dev selesai dalam ~2 menit; cache akan memakan ~1 GB tanpa manfaat berarti.

## Tahap 4 - seleksi di dev

2026-10-20 | Kriteria seleksi dihitung dengan menggabungkan seed 1-3 (mikro atas query, makro atas skenario): MRR dihitung atas semua unit TPL, lalu atas semua unit PARA, lalu dirata-ratakan | Spesifikasi menyebut "rata-rata MRR(TPL) dan MRR(PARA)" tanpa menetapkan tingkat agregasi. Makro atas skenario mencegah skenario dengan unit lebih banyak mendominasi; mikro atas query adalah default yang lazim.

2026-10-20 | `(k, n, gamma)` dipilih **terpisah untuk LSA-QE dan untuk RM3+LSA** | Kalau keduanya dipaksa memakai satu config, salah satu metode akan dibandingkan pada parameter yang bukan miliknya, sehingga perbandingan RM3+LSA vs LSA-QE (bagian dari keluarga uji Tahap 5) menjadi berat sebelah. RM3 tetap tanpa tuning sesuai spesifikasi.

2026-10-20 | Optimum `gamma` untuk RM3+LSA jatuh di batas atas grid (gamma = 4) | Dilaporkan apa adanya sesuai spesifikasi; grid **tidak** diperluas. Lihat `reports/02_dev_selection.md`.

2026-10-20 | Estimasi LOSO dihitung dengan memilih config pada dua seed lalu menilainya pada seed yang ditahan, diulang untuk ketiga seed dan dirata-ratakan | Memberi ukuran bias seleksi tanpa menyentuh test.

## Struktur kerja

2026-10-20 | Satu skrip per tahap, kecuali Tahap 4 yang dipecah menjadi `scripts/02_dev_grid.py` (komputasi, menulis ke `results/`) dan `scripts/02_dev_report.py` (laporan, membaca dari `results/`) | Menjamin aturan tetap no. 4 secara harfiah: penulis laporan tidak punya akses ke angka selain yang ada di `results/`.

2026-10-20 | Laporan tidak memuat kutipan teks klinis maupun nama file laporan; dokumen dirujuk dengan `doc_id` | Aturan tetap no. 5. Pemetaan `doc_id` -> nama file disimpan di `results/00_preprocess.json` untuk penelusuran.

2026-10-20 | Repo belum di-push ke mana pun | Aturan tetap no. 5 menuntut konfirmasi pengguna bahwa repo private sebelum push.

2026-10-20 | Penomoran berkas laporan mengikuti urutan penulisan, bukan nomor tahap: `00_preprocessing.md` (Tahap 0), `01_split_scenario.md` (Tahap 1), `02_dev_selection.md` (Tahap 4), `03_metode.md` (Tahap 2-3), nanti `04_test.md` (Tahap 5) dan `05_analisis.md` (Tahap 6) | Nama `reports/02_dev_selection.md` untuk Tahap 4 sudah ditetapkan spesifikasi, sehingga penomoran tidak bisa sekaligus sejajar dengan nomor tahap.

2026-10-20 | `scripts/04_test.py` menerima `--partition dev --out-dir <dir>` dan divalidasi lebih dulu pada dev sebelum dijalankan pada test | Aturan tetap no. 3 menuntut test dibuka sekali. Men-debug skrip di atas data test akan melanggar semangat aturan itu; dry run pada dev (seed 1-3) menghasilkan MRR yang identik dengan `scripts/02_dev_grid.py` (B1 0,2635; RM3 0,2736; LSA-QE 0,3281; RM3+LSA 0,3207), sehingga kedua jalur kode saling memverifikasi. Keluaran dry run ditulis ke direktori sementara, bukan `results/`.

## Penguncian config (sebelum test dibuka)

2026-10-20 | Atas keputusan pengguna, `gamma = 2` dipakai untuk **kedua** metode berparameter, bukan `gamma = 4` yang menang di dev untuk RM3+LSA | Dengan gamma yang sama, perbandingan RM3+LSA vs LSA-QE hanya menyisakan perbedaan basis (RM3 vs B1) dan tidak tercampur perbedaan bobot komponen LSA; sekaligus tidak ada parameter terpilih yang jatuh di batas grid. Biayanya: kriteria dev RM3+LSA turun dari 0,3207 (gamma=4) ke 0,3113 (gamma=2), jadi angka test RM3+LSA adalah batas bawah yang konservatif.

2026-10-20 | Atas keputusan pengguna, normalisasi ulang `P(t|R)` dipertahankan (`rm3_renormalize = true`) | Lihat alasan di bagian Tahap 3.

2026-10-20 | Config dikunci di `configs/locked_config.json`, SHA-256 `cf1e14b2d9c967cffc4283b30b63eea67139aaf9957c507d224e021087667ab3` | Aturan tetap no. 3. Test dibuka sekali setelah baris ini dicatat.

## Tahap 5 - test

2026-10-20 | Test dibuka satu kali pada seed 1-10 (3.444 unit) dengan config SHA-256 `cf1e14b2...`; tidak ada parameter yang disetel setelah itu | Aturan tetap no. 3.

2026-10-20 | Wilcoxon memakai `zero_method="wilcox"` (selisih nol dibuang) dan dua sisi | Default scipy dan praktik standar. Dari 3.444 unit, 2.276 berubah peringkat pada perbandingan RM3+LSA vs LSA-QE.

2026-10-20 | "Rata-rata" antar skenario dilaporkan sebagai rata-rata makro MRR(TPL) dan MRR(PARA), konsisten dengan kriteria dev | Kedua skenario punya jumlah unit hampir sama sehingga nilainya praktis sama dengan rata-rata mikro; disimpan sebagai `macro_MRR` di `results/04_test.json`.

2026-10-20 | CI bootstrap untuk RM3+LSA vs LSA-QE tidak memuat nol sementara Wilcoxon tidak signifikan; keduanya dilaporkan dan ditafsirkan sebagai "setara pada MRR" | Kedua uji menanyakan hal berbeda (selisih rata-rata vs arah perubahan). Mengambil hanya yang mendukung satu kesimpulan akan menyesatkan.

## Tahap 6 - analisis

2026-10-20 | Pasangan simetris dari "diselamatkan hanya oleh gabungan" didefinisikan sebagai "kedua komponen berhasil, gabungan gagal" (24 vs 12) | Perhitungan pertama memakai "salah satu komponen berhasil, gabungan gagal" (190) yang syaratnya jauh lebih longgar; membandingkan 24 dengan 190 akan menyesatkan. Angka 190 tetap dilaporkan tetapi diberi label eksplisit.

2026-10-20 | Contoh term ekspansi dipilih dengan aturan yang tidak bergantung metrik: tipe pada indeks 0, 2, 4, 6, 7 dari TYPE_ORDER; tiap tipe diwakili unit seed 1 dengan qid terkecil; skenario PARA untuk contoh ke-2 dan ke-4 | Memenuhi syarat "deterministik, dicatat caranya, minimal 2 parafrasa" tanpa membuka peluang memilih contoh yang menguntungkan.

2026-10-20 | Porsi term ekspansi RM3 yang berupa angka murni diukur (16,7% term, 16,7% massa probabilitas) alih-alih disimpulkan dari beberapa contoh | Klaim mekanisme harus punya angka pendukung; sekarang ada di `results/05_analysis.json`.

2026-10-20 | Penanda format PES didefinisikan sebagai {berkaitan, ditandai, disebabkan} dan diukur, bukan diasumsikan | Ketiganya adalah konektor baku pernyataan PES. Hasil ukur: penanda muncul di 98,3% chunk gold tipe `pes` tetapi juga di 16,6% chunk mana pun pada laporan yang sama - nyaris selalu benar, hampir tidak membedakan.

2026-10-20 | Laporan Tahap 5 di `reports/04_test.md`, Tahap 6 di `reports/05_analisis.md` | Melanjutkan penomoran menurut urutan penulisan.

## Verifikasi determinisme

2026-10-20 | Seluruh pipeline dijalankan ulang dan keluarannya dibandingkan per-bita: semua berkas di `results/` identik kecuali satu field, `runtime_seconds`, yang memang mengukur waktu dan bukan hasil | Aturan tetap no. 6. Jadi determinisme berlaku untuk setiap angka hasil; pemeriksaan ulang sebaiknya mengabaikan `runtime_seconds` (mis. `jq 'del(.runtime_seconds)'`) sebelum membandingkan hash.

## Perubahan lingkup: metode yang diusulkan menjadi LSA saja

2026-10-20 | Atas keputusan pengguna, metode yang diusulkan dipersempit menjadi **LSA-QE saja**; RM3 dan RM3+LSA tetap diukur dan dilaporkan sebagai pembanding, bukan sebagai metode yang diusulkan | Alasan pengguna: kesederhanaan. Tidak ada angka yang berubah dan tidak ada yang dijalankan ulang - hanya penekanan laporan yang bergeser. B1 tetap wajib sebagai baseline. Keputusan ini didukung bukti: LSA-QE menanggung bagian terbesar penurunan kegagalan (0,425 -> 0,369), sedangkan RM3 hanya menambah 0,369 -> 0,350.

2026-10-20 | Empat arah lanjutan diuji dan semuanya gugur: ekspansi morfologis (kontribusi unik 1,3% unit), selective query expansion (held-out -0,0006 logistic / -0,0052 ridge), rank fusion RRF/CombSUM termasuk dengan LM Dirichlet (kegagalan terbaik 0,347 vs 0,339 RM3+LSA), dan penggabungan ekspansi lokal | Seluruhnya dihitung **hanya di dev** (`scripts/06_explorations.py` -> `results/06_explorations.json`). Karena tidak ada kandidat yang mengalahkan acuan, tidak ada yang dipromosikan ke test; config terkunci dan hasil test Tahap 5 tetap utuh dan sah.

2026-10-20 | Gagasan selective query expansion lahir dari analisis pada partisi test (Tahap 6), meskipun pengembangan dan pengujiannya dilakukan sepenuhnya di dev | Dicatat demi transparansi: arah eksplorasi dipengaruhi pengamatan di test, jadi hasil negatifnya tidak boleh diperlakukan sebagai uji yang sepenuhnya pre-registered.

2026-10-20 | Klaim "ekspansi khusus mengatasi variasi parafrasa" diturunkan menjadi "ekspansi menolong sama baiknya pada parafrasa maupun template" | Di dev celah TPL-PARA tampak menyempit tajam, tetapi di test penyempitannya hanya 0,0642 -> 0,0612. Keuntungan relatif memang lebih besar pada parafrasa (+14,7% vs +10,6%), namun terlalu tipis untuk menopang klaim yang lebih kuat. Berpengaruh pada pemilihan judul skripsi.

## Penulisan Bab 4 (Perancangan)

2026-10-20 | Angka pada subbab Perhitungan Manual dihasilkan `scripts/manual_trace_bab4.py` ke `results/manual_trace_bab4.json`, memakai modul `src/qe/` yang sama dengan pipeline | Aturan tetap no. 4 menuntut setiap angka di laporan berasal dari `results/`. Menghitung ulang dengan tangan berisiko meleset dari implementasi, sedangkan menelusuri lewat modul yang sama menjamin rumus di bab itu memang rumus yang dijalankan.

2026-10-20 | Korpus pada Perhitungan Manual dikarang sendiri (10 chunk koleksi, 6 chunk laporan sasaran) dan bukan cuplikan laporan pasien | Aturan tetap no. 5 melarang pengutipan teks klinis pasien. Korpus ilustratif juga membuat setiap langkah dapat diperiksa pembaca, yang tidak mungkin dilakukan pada chunk asli berisi ratusan token.

2026-10-20 | Parameter pada Perhitungan Manual diperkecil menjadi k=3, n=3, 2 chunk feedback, 3 term RM3; gamma=2,0, lambda=0,5, dan seluruh parameter BM25 tetap memakai nilai sistem | k=300 mustahil pada korpus 10 dokumen (rank maksimum 10), dan tabel untuk n=10 tidak muat di halaman. Perbedaan ini dinyatakan terbuka di awal subbab agar tidak dikira nilai yang dipakai saat pengujian.

2026-10-20 | Alasan pemilihan centroid vektor term (bukan folding-in), cosine (bukan dot product atau jarak Euclidean), dan penyaringan cosine > 0 ditulis di Subbab 4.5 | Ketiganya sebelumnya hanya ada sebagai perilaku kode tanpa pembenaran tertulis di mana pun. Pertanyaan penguji yang paling mungkin menyasar tepat ketiga titik itu.

2026-10-20 | Kolom LaTeX `M{}` (lebar tetap, rata tengah) ditambahkan ke preamble `skripsi.tex` | `SKILL.md` dan `tools/latex_to_docx.py` sudah mengenali `L{}`/`M{}`/`P{}`, tetapi hanya `L`, `N`, dan `P` yang terdefinisi di preamble sehingga `M` memicu galat `Illegal pream-token`. Definisinya disamakan dengan yang diharapkan konverter: `>{\centering\arraybackslash}p{#1}`.

2026-10-20 | Empat rujukan metode inti ditambahkan lengkap ke `daftar-pustaka.bib`: Deerwester et al. (1990), Lavrenko & Croft (2001), Xu & Croft (1996), Robertson & Zaragoza (2009) | Keempatnya dirujuk Bab 4 tetapi belum ada di daftar pustaka. Ditulis lengkap, bukan sebagai placeholder `[LENGKAPI]`.

2026-10-20 | Subbab terakhir Bab 4 dinamai "Perancangan Pengujian", bukan "Skenario Pengujian" seperti pada kerangka draft | Isinya bukan sekadar daftar skenario melainkan rancangan empat pengujian beserta matriks hasil kosong, mengikuti pola skripsi pembanding. Nama lama dapat dikembalikan tanpa mengubah isi.
