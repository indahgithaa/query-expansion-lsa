# Log Keputusan

| Tanggal | Keputusan | Alasan |
|---|---|---|
| 2026-11-11 | Jalur retrieval ditulis ulang dari Bab 4 dalam paket `qe_lsa/` | Repo belum berisi implementasi; Bab 4 adalah spesifikasi |
| 2026-11-11 | Pengacakan dokumen memakai `random.Random(seed*1000003+attempt).shuffle` | Satu-satunya kandidat yang mereproduksi Tabel 4.14 (`results/fase0_reproduce_4_12_eq29_set.txt`) |
| 2026-11-11 | Bootstrap memakai `np.random.default_rng(20261019).integers(0, n, n)` per strata, strata diurutkan | Mereproduksi tiga *draw* Tabel 4.51 secara persis |
| 2026-11-11 | BM25 diimplementasikan sendiri dengan semantik `rank_bm25.BM25Okapi` (rata-rata idf termasuk nilai negatif) | Memerlukan bobot per term (Persamaan 2.3); angka 4.12.4–4.12.5 cocok |
| 2026-11-11 | Wilcoxon: `scipy.stats.wilcoxon(zero_method="wilcox", alternative="two-sided")`, metode bawaan | Mereproduksi Tabel 4.49 |
| 2026-11-11 | Angka murni = `\d+` *full match*; pola #Uxxxx = tepat 4 digit heksadesimal | Tafsiran Subbab 4.4.1 dan 4.6.1; jumlah 1.464 cocok Tabel 4.25 |
| 2026-11-11 | Lingkungan: Python 3.14.7 dengan versi library di `requirements.txt` | Lingkungan lokal yang sebenarnya; Bab 3.8 perlu disesuaikan (Fase 5) |
| (tertunda) | Centroid LSA: himpunan (Pers. 2.9) vs barisan token (Gbr. 4.18) | Lihat `reports/07_audit_bab4.md` A1 |
| (tertunda) | Daftar parafrasa validasi/pengujian | Lihat audit B1 |
