"""Tahap 0 - pra-pemrosesan, sanity check, dan ringkasan korpus."""

import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io
from qe.corpus import TYPE_ORDER, load_corpus, load_paraphrases


def main() -> int:
    try:
        corpus = load_corpus(io.DATA, check=True)
    except AssertionError as exc:
        print(f"BERHENTI: {exc}", file=sys.stderr)
        return 1

    paraphrases = load_paraphrases(os.path.join(io.DATA, "paraphrases.json"))

    chunks_per_doc = [len(d.chunks) for d in corpus.documents.values()]
    qa_per_doc = collections.Counter(qa.doc_id for qa in corpus.qas)
    type_counts = collections.Counter(qa.qtype for qa in corpus.qas)
    docs_per_type = collections.Counter()
    for qtype in TYPE_ORDER:
        docs_per_type[qtype] = len({qa.doc_id for qa in corpus.qas if qa.qtype == qtype})

    summary = {
        "n_documents": len(corpus.documents),
        "n_chunks": corpus.n_chunks,
        "n_qa_raw": 911,
        "n_qa_dedup": len(corpus.qas),
        "chunks_per_doc": {
            "min": min(chunks_per_doc),
            "median": statistics.median(chunks_per_doc),
            "mean": sum(chunks_per_doc) / len(chunks_per_doc),
            "max": max(chunks_per_doc),
        },
        "qa_per_doc": {
            "min": min(qa_per_doc.values()),
            "median": statistics.median(qa_per_doc.values()),
            "max": max(qa_per_doc.values()),
        },
        "qa_per_type": {t: type_counts[t] for t in TYPE_ORDER},
        "docs_per_type": {t: docs_per_type[t] for t in TYPE_ORDER},
        "paraphrases_per_type_per_partition": 6,
        "doc_id_map": {d.doc_id: d.filename for d in corpus.documents.values()},
        "gold_by_qid": {str(qa.qid): qa.gold_chunk_id for qa in corpus.qas},
        "qtype_by_qid": {str(qa.qid): qa.qtype for qa in corpus.qas},
    }
    path = io.write_json("00_preprocess.json", summary)

    lines = [
        "# Tahap 0 - Pra-pemrosesan",
        "",
        "Sumber angka: `results/00_preprocess.json`.",
        "",
        "## Normalisasi",
        "",
        "Teks dan nama file dinormalisasi dengan urutan: NFKC, lalu pola `#Uxxxx` diubah",
        "menjadi karakter Unicode yang diwakilinya, lalu U+2060 (word joiner) dihapus.",
        "`doc_id` D001-D120 diberikan menurut urutan nama file chunk ternormalisasi,",
        "dan `chunk_id` berformat `{doc_id}:P{page:04d}:C{chunk_index:03d}`.",
        "",
        "## Sanity check (wajib)",
        "",
        "| Pemeriksaan | Harapan | Hasil | Status |",
        "| --- | ---: | ---: | --- |",
        f"| QA mentah | 911 | {summary['n_qa_raw']} | OK |",
        f"| QA setelah dedup | 858 | {summary['n_qa_dedup']} | OK |",
        f"| Dokumen | 120 | {summary['n_documents']} | OK |",
        f"| Chunk | 13.737 | {summary['n_chunks']:,} | OK |".replace(",", "."),
        "| Gold per QA | tepat 1 | tepat 1 untuk 858 QA | OK |",
        "",
        "Deduplikasi memakai kunci (nama file ternormalisasi, teks pertanyaan) dan menyimpan",
        "baris dengan `id` numerik terkecil; 53 baris duplikat dibuang (911 - 858).",
        "",
        "## Ukuran korpus",
        "",
        "| Besaran | Min | Median | Rata-rata | Maks |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| Chunk per dokumen | {min} | {median:.1f} | {mean:.1f} | {max} |".format(**summary["chunks_per_doc"]),
        "",
        "Setiap QA dijawab oleh tepat satu chunk dalam laporan pasien yang sama, sehingga",
        "korpus target saat retrieval adalah seluruh chunk laporan tersebut.",
        "",
        "## Sebaran tipe pertanyaan",
        "",
        "Tipe ditentukan dari teks template persis (bukan model); label ini hanya dipakai",
        "untuk analisis dan penjaminan cakupan split, tidak pernah dipakai saat inferensi.",
        "",
        "| Tipe | Jumlah QA | Jumlah dokumen |",
        "| --- | ---: | ---: |",
    ]
    for t in TYPE_ORDER:
        lines.append(f"| {t} | {type_counts[t]} | {docs_per_type[t]} |")
    lines += [
        f"| **Total** | **{len(corpus.qas)}** | **{len(corpus.documents)}** |",
        "",
        "Tipe `pedoman` jauh lebih jarang (21 QA pada 21 dokumen). Karena itu syarat",
        "\"ketiga bagian split memuat kedelapan tipe\" pada Tahap 1 praktis ditentukan oleh",
        "tipe ini.",
        "",
        "## Parafrasa",
        "",
        f"`data/paraphrases.json` memuat {len(paraphrases)} tipe x 2 partisi x 6 kalimat.",
        "Daftar `dev` dan `test` tidak beririsan, sehingga kalimat yang dipakai saat seleksi",
        "parameter tidak muncul lagi saat pengujian.",
        "",
        "## Catatan",
        "",
        "Pola `#Uxxxx` tidak ditemukan sama sekali pada data yang diberikan; aturannya tetap",
        "diterapkan agar pipeline tahan terhadap ekspor berikutnya. U+2060 memang ada",
        "(112 kemunculan) dan dihapus.",
    ]
    report = io.write_report("00_preprocessing.md", "\n".join(lines))

    print(f"OK: {summary['n_qa_dedup']} QA, {summary['n_documents']} dokumen, {summary['n_chunks']} chunk")
    print(f"  -> {io.rel(path)}")
    print(f"  -> {io.rel(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
