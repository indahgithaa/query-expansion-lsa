"""Tahap 1 - split per dokumen dan skenario query."""

import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io
from qe.corpus import TYPE_ORDER, load_corpus, load_paraphrases
from qe.scenarios import SCENARIOS, paraphrase_index
from qe.splits import make_split, types_by_doc

SEEDS = tuple(range(1, 11))
DEV_SEEDS = (1, 2, 3)


def main() -> int:
    corpus = load_corpus(io.DATA)
    paraphrases = load_paraphrases(os.path.join(io.DATA, "paraphrases.json"))
    tbd = types_by_doc(corpus)

    splits = {}
    rows = []
    for seed in SEEDS:
        s = make_split(corpus, seed, tbd)
        parts = {"collection": s.collection, "dev": s.dev, "test": s.test}
        n_qa = {name: len(corpus.qas_in(docs)) for name, docs in parts.items()}
        coverage = {
            name: sorted(set().union(*(tbd[d] for d in docs))) for name, docs in parts.items()
        }
        splits[str(seed)] = {
            "attempt": s.attempt,
            "collection": list(s.collection),
            "dev": list(s.dev),
            "test": list(s.test),
            "n_qa": n_qa,
            "n_chunks": {name: sum(len(corpus.documents[d].chunks) for d in docs)
                         for name, docs in parts.items()},
            "all_types_covered": {name: sorted(coverage[name]) == sorted(TYPE_ORDER)
                                  for name in parts},
        }
        rows.append({
            "seed": seed, "attempt": s.attempt,
            "qa_collection": n_qa["collection"], "qa_dev": n_qa["dev"], "qa_test": n_qa["test"],
            "chunks_collection": splits[str(seed)]["n_chunks"]["collection"],
            "chunks_dev": splits[str(seed)]["n_chunks"]["dev"],
            "chunks_test": splits[str(seed)]["n_chunks"]["test"],
        })

    # Sebaran indeks parafrasa (pemeriksaan keseragaman, bukan parameter).
    para_counts = {}
    for partition, seeds in (("dev", DEV_SEEDS), ("test", SEEDS)):
        c = collections.Counter()
        for seed in seeds:
            docs = splits[str(seed)][partition]
            for qa in corpus.qas_in(docs):
                c[paraphrase_index(seed, qa.qid)] += 1
        para_counts[partition] = {str(i): c[i] for i in range(6)}

    payload = {
        "seeds": list(SEEDS),
        "dev_seeds": list(DEV_SEEDS),
        "scenarios": list(SCENARIOS),
        "sizes": {"collection": 72, "dev": 24, "test": 24},
        "splits": splits,
        "paraphrase_index_counts": para_counts,
    }
    path = io.write_json("01_splits.json", payload)
    csv_path = io.write_csv("01_splits_summary.csv", rows)

    dev_rows = [r for r in rows if r["seed"] in DEV_SEEDS]
    lines = [
        "# Tahap 1 - Split dan skenario",
        "",
        "Sumber angka: `results/01_splits.json`, `results/01_splits_summary.csv`.",
        "",
        "## Prosedur split",
        "",
        "Split dilakukan **per dokumen**, bukan per QA, supaya tidak ada chunk dari satu",
        "laporan yang muncul di dua bagian sekaligus. Untuk tiap seed, daftar `doc_id`",
        "terurut diacak dengan `random.Random(seed * 1_000_003 + attempt).shuffle`, lalu",
        "dipotong 72 koleksi / 24 dev / 24 test. Percobaan dinaikkan sampai ketiga bagian",
        "memuat kedelapan tipe pertanyaan.",
        "",
        "Bagian **koleksi** hanya dipakai untuk melatih LSA (Tahap 3). Bagian **dev**",
        "dipakai untuk memilih parameter (seed 1-3), bagian **test** untuk pengujian akhir",
        "(seed 1-10). Dokumen dev dan test tidak pernah ikut melatih LSA.",
        "",
        "## Hasil per seed",
        "",
        "| Seed | attempt | QA koleksi | QA dev | QA test | Chunk koleksi | Chunk dev | Chunk test |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {seed} | {attempt} | {qa_collection} | {qa_dev} | {qa_test} | "
            "{chunks_collection} | {chunks_dev} | {chunks_test} |".format(**r)
        )
    mean_dev = sum(r["qa_dev"] for r in dev_rows) / len(dev_rows)
    lines += [
        "",
        f"Semua seed menemukan split yang sah pada attempt = {max(r['attempt'] for r in rows)}",
        "atau lebih kecil, dan ketiga bagian setiap seed memuat kedelapan tipe",
        "(`all_types_covered` bernilai true di seluruh entri).",
        "",
        "## Skenario",
        "",
        "- **TPL** - pertanyaan template asli, delapan kalimat tetap.",
        "- **PARA** - satu dari enam parafrasa milik tipe pertanyaan tersebut. Indeksnya",
        "  `int.from_bytes(sha256(f\"{seed}|{qid}\").digest()[:8], \"big\") % 6`, sehingga",
        "  penugasan deterministik, bebas dari urutan pemrosesan, dan berubah antar seed.",
        "  Partisi dev memakai daftar parafrasa `dev`, partisi test memakai daftar `test`.",
        "",
        "Unit evaluasi adalah (seed, skenario, qid). Untuk seleksi dev terdapat",
        f"{int(mean_dev)} QA rata-rata per seed x 3 seed x 2 skenario ~= "
        f"{sum(r['qa_dev'] for r in dev_rows) * 2} unit.",
        "",
        "## Keseragaman indeks parafrasa",
        "",
        "| Indeks | 0 | 1 | 2 | 3 | 4 | 5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| dev (seed 1-3) | " + " | ".join(str(para_counts["dev"][str(i)]) for i in range(6)) + " |",
        "| test (seed 1-10) | " + " | ".join(str(para_counts["test"][str(i)]) for i in range(6)) + " |",
        "",
        "Sebaran mendekati seragam, jadi tidak ada parafrasa yang mendominasi evaluasi.",
    ]
    report = io.write_report("01_split_scenario.md", "\n".join(lines))

    print(f"OK: {len(SEEDS)} seed, attempt maksimum {max(r['attempt'] for r in rows)}")
    for p in (path, csv_path, report):
        print(f"  -> {io.rel(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
