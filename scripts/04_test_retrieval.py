"""Retrieval pada DATA PENGUJIAN seed 1–10, dua skenario, tiga metode, dengan konfigurasi terkunci.

Sebelum berjalan, hash kode dan parafrasa dicocokkan dengan config/locked_config.json; bila berbeda, skrip berhenti.
Keluaran: results/04_test_runs.jsonl — satu baris per (seed, skenario, qid, metode) berisi peringkat chunk acuan
dan 20 chunk teratas. Metrik dihitung terpisah pada Fase 3.
"""
import hashlib
import json
import time

from qe_lsa import config
from qe_lsa.data import load_corpus
from qe_lsa.pipeline import METHODS, prepare_seed, retrieve
from qe_lsa.split import load_paraphrases, question_text

TOP = 20


def verify_lock():
    lock = json.loads((config.ROOT / "config" / "locked_config.json").read_text(encoding="utf-8"))
    for rel, digest in lock["sha256"].items():
        actual = hashlib.sha256((config.ROOT / rel).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"Berkas {rel} berubah sejak konfigurasi dikunci. Hentikan.")
    return lock


def main():
    verify_lock()
    t0 = time.time()
    corpus = load_corpus()
    para = load_paraphrases()
    out = config.RESULTS_DIR / "04_test_runs.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for seed in config.SEEDS:
            ctx = prepare_seed(corpus, seed, index_parts=("test",))
            units = sorted((q for q in corpus.qas if q.doc_id in ctx.parts["test"]), key=lambda q: q.qid)
            for qa in units:
                for scen in ("TPL", "PARA"):
                    question = question_text(qa, seed, "test", scen, para)
                    for m in METHODS:
                        ranked, scores = retrieve(ctx, qa.doc_id, question, m)
                        f.write(json.dumps({
                            "seed": seed, "scenario": scen, "qid": qa.qid, "qtype": qa.qtype, "doc_id": qa.doc_id,
                            "method": m, "question": question, "ref_cid": qa.ref_cid,
                            "rank": ranked.index(qa.ref_cid) + 1,
                            "top": ranked[:TOP], "top_scores": [round(float(s), 6) for s in scores[:TOP]],
                        }, ensure_ascii=False) + "\n")
                        n += 1
            print(f"seed {seed}: {len(units)} QA, {time.time() - t0:.0f}s", flush=True)
    print(f"{n} baris -> {out}")


if __name__ == "__main__":
    main()
