"""Skor kualitas jawaban yang dihitung lokal (CPU): BLEU, ROUGE-L, dan BERTScore, satu baris per prompt unik.

Masukan : results/generation/answers.jsonl, kaggle/input/prompts.jsonl (jawaban acuan)
Keluaran: results/09_answer_scores_local.csv, results/09_answer_scores_local.json (versi dan ringkasan)

BERTScore 0.3.13 dengan lang="id" -> bert-base-multilingual-cased lapisan 9 (bawaan pustaka), tanpa idf dan tanpa
rescale baseline. Teks dibiarkan apa adanya karena tokenizer BERT menangani tanda baca sendiri.
"""
import csv
import json
import time
from importlib.metadata import version

import bert_score
import torch

from qe_lsa import config
from qe_lsa.answer_metrics import bleu, rouge_l


def main():
    torch.manual_seed(0)
    answers = {}
    for line in (config.RESULTS_DIR / "generation" / "answers.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        answers[r["pid"]] = r["answer"]
    refs = {}
    for line in (config.ROOT / "kaggle" / "input" / "prompts.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        refs[r["pid"]] = r["reference"]
    pids = sorted(answers)
    assert set(pids) == set(refs)
    cands, golds = [answers[p] for p in pids], [refs[p] for p in pids]

    rows = []
    for p, c, g in zip(pids, cands, golds):
        rl = rouge_l(c, g)
        rows.append({"pid": p, "bleu": bleu(c, g), "rouge_l": rl["f"], "rouge_l_p": rl["p"], "rouge_l_r": rl["r"]})

    t0 = time.time()
    (P, R, F), hash_code = bert_score.score(cands, golds, lang="id", batch_size=16, idf=False,
                                          rescale_with_baseline=False, return_hash=True, verbose=True)
    seconds = time.time() - t0
    for row, p, r, f in zip(rows, P.tolist(), R.tolist(), F.tolist()):
        row |= {"bertscore_f1": f, "bertscore_p": p, "bertscore_r": r}

    tok = bert_score.utils.get_tokenizer(bert_score.utils.lang2model["id"], use_fast=False)
    n_trunc = sum(len(tok.encode(t, add_special_tokens=True)) > tok.model_max_length for t in cands + golds)

    out = config.RESULTS_DIR / "09_answer_scores_local.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {"n_pids": len(rows), "bertscore_hash": hash_code, "bertscore_seconds": round(seconds, 1),
            "n_texts_truncated_512": n_trunc,
            "versions": {k: version(k) for k in ("bert-score", "sacrebleu", "torch", "transformers")}}
    (config.RESULTS_DIR / "09_answer_scores_local.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
