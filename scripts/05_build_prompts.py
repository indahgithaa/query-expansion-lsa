"""Menyusun prompt generasi dari hasil retrieval data uji (5 chunk teratas).

Prompt identik (metode berbeda tetapi 5 chunk teratas sama dan urutannya sama) hanya dibangkitkan sekali;
karena dekode greedy, jawabannya dipakai bersama. Keluaran:
- kaggle/input/prompts.jsonl  : prompt unik {pid, prompt, question, reference}
- results/05_prompt_map.csv    : pemetaan (seed, skenario, qid, metode) -> pid
"""
import csv
import json

from qe_lsa import config
from qe_lsa.data import load_corpus
from qe_lsa.generation import build_prompt, prompt_id


def main():
    corpus = load_corpus()
    text_of = {c.cid: c.text for cs in corpus.chunks.values() for c in cs}
    answer_of = {q.qid: q.answer for q in corpus.qas}
    unique, rows = {}, []
    with (config.RESULTS_DIR / "04_test_runs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            top5 = r["top"][:config.GEN_TOP_CHUNKS]
            prompt = build_prompt(r["question"], [text_of[c] for c in top5])
            pid = prompt_id(prompt)
            unique.setdefault(pid, {"pid": pid, "prompt": prompt, "question": r["question"],
                                    "reference": answer_of[r["qid"]], "contexts": top5})
            rows.append({k: r[k] for k in ("seed", "scenario", "qid", "qtype", "method")} | {"pid": pid})
    out_dir = config.ROOT / "kaggle" / "input"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for p in sorted(unique.values(), key=lambda p: p["pid"]):
            f.write(json.dumps({k: p[k] for k in ("pid", "prompt", "question", "reference")}, ensure_ascii=False) + "\n")
    with (config.RESULTS_DIR / "05_prompt_map.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    chars = [len(p["prompt"]) for p in unique.values()]
    print(f"unit x metode: {len(rows)}, prompt unik: {len(unique)}, "
          f"rata-rata {sum(chars) / len(chars):.0f} karakter, maks {max(chars)} karakter")


if __name__ == "__main__":
    main()
