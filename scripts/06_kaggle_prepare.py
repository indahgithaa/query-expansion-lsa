"""Menyiapkan folder unggah Kaggle: dataset prompt dan notebook generasi Qwen (Subbab 4.10).

Pemakaian: python scripts/06_kaggle_prepare.py <username_kaggle> <smoke|full>
Hasil:
- kaggle/input/   : prompts.jsonl, smoke_pids.json, dataset-metadata.json  (dataset privat)
- notebooks/generate_qwen.ipynb dan kaggle/kernel/{generate_qwen.ipynb, kernel-metadata.json}
"""
import csv
import json
import sys

from qe_lsa import config

MODEL_ID = "RedHatAI/Qwen3.8-27B-INT4"  # kuantisasi W4A16 dari Qwen/Qwen3.8-27B; lihat DECISIONS.md
DATASET_SLUG = "qe-lsa-prompts"
KERNEL_SLUG = "qe-lsa-qwen-generation"

CELLS = [
    ("markdown", """# Generasi jawaban Qwen3.8-27B (Subbab 4.10)
Prompt sudah disusun lokal (instruksi, konteks [1]–[5], pertanyaan). Notebook ini hanya membangkitkan jawaban:
*greedy decoding*, maksimum 1.024 token, mode berpikir dinonaktifkan. Prompt identik antarmetode dibangkitkan sekali."""),
    ("code", """MODE = "{MODE}"            # "smoke" = 5 unit x 3 metode, "full" = seluruh prompt unik
MODEL_ID = "{MODEL_ID}"
MAX_NEW_TOKENS = {MAX_NEW_TOKENS}
BATCH = 512
TIME_BUDGET_S = 11 * 3600    # berhenti memulai batch baru setelah 11 jam (batas sesi 12 jam)"""),
    ("code", """!pip install -q -U vllm 2>&1 | tail -2"""),
    ("code", """import glob, json, os, platform, subprocess, time
import torch, vllm
from huggingface_hub import model_info
t_start = time.time()
env = {"vllm": vllm.__version__, "torch": torch.__version__, "python": platform.python_version(),
       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
       "model_id": MODEL_ID, "model_revision": model_info(MODEL_ID).sha, "mode": MODE}
print(json.dumps(env, indent=2))
json.dump(env, open("/kaggle/working/environment.json", "w"), indent=2)"""),
    ("code", """def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return hits
prompts = [json.loads(l) for l in open(find("prompts.jsonl")[0], encoding="utf-8")]
if MODE == "smoke":
    keep = set(json.load(open(find("smoke_pids.json")[0])))
    prompts = [p for p in prompts if p["pid"] in keep]
done = set()
for path in find("answers.jsonl"):          # keluaran sesi sebelumnya (bila dilampirkan) untuk melanjutkan
    done |= {json.loads(l)["pid"] for l in open(path, encoding="utf-8")}
out_path = "/kaggle/working/answers.jsonl"
if os.path.exists(out_path):
    done |= {json.loads(l)["pid"] for l in open(out_path, encoding="utf-8")}
todo = sorted((p for p in prompts if p["pid"] not in done), key=lambda p: p["prompt"])
print(f"prompt: {len(prompts)}, sudah selesai: {len(done & {p['pid'] for p in prompts})}, sisa: {len(todo)}")"""),
    ("code", """from vllm import LLM, SamplingParams
llm = LLM(model=MODEL_ID, tensor_parallel_size=torch.cuda.device_count(), dtype="float16",
          max_model_len=4096, gpu_memory_utilization=0.92, seed=0, enable_prefix_caching=True,
          kv_cache_dtype="float16",            # T4 (SM75) tidak mendukung KV cache FP8 bawaan model
          max_num_batched_tokens=2048, max_num_seqs=32, enforce_eager=True,   # muat di 2x15 GB
          disable_custom_all_reduce=True, limit_mm_per_prompt={"image": 0, "video": 0})
sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=0)   # greedy
CHAT_KW = {"enable_thinking": False}                                       # mode berpikir dimatikan"""),
    ("code", """with open(out_path, "a", encoding="utf-8") as f:
    for i in range(0, len(todo), BATCH):
        if time.time() - t_start > TIME_BUDGET_S:
            print("batas waktu sesi tercapai; lanjutkan di sesi berikutnya"); break
        batch = todo[i:i + BATCH]
        t0 = time.time()
        outs = llm.chat([[{"role": "user", "content": p["prompt"]}] for p in batch], sp,
                        chat_template_kwargs=CHAT_KW, use_tqdm=False)
        dt = time.time() - t0
        for p, o in zip(batch, outs):
            c = o.outputs[0]
            f.write(json.dumps({"pid": p["pid"], "answer": c.text, "n_output_tokens": len(c.token_ids),
                                "n_prompt_tokens": len(o.prompt_token_ids), "finish_reason": c.finish_reason,
                                "batch_seconds": round(dt, 3), "batch_size": len(batch),
                                "seconds_per_prompt": round(dt / len(batch), 3)}, ensure_ascii=False) + "\\n")
        f.flush()
        print(f"{min(i + BATCH, len(todo))}/{len(todo)}  {dt:.0f}s  total {(time.time() - t_start) / 60:.0f} menit", flush=True)"""),
    ("code", """rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
think = sum("<think>" in r["answer"] for r in rows)
trunc = sum(r["finish_reason"] == "length" for r in rows)
print(f"jawaban: {len(rows)}, memuat <think>: {think}, terpotong 1.024 token: {trunc}")
for r in rows[:5]:
    print("-" * 80); print(r["pid"], r["n_output_tokens"], "token"); print(r["answer"][:800])"""),
]


def notebook(mode: str) -> dict:
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        src = src.replace("{MODE}", mode).replace("{MODEL_ID}", MODEL_ID) \
                 .replace("{MAX_NEW_TOKENS}", str(config.GEN_MAX_NEW_TOKENS))
        cell = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                         "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def smoke_pids() -> list[str]:
    """5 unit pertama (qid terkecil) pengujian seed 1 skenario TPL, ketiga metode."""
    rows = list(csv.DictReader((config.RESULTS_DIR / "05_prompt_map.csv").open(encoding="utf-8")))
    sel = [r for r in rows if r["seed"] == "1" and r["scenario"] == "TPL"]
    qids = sorted({int(r["qid"]) for r in sel})[:5]
    return sorted({r["pid"] for r in sel if int(r["qid"]) in qids})


def main(user: str, mode: str):
    inp = config.ROOT / "kaggle" / "input"
    (inp / "smoke_pids.json").write_text(json.dumps(smoke_pids()), encoding="utf-8")
    (inp / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}, indent=2))
    nb = json.dumps(notebook(mode), indent=1, ensure_ascii=False)
    (config.ROOT / "notebooks").mkdir(exist_ok=True)
    (config.ROOT / "notebooks" / "generate_qwen.ipynb").write_text(nb, encoding="utf-8")
    kdir = config.ROOT / "kaggle" / "kernel"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "generate_qwen.ipynb").write_text(nb, encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "generate_qwen.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [f"{user}/{DATASET_SLUG}"], "kernel_sources": [],
        "competition_sources": []}, indent=2))
    print(f"siap: mode={mode}, smoke pid={len(smoke_pids())}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
