"""Menyiapkan evaluasi Faithfulness (RAGAS 0.4.3) di Kaggle dengan Qwen yang sama sebagai model juri.

Pemakaian: python scripts/08_kaggle_faithfulness.py <username_kaggle> <smoke|full>
Hasil:
- kaggle/faith/input/  : faith_inputs.jsonl, smoke_pids.json, dataset-metadata.json  (dataset privat)
- notebooks/faithfulness_ragas.ipynb dan kaggle/faith/kernel/{faithfulness_ragas.ipynb, kernel-metadata.json}

Satu baris masukan per prompt unik (pid): pertanyaan, jawaban Qwen, dan lima teks chunk konteks persis seperti
yang dimasukkan ke prompt. Faithfulness bergantung hanya pada (pertanyaan, jawaban, konteks), sehingga nilainya
dipakai bersama oleh semua unit x metode yang memakai pid yang sama.
"""
import csv
import json
import sys

from qe_lsa import config
from qe_lsa.data import load_corpus
from qe_lsa.generation import build_prompt, prompt_id

MODEL_ID = "RedHatAI/Qwen3.8-27B-INT4"
MODEL_REVISION = "e45c27e8f463817db788d141a73a3c235a0a7c0c"   # sama dengan revisi saat generasi
DATASET_SLUG = "qe-lsa-faithfulness-inputs"
KERNEL_SLUG = "qe-lsa-faithfulness"

CELLS = [
    ("markdown", """# Faithfulness RAGAS 0.4.3
Model juri: Qwen3.8-27B INT4 yang sama dengan generator, dilayani vLLM (OpenAI API lokal), temperatur 0,
mode berpikir dimatikan. Prompt RAGAS bawaan tidak diubah. Langkah per jawaban sama dengan
`Faithfulness.ascore`: (1) pecah jawaban menjadi pernyataan, (2) NLI setiap pernyataan terhadap konteks
yang digabung dengan baris baru, (3) skor = pernyataan didukung / seluruh pernyataan."""),
    ("code", """MODE = "{MODE}"
MODEL_ID = "{MODEL_ID}"
MODEL_REVISION = "{MODEL_REVISION}"
CONCURRENCY = 32            # KV cache T4x2 ~75 ribu token; ~25-30 permintaan ~3 ribu token muat bersamaan
PRINT_EVERY = 256
MAX_TOKENS = 2048
TIME_BUDGET_S = 11 * 3600"""),
    ("code", """!pip install -q -U vllm 2>&1 | tail -2
!pip install -q ragas==0.4.3 langchain-community==0.4.1 2>&1 | tail -2   # 0.4.2 menghapus chat_models.vertexai yang diimpor ragas 0.4.3"""),
    ("code", """import asyncio, glob, json, math, os, platform, subprocess, time, urllib.request
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
from importlib.metadata import version
import torch
t_start = time.time()
env = {k: version(k) for k in ("vllm", "torch", "ragas", "langchain-community", "instructor", "openai", "pydantic")}
env |= {"python": platform.python_version(), "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "mode": MODE}
print(json.dumps(env, indent=2))
json.dump(env, open("/kaggle/working/environment.json", "w"), indent=2)"""),
    ("code", """server_log = open("/kaggle/working/vllm_server.log", "w")
server = subprocess.Popen(
    ["vllm", "serve", MODEL_ID, "--revision", MODEL_REVISION, "--served-model-name", "qwen", "--port", "8000",
     "--tensor-parallel-size", str(torch.cuda.device_count()), "--dtype", "float16", "--max-model-len", "8192",
     "--gpu-memory-utilization", "0.92", "--seed", "0", "--enable-prefix-caching", "--kv-cache-dtype", "float16",
     "--max-num-batched-tokens", "2048", "--max-num-seqs", "32", "--enforce-eager", "--disable-custom-all-reduce",
     "--limit-mm-per-prompt", '{"image": 0, "video": 0}'],
    stdout=server_log, stderr=subprocess.STDOUT)
for _ in range(240):
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=5); break
    except Exception:
        if server.poll() is not None:
            raise RuntimeError(open("/kaggle/working/vllm_server.log").read()[-5000:])
        time.sleep(10)
print(f"server siap setelah {(time.time() - t_start) / 60:.1f} menit")"""),
    ("code", """def find(name):
    return sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
items = [json.loads(l) for l in open(find("faith_inputs.jsonl")[0], encoding="utf-8")]
if MODE == "smoke":
    keep = set(json.load(open(find("smoke_pids.json")[0])))
    items = [r for r in items if r["pid"] in keep]
out_path = "/kaggle/working/faithfulness.jsonl"
done = set()
for path in find("faithfulness.jsonl") + ([out_path] if os.path.exists(out_path) else []):
    done |= {json.loads(l)["pid"] for l in open(path, encoding="utf-8")}
todo = sorted((r for r in items if r["pid"] not in done), key=lambda r: r["pid"])
print(f"jawaban: {len(items)}, sudah: {len(done & {r['pid'] for r in items})}, sisa: {len(todo)}")"""),
    ("code", """from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY", timeout=3600, max_retries=0)
llm = llm_factory("qwen", provider="openai", client=client, temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS,
                  seed=0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
metric = Faithfulness(llm=llm)
sem = asyncio.Semaphore(CONCURRENCY)

async def judge(r):
    # urutan langkah identik dengan Faithfulness.ascore; detail pernyataan disimpan untuk audit
    async with sem:
        if time.time() - t_start > TIME_BUDGET_S:
            return None                   # tidak memulai penilaian baru setelah batas waktu sesi
        t0 = time.time()
        rec = {"pid": r["pid"], "faithfulness": None, "n_statements": 0, "n_supported": 0,
               "statements": [], "verdicts": [], "error": None}
        try:
            statements = await metric._create_statements(r["question"], r["answer"])
            rec["statements"] = statements
            if statements:
                v = await metric._create_verdicts(statements, "\\n".join(r["contexts"]))
                score = metric._compute_score(v)
                rec["verdicts"] = [{"statement": s.statement, "verdict": s.verdict, "reason": s.reason}
                                   for s in v.statements]
                rec["n_statements"] = len(v.statements)
                rec["n_supported"] = sum(1 for s in v.statements if s.verdict)
                rec["faithfulness"] = None if math.isnan(score) else score
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:1000]
        rec["seconds"] = round(time.time() - t0, 2)
        return rec

async def run():
    n = err = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for fut in asyncio.as_completed([judge(r) for r in todo]):   # antrean mengalir
            rec = await fut
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n"); f.flush()
            n += 1; err += rec["error"] is not None
            if n % PRINT_EVERY == 0:
                print(f"{n}/{len(todo)}  galat {err}  total {(time.time() - t_start) / 60:.0f} menit", flush=True)
    print(f"selesai sesi ini: {n}/{len(todo)}, galat {err}; sisa {len(todo) - n}", flush=True)

await run()"""),
    ("code", """rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
vals = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
print(f"baris: {len(rows)}, bernilai: {len(vals)}, tanpa pernyataan: {sum(r['n_statements'] == 0 and r['error'] is None for r in rows)}, "
      f"galat: {sum(r['error'] is not None for r in rows)}, rata-rata: {sum(vals) / max(len(vals), 1):.4f}")
for r in rows[:3]:
    print("-" * 80); print(json.dumps(r, ensure_ascii=False, indent=1)[:2500])
for r in [r for r in rows if r["error"]][:3]:
    print("GALAT", r["pid"], r["error"][:500])
server.terminate()"""),
]


def notebook(mode: str) -> dict:
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        src = src.replace("{MODE}", mode).replace("{MODEL_ID}", MODEL_ID).replace("{MODEL_REVISION}", MODEL_REVISION)
        cell = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                         "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def build_inputs() -> list[dict]:
    corpus = load_corpus()
    text_of = {c.cid: c.text for cs in corpus.chunks.values() for c in cs}
    answers = {}
    for line in (config.RESULTS_DIR / "generation" / "answers.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        answers[r["pid"]] = r["answer"]
    seen = {}
    for line in (config.RESULTS_DIR / "04_test_runs.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        ctx = [text_of[c].strip() for c in r["top"][:config.GEN_TOP_CHUNKS]]
        pid = prompt_id(build_prompt(r["question"], [text_of[c] for c in r["top"][:config.GEN_TOP_CHUNKS]]))
        seen.setdefault(pid, {"pid": pid, "question": r["question"].strip(), "answer": answers[pid], "contexts": ctx})
    assert set(seen) == set(answers), "pid masukan Faithfulness tidak cocok dengan answers.jsonl"
    return sorted(seen.values(), key=lambda r: r["pid"])


def smoke_pids() -> list[str]:
    """Prompt unik dari 5 unit pertama (qid terkecil) seed 1 skenario TPL, sama dengan uji generasi."""
    rows = list(csv.DictReader((config.RESULTS_DIR / "05_prompt_map.csv").open(encoding="utf-8")))
    sel = [r for r in rows if r["seed"] == "1" and r["scenario"] == "TPL"]
    qids = sorted({int(r["qid"]) for r in sel})[:5]
    return sorted({r["pid"] for r in sel if int(r["qid"]) in qids})


def main(user: str, mode: str):
    base = config.ROOT / "kaggle" / "faith"
    inp, kdir = base / "input", base / "kernel"
    inp.mkdir(parents=True, exist_ok=True)
    kdir.mkdir(parents=True, exist_ok=True)
    items = build_inputs()
    with (inp / "faith_inputs.jsonl").open("w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (inp / "smoke_pids.json").write_text(json.dumps(smoke_pids()), encoding="utf-8")
    (inp / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}, indent=2))
    nb = json.dumps(notebook(mode), indent=1, ensure_ascii=False)
    (config.ROOT / "notebooks" / "faithfulness_ragas.ipynb").write_text(nb, encoding="utf-8")
    (kdir / "faithfulness_ragas.ipynb").write_text(nb, encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "faithfulness_ragas.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [f"{user}/{DATASET_SLUG}"], "kernel_sources": [],
        "competition_sources": []}, indent=2))
    print(f"siap: mode={mode}, masukan={len(items)}, smoke pid={len(smoke_pids())}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
