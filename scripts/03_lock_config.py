"""Mengunci konfigurasi sebelum data pengujian dipakai. Menyimpan hash kode, parafrasa, dan hasil seleksi dev."""
import hashlib
import json
from datetime import date

from qe_lsa import config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


sel = json.loads((config.RESULTS_DIR / "02_dev_selection.json").read_text(encoding="utf-8"))
assert sel["selected"]["LSA-QE"]["config"] == "k=300,n=10,gamma=2.0"
assert sel["selected"]["RM3"]["config"] == "fbDocs=3,fbTerms=20,lambda=0.5"
lock = {
    "locked_at": date.today().isoformat(),
    "source": "results/02_dev_selection.json (dev seed 1-3, 1.026 unit, sama dengan Tabel 4.2)",
    "bm25": {"k1": config.BM25_K1, "b": config.BM25_B, "epsilon": config.BM25_EPSILON},
    "LSA-QE": {"k": config.LSA_K, "n_terms": config.LSAQE_N_TERMS, "gamma": config.LSAQE_GAMMA,
               "min_df": config.LSA_MIN_DF, "random_state": config.LSA_RANDOM_STATE, "centroid": "token (Gambar 4.18)"},
    "RM3": {"fb_docs": config.RM3_FB_DOCS, "fb_terms": config.RM3_FB_TERMS, "lambda": config.RM3_LAMBDA},
    "test": {"seeds": config.SEEDS, "scenarios": ["TPL", "PARA"], "pairs": config.PAIRS,
             "bootstrap": {"reps": config.BOOTSTRAP_REPS, "seed": config.BOOTSTRAP_SEED, "strata": "seed x skenario"},
             "multiple_testing": "Holm (3 pasangan)", "rule": "p_Holm < 0,05 dan CI 95% tidak memuat 0"},
    "sha256": {str(p.relative_to(config.ROOT)).replace("\\", "/"): sha(p)
               for p in sorted(config.ROOT.glob("qe_lsa/*.py")) + [config.PARAPHRASE_PATH]},
}
out = config.ROOT / "config" / "locked_config.json"
out.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
print(out.read_text(encoding="utf-8"))
