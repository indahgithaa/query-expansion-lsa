"""Parameter sistem sesuai Tabel 4.2 Bab 4. Nilai di sini adalah spesifikasi, bukan hasil penyetelan."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHUNK_DIR = DATA_DIR / "chunks"
QA_PATH = DATA_DIR / "dataset" / "dataset.json"
PARAPHRASE_PATH = ROOT / "config" / "paraphrases.json"
RESULTS_DIR = ROOT / "results"

# BM25 Okapi (Subbab 4.8)
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25

# RM3 (Subbab 4.9) — konfigurasi terkunci menurut Tabel 4.2
RM3_FB_DOCS = 3
RM3_FB_TERMS = 20
RM3_LAMBDA = 0.5

# LSA (Subbab 4.6) dan LSA-QE (Subbab 4.7)
LSA_MIN_DF = 2
LSA_K = 300
LSA_RANDOM_STATE = 0
LSAQE_N_TERMS = 10
LSAQE_GAMMA = 2.0

# Grid pemilihan parameter (Tabel 4.2)
RM3_GRID = {"fb_docs": [3, 5, 10], "fb_terms": [5, 10, 20, 30, 50], "lam": [0.2, 0.35, 0.5, 0.65, 0.8]}
LSAQE_GRID = {"k": [100, 200, 300, 500], "n_terms": [5, 10, 20, 30], "gamma": [0.5, 1.0, 2.0, 4.0]}
TUNING_SEEDS = [1, 2, 3]

# Pembagian data (Subbab 4.5)
N_COLLECTION, N_DEV, N_TEST = 72, 24, 24
SEEDS = list(range(1, 11))
SEED_MULTIPLIER = 1_000_003
N_PARAPHRASES = 6

# Evaluasi (Subbab 4.11)
RECALL_KS = (1, 3, 5, 10, 20)
NDCG_K = 10
FAIL_RANK = 10
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261019
ALPHA = 0.05
PAIRS = (("RM3", "Baseline"), ("LSA-QE", "Baseline"), ("LSA-QE", "RM3"))

# Generasi (Subbab 4.10)
GEN_TOP_CHUNKS = 5
GEN_MAX_NEW_TOKENS = 1024
