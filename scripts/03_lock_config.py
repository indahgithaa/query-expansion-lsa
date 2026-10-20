"""Tahap 4/5 - kunci config sebelum test dibuka.

Menulis `configs/locked_config.json` dari konfigurasi terpilih di
`results/02_dev_selection.json`, lalu mencetak SHA-256-nya untuk dicatat di
`reports/DECISIONS.md`. Jalankan hanya setelah pengguna menyetujui konfigurasi.
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io
from qe.rm3 import LAMBDA, N_FEEDBACK, N_TERMS

LOCKED_PATH = os.path.join(io.CONFIGS, "locked_config.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--override", action="append", default=[],
                   help="METODE:k,n,gamma untuk memakai config selain pilihan dev, "
                        "mis. --override 'RM3+LSA:300,10,2'")
    p.add_argument("--force", action="store_true",
                   help="tulis ulang config yang sudah terkunci")
    return p.parse_args()


def main():
    args = parse_args()
    if os.path.exists(LOCKED_PATH) and not args.force:
        print(f"BERHENTI: {io.rel(LOCKED_PATH)} sudah ada. Config yang sudah dikunci tidak "
              f"boleh diubah diam-diam; pakai --force kalau memang disengaja.", file=sys.stderr)
        return 1

    sel = io.read_json("02_dev_selection.json")
    params = {m: dict(v["best"]) for m, v in sel["selection"].items()}

    for spec in args.override:
        method, _, values = spec.partition(":")
        if method not in params:
            print(f"BERHENTI: metode {method!r} tidak dikenal", file=sys.stderr)
            return 1
        k, n, g = (x.strip() for x in values.split(","))
        params[method] = {"k": int(k), "n": int(n), "gamma": float(g)}

    config = {
        "locked_at": datetime.date.today().isoformat(),
        "selected_on": {"partition": "dev", "seeds": sel["dev_seeds"],
                        "criterion": sel["criterion"], "tie_break": sel["tie_break"]},
        "grid": sel["grid"],
        "retrieval": {"bm25": {"k1": 1.5, "b": 0.75, "epsilon": 0.25},
                      "tokenizer": "lowercase + \\w+",
                      "query_stopwords": "Sastrawi + {apa, bagaimana, yang, ini, pasien}",
                      "index_stopwords": "tidak dihapus"},
        "rm3": {"n_feedback": N_FEEDBACK, "n_terms": N_TERMS, "lambda": LAMBDA,
                "renormalize": True, "tuned": False},
        "lsa": {"min_df": 2, "drop_pure_digits": True,
                "cell_weight": "(1 + ln tf) * ln(N/df)", "random_state": 0},
        "methods": params,
        "test": {"seeds": list(range(1, 11)), "scenarios": ["TPL", "PARA"],
                 "bootstrap": {"n_resamples": 10000, "rng_seed": 20261019,
                               "strata": "seed x skenario"},
                 "comparisons": [["RM3", "B1"], ["LSA-QE", "B1"], ["RM3+LSA", "B1"],
                                 ["RM3+LSA", "RM3"], ["RM3+LSA", "LSA-QE"]],
                 "multiple_testing": "Holm"},
        "overrides_applied": args.override,
    }

    os.makedirs(io.CONFIGS, exist_ok=True)
    with open(LOCKED_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    digest = io.sha256_file(LOCKED_PATH)
    print(f"Config dikunci: {io.rel(LOCKED_PATH)}")
    for m, p in params.items():
        print(f"  {m:<8} k={p['k']} n={p['n']} gamma={p['gamma']}")
    print(f"\nSHA-256: {digest}")
    print("\nCatat baris berikut di reports/DECISIONS.md:")
    print(f"{config['locked_at']} | Config dikunci di `configs/locked_config.json`, "
          f"SHA-256 `{digest}` | Test dibuka sekali setelah ini.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
