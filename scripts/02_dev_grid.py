"""Tahap 2-4 - grid parameter pada dev (seed 1-3) dan seleksi config.

Grid ditetapkan sebelum melihat hasil: k in {100,200,300,500}, n in {5,10,20,30},
gamma in {0.5,1,2,4}. Kriteria seleksi: rata-rata MRR(TPL) dan MRR(PARA).
Seri dipecah dengan k, lalu n, lalu gamma yang lebih kecil.
"""

import collections
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qe import io
from qe.corpus import load_corpus, load_paraphrases
from qe.engine import Engine, expansion_for_n
from qe.metrics import summarize
from qe.scenarios import SCENARIOS
from qe.splits import make_split, types_by_doc

K_GRID = (100, 200, 300, 500)
N_GRID = (5, 10, 20, 30)
GAMMA_GRID = (0.5, 1.0, 2.0, 4.0)
DEV_SEEDS = (1, 2, 3)
PARAM_METHODS = ("LSA-QE", "RM3+LSA")


def selection_score(mrr_by_scenario):
    """Rata-rata MRR(TPL) dan MRR(PARA)."""
    return sum(mrr_by_scenario[s] for s in SCENARIOS) / len(SCENARIOS)


def best_config(score_by_config):
    """Skor tertinggi; seri -> k, lalu n, lalu gamma terkecil."""
    return min(score_by_config, key=lambda c: (-score_by_config[c], c))


def main():
    t0 = time.time()
    corpus = load_corpus(io.DATA)
    paraphrases = load_paraphrases(os.path.join(io.DATA, "paraphrases.json"))
    engine = Engine(corpus, paraphrases)
    tbd = types_by_doc(corpus)

    # ranks[(method, k, n, gamma)][seed][scenario] = daftar peringkat gold
    ranks = {}

    def store(key, seed, scenario, values):
        ranks.setdefault(key, {}).setdefault(seed, {})[scenario] = values

    for seed in DEV_SEEDS:
        split = make_split(corpus, seed, tbd)
        units = engine.prepare_units(seed, split.dev, "dev", scenarios=SCENARIOS)
        by_scenario = collections.defaultdict(list)
        for u in units:
            by_scenario[u.scenario].append(u)

        for scenario, us in by_scenario.items():
            store(("B1", None, None, None), seed, scenario, [engine.rank_b1(u) for u in us])
            store(("RM3", None, None, None), seed, scenario, [engine.rank_rm3(u) for u in us])
        print(f"[seed {seed}] {len(units)} unit dev siap ({time.time() - t0:.0f}s)", flush=True)

        for k in K_GRID:
            t1 = time.time()
            model = engine.fit_lsa(split.collection, k)
            cands = {sc: engine.expansion_candidates(model, us, max(N_GRID))
                     for sc, us in by_scenario.items()}
            for n in N_GRID:
                exps = {sc: [expansion_for_n(c, n) for c in cands[sc]] for sc in by_scenario}
                for g in GAMMA_GRID:
                    for sc, us in by_scenario.items():
                        e = exps[sc]
                        store(("LSA-QE", k, n, g), seed, sc,
                              [engine.rank_lsa_qe(u, ex, g) for u, ex in zip(us, e)])
                        store(("RM3+LSA", k, n, g), seed, sc,
                              [engine.rank_rm3_lsa(u, ex, g) for u, ex in zip(us, e)])
            del model, cands
            n_cfg = len(N_GRID) * len(GAMMA_GRID) * len(PARAM_METHODS)
            print(f"  k={k:<4} {n_cfg} konfigurasi ({time.time() - t1:.0f}s)", flush=True)

    # --- agregasi ---
    def pooled(key, seeds, scenario=None):
        out = []
        for s in seeds:
            for sc, rs in ranks[key][s].items():
                if scenario is None or sc == scenario:
                    out.extend(rs)
        return out

    def score_of(key, seeds):
        return selection_score(
            {sc: summarize(pooled(key, seeds, sc))["MRR"] for sc in SCENARIOS})

    rows = []
    for key in ranks:
        method, k, n, g = key
        per_sc = {sc: summarize(pooled(key, DEV_SEEDS, sc))["MRR"] for sc in SCENARIOS}
        overall = summarize(pooled(key, DEV_SEEDS))
        row = {
            "method": method, "k": k, "n": n, "gamma": g,
            "MRR_TPL": per_sc["TPL"], "MRR_PARA": per_sc["PARA"],
            "selection_score": selection_score(per_sc),
            "MRR_pooled": overall["MRR"], "nDCG@10": overall["nDCG@10"],
            "Recall@1": overall["Recall@1"], "Recall@5": overall["Recall@5"],
            "Recall@10": overall["Recall@10"], "n_units": overall["n"],
        }
        for s in DEV_SEEDS:
            row[f"score_seed{s}"] = score_of(key, (s,))
        rows.append(row)
    rows.sort(key=lambda r: (r["method"], -r["selection_score"]))

    # --- seleksi per metode berparameter + LOSO ---
    selection = {}
    for method in PARAM_METHODS:
        scores = {(r["k"], r["n"], r["gamma"]): r["selection_score"]
                  for r in rows if r["method"] == method}
        best = best_config(scores)

        loso = []
        for held in DEV_SEEDS:
            train = tuple(s for s in DEV_SEEDS if s != held)
            train_scores = {(k, n, g): score_of((method, k, n, g), train)
                            for k in K_GRID for n in N_GRID for g in GAMMA_GRID}
            picked = best_config(train_scores)
            loso.append({
                "held_out_seed": held,
                "config": {"k": picked[0], "n": picked[1], "gamma": picked[2]},
                "score_on_held_out": score_of((method, *picked), (held,)),
            })
        in_sample = [score_of((method, *best), (s,)) for s in DEV_SEEDS]
        loso_mean = sum(x["score_on_held_out"] for x in loso) / len(loso)
        in_sample_mean = sum(in_sample) / len(in_sample)
        selection[method] = {
            "best": {"k": best[0], "n": best[1], "gamma": best[2]},
            "best_score": scores[best],
            "at_grid_boundary": {
                "k": best[0] in (min(K_GRID), max(K_GRID)),
                "n": best[1] in (min(N_GRID), max(N_GRID)),
                "gamma": best[2] in (min(GAMMA_GRID), max(GAMMA_GRID)),
            },
            "loso": loso,
            "loso_mean": loso_mean,
            "in_sample_mean_per_seed": in_sample_mean,
            "selection_bias": in_sample_mean - loso_mean,
        }

    # --- sensitivitas marginal: skor terbaik untuk tiap nilai parameter ---
    sensitivity = {}
    for method in PARAM_METHODS:
        subset = [r for r in rows if r["method"] == method]
        sensitivity[method] = {
            "k": {str(k): max(r["selection_score"] for r in subset if r["k"] == k) for k in K_GRID},
            "n": {str(n): max(r["selection_score"] for r in subset if r["n"] == n) for n in N_GRID},
            "gamma": {str(g): max(r["selection_score"] for r in subset if r["gamma"] == g)
                      for g in GAMMA_GRID},
        }

    baselines = {m: next(r for r in rows if r["method"] == m) for m in ("B1", "RM3")}
    payload = {
        "grid": {"k": list(K_GRID), "n": list(N_GRID), "gamma": list(GAMMA_GRID)},
        "dev_seeds": list(DEV_SEEDS),
        "scenarios": list(SCENARIOS),
        "criterion": "mean(MRR(TPL), MRR(PARA)) pada seed 1-3 digabung",
        "tie_break": "k terkecil, lalu n, lalu gamma",
        "rm3_fixed": {"n_feedback": 5, "n_terms": 30, "lambda": 0.5, "renormalize": True},
        "baselines": baselines,
        "selection": selection,
        "sensitivity": sensitivity,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    io.write_json("02_dev_selection.json", payload)
    io.write_csv("02_dev_grid.csv", rows, fieldnames=list(rows[0]))

    print(f"\nSelesai dalam {time.time() - t0:.0f}s")
    print(f"  B1      {baselines['B1']['selection_score']:.4f}")
    print(f"  RM3     {baselines['RM3']['selection_score']:.4f}")
    for m, s in selection.items():
        b = s["best"]
        print(f"  {m:<8} k={b['k']} n={b['n']} gamma={b['gamma']} "
              f"score={s['best_score']:.4f} LOSO={s['loso_mean']:.4f} "
              f"bias={s['selection_bias']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
