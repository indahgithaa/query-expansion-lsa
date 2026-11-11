"""Metrik performa retrieval dan uji signifikansi sesuai Subbab 4.11."""
import math

import numpy as np
from scipy.stats import wilcoxon

from . import config


def unit_metrics(rank: int) -> dict:
    out = {"rank": rank, "rr": 1.0 / rank,
           "ndcg10": 1.0 / math.log2(rank + 1) if rank <= config.NDCG_K else 0.0,
           "fail": int(rank > config.FAIL_RANK)}
    for k in config.RECALL_KS:
        out[f"r@{k}"] = int(rank <= k)
    return out


def aggregate(ranks) -> dict:
    ranks = np.asarray(ranks)
    rows = [unit_metrics(int(r)) for r in ranks]
    out = {"n": len(rows), "mrr": float(np.mean([r["rr"] for r in rows]))}
    for k in config.RECALL_KS:
        out[f"r@{k}"] = float(np.mean([r[f"r@{k}"] for r in rows]))
    out["ndcg10"] = float(np.mean([r["ndcg10"] for r in rows]))
    out["median_rank"] = float(np.median(ranks))
    out["fail_rate"] = float(np.mean([r["fail"] for r in rows]))
    return out


def stratified_bootstrap(diff, strata, reps: int = config.BOOTSTRAP_REPS, seed: int = config.BOOTSTRAP_SEED,
                         return_samples: bool = False):
    """Subbab 4.11.2: ambil ulang dengan pengembalian di dalam setiap strata (pengulangan x skenario)."""
    diff = np.asarray(diff, dtype=float)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    groups = [np.nonzero(strata == s)[0] for s in sorted(set(strata.tolist()))]
    means = np.empty(reps)
    draws = []
    for b in range(reps):
        idx = np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])
        means[b] = diff[idx].mean()
        if return_samples and b < 3:
            draws.append(idx)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi), draws) if return_samples else (float(lo), float(hi))


def wilcoxon_test(diff) -> dict:
    """Subbab 4.11.2: selisih nol dikeluarkan; bila semua nol, p = 1."""
    diff = np.asarray(diff, dtype=float)
    nz = diff[diff != 0]
    if nz.size == 0:
        return {"n_nonzero": 0, "w_plus": 0.0, "w_minus": 0.0, "statistic": 0.0, "p": 1.0}
    from scipy.stats import rankdata
    r = rankdata(np.abs(nz))
    w_plus, w_minus = float(r[nz > 0].sum()), float(r[nz < 0].sum())
    res = wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
    return {"n_nonzero": int(nz.size), "w_plus": w_plus, "w_minus": w_minus,
            "statistic": float(res.statistic), "p": float(res.pvalue)}


def holm(pvalues) -> list[float]:
    """Persamaan 2.24: p*_(i) = min(1, max_{j<=i} (m-j+1) p_(j))."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for j, i in enumerate(order):
        running = max(running, (m - j) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def verdict(p_holm: float, ci: tuple[float, float]) -> bool:
    """Aturan dua syarat: p Holm < 0,05 DAN selang 95% tidak memuat nol."""
    return p_holm < config.ALPHA and not (ci[0] <= 0.0 <= ci[1])
