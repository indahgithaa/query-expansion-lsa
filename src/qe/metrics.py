"""Metrik evaluasi dengan tepat satu gold per query."""

import math
import statistics

CUTOFFS = (1, 3, 5, 10, 20)
FAIL_CUTOFF = 10


def reciprocal_rank(rank: int) -> float:
    return 1.0 / rank


def ndcg_at_10(rank: int) -> float:
    """Satu gold: 1/log2(rank + 1) bila rank <= 10, selain itu 0."""
    return 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0


def summarize(ranks):
    """Ringkasan metrik dari daftar peringkat gold (1-based)."""
    ranks = list(ranks)
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    out = {
        "n": n,
        "MRR": sum(reciprocal_rank(r) for r in ranks) / n,
        "nDCG@10": sum(ndcg_at_10(r) for r in ranks) / n,
        "median_rank": float(statistics.median(ranks)),
        "fail_rate": sum(1 for r in ranks if r > FAIL_CUTOFF) / n,
    }
    for k in CUTOFFS:
        out[f"Recall@{k}"] = sum(1 for r in ranks if r <= k) / n
    return out


def robustness(ranks_method, ranks_base):
    """Perbandingan per query terhadap baseline B1."""
    a = list(ranks_method)
    b = list(ranks_base)
    if len(a) != len(b):
        raise ValueError("jumlah query tidak sama")
    n = len(a)
    if n == 0:
        return {"n": 0}
    improved = sum(1 for x, y in zip(a, b) if x < y)
    worsened = sum(1 for x, y in zip(a, b) if x > y)
    unchanged = n - improved - worsened
    deltas = [reciprocal_rank(x) - reciprocal_rank(y) for x, y in zip(a, b)]
    return {
        "n": n,
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
        "pct_improved": improved / n,
        "pct_unchanged": unchanged / n,
        "pct_worsened": worsened / n,
        "robustness_index": (improved - worsened) / n,
        "entered_top10": sum(1 for x, y in zip(a, b) if x <= 10 < y),
        "left_top10": sum(1 for x, y in zip(a, b) if y <= 10 < x),
        "big_drops": sum(1 for d in deltas if d < -0.5),
        "mean_delta_rr": sum(deltas) / n,
    }
