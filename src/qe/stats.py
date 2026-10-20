"""Uji signifikansi: paired bootstrap berstrata, Wilcoxon, koreksi Holm (Tahap 5)."""

import numpy as np
from scipy import stats as sps

BOOTSTRAP_SEED = 20261019
N_RESAMPLES = 10_000


def stratified_paired_bootstrap(rr_a, rr_b, strata, n_resamples: int = N_RESAMPLES,
                                seed: int = BOOTSTRAP_SEED, alpha: float = 0.05):
    """CI persentil untuk selisih rata-rata RR (A - B), resample dalam tiap strata.

    `strata` = label strata per unit (mis. (seed, skenario)). Pasangan dipertahankan:
    unit yang sama diambil untuk A dan B, sehingga mean(RR_A - RR_B) = selisih mean.
    """
    rr_a = np.asarray(rr_a, dtype=np.float64)
    rr_b = np.asarray(rr_b, dtype=np.float64)
    if rr_a.shape != rr_b.shape:
        raise ValueError("rr_a dan rr_b harus sama panjang")
    diff = rr_a - rr_b
    n_total = diff.size
    rng = np.random.default_rng(seed)

    groups = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)

    sums = np.zeros(n_resamples, dtype=np.float64)
    for key in sorted(groups, key=repr):
        d = diff[np.asarray(groups[key], dtype=np.int64)]
        m = d.size
        idx = rng.integers(0, m, size=(n_resamples, m))
        sums += d[idx].sum(axis=1)

    boot = sums / n_total
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n_total),
        "n_strata": len(groups),
        "n_resamples": int(n_resamples),
    }


def wilcoxon(rr_a, rr_b):
    """Wilcoxon signed-rank dua sisi pada selisih RR berpasangan."""
    a = np.asarray(rr_a, dtype=np.float64)
    b = np.asarray(rr_b, dtype=np.float64)
    d = a - b
    n_nonzero = int(np.count_nonzero(d))
    if n_nonzero == 0:
        return {"statistic": float("nan"), "p_value": 1.0, "n_nonzero": 0}
    res = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n_nonzero": n_nonzero}


def holm(p_values):
    """Koreksi Holm-Bonferroni; mengembalikan p terkoreksi pada urutan masukan."""
    p = list(p_values)
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank_, i in enumerate(order):
        running = max(running, (m - rank_) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted
