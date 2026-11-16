"""Metrik kualitas jawaban berbasis kata: BLEU (Persamaan 2.17) dan ROUGE-L (Persamaan 2.18).

Kedua teks ditokenisasi dengan `text.tokenize` (normalisasi, huruf kecil, token \\w+), sama seperti jalur
retrieval. Tanda markdown (**, *, -, #) dan tanda baca otomatis terbuang. Stopword tidak dihapus.
BERTScore dan Faithfulness dihitung di skrip terpisah karena memerlukan model.
"""
from sacrebleu.metrics import BLEU

from .text import tokenize

_BLEU = BLEU(tokenize="none", smooth_method="exp", max_ngram_order=4, effective_order=True)


def bleu(candidate: str, reference: str) -> float:
    """BLEU kalimat, N = 4, bobot seragam, brevity penalty, smoothing eksponensial; rentang 0-1."""
    c, r = " ".join(tokenize(candidate)), " ".join(tokenize(reference))
    return _BLEU.sentence_score(c, [r]).score / 100.0


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str, beta: float = 1.0) -> dict:
    """ROUGE-L berbasis LCS token; F dengan beta = 1 (rata-rata harmonik P dan R)."""
    c, r = tokenize(candidate), tokenize(reference)
    if not c or not r:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    lcs = _lcs(c, r)
    p, rec = lcs / len(c), lcs / len(r)
    f = 0.0 if lcs == 0 else (1 + beta ** 2) * rec * p / (rec + beta ** 2 * p)
    return {"p": p, "r": rec, "f": f}
