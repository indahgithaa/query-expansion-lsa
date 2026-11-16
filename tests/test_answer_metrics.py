"""Uji metrik jawaban berbasis kata dengan contoh yang dapat dihitung tangan."""
import math

import pytest

from qe_lsa.answer_metrics import bleu, rouge_l


def test_bleu_identical_is_one():
    assert bleu("Diet rendah garam 3 g.", "diet RENDAH garam 3 g") == pytest.approx(1.0)


def test_bleu_brevity_penalty_only():
    # seluruh presisi n-gram = 1, kandidat 5 token, referensi 7 token -> BLEU = exp(1 - 7/5)
    assert bleu("a b c d e", "a b c d e f g") == pytest.approx(math.exp(1 - 7 / 5), abs=1e-4)


def test_bleu_markdown_ignored():
    assert bleu("**Diet** rendah garam", "diet rendah garam") == pytest.approx(1.0)


def test_rouge_l_hand_example():
    # LCS("a b c d", "a c d e") = "a c d" = 3 -> P = R = 3/4, F1 = 0.75
    s = rouge_l("a b c d", "a c d e")
    assert (s["p"], s["r"], s["f"]) == pytest.approx((0.75, 0.75, 0.75))


def test_rouge_l_asymmetric():
    # LCS = 2 ("a b"), P = 2/2, R = 2/4 -> F1 = 2*1*0.5/1.5
    s = rouge_l("a b", "a x b y")
    assert s["f"] == pytest.approx(2 / 3)


def test_rouge_l_no_overlap():
    assert rouge_l("a b", "c d")["f"] == 0.0
