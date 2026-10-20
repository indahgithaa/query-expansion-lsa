"""Skenario query TPL dan PARA (Tahap 1)."""

import hashlib

SCENARIOS = ("TPL", "PARA")
N_PARAPHRASES = 6


def paraphrase_index(seed: int, qid: int, n: int = N_PARAPHRASES) -> int:
    """int.from_bytes(sha256(f"{seed}|{qid}").digest()[:8], "big") % n."""
    digest = hashlib.sha256(f"{seed}|{qid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def query_text(qa, scenario: str, seed: int, paraphrases, partition: str) -> str:
    """Teks query untuk satu unit evaluasi (seed, skenario, qid)."""
    if scenario == "TPL":
        return qa.question
    if scenario != "PARA":
        raise ValueError(f"skenario tidak dikenal: {scenario}")
    if partition not in ("dev", "test"):
        raise ValueError(f"partisi parafrasa harus dev/test, bukan {partition!r}")
    return paraphrases[qa.qtype][partition][paraphrase_index(seed, qa.qid)]
