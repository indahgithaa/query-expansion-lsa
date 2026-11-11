"""Pembagian dokumen dan skenario pertanyaan sesuai Subbab 4.5."""
import hashlib
import json
import random

from . import config
from .data import QTYPES, Corpus


def split_documents(corpus: Corpus, seed: int) -> dict[str, list[str]]:
    """Subbab 4.5.1: acak dengan nilai awal seed*1.000.003+attempt sampai tiap bagian memuat 8 tipe."""
    docs = sorted(corpus.doc_ids)
    types_of = {d: set() for d in docs}
    for qa in corpus.qas:
        types_of[qa.doc_id].add(qa.qtype)
    attempt = 0
    while True:
        order = docs[:]
        random.Random(seed * config.SEED_MULTIPLIER + attempt).shuffle(order)
        a, b = config.N_COLLECTION, config.N_COLLECTION + config.N_DEV
        parts = {"collection": order[:a], "dev": order[a:b], "test": order[b:]}
        if all(set().union(*(types_of[d] for d in p)) == set(QTYPES) for p in parts.values()):
            parts["attempt"] = attempt
            return parts
        attempt += 1


def load_paraphrases() -> dict:
    """Format: {"dev": {tipe: [6 kalimat]}, "test": {tipe: [6 kalimat]}}."""
    if not config.PARAPHRASE_PATH.exists():
        raise FileNotFoundError(f"Daftar parafrasa belum tersedia: {config.PARAPHRASE_PATH}")
    return json.loads(config.PARAPHRASE_PATH.read_text(encoding="utf-8"))


def paraphrase_index(seed: int, qid: int) -> int:
    """Subbab 4.5.2: 8 byte pertama SHA-256("seed|qid") sebagai bilangan bulat, mod 6."""
    digest = hashlib.sha256(f"{seed}|{qid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % config.N_PARAPHRASES


def question_text(qa, seed: int, part: str, scenario: str, paraphrases: dict | None = None) -> str:
    if scenario == "TPL":
        return qa.question
    if paraphrases is None:
        paraphrases = load_paraphrases()
    return paraphrases[part][qa.qtype][paraphrase_index(seed, qa.qid)]
