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


PARAPHRASE_KEYS = {"skrining": "Skrining", "status_gizi": "Status gizi", "biokimia": "Biokimia", "pes": "PES",
                   "tujuan_diet": "Tujuan diet", "prinsip_diet": "Prinsip diet", "syarat_diet": "Syarat diet",
                   "pedoman": "Pedoman"}


def load_paraphrases() -> dict:
    """Berkas asli: {kunci_tipe: {"dev": [6], "test": [6]}} -> {"dev": {tipe: [6]}, "test": {tipe: [6]}}."""
    raw = json.loads(config.PARAPHRASE_PATH.read_text(encoding="utf-8"))
    out = {"dev": {}, "test": {}}
    for key, qtype in PARAPHRASE_KEYS.items():
        for part in ("dev", "test"):
            sentences = raw[key][part]
            assert len(sentences) == config.N_PARAPHRASES
            out[part][qtype] = sentences
        assert not set(raw[key]["dev"]) & set(raw[key]["test"]), "parafrasa dev dan test harus tidak beririsan"
    assert set(out["dev"]) == set(QTYPES)
    return out


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
