"""Pemuatan data sesuai Subbab 4.3."""
import json
from dataclasses import dataclass, field

from . import config
from .text import normalize, tokenize

TEMPLATES = {
    "Apa metode skrining status gizi yang direkomendasikan?": "Skrining",
    "Bagaimana status gizi pada pasien ini?": "Status gizi",
    "Bagaimana data biokimia/parameter lab pada pasien (yg abnormal)?": "Biokimia",
    "Bagaimana menyusun diagnosis gizi dengan format PES pada pasien ini?": "PES",
    "Apa tujuan diet pada pasien ini?": "Tujuan diet",
    "Bagaimana prinsip diet yang tepat sesuai dengan kondisi pasien ini?": "Prinsip diet",
    "Bagaimana syarat diet yang harus dipenuhi dalam pengaturan diet pasien ini?": "Syarat diet",
    "Apa pedoman diet yang digunakan dalam perencanaan diet pasien ini?": "Pedoman",
}
QTYPES = list(TEMPLATES.values())


def chunk_id(doc_id: str, page: int, index: int) -> str:
    return f"{doc_id}:P{page:04d}:C{index:03d}"


@dataclass
class Chunk:
    cid: str
    doc_id: str
    text: str
    tokens: list[str]


@dataclass
class QA:
    qid: int
    doc_id: str
    ref_cid: str
    qtype: str
    question: str
    answer: str
    context: str


@dataclass
class Corpus:
    doc_ids: list[str]
    chunks: dict[str, list[Chunk]] = field(default_factory=dict)  # doc_id -> chunk terurut
    qas: list[QA] = field(default_factory=list)
    file_to_doc: dict[str, str] = field(default_factory=dict)
    n_qa_raw: int = 0


def load_corpus() -> Corpus:
    files = sorted(config.CHUNK_DIR.glob("*.chunks.json"), key=lambda p: normalize(p.name))
    file_to_doc = {p.name: f"D{i + 1:03d}" for i, p in enumerate(files)}
    corpus = Corpus(doc_ids=sorted(file_to_doc.values()), file_to_doc=file_to_doc)

    for path in files:
        doc_id = file_to_doc[path.name]
        records = json.loads(path.read_text(encoding="utf-8"))
        chunks = []
        for r in records:
            text = normalize(r["text"])
            chunks.append(Chunk(chunk_id(doc_id, r["page_number"], r["chunk_index"]), doc_id, text, tokenize(text)))
        chunks.sort(key=lambda c: c.cid)
        corpus.chunks[doc_id] = chunks

    raw = json.loads(config.QA_PATH.read_text(encoding="utf-8"))
    corpus.n_qa_raw = len(raw)
    kept: dict[tuple[str, str], dict] = {}
    for r in raw:
        key = (r["file"], r["question"])
        if key not in kept or r["id"] < kept[key]["id"]:
            kept[key] = r

    cid_sets = {d: {c.cid for c in cs} for d, cs in corpus.chunks.items()}
    for r in sorted(kept.values(), key=lambda r: r["id"]):
        doc_id = file_to_doc[r["file"]]
        ref = chunk_id(doc_id, r["page_number"], r["chunk_index"])
        n_match = int(ref in cid_sets[doc_id])
        if n_match != 1:
            raise ValueError(f"QA id={r['id']}: chunk acuan {ref} ditemukan {n_match} kali")
        corpus.qas.append(QA(
            qid=r["id"], doc_id=doc_id, ref_cid=ref, qtype=TEMPLATES[r["question"]],
            question=r["question"], answer=" ".join(r["answer"]["text"]), context=r["context"],
        ))
    return corpus
