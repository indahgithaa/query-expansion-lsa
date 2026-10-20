"""Pemuatan korpus dan dataset QA (Tahap 0)."""

import dataclasses
import glob
import json
import os

from .normalize import normalize_filename, normalize_text

# Tipe pertanyaan ditentukan dari teks template persis.
QUESTION_TYPES = {
    "Apa metode skrining status gizi yang direkomendasikan?": "skrining",
    "Bagaimana status gizi pada pasien ini?": "status_gizi",
    "Bagaimana data biokimia/parameter lab pada pasien (yg abnormal)?": "biokimia",
    "Bagaimana menyusun diagnosis gizi dengan format PES pada pasien ini?": "pes",
    "Apa tujuan diet pada pasien ini?": "tujuan_diet",
    "Bagaimana prinsip diet yang tepat sesuai dengan kondisi pasien ini?": "prinsip_diet",
    "Bagaimana syarat diet yang harus dipenuhi dalam pengaturan diet pasien ini?": "syarat_diet",
    "Apa pedoman diet yang digunakan dalam perencanaan diet pasien ini?": "pedoman",
}
TYPE_ORDER = (
    "skrining",
    "status_gizi",
    "biokimia",
    "pes",
    "tujuan_diet",
    "prinsip_diet",
    "syarat_diet",
    "pedoman",
)

# Angka sanity check wajib (Tahap 0).
EXPECTED_RAW_QA = 911
EXPECTED_QA = 858
EXPECTED_DOCS = 120
EXPECTED_CHUNKS = 13737


def make_chunk_id(doc_id: str, page: int, chunk_index: int) -> str:
    return f"{doc_id}:P{page:04d}:C{chunk_index:03d}"


@dataclasses.dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_index: int
    text: str
    tokens: tuple


@dataclasses.dataclass(frozen=True)
class Document:
    doc_id: str
    filename: str
    chunks: tuple

    def index_of(self, chunk_id: str) -> int:
        for i, c in enumerate(self.chunks):
            if c.chunk_id == chunk_id:
                return i
        raise KeyError(chunk_id)


@dataclasses.dataclass(frozen=True)
class QA:
    qid: int
    doc_id: str
    filename: str
    question: str
    qtype: str
    gold_chunk_id: str


class Corpus:
    """Koleksi 120 laporan beserta chunk-nya dan QA yang sudah dideduplikasi."""

    def __init__(self, documents, qas):
        self.documents = {d.doc_id: d for d in documents}
        self.doc_ids = tuple(d.doc_id for d in documents)
        self.by_filename = {d.filename: d for d in documents}
        self.qas = tuple(qas)
        self.qa_by_id = {q.qid: q for q in self.qas}

    @property
    def n_chunks(self) -> int:
        return sum(len(d.chunks) for d in self.documents.values())

    def chunks_of(self, doc_ids):
        for did in doc_ids:
            yield from self.documents[did].chunks

    def qas_in(self, doc_ids):
        s = set(doc_ids)
        return [q for q in self.qas if q.doc_id in s]


def load_chunks(chunks_dir: str):
    """Baca semua `*.chunks.json`, beri doc_id D001..D120 menurut urutan nama ternormalisasi."""
    from .tokenize import tokenize_doc

    paths = glob.glob(os.path.join(chunks_dir, "*.chunks.json"))
    by_name = {}
    for p in paths:
        name = normalize_filename(os.path.basename(p))
        if name in by_name:
            raise ValueError(f"nama file bentrok setelah normalisasi: {name}")
        by_name[name] = p

    documents = []
    for i, name in enumerate(sorted(by_name)):
        doc_id = f"D{i + 1:03d}"
        with open(by_name[name], encoding="utf-8") as fh:
            raw = json.load(fh)
        chunks = []
        for rec in raw:
            page = int(rec["page_number"])
            ci = int(rec["chunk_index"])
            text = normalize_text(rec["text"])
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, page, ci),
                    doc_id=doc_id,
                    page_number=page,
                    chunk_index=ci,
                    text=text,
                    tokens=tuple(tokenize_doc(text)),
                )
            )
        chunks.sort(key=lambda c: c.chunk_id)
        documents.append(Document(doc_id=doc_id, filename=name, chunks=tuple(chunks)))
    return documents


def load_qa(dataset_path: str, documents):
    """Baca dataset QA, deduplikasi, tentukan tipe dan gold chunk.

    Dedup: kunci (file ternormalisasi, question), simpan id numerik terkecil.
    Gold: tepat satu chunk pada (file, page_number, chunk_index).
    """
    by_filename = {d.filename: d for d in documents}
    with open(dataset_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    best = {}
    for rec in raw:
        key = (normalize_filename(rec["file"]), rec["question"])
        qid = int(rec["id"])
        if key not in best or qid < best[key][0]:
            best[key] = (qid, rec)

    qas = []
    for (fname, question), (qid, rec) in best.items():
        doc = by_filename.get(fname)
        if doc is None:
            raise ValueError(f"file QA tidak punya file chunk: {fname!r}")
        qtype = QUESTION_TYPES.get(question)
        if qtype is None:
            raise ValueError(f"pertanyaan di luar 8 template: {question!r}")
        gold = make_chunk_id(doc.doc_id, int(rec["page_number"]), int(rec["chunk_index"]))
        matches = [c for c in doc.chunks if c.chunk_id == gold]
        if len(matches) != 1:
            raise ValueError(f"qid {qid}: gold chunk tidak tepat satu ({len(matches)}) untuk {gold}")
        qas.append(
            QA(
                qid=qid,
                doc_id=doc.doc_id,
                filename=fname,
                question=question,
                qtype=qtype,
                gold_chunk_id=gold,
            )
        )
    qas.sort(key=lambda q: q.qid)
    return qas, len(raw)


def load_corpus(data_dir: str = "data", check: bool = True):
    """Muat korpus lengkap dan jalankan sanity check Tahap 0."""
    documents = load_chunks(os.path.join(data_dir, "chunks"))
    qas, n_raw = load_qa(os.path.join(data_dir, "dataset", "dataset.json"), documents)
    corpus = Corpus(documents, qas)
    if check:
        problems = []
        if n_raw != EXPECTED_RAW_QA:
            problems.append(f"QA mentah {n_raw} != {EXPECTED_RAW_QA}")
        if len(qas) != EXPECTED_QA:
            problems.append(f"QA setelah dedup {len(qas)} != {EXPECTED_QA}")
        if len(documents) != EXPECTED_DOCS:
            problems.append(f"dokumen {len(documents)} != {EXPECTED_DOCS}")
        if corpus.n_chunks != EXPECTED_CHUNKS:
            problems.append(f"chunk {corpus.n_chunks} != {EXPECTED_CHUNKS}")
        if problems:
            raise AssertionError("Sanity check Tahap 0 gagal: " + "; ".join(problems))
    return corpus


def load_paraphrases(path: str):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for qtype in TYPE_ORDER:
        if qtype not in data:
            raise ValueError(f"paraphrases.json tidak punya tipe {qtype}")
        for part in ("dev", "test"):
            items = data[qtype][part]
            if len(items) != 6:
                raise ValueError(f"{qtype}/{part}: {len(items)} parafrasa, harus 6")
    return {t: {p: [normalize_text(s) for s in v[p]] for p in ("dev", "test")} for t, v in data.items()}
