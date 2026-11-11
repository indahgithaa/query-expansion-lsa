"""Penyusunan prompt untuk lapisan generasi (Subbab 4.10). Pemanggilan model dilakukan di notebook Kaggle."""
import hashlib

from . import config

INSTRUCTION = (
    "Anda adalah asisten ahli gizi klinik. Jawablah pertanyaan hanya berdasarkan konteks yang diberikan, "
    "yang berasal dari laporan asuhan gizi seorang pasien. Jangan menambahkan informasi yang tidak terdapat "
    "pada konteks. Apabila konteks tidak memuat jawaban, nyatakan bahwa informasi tersebut tidak tersedia "
    "dalam konteks. Jawablah dalam bahasa Indonesia secara ringkas."
)


def build_prompt(question: str, chunk_texts: list[str]) -> str:
    """Tiga bagian: instruksi, konteks berpenanda [1]–[5] sesuai urutan peringkat, dan pertanyaan."""
    assert len(chunk_texts) == config.GEN_TOP_CHUNKS
    context = "\n\n".join(f"[{i}] {t.strip()}" for i, t in enumerate(chunk_texts, 1))
    return f"Instruksi:\n{INSTRUCTION}\n\nKonteks:\n{context}\n\nPertanyaan:\n{question.strip()}"


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
