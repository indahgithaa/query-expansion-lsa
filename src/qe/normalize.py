"""Normalisasi teks dan nama file (Tahap 0)."""

import re
import unicodedata

WORD_JOINER = "\u2060"
_HASH_U = re.compile(r"#U([0-9a-fA-F]{4})")


def _decode_hash_u(s: str) -> str:
    """Ubah pola `#Uxxxx` menjadi karakter Unicode yang diwakilinya."""
    return _HASH_U.sub(lambda m: chr(int(m.group(1), 16)), s)


def normalize_text(s: str) -> str:
    """NFKC -> dekode `#Uxxxx` -> hapus U+2060 (word joiner)."""
    s = unicodedata.normalize("NFKC", s)
    s = _decode_hash_u(s)
    return s.replace(WORD_JOINER, "")


def normalize_filename(s: str) -> str:
    """Nama file memakai normalisasi yang sama dengan teks."""
    return normalize_text(s)
