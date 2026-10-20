"""Tokenisasi dan stopword (Tahap 2).

Query: stopword dihapus. Chunk yang diindeks: stopword TIDAK dihapus.
"""

import functools
import re

from .normalize import normalize_text

_WORD = re.compile(r"\w+", re.UNICODE)

# Stopword tambahan di luar Sastrawi: kata tanya/penunjuk yang muncul di tiap
# template pertanyaan sehingga tidak membawa daya pembeda.
EXTRA_STOPWORDS = frozenset({"apa", "bagaimana", "yang", "ini", "pasien"})


@functools.lru_cache(maxsize=1)
def stopwords() -> frozenset:
    """Stopword Sastrawi + EXTRA_STOPWORDS, dinormalisasi dan di-lowercase."""
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    base = {normalize_text(w).lower() for w in StopWordRemoverFactory().get_stop_words()}
    return frozenset(base | EXTRA_STOPWORDS)


def tokenize(text: str) -> list:
    r"""Lowercase lalu ambil semua `\w+`. Teks diasumsikan sudah ternormalisasi."""
    return _WORD.findall(text.lower())


def tokenize_query(text: str) -> list:
    """Tokenisasi query: stopword dihapus."""
    sw = stopwords()
    return [t for t in tokenize(text) if t not in sw]


def tokenize_doc(text: str) -> list:
    """Tokenisasi chunk untuk indeks BM25: stopword dipertahankan."""
    return tokenize(text)


def strip_stopwords(tokens) -> list:
    sw = stopwords()
    return [t for t in tokens if t not in sw]
