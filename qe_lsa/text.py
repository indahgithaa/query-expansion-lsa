"""Preprocessing teks sesuai Subbab 4.4."""
import re
import unicodedata
from functools import lru_cache

_UXXXX = re.compile(r"#U([0-9A-Fa-f]{4})")
_TOKEN = re.compile(r"\w+")
_DIGITS = re.compile(r"\d+")
EXTRA_STOPWORDS = ("apa", "bagaimana", "yang", "ini", "pasien")


def normalize(text: str) -> str:
    """Subbab 4.4.1: NFKC -> dekode #Uxxxx -> hapus U+2060."""
    text = unicodedata.normalize("NFKC", text)
    text = _UXXXX.sub(lambda m: chr(int(m.group(1), 16)), text)
    return text.replace("⁠", "")


def tokenize(text: str) -> list[str]:
    """Subbab 4.4.2: normalisasi, case folding, lalu pola \\w+."""
    return _TOKEN.findall(normalize(text).lower())


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    """Subbab 4.4.3: daftar Sastrawi yang dinormalisasi + lima kata tambahan."""
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    base = StopWordRemoverFactory().get_stop_words()
    return frozenset({normalize(w).lower() for w in base} | set(EXTRA_STOPWORDS))


def remove_stopwords(tokens: list[str]) -> list[str]:
    sw = stopwords()
    return [t for t in tokens if t not in sw]


def is_pure_number(term: str) -> bool:
    """Subbab 4.6.1: token yang seluruhnya berupa digit."""
    return _DIGITS.fullmatch(term) is not None
