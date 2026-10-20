import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest

from qe.corpus import load_corpus, load_paraphrases

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(os.path.join(_ROOT, "data"))


@pytest.fixture(scope="session")
def paraphrases():
    return load_paraphrases(os.path.join(_ROOT, "data", "paraphrases.json"))
