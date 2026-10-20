from qe.tokenize import EXTRA_STOPWORDS, stopwords, strip_stopwords, tokenize, tokenize_doc, tokenize_query


def test_tokenize_lowercases_and_splits_on_non_word():
    assert tokenize("Status Gizi/Lab (abnormal)?") == ["status", "gizi", "lab", "abnormal"]


def test_tokenize_keeps_digits():
    assert tokenize("Hb 10,5 g/dL") == ["hb", "10", "5", "g", "dl"]


def test_query_drops_stopwords_doc_keeps_them():
    q = "Bagaimana status gizi pada pasien ini?"
    assert tokenize_query(q) == ["status", "gizi"]
    assert tokenize_doc(q) == ["bagaimana", "status", "gizi", "pada", "pasien", "ini"]


def test_extra_stopwords_are_included():
    sw = stopwords()
    assert EXTRA_STOPWORDS <= sw
    assert "gizi" not in sw


def test_strip_stopwords_matches_query_tokenizer():
    text = "Apa tujuan diet pada pasien ini?"
    assert strip_stopwords(tokenize(text)) == tokenize_query(text)
