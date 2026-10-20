from qe.normalize import normalize_filename, normalize_text


def test_removes_word_joiner():
    assert normalize_text("A\u2060B") == "AB"


def test_decodes_hash_u():
    assert normalize_text("MECKEL#U2019S") == "MECKEL\u2019S"
    assert normalize_text("a#U00e9b") == "a\u00e9b"


def test_nfkc_folds_compatibility_characters():
    # MATHEMATICAL ITALIC SMALL A -> a, MICRO SIGN -> GREEK SMALL LETTER MU
    assert normalize_text("\U0001d44e") == "a"
    assert normalize_text("\u00b5g") == "\u03bcg"


def test_leaves_plain_text_alone():
    assert normalize_text("status gizi pasien") == "status gizi pasien"


def test_filename_uses_same_rule():
    assert normalize_filename("x\u2060y.chunks.json") == "xy.chunks.json"
