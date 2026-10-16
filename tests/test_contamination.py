"""Tests for crux/contamination.py: the planted 8-gram copy and the disjoint text."""
import numpy as np

from crux import contamination as C

TEXT = "Alice is kind. If someone is kind then they are green. Bob is not green. Question: Alice is green."


def found(tmp_path, corpus, n=8):
    """Index one test record. Return the fraction of its n-grams that occur in the corpus texts."""
    C.build_index([("r0", "g", TEXT)], tmp_path, n)
    C.load_index(tmp_path)
    hit = np.zeros(len(C._ix["H"]), bool)
    C.scan_batch(corpus, hit)
    return hit.mean()


def test_planted_copy_is_found(tmp_path):
    copy = "NOISE one two three. " + TEXT.upper() + " four five."  # case and punctuation do not matter
    assert found(tmp_path, ["unrelated words here", copy]) == 1.0


def test_disjoint_text_and_document_borders_give_no_match(tmp_path):
    words = C.tokens(TEXT)
    parts = [" ".join(words[i:i + 7]) for i in range(0, len(words), 7)]  # each part is shorter than 8 tokens
    assert found(tmp_path, ["the quick brown fox " * 20] + parts) == 0.0
