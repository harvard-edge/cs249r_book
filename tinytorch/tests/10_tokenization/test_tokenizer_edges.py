"""Edge cases for CharTokenizer's user-supplied vocabulary, run against Module 10's source."""
from pathlib import Path
import runpy

import pytest


@pytest.fixture(scope="module")
def tokenization_source():
    source = Path(__file__).resolve().parents[2] / "src/10_tokenization/10_tokenization.py"
    return runpy.run_path(str(source), run_name="tokenization_edge_audit")


def test_char_tokenizer_dedupes_user_vocab(tokenization_source):
    tokenizer = tokenization_source["CharTokenizer"](["a", "b", "a", "c", "b"])
    unk = tokenization_source["Tokenizer"].TOK_UNKNOWN
    assert tokenizer.vocab == [unk, "a", "b", "c"]
    assert tokenizer.vocab_size == 4
    # Every token has exactly one ID, and the two mappings invert each other.
    assert len(tokenizer.char_to_id) == len(tokenizer.id_to_char) == tokenizer.vocab_size
    assert all(tokenizer.char_to_id[tokenizer.id_to_char[i]] == i for i in range(tokenizer.vocab_size))
    ids = tokenizer.encode("abcab")
    assert all(0 < i < tokenizer.vocab_size for i in ids)
    assert tokenizer.decode(ids) == "abcab"


def test_char_tokenizer_does_not_duplicate_unk(tokenization_source):
    unk = tokenization_source["Tokenizer"].TOK_UNKNOWN
    tokenizer = tokenization_source["CharTokenizer"]([unk, "x"])
    assert tokenizer.vocab == [unk, "x"]
    assert tokenizer.unk_id == tokenizer.char_to_id[unk] == 0


def test_char_tokenizer_accepts_any_iterable_vocab(tokenization_source):
    tokenizer = tokenization_source["CharTokenizer"](("x", "y", "x"))
    assert tokenizer.vocab_size == 3
    assert tokenizer.decode(tokenizer.encode("xyx")) == "xyx"
