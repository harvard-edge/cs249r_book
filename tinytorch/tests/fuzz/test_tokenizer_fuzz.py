"""
Property-based fuzz tests for CharTokenizer.encode()/decode().

Requires the optional `fuzz` dependency group; skips cleanly if
hypothesis isn't installed. See test_tensor_fuzz.py for the rationale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

hypothesis = pytest.importorskip("hypothesis", reason="fuzz tests need the optional 'fuzz' dependency group")
from hypothesis import given, settings, strategies as st

from tinytorch.core.tokenization import CharTokenizer


# Include a broad range of unicode text: punctuation, whitespace, emoji,
# surrogate-adjacent characters, not just ASCII letters.
text_strategy = st.text(min_size=0, max_size=200)


@given(text_strategy)
@settings(max_examples=200)
def test_encode_decode_round_trips_when_vocab_covers_the_text(text):
    """If the vocabulary was built from the text itself (so every
    character is known), decode(encode(text)) must reproduce it exactly."""
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([text])

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert decoded == text


@given(text_strategy)
@settings(max_examples=200)
def test_encode_never_crashes_on_arbitrary_text(text):
    """encode() must handle any string, including text containing
    characters absent from the vocabulary (mapped to <UNK>), without
    raising."""
    tokenizer = CharTokenizer(vocab=["a", "b", "c"])
    tokens = tokenizer.encode(text)

    assert len(tokens) == len(text)
    assert all(isinstance(t, int) for t in tokens)


@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=200))
@settings(max_examples=200)
def test_decode_never_crashes_on_arbitrary_token_ids(tokens):
    """decode() must handle any list of integers, including IDs outside
    the vocabulary's valid range (negative or beyond vocab_size), without
    raising, since a malformed or corrupted token stream is exactly the
    kind of input this must survive gracefully."""
    tokenizer = CharTokenizer(vocab=["a", "b", "c"])
    result = tokenizer.decode(tokens)

    assert isinstance(result, str)


@given(text_strategy)
@settings(max_examples=100)
def test_encode_output_length_matches_input_length(text):
    """CharTokenizer maps one token per character; encode()'s output
    length must always match the input length exactly, known or not."""
    tokenizer = CharTokenizer(vocab=["x", "y", "z"])
    tokens = tokenizer.encode(text)

    assert len(tokens) == len(text)
