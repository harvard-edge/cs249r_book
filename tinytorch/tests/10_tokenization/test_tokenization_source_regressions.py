"""Regression coverage against Module 10's source of truth, without exporting."""
from pathlib import Path
import runpy
import warnings

import pytest


@pytest.fixture(scope="module")
def tokenization_source():
    source = Path(__file__).resolve().parents[2] / "src/10_tokenization/10_tokenization.py"
    return runpy.run_path(str(source), run_name="tokenization_audit")


def test_actual_vocabulary_size_supports_embedding_allocation(tokenization_source):
    tokenizer = tokenization_source["BPETokenizer"](vocab_size=2)
    tokenizer.train(["abcdef"])
    assert tokenizer.target_vocab_size == 2
    assert tokenizer.vocab_size == len(tokenizer.vocab) > 2
    assert all(0 <= token < tokenizer.vocab_size for token in tokenizer.encode("fedcba"))
    assert tokenizer.decode(tokenizer.encode("fedcba")) == "fedcba"
    tokenizer.train(["hello"], vocab_size=1000)
    assert tokenizer.vocab_size == len(tokenizer.vocab) < 1000
    stats = tokenization_source["analyze_tokenization"](["hello"], tokenizer)
    assert stats["vocab_size"] == len(tokenizer.vocab)


def test_bpe_recombines_seen_characters_in_new_word_positions(tokenization_source):
    tokenizer = tokenization_source["BPETokenizer"](20)
    tokenizer.train(["ab"])
    for text in ["ba", "a", "b", "abba", "ba ab", "  ba\t ab  "]:
        ids = tokenizer.encode(text)
        assert 0 not in ids
        assert tokenizer.decode(ids) == " ".join(text.split())


@pytest.mark.parametrize("target", [1, 20, 100])
def test_literal_marker_text_round_trips_at_every_merge_depth(tokenization_source, target):
    tokenizer = tokenization_source["BPETokenizer"](target)
    corpus = ["x</w>y", "</w>", "<UNK>x", "x<UNK>", "a&<b", "x</w></w>y"]
    tokenizer.train(corpus)
    for text in corpus:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert len(tokenizer.vocab) == len(set(tokenizer.vocab))


def test_empty_tokenization_analysis_is_finite_and_quiet(tokenization_source):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = tokenization_source["analyze_tokenization"](
            [], tokenization_source["CharTokenizer"]()
        )
    assert stats == {
        "vocab_size": 1, "avg_sequence_length": 0.0, "max_sequence_length": 0,
        "total_tokens": 0, "compression_ratio": 0, "unique_tokens": 0,
    }


@pytest.mark.parametrize("target", [0, -1, 1.5, True])
def test_invalid_bpe_target_fails_before_training(tokenization_source, target):
    cls = tokenization_source["BPETokenizer"]
    with pytest.raises(ValueError, match="positive integer"):
        cls(target)
    tokenizer = cls(10)
    with pytest.raises(ValueError, match="positive integer"):
        tokenizer.train(["abc"], vocab_size=target)


def test_source_module_integration(tokenization_source):
    tokenization_source["test_module"]()
