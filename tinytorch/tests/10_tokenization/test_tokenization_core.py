"""
Module 10: Tokenization - Core Functionality Tests
===================================================

WHY TOKENIZATION MATTERS:
------------------------
Models can't read text - they need numbers. Tokenization:
- Splits text into tokens (words or subwords)
- Maps tokens to integer IDs
- Enables text → numbers conversion

WHAT STUDENTS LEARN:
-------------------
1. Vocabulary: mapping token ↔ ID
2. Subword tokenization (BPE): handle unknown words
3. Special tokens: [CLS], [SEP], [PAD]
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tokenization import CharTokenizer, create_tokenizer, tokenize_dataset


class TestTokenizerBasics:
    """Test basic tokenization functionality."""

    def test_tokenizer_encode(self):
        """
        WHAT: Verify tokenizer converts text to IDs.

        WHY: encode("hello world") should give [id1, id2]
        where id1 and id2 are integers.

        STUDENT LEARNING: Each token gets a unique integer ID.
        "hello" might be 156, "world" might be 234.
        """
        # Build vocab from test text
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])

        text = "hello world"
        token_ids = tokenizer.encode(text)

        assert isinstance(token_ids, (list, np.ndarray)), (
            "encode() should return list or array of IDs"
        )
        assert all(isinstance(id, (int, np.integer)) for id in token_ids), (
            "Token IDs should be integers"
        )

    def test_tokenizer_decode(self):
        """
        WHAT: Verify tokenizer converts IDs back to text.

        WHY: decode(encode(text)) should give back something close
        to the original text.

        STUDENT LEARNING: Tokenization should be (mostly) reversible.
        Some normalization may occur (case, whitespace).
        """
        # Build vocab from test text
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])

        text = "hello world"
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        assert "hello" in decoded.lower() and "world" in decoded.lower(), (
            f"decode(encode(text)) should recover the text.\n"
            f"  Original: '{text}'\n"
            f"  Recovered: '{decoded}'"
        )

    def test_vocabulary_size(self):
        """
        WHAT: Verify tokenizer has a defined vocabulary.

        WHY: Vocabulary size determines embedding table size.
        GPT-2: ~50k tokens, LLaMA: ~32k tokens.

        STUDENT LEARNING: Larger vocab = more precise tokens but
        larger embedding matrix. Trade-off!
        """
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])

        vocab_size = tokenizer.vocab_size
        assert isinstance(vocab_size, int) and vocab_size > 0, (
            "Tokenizer should have positive vocab_size"
        )


class TestTokenizeDataset:
    """Test tokenize_dataset behavior around max_length."""

    def test_no_truncation_when_max_length_omitted(self):
        """
        WHAT: With no max_length argument, long sequences are preserved in full.

        WHY: Truncation should only happen when the caller explicitly asks
        for it. Silent truncation would drop data users didn't opt into
        losing.
        """
        tokenizer = CharTokenizer()
        long_text = "abcdefghij" * 20  # 200 characters
        tokenizer.build_vocab([long_text])

        full_length = len(tokenizer.encode(long_text))
        tokenized = tokenize_dataset([long_text], tokenizer)

        assert len(tokenized[0]) == full_length, (
            "tokenize_dataset without max_length should not truncate the sequence"
        )


class TestCreateTokenizer:
    """Test create_tokenizer factory validation."""

    def test_unsupported_strategy_raises_value_error(self):
        """WHAT: An unsupported strategy string raises ValueError."""
        with pytest.raises(ValueError):
            create_tokenizer("word", corpus=["hello world"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
