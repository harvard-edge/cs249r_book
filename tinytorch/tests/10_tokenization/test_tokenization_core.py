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

from tinytorch.core.tokenization import CharTokenizer


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
