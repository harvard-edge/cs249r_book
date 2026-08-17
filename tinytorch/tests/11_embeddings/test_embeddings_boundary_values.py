"""
Boundary-value tests for Embedding and PositionalEncoding.

Systematic sweep: vocab_size=1 (the smallest valid vocabulary), an
out-of-range token index (must raise a clean error, not crash
confusingly), and embed_dim=1 / max_seq_len=1 for positional encoding
(the smallest valid dimensions, where the sin/cos formula's exponent
degenerates to a single term). None of these were previously exercised.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from tinytorch.core.embeddings import Embedding, PositionalEncoding
from tinytorch.core.tensor import Tensor


class TestEmbeddingBoundaryValues:
    def test_vocab_size_one(self):
        """The smallest valid vocabulary: every token maps to index 0."""
        embed = Embedding(vocab_size=1, embed_dim=4)
        x = Tensor(np.array([[0, 0, 0]]))

        out = embed(x)

        assert out.shape == (1, 3, 4)
        assert np.all(np.isfinite(out.data))

    def test_out_of_range_index_raises_clean_error_not_crash(self):
        """An index >= vocab_size (e.g. from a tokenizer/vocab mismatch)
        must raise a clear ValueError, not an obscure IndexError from
        deep inside a numpy indexing operation."""
        embed = Embedding(vocab_size=5, embed_dim=4)
        x = Tensor(np.array([[5]]))

        with pytest.raises(ValueError, match="out of range"):
            embed(x)

    def test_negative_index_raises_clean_error_not_silent_wraparound(self):
        """A negative index must not silently wrap around to the end of
        the vocabulary via numpy's negative-indexing semantics."""
        embed = Embedding(vocab_size=5, embed_dim=4)
        x = Tensor(np.array([[-1]]))

        with pytest.raises(ValueError):
            embed(x)

    def test_embed_dim_one(self):
        embed = Embedding(vocab_size=10, embed_dim=1)
        x = Tensor(np.array([[0, 5, 9]]))

        out = embed(x)

        assert out.shape == (1, 3, 1)
        assert np.all(np.isfinite(out.data))


class TestPositionalEncodingBoundaryValues:
    def test_embed_dim_one(self):
        """embed_dim=1 is odd, so the sin/cos formula's even/odd index
        pairing degenerates to a single sin-only term."""
        pe = PositionalEncoding(max_seq_len=5, embed_dim=1)
        x = Tensor(np.zeros((1, 5, 1)))

        out = pe(x)

        assert out.shape == (1, 5, 1)
        assert np.all(np.isfinite(out.data))

    def test_max_seq_len_one(self):
        """A single-position sequence (the first token of generation)."""
        pe = PositionalEncoding(max_seq_len=1, embed_dim=8)
        x = Tensor(np.zeros((1, 1, 8)))

        out = pe(x)

        assert out.shape == (1, 1, 8)
        assert np.all(np.isfinite(out.data))
