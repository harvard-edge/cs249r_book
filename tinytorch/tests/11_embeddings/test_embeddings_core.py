"""
Module 11: Embeddings - Core Functionality Tests
=================================================

WHY EMBEDDINGS MATTER:
---------------------
Embeddings turn discrete IDs into dense vectors:
- Token ID 156 → [0.2, -0.5, 0.8, ...]  (512 dims)
- These vectors capture meaning
- Similar words have similar embeddings

WHAT STUDENTS LEARN:
-------------------
1. Embedding is just a lookup table
2. Embeddings are learned during training
3. Positional encoding adds position information
"""

import numpy as np
rng = np.random.default_rng(7)
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.embeddings import Embedding, PositionalEncoding
from tinytorch.core.tensor import Tensor


class TestEmbeddingLayer:
    """Test Embedding layer functionality."""

    def test_embedding_lookup(self):
        """
        WHAT: Verify embedding maps IDs to vectors.

        WHY: Input [3, 7, 2] should give 3 embedding vectors,
        one for each token ID.

        STUDENT LEARNING: Embedding is just:
        embedding_matrix[token_id] → vector
        """
        vocab_size = 100
        embed_dim = 64

        embed = Embedding(vocab_size, embed_dim)

        # Token IDs
        tokens = Tensor(np.array([3, 7, 2]))

        output = embed(tokens)

        assert output.shape == (3, 64), (
            f"Embedding output shape wrong.\n"
            f"  Input: 3 token IDs\n"
            f"  Expected: (3, 64)\n"
            f"  Got: {output.shape}"
        )

    def test_embedding_batch(self):
        """
        WHAT: Verify embedding handles batched sequences.

        WHY: Training uses batches of sequences.
        (batch, seq_len) → (batch, seq_len, embed_dim)

        STUDENT LEARNING: Embedding adds a dimension.
        Input: (batch, seq_len) of integers
        Output: (batch, seq_len, embed_dim) of floats
        """
        embed = Embedding(vocab_size=100, embed_dim=32)

        # Batch of 4 sequences, each length 10
        tokens = Tensor(rng.integers(0, 100, (4, 10)))

        output = embed(tokens)

        assert output.shape == (4, 10, 32), (
            f"Batched embedding shape wrong.\n"
            f"  Input: (4, 10) token IDs\n"
            f"  Expected: (4, 10, 32)\n"
            f"  Got: {output.shape}"
        )


class TestPositionalEncoding:
    """Test positional encoding."""

    def test_positional_encoding_shape(self):
        """
        WHAT: Verify positional encoding has correct shape.

        WHY: Must match embedding dimensions to be added.

        STUDENT LEARNING: Transformers have no notion of position.
        Positional encoding adds position information:
        final_embedding = token_embedding + position_encoding
        """
        max_len = 100
        embed_dim = 64

        pos_enc = PositionalEncoding(max_len, embed_dim)

        # Sequence of embeddings
        x = Tensor(rng.standard_normal((2, 50, 64)))  # (batch, seq, embed)

        output = pos_enc(x)

        assert output.shape == x.shape, (
            "Positional encoding should preserve shape"
        )


class TestEmbeddingValidation:
    """Test Embedding index validation."""

    def test_index_at_or_above_vocab_size_raises(self):
        """WHAT: An index >= vocab_size raises ValueError."""
        embed = Embedding(vocab_size=10, embed_dim=4)
        with pytest.raises(ValueError):
            embed.forward(Tensor([15]))

    def test_negative_index_raises(self):
        """WHAT: A negative index raises ValueError."""
        embed = Embedding(vocab_size=10, embed_dim=4)
        with pytest.raises(ValueError):
            embed.forward(Tensor([-1]))


class TestPositionalEncodingValidation:
    """Test PositionalEncoding input validation."""

    def test_2d_input_missing_batch_dim_raises(self):
        """WHAT: A 2D input (missing batch dim) raises ValueError."""
        pos_enc = PositionalEncoding(max_seq_len=100, embed_dim=64)
        x = Tensor(rng.standard_normal((50, 64)))
        with pytest.raises(ValueError):
            pos_enc.forward(x)

    def test_seq_len_exceeding_max_raises(self):
        """WHAT: seq_len > max_seq_len raises ValueError."""
        pos_enc = PositionalEncoding(max_seq_len=10, embed_dim=64)
        x = Tensor(rng.standard_normal((2, 20, 64)))
        with pytest.raises(ValueError):
            pos_enc.forward(x)

    def test_wrong_ndim_input_raises(self):
        """WHAT: A non-3D input (e.g. 4D) raises ValueError."""
        pos_enc = PositionalEncoding(max_seq_len=100, embed_dim=64)
        x = Tensor(rng.standard_normal((2, 3, 50, 64)))
        with pytest.raises(ValueError):
            pos_enc.forward(x)

    def test_embed_dim_mismatch_raises(self):
        """WHAT: An embed_dim mismatch between input and config raises ValueError."""
        pos_enc = PositionalEncoding(max_seq_len=100, embed_dim=64)
        x = Tensor(rng.standard_normal((2, 50, 32)))
        with pytest.raises(ValueError):
            pos_enc.forward(x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
