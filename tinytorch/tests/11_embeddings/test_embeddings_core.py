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
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
        try:
            from tinytorch.nn import Embedding
            from tinytorch.core.tensor import Tensor

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

        except ImportError:
            pytest.skip("Embedding not implemented yet")

    def test_embedding_batch(self):
        """
        WHAT: Verify embedding handles batched sequences.

        WHY: Training uses batches of sequences.
        (batch, seq_len) → (batch, seq_len, embed_dim)

        STUDENT LEARNING: Embedding adds a dimension.
        Input: (batch, seq_len) of integers
        Output: (batch, seq_len, embed_dim) of floats
        """
        try:
            from tinytorch.nn import Embedding
            from tinytorch.core.tensor import Tensor

            embed = Embedding(vocab_size=100, embed_dim=32)

            # Batch of 4 sequences, each length 10
            tokens = Tensor(np.random.randint(0, 100, (4, 10)))

            output = embed(tokens)

            assert output.shape == (4, 10, 32), (
                f"Batched embedding shape wrong.\n"
                f"  Input: (4, 10) token IDs\n"
                f"  Expected: (4, 10, 32)\n"
                f"  Got: {output.shape}"
            )

        except ImportError:
            pytest.skip("Embedding batch not implemented yet")


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
        try:
            from tinytorch.nn import PositionalEncoding
            from tinytorch.core.tensor import Tensor

            max_len = 100
            embed_dim = 64

            pos_enc = PositionalEncoding(max_len, embed_dim)

            # Sequence of embeddings
            x = Tensor(np.random.randn(2, 50, 64))  # (batch, seq, embed)

            output = pos_enc(x)

            assert output.shape == x.shape, (
                "Positional encoding should preserve shape"
            )

        except ImportError:
            pytest.skip("PositionalEncoding not implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
