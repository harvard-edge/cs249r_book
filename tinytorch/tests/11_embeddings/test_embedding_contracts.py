"""Token IDs are integer identities; positions address a finite table."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.core.embeddings import Embedding, EmbeddingLayer, PositionalEncoding


@pytest.mark.parametrize('value', [0.5, np.nan, np.inf, -np.inf])
def test_embedding_rejects_noninteger_or_nonfinite_ids(value):
    with pytest.raises(ValueError, match='finite integers'):
        Embedding(3, 2)(Tensor([value]))


@pytest.mark.parametrize('kind', ['learned', 'sinusoidal'])
@pytest.mark.parametrize('start', [-1, 0.5, 3])
def test_position_lookup_rejects_invalid_range(kind, start):
    with pytest.raises(ValueError):
        EmbeddingLayer(4, 2, max_seq_len=4, pos_encoding=kind)(Tensor([[1, 2]]), start_pos=start)


def test_repeated_token_gradient_adds_contributions():
    import tinytorch.core.autograd
    embedding = Embedding(4, 2)
    embedding(Tensor([[1, 1, 2]])).sum().backward()
    np.testing.assert_array_equal(embedding.weight.grad, [[0, 0], [2, 2], [1, 1], [0, 0]])
