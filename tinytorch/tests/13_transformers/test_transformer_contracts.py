"""Small GPT generation and one-axis LayerNorm have explicit contracts."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.core.transformers import GPT, LayerNorm


def test_zero_temperature_is_greedy():
    model = GPT(4, 4, 1, 1, max_seq_len=5)
    assert model._sample_next_token(np.array([[1., 4., 2., 3.]]), 0) == 1
    assert model.generate(Tensor([[1, 2]]), max_new_tokens=2, temperature=0).shape == (1, 4)


@pytest.mark.parametrize('temperature', [-1, np.nan, np.inf])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match='temperature'):
        GPT(4, 4, 1, 1)._sample_next_token(np.ones((1, 4)), temperature)


@pytest.mark.parametrize('prompt,length', [([[1], [2]], 1), ([[]], 1), ([[1]], -1), ([[1, 2]], 4)])
def test_invalid_generation_request_is_rejected(prompt, length):
    with pytest.raises(ValueError):
        GPT(4, 4, 1, 1, max_seq_len=5).generate(Tensor(prompt), length)


def test_layernorm_does_not_broadcast_wrong_normalized_shape():
    with pytest.raises(ValueError):
        LayerNorm(1)(Tensor([[1, 2, 3]]))
    with pytest.raises(ValueError):
        LayerNorm((2, 3))


def test_unsupported_dropout_is_explicit():
    from tinytorch.core.transformers import MLP, TransformerBlock
    assert MLP(4).dropout_prob == 0
    for constructor in [lambda: MLP(4, dropout_prob=0.1),
                        lambda: TransformerBlock(4, 1, dropout_prob=0.1)]:
        with pytest.raises(ValueError, match='dropout_prob=0'):
            constructor()
