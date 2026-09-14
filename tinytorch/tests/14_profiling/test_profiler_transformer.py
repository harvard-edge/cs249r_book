"""Module 14 Profiler on Module 13's GPT: FLOP counting and token-model profiling."""
import numpy as np
import pytest

from tinytorch.core.tensor import Tensor
from tinytorch.core.transformers import GPT
from tinytorch.perf.profiling import Profiler


@pytest.fixture(scope="module")
def tiny_gpt():
    return GPT(vocab_size=1000, embed_dim=128, num_layers=4, num_heads=4, max_seq_len=256)


def test_transformer_flops_match_hand_count(tiny_gpt):
    """count_flops on a GPT counts one sequence: Linear projections, attention products, and the head."""
    D, V, L = 128, 1000, 4
    per_row_linear = 2 * (4 * D * D + 2 * D * 4 * D)   # six Linear layers per block, one row
    head = 2 * D * V
    profiler = Profiler()
    for S in (1, 64):
        expected = L * S * per_row_linear + L * 4 * S * S * D + S * head
        assert profiler.count_flops(tiny_gpt, (1, S)) == expected


def test_transformer_parameters(tiny_gpt):
    assert Profiler().count_parameters(tiny_gpt) == 1_082_112


def test_profile_forward_pass_on_token_model(tiny_gpt):
    """A GPT takes integer ids; the profiler must build a valid dummy input for it."""
    ids = Tensor(np.random.default_rng(0).integers(0, 1000, size=(1, 8)))
    profile = Profiler().profile_forward_pass(tiny_gpt, ids)
    assert profile['parameters'] == 1_082_112
    assert profile['flops'] == Profiler().count_flops(tiny_gpt, (1, 8))
    assert profile['latency_ms'] > 0
    assert profile['bottleneck'] in ('memory', 'compute')
