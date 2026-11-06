"""Independent tensor counts and composition invariants for shared accounting."""

import pytest

from mlsysim.core.units import Q_
from mlsysim.physics.agents import calc_kv_cache_bytes_per_token
from mlsysim.physics.memory import calc_decoder_matmul_weights, calc_paged_kv_cache_size
from mlsysim.physics.transformer import (
    calc_attention_matmul_flops,
    calc_transformer_prefill_flops,
)


def test_attention_counts_both_rectangular_matrix_products():
    # Construct both matrix products directly: each output is a dot product,
    # whose convention counts one multiply and one add per reduction element.
    layers, width, queries, keys, batch = 2, 5, 3, 7, 2
    score_ops = sum(
        2
        for _b in range(batch)
        for _l in range(layers)
        for _q in range(queries)
        for _k in range(keys)
        for _d in range(width)
    )
    value_ops = sum(
        2
        for _b in range(batch)
        for _l in range(layers)
        for _q in range(queries)
        for _d in range(width)
        for _k in range(keys)
    )
    work = calc_attention_matmul_flops(layers, width, queries, keys, batch)
    assert work.m_as("flop") == score_ops + value_ops


def test_fixed_context_query_chunks_conserve_attention_work():
    # Only the query partition changes. Every chunk visits the same final
    # context, matching the serving solver's explicit upper-bound convention.
    chunks = [3, 3, 1]
    whole = calc_attention_matmul_flops(4, 32, sum(chunks), 128, 2)
    pieces = sum((calc_attention_matmul_flops(4, 32, q, 128, 2) for q in chunks), Q_(0, "flop"))
    assert pieces == whole


def test_causal_prefill_keeps_its_distinct_half_square_convention():
    params, seq, layers, width = 100, 8, 2, 16
    causal = calc_transformer_prefill_flops(params, seq, layers, width, causal=True)
    full = calc_transformer_prefill_flops(params, seq, layers, width, causal=False)
    # The public prefill function approximates a triangular matrix by half
    # the square. This differs intentionally from rectangular serving work.
    projection = 2 * params * seq
    causal_attention = 2 * layers * width * seq**2
    assert causal.m_as("flop") == projection + causal_attention
    assert full.m_as("flop") == projection + 2 * causal_attention


@pytest.mark.parametrize("tokens", [0, 1, 7, 8, 9, 23])
@pytest.mark.parametrize("bytes_per_element", [Q_(1, "byte"), Q_(16, "bit")])
def test_paged_kv_counts_allocated_tensor_slots(tokens, bytes_per_element):
    layers, heads, head_dim, batch, page = 2, 3, 4, 2, 8
    allocated = []
    while len(allocated) < tokens:
        allocated.extend([None] * page)
    # K and V each hold [batch, layer, kv_head, allocated_token, head_dim].
    entries = sum(
        1
        for _kind in ("K", "V")
        for _b in range(batch)
        for _l in range(layers)
        for _h in range(heads)
        for _token in allocated
        for _d in range(head_dim)
    )
    size, waste = calc_paged_kv_cache_size(
        layers,
        heads,
        head_dim,
        tokens,
        batch,
        page,
        bytes_per_element,
    )
    assert size == entries * bytes_per_element
    expected_waste = (len(allocated) - tokens) / len(allocated) if allocated else 0
    assert waste == pytest.approx(expected_waste)


@pytest.mark.parametrize("heads", [1, 4, 8])
def test_agent_kv_token_count_uses_key_value_heads(heads):
    layers, width = 3, 8
    actual = calc_kv_cache_bytes_per_token(
        n_layers=layers,
        n_kv_heads=heads,
        head_dim=width,
        bytes_per_elem=Q_(16, "bit"),
    )
    elements = sum(1 for _kind in ("K", "V") for _l in range(layers) for _h in range(heads) for _d in range(width))
    assert actual == elements * Q_(16, "bit")


def test_decoder_storage_counts_named_weight_matrix_shapes():
    layers, width, vocab, ratio = 3, 16, 32, 4
    shapes = [(width, 3 * width), (width, width), (width, ratio * width), (ratio * width, width)]
    expected_block = sum(rows * columns for rows, columns in shapes)
    weights = calc_decoder_matmul_weights(layers, width, vocab, ratio)
    assert weights == {
        "block": layers * expected_block,
        "block_matrices": layers * len(shapes),
        "output": vocab * width,
    }
