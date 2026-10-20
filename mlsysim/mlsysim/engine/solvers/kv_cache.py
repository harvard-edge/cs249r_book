"""KV cache management solvers: what retained attention state costs a serving node.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each
model is arithmetic on a model's KV geometry and a node's registry specs:
the per-token payload and the pool left after the weights, the bytes per
token each attention design keeps, full-cap reservation against paged
blocks, the prefill a prefix hit avoids, chunked prefill against a
monolithic one, keeping, moving, or dropping a paused request's state, and
block-rounded resident fit turned into a turn rate by Little's law. None
models attention traffic during a pass, link contention, or an engine's
scheduler; every rate is a memory-only or weight-only bound.
"""

from __future__ import annotations

from typing import Dict, Sequence

from ..results import (
    ChunkedPrefillResult,
    KvAttentionDesignResult,
    KvBlockSharingResult,
    KvResidencyArchetypeResult,
    PagedAllocationResult,
    PausedKvStateResult,
    PrefixRebuildResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...core.units import resolve_precision, ureg as _ureg
from ...physics.memory import (
    calc_expected_paged_kv_tokens,
    calc_kv_cache_size,
    calc_kv_attention_design,
    calc_kv_block_count,
    calc_kv_resident_fit,
    calc_shared_prefix_fit,
    calc_sharing_gain_limit,
)
from ...physics.serving import calc_littles_law_rate, calc_residency_factor
from ...physics.transformer import (
    calc_transformer_decode_flops,
    calc_transformer_prefill_flops,
    calc_weight_only_step_time,
)


def _precision(precision):
    return resolve_precision(precision)[1] if isinstance(precision, str) else precision


def _per_token(workload, precision) -> Quantity:
    return workload.get_kv_cache_size(seq_len=1, batch_size=1, precision=_precision(precision))


def _ratio(a, b) -> float:
    return (a / b).to("").magnitude


# Copied verbatim from the v2-10 inference branch (engine/solvers/serving_scale.py);
# on merge keep one definition.
class KvAttentionDesignModel(ForwardModel):
    """
    Bytes each token keeps under multi-head, grouped-query, and latent attention.

    The design is fixed in pretraining: MHA stores a key and value per query
    head, GQA shares each stored head among a group, and MLA stores one
    compressed latent per layer. Parameter count does not enter.
    """
    requires = ()
    produces = KvAttentionDesignResult

    def solve(self, models: Sequence, mha, gqa, mla, precision: str,
              session_tokens: int) -> KvAttentionDesignResult:
        per = {m.name: _per_token(m, precision) for m in models}
        design = {m.name: calc_kv_attention_design(m.heads, m.kv_heads, m.kv_lora_rank) for m in models}
        return KvAttentionDesignResult(
            per_token=per,
            design=design,
            mha_over_mla=_ratio(per[mha.name], per[mla.name]),
            gqa_over_mla=_ratio(per[gqa.name], per[mla.name]),
            gqa_session=per[gqa.name] * session_tokens,
            mla_session=per[mla.name] * session_tokens,
        )


class PagedAllocationModel(ForwardModel):
    """
    Full-cap reservation against fixed-size blocks allocated on demand.

    A reservation holds ``S_max`` slots from admission; a request ending at a
    mean of ``S_mean`` tokens fills ``S_mean / S_max`` of it. Paged blocks
    waste at most the unused tail of the last block; under capped
    exponential final lengths (a modeling assumption,
    ``calc_expected_paged_kv_tokens``) the expected allocation sets a
    final-length snapshot count, not a live admission capacity.
    """
    requires = ()
    produces = PagedAllocationResult

    def solve(self, bytes_per_token: Quantity, pool: Quantity, cap: int, mean_final: int,
              small_block: int, large_block: int) -> PagedAllocationResult:
        reservation = cap * bytes_per_token
        reservations = calc_kv_resident_fit(pool, reservation)
        small_alloc, small_waste = calc_expected_paged_kv_tokens(mean_final, cap, small_block)
        large_alloc, large_waste = calc_expected_paged_kv_tokens(mean_final, cap, large_block)
        paged_fit = calc_kv_resident_fit(pool, small_alloc * bytes_per_token)
        small_tail, large_tail = small_block - 1, large_block - 1
        return PagedAllocationResult(
            reservation=reservation,
            reservations=reservations,
            final_length_efficiency=mean_final / cap,
            small_block=small_block * bytes_per_token,
            large_block=large_block * bytes_per_token,
            small_tail_tokens=small_tail,
            large_tail_tokens=large_tail,
            small_tail=small_tail * bytes_per_token,
            large_tail=large_tail * bytes_per_token,
            large_tail_share=large_tail / cap,
            small_allocated_tokens=small_alloc,
            small_waste=small_waste,
            large_allocated_tokens=large_alloc,
            large_waste=large_waste,
            paged_fit=paged_fit,
            paged_gain=paged_fit / reservations,
        )


class PrefixRebuildModel(ForwardModel):
    """
    The prefill a prefix-cache hit avoids, against one short suffix pass.

        T_rebuild = (2 P S + 2 L d S^2) / R,   R = n R_peak u
        T_hit = max(W / (n B_mem), 2 P S_suffix / R)

    A miss rebuilds the prefix's linear and causal-attention work at the
    sustained rate; a hit runs one pass over the suffix, bounded by reading
    the weights once when the suffix is short.
    """
    requires = ()
    produces = PrefixRebuildResult

    def solve(self, workload, node, precision: str, prefix_tokens: int, suffix_tokens: int,
              utilization: float) -> PrefixRebuildResult:
        n_params = workload.parameters.to("count").magnitude
        linear = calc_transformer_prefill_flops(workload.parameters, prefix_tokens)
        total = calc_transformer_prefill_flops(workload.parameters, prefix_tokens,
                                               workload.layers, workload.hidden_dim)
        rate = node.accelerators_per_node * node.accelerator.compute.peak_flops * utilization
        weights = workload.size_in_bytes(_precision(precision))
        node_bw = node.accelerators_per_node * node.accelerator.memory.bandwidth
        t_read = (weights / node_bw).to("ms")
        t_rebuild = (total / rate).to("ms")
        t_suffix = (calc_transformer_decode_flops(workload.parameters, suffix_tokens) / rate).to("ms")
        return PrefixRebuildResult(
            n_params=n_params,
            state=prefix_tokens * _per_token(workload, precision),
            linear_ops=linear,
            attention_ops=total - linear,
            sustained_rate=rate,
            rebuild_time=t_rebuild,
            weight_read_time=t_read,
            suffix_ops_time=t_suffix,
            rebuild_over_hit=_ratio(t_rebuild, max(t_read, t_suffix)),
        )


class ChunkedPrefillModel(ForwardModel):
    """
    The decode gap a new prompt causes, run whole or in chunks.

        T(N) = max(T_weights, 2 P N / R)

    Run whole, the prompt sits between two decode steps, so every ongoing
    stream waits ``T(decodes) + T(prompt)``. Chunked, each iteration runs one
    chunk beside the decodes, so the gap is ``T(chunk + decodes)`` and the
    prompt finishes after ``prompt / chunk`` iterations. Weight work only.
    """
    requires = ()
    produces = ChunkedPrefillResult

    def solve(self, n_params: Quantity, sustained_rate: Quantity, weight_read_time: Quantity,
              decodes: int, prompt_tokens: int, chunk_tokens: int) -> ChunkedPrefillResult:
        def step(n):
            return calc_weight_only_step_time(n_params, n, sustained_rate, weight_read_time)

        t_decode, t_mono = step(decodes), step(prompt_tokens)
        gap = t_decode + t_mono
        chunks = prompt_tokens // chunk_tokens
        t_iter = step(chunk_tokens + decodes)
        return ChunkedPrefillResult(
            decode_step=t_decode,
            monolithic_prefill=t_mono,
            monolithic_gap=gap,
            chunks=chunks,
            chunked_iteration=t_iter,
            chunked_done=chunks * t_iter,
            gap_ratio=_ratio(gap, t_iter),
        )


class PausedKvStateModel(ForwardModel):
    """
    Keep, move, or drop a paused request's KV state during a tool wait.

        T_reload = M / (n B_link),   T_recompute = 2 P S / R

    Both grow linearly with the context, so their ratio does not depend on
    length. Keeping holds ``M x T_wait`` of memory time; moving frees the
    memory for the wait less the copy-out (on demand) or less the round trip
    (prefetch). Correlated returns serialize on the shared links.
    """
    requires = ()
    produces = PausedKvStateResult

    def solve(self, bytes_per_token: Quantity, n_params: Quantity, sustained_rate: Quantity,
              link: Quantity, context_tokens: int, block_tokens: int, wait: Quantity,
              link_fraction: float, returns: int) -> PausedKvStateResult:
        blocks = calc_kv_block_count(context_tokens, block_tokens)
        state = bytes_per_token * (blocks * block_tokens)
        copy_out = (state / link).to("ms")
        copy_in = (state / link).to("ms")
        recompute = (calc_transformer_decode_flops(n_params, context_tokens) / sustained_rate).to("second")
        ratio = _ratio(recompute, copy_in)
        burst = returns * state
        burst_time = (burst / link).to("second")
        return PausedKvStateResult(
            blocks=blocks,
            state=state,
            link=link,
            copy_out=copy_out,
            copy_in=copy_in,
            round_trip=copy_out + copy_in,
            recompute=recompute,
            recompute_over_reload=ratio,
            recompute_over_reload_derated=ratio * link_fraction,
            held_occupancy=state * wait,
            released_on_demand=wait - copy_out,
            released_prefetch=wait - copy_out - copy_in,
            burst=burst,
            burst_time=burst_time,
            burst_slowdown=_ratio(burst_time, copy_in),
        )


class KvBlockSharingModel(ForwardModel):
    """
    Block-rounded resident fit with and without a shared prefix, as a turn rate.

        B = floor(N_blocks / N_req),   B_shared = floor((N_blocks - N_common) / N_private)
        lambda = B / W,   W = T_gen (+ T_wait if the state is kept)

    Each request reserves its input and output allowance in whole blocks; a
    fraction of the pool is withheld as headroom. The gain from sharing
    approaches ``1 / (1 - f)`` in a large pool.
    """
    requires = ()
    produces = KvBlockSharingResult

    def solve(self, bytes_per_token: Quantity, pool: Quantity, headroom: float, context: int,
              output: int, common: int, block_tokens: int, generation: Quantity,
              wait: Quantity) -> KvBlockSharingResult:
        block_bytes = block_tokens * bytes_per_token
        usable_blocks = calc_kv_resident_fit((1 - headroom) * pool, block_bytes)
        unshared = calc_kv_block_count(context + output, block_tokens)
        common_blocks = calc_kv_block_count(common, block_tokens)
        private = calc_kv_block_count(context + output - common, block_tokens)
        fit_unshared = usable_blocks // unshared
        fit_shared = calc_shared_prefix_fit(usable_blocks, common_blocks, private)
        retained = generation + wait
        f = common_blocks / unshared
        return KvBlockSharingResult(
            block_bytes=block_bytes,
            usable_blocks=usable_blocks,
            usable=usable_blocks * block_bytes,
            unshared_blocks=unshared,
            common_blocks=common_blocks,
            private_blocks=private,
            unshared_bytes=unshared * block_bytes,
            common_bytes=common_blocks * block_bytes,
            private_bytes=private * block_bytes,
            fit_unshared=fit_unshared,
            fit_shared=fit_shared,
            retained_time=retained,
            retained_rate=calc_littles_law_rate(fit_unshared, retained),
            released_rate=calc_littles_law_rate(fit_unshared, generation),
            shared_retained_rate=calc_littles_law_rate(fit_shared, retained),
            shared_released_rate=calc_littles_law_rate(fit_shared, generation),
            residency_factor=calc_residency_factor(generation, wait),
            shared_fraction=f,
            share_gain=fit_shared / fit_unshared,
            share_bound=calc_sharing_gain_limit(f),
        )


class KvResidencyArchetypeModel(ForwardModel):
    """
    Sharing gain, residency tax, and resume cost for one agent workload.

    Unrounded resident fit with a shared prefix stored once; the residency
    tax ``1 + T_wait / T_gen``; and the two resume paths for a dropped or
    moved state, recompute ``2 P S / R`` against reload ``S m_token / B_link``.
    """
    requires = ()
    produces = KvResidencyArchetypeResult

    def solve(self, bytes_per_token: Quantity, pool: Quantity, n_params: Quantity,
              sustained_rate: Quantity, link: Quantity, context_tokens: int, shared_prefix: int,
              generation: Quantity, wait: Quantity) -> KvResidencyArchetypeResult:
        def fit(shared):
            return calc_kv_resident_fit(pool - shared * bytes_per_token,
                                        (context_tokens - shared) * bytes_per_token)

        unshared, shared = fit(0), fit(shared_prefix)
        cycle = generation + wait
        recompute = (calc_transformer_decode_flops(n_params, context_tokens) / sustained_rate).to("second")
        return KvResidencyArchetypeResult(
            shared_fraction=shared_prefix / context_tokens,
            fit_unshared=unshared,
            fit_shared=shared,
            share_gain=shared / unshared,
            residency_factor=calc_residency_factor(generation, wait),
            turn_cycle=cycle,
            recompute=recompute,
            reload=(context_tokens * bytes_per_token / link).to("ms"),
            recompute_share=_ratio(recompute, cycle),
            shared_retained_rate=calc_littles_law_rate(shared, cycle),
            shared_released_rate=calc_littles_law_rate(shared, generation),
        )
