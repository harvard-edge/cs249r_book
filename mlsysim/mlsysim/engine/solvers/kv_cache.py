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

from typing import Sequence

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
from ...core.units import resolve_precision
from ...physics.memory import (
    calc_expected_paged_kv_tokens,
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
        """Compare retained KV bytes across attention designs.

        Parameters
        ----------
        models : Sequence
            Sequence of workloads whose names identify the comparison.
        mha : object
            MHA comparison workload included in models.
        gqa : object
            GQA comparison workload included in models.
        mla : object
            MLA comparison workload included in models.
        precision : str
            Weight or KV precision name; gradient synchronization also accepts bytes per element.
        session_tokens : int
            Nonnegative session length in tokens.

        Returns
        -------
        KvAttentionDesignResult
            Bytes per token and attention-design labels keyed by workload name,
            dimensionless comparison ratios, and session-state byte quantities.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: session_bytes = per_token_bytes * session_tokens; ratios divide MHA or GQA per-token bytes by MLA bytes.

        Assumptions: Named comparison models are present in models and their KV geometry is defined.

        Excludes: Weights, activations, allocator reserve, and attention execution time.

        References
        ----------
        Derivation: session_bytes = per_token_bytes * session_tokens; ratios divide MHA or GQA per-token bytes by MLA bytes. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import KvAttentionDesignModel
        >>> m = Models.Language.Llama3_8B
        >>> KvAttentionDesignModel().solve([m], m, m, m, 'fp16', 8).mha_over_mla == 1
        True
        """
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
        """Compare full-context reservation with expected block allocation.

        Parameters
        ----------
        bytes_per_token : Quantity
            Positive bytes retained per token across all layers.
        pool : Quantity
            Positive bytes available for KV state.
        cap : int
            Positive maximum request length in tokens.
        mean_final : int
            Positive mean final token count below the cap.
        small_block : int
            Positive smaller allocation block size in tokens.
        large_block : int
            Positive larger allocation block size in tokens.

        Returns
        -------
        PagedAllocationResult
            Reserved bytes and request counts, expected allocated token counts,
            wasted fractions, block-tail bytes, and dimensionless residency gain.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: reservation = cap * bytes_per_token; fit = floor(pool / allocation_bytes); efficiency = mean_final / cap.

        Assumptions: Capped exponential final lengths and positive blocks; pool holds at least one full reservation.

        Excludes: Live scheduler admission, fragmentation between requests, and transient growth.

        References
        ----------
        Derivation: reservation = cap * bytes_per_token; fit = floor(pool / allocation_bytes); efficiency = mean_final / cap. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import PagedAllocationModel
        >>> PagedAllocationModel().solve(Q_('1 KB'), Q_('1 GB'), 128, 32, 8, 16).reservations > 0
        True
        """
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
        """Compare rebuilding a cached prefix with a suffix-only pass.

        Parameters
        ----------
        workload : object
            Transformer workload with parameter and KV geometry.
        node : object
            System node containing accelerator count and specifications.
        precision : str
            Weight or KV precision name; gradient synchronization also accepts bytes per element.
        prefix_tokens : int
            Positive cached-prefix token count.
        suffix_tokens : int
            Positive suffix token count.
        utilization : float
            Compute utilization fraction above zero and at most one.

        Returns
        -------
        PrefixRebuildResult
            Prefix-state bytes, linear and attention FLOPs, sustained FLOP/s,
            rebuild/read/suffix milliseconds, and the rebuild-to-hit time ratio.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: rebuild = (2PS + 2LdS^2) / rate; hit = max(weights / aggregate_bandwidth, ``2P*suffix`` / rate).

        Assumptions: Aggregate device bandwidth and compute add ideally; utilization is a declared fraction.

        Excludes: Inter-device collectives, suffix attention, scheduler overhead, and capacity validation.

        References
        ----------
        Derivation: rebuild = (2PS + 2LdS^2) / rate; hit = max(weights / aggregate_bandwidth, ``2P*suffix`` / rate). This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import PrefixRebuildModel
        >>> from mlsysim import Systems
        >>> PrefixRebuildModel().solve(Models.Language.Llama3_8B, Systems.Nodes.DGX_H100, 'fp16', 128, 1, 0.5).rebuild_time.magnitude > 0
        True
        """
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
        """Compare monolithic and chunked weight-only prefill intervals.

        Parameters
        ----------
        n_params : Quantity
            Positive model parameter count in param.
        sustained_rate : Quantity
            Positive declared sustained compute rate in FLOP/s.
        weight_read_time : Quantity
            Nonnegative time to read model weights once.
        decodes : int
            Nonnegative active decode tokens per iteration.
        prompt_tokens : int
            Positive prompt length, evenly divisible by chunk_tokens.
        chunk_tokens : int
            Positive prefill chunk length in tokens.

        Returns
        -------
        ChunkedPrefillResult
            Decode and prefill time quantities, complete chunk count, total
            chunked duration, and monolithic-to-chunked gap ratio.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: T(n) = max(weight_read_time, ``2P*n`` / rate); chunks = prompt_tokens // chunk_tokens; done = chunks * T(chunk_tokens+decodes).

        Assumptions: Prompt tokens divide evenly into positive chunks; each iteration shares weight reads.

        Excludes: Attention work, residual partial chunks, dispatch overhead, and scheduling variation.

        References
        ----------
        Derivation: T(n) = max(weight_read_time, ``2P*n`` / rate); chunks = prompt_tokens // chunk_tokens; done = chunks * T(chunk_tokens+decodes). This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import ChunkedPrefillModel
        >>> ChunkedPrefillModel().solve(Q_('1000 param'), Q_('1 GFLOP/s'), Q_('1 ms'), 1, 16, 4).chunks == 4
        True
        """
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
        """Compare keeping, copying, and rebuilding paused KV state.

        Parameters
        ----------
        bytes_per_token : Quantity
            Positive bytes retained per token across all layers.
        n_params : Quantity
            Positive model parameter count in param.
        sustained_rate : Quantity
            Positive declared sustained compute rate in FLOP/s.
        link : Quantity
            Positive aggregate copy bandwidth in byte/s.
        context_tokens : int
            Positive retained context length in tokens.
        block_tokens : int
            Positive allocation block length in tokens.
        wait : Quantity
            Nonnegative paused residence time.
        link_fraction : float
            Positive fraction of peak link rate, at most one.
        returns : int
            Positive number of simultaneous returning requests.

        Returns
        -------
        PausedKvStateResult
            Block count, state bytes, copy milliseconds, recompute seconds,
            byte-time occupancy, released durations, and serialized burst costs.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: state = ceil(context/block_tokens) * block_tokens * bytes_per_token; copy = state/link; recompute = ``2P*context/rate``.

        Assumptions: Whole-block state, equal inbound/outbound bandwidth, and serialized simultaneous returns.

        Excludes: Attention recomputation, copy overlap, contention, and feasibility clipping of negative released time.

        References
        ----------
        Derivation: state = ceil(context/block_tokens) * block_tokens * bytes_per_token; copy = state/link; recompute = ``2P*context/rate``. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import PausedKvStateModel
        >>> PausedKvStateModel().solve(Q_('1 KB'), Q_('1000 param'), Q_('1 GFLOP/s'), Q_('1 GB/s'), 16, 4, Q_('1 s'), 1, 2).blocks == 4
        True
        """
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
        """Turn block-rounded KV residency into a memory-limited turn rate.

        Parameters
        ----------
        bytes_per_token : Quantity
            Positive bytes retained per token across all layers.
        pool : Quantity
            Positive bytes available for KV state.
        headroom : float
            Pool fraction withheld, between zero and one (exclusive of one).
        context : int
            Positive input-context token count.
        output : int
            Nonnegative reserved output-token allowance.
        common : int
            Nonnegative shared prefix shorter than total request length.
        block_tokens : int
            Positive allocation block length in tokens.
        generation : Quantity
            Positive generation duration.
        wait : Quantity
            Nonnegative paused residence time.

        Returns
        -------
        KvBlockSharingResult
            Block-rounded byte counts and resident counts, retained duration,
            turn rates, and dimensionless residency and sharing factors.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: fit = usable_blocks // request_blocks; shared_fit = (usable_blocks-common_blocks) // private_blocks; rate = fit / residence_time.

        Assumptions: Common prefix stored once, positive private state and generation time, and sufficient pool for one request.

        Excludes: Actual serving throughput, scheduler contention, and fragmentation beyond block rounding.

        References
        ----------
        Derivation: fit = usable_blocks // request_blocks; shared_fit = (usable_blocks-common_blocks) // private_blocks; rate = fit / residence_time. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import KvBlockSharingModel
        >>> KvBlockSharingModel().solve(Q_('1 KB'), Q_('1 GB'), 0.1, 16, 8, 4, 4, Q_('1 s'), Q_('1 s')).fit_shared >= 1
        True
        """
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
        """Compare shared-prefix residency and paused-state resume paths.

        Parameters
        ----------
        bytes_per_token : Quantity
            Positive bytes retained per token across all layers.
        pool : Quantity
            Positive bytes available for KV state.
        n_params : Quantity
            Positive model parameter count in param.
        sustained_rate : Quantity
            Positive declared sustained compute rate in FLOP/s.
        link : Quantity
            Positive aggregate copy bandwidth in byte/s.
        context_tokens : int
            Positive retained context length in tokens.
        shared_prefix : int
            Nonnegative shared prefix shorter than context_tokens.
        generation : Quantity
            Positive generation duration.
        wait : Quantity
            Nonnegative paused residence time.

        Returns
        -------
        KvResidencyArchetypeResult
            Resident counts, sharing and residency factors, cycle/recompute
            seconds, reload milliseconds, and retained/released turn rates.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: fit_shared = ``floor((pool-shared_prefix*bytes_per_token)/((context-shared_prefix)*bytes_per_token))``; cycle = generation+wait; recompute = ``2P*context/rate``.

        Assumptions: Shared prefix shorter than context, positive rates, and enough pool for one request.

        Excludes: Block rounding, attention recomputation, contention, and execution-capacity limits.

        References
        ----------
        Derivation: fit_shared = ``floor((pool-shared_prefix*bytes_per_token)/((context-shared_prefix)*bytes_per_token))``; cycle = generation+wait; recompute = ``2P*context/rate``. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import KvResidencyArchetypeModel
        >>> KvResidencyArchetypeModel().solve(Q_('1 KB'), Q_('1 GB'), Q_('1000 param'), Q_('1 GFLOP/s'), Q_('1 GB/s'), 16, 4, Q_('1 s'), Q_('1 s')).fit_shared > 0
        True
        """
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
