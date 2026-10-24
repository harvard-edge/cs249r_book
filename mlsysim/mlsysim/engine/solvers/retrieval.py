"""Retrieval solvers: what a read from long-term memory stages, ranks, streams, and maintains.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each
model is arithmetic on stated counts: whole serialized records in a token
allowance, one BM25 term in two documents, the bandwidth floor of an exact
vector scan, the resident bytes of approximate indexes, an exact tree's
neighborhood against an allowance, the write-path cost of keeping an index
current, the auxiliary bytes and re-index time of two stores, cost per
accepted task with store upkeep, and the words two texts share. None models
a tokenizer, a trained encoder, cache effects, or a real code graph's cycles.
"""

from __future__ import annotations

import math
from typing import Optional

from ..results import (
    AnnFootprintResult,
    Bm25PairResult,
    ExactScanFloorResult,
    GraphNeighborhoodResult,
    IndexMaintenanceResult,
    RetrievalPayloadResult,
    StorageAmplificationResult,
)
from .base import ForwardModel
from ...core.units import KB, byte, flop, millisecond, microsecond, second
from ...physics.agents import calc_cost_per_accepted
from ...physics.context import calc_observation_budget
from ...physics.retrieval import (
    calc_bm25_idf,
    calc_bm25_length_normalizer,
    calc_bm25_term_score,
    calc_full_tree_nodes,
    calc_records_that_fit,
    calc_vectors_in_budget,
    calc_word_overlap,
)


class RetrievalPayloadModel(ForwardModel):
    """
    Whole serialized records in the retrieval allowance of one model call.

    The allowance is what the window leaves after the output reserve and every
    other input role: ``S_max - K_max - S_other``. The envelope is paid once;
    each record pays its body and its metadata, so the count is
    ``floor((allowance - envelope) / (body + metadata))``. A count of bodies
    alone, ``floor(allowance / body)``, overshoots once serialized.
    """
    requires = ()
    produces = RetrievalPayloadResult

    def solve(self, window: int, output_reserve: int, other_input: int, envelope: int, body: int,
              metadata: int, metadata_scale: Optional[int] = None) -> RetrievalPayloadResult:
        allowance = calc_observation_budget(window, output_reserve, other_input)
        record = body + metadata
        records = calc_records_that_fit(allowance, envelope, record)
        used = envelope + records * record
        body_only = allowance // body
        body_only_used = envelope + body_only * record
        scaled = scaled_records = None
        if metadata_scale is not None:
            scaled = metadata * metadata_scale
            scaled_records = calc_records_that_fit(allowance, envelope, body + scaled)
        return RetrievalPayloadResult(
            allowance=allowance, record=record, records=records, used=used,
            remaining=allowance - used, body_only=body_only, body_only_used=body_only_used,
            overshoot=body_only_used - allowance, overcount=body_only - records,
            scaled_metadata=scaled, scaled_metadata_records=scaled_records,
        )


class Bm25PairModel(ForwardModel):
    """
    One query term's BM25 score in a short and a long document of one corpus.

    The same IDF multiplies both; term-frequency saturation (``k1``) and length
    normalization (``b``) decide the order. Lengths and frequencies are
    analyzer-term counts, not model tokens.
    """
    requires = ()
    produces = Bm25PairResult

    def solve(self, documents: int, matching_documents: int, average_length: float,
              short_length: float, short_frequency: float, long_length: float,
              long_frequency: float, k1: float, b: float) -> Bm25PairResult:
        idf = calc_bm25_idf(documents, matching_documents)
        return Bm25PairResult(
            idf=idf,
            short_normalizer=calc_bm25_length_normalizer(short_length, average_length, k1, b),
            long_normalizer=calc_bm25_length_normalizer(long_length, average_length, k1, b),
            short_score=calc_bm25_term_score(idf, short_frequency, short_length, average_length, k1, b),
            long_score=calc_bm25_term_score(idf, long_frequency, long_length, average_length, k1, b),
        )


class ExactScanFloorModel(ForwardModel):
    """
    The time floor of an exact scan: every eligible vector streamed once.

    ``N d b`` bytes cross memory and ``2 N d`` FLOPs score them, so the
    intensity is ``2 / b`` FLOP per byte at any corpus size. Below the host's
    ridge point bandwidth sets the floor, ``t >= N d b / B``. A floor, not a
    forecast: query encoding, record fetches, and caches are left out.
    """
    requires = ()
    produces = ExactScanFloorResult

    def solve(self, host, vectors: int, dimension: int, bytes_per_value, searches: Optional[int] = None,
              filtered_vectors: Optional[int] = None, budget=None) -> ExactScanFloorResult:
        bw = host.memory.bandwidth
        peak = host.compute.peak_flops
        footprint = vectors * dimension * bytes_per_value
        flops = 2 * vectors * dimension * flop
        stream = (footprint / bw).to(millisecond)
        return ExactScanFloorResult(
            footprint=footprint,
            bytes_per_vector=dimension * bytes_per_value,
            flops=flops,
            intensity=(flops / footprint).to(flop / byte),
            ridge=(peak / bw).to(flop / byte),
            stream_time=stream,
            math_time=(flops / peak).to(millisecond),
            trajectory_time=None if searches is None else (searches * stream).to(second),
            filtered_time=(None if filtered_vectors is None
                           else (filtered_vectors * dimension * bytes_per_value / bw).to(millisecond)),
            budget_vectors=(None if budget is None
                            else calc_vectors_in_budget(budget, bw, dimension, bytes_per_value)),
        )


class AnnFootprintModel(ForwardModel):
    """
    Resident bytes of two approximate indexes over the same vectors.

    A navigable-small-world graph keeps every vector plus ``M`` links per node;
    product quantization replaces each vector with ``m`` codes of one byte and
    scores through a lookup table of ``m x centroids`` entries per query.
    """
    requires = ()
    produces = AnnFootprintResult

    def solve(self, vectors: int, dimension: int, bytes_per_value, links_per_node: int, link_bytes,
              subvectors: int, centroids: int, code_bytes, table_entry_bytes) -> AnnFootprintResult:
        raw = vectors * dimension * bytes_per_value
        graph = vectors * links_per_node * link_bytes
        codes = vectors * subvectors * code_bytes
        return AnnFootprintResult(
            raw=raw, graph=graph, graph_total=raw + graph, codes=codes,
            compression=(raw / codes).to("").magnitude,
            lookup_table=(subvectors * centroids * table_entry_bytes).to(KB),
            sub_dim=dimension // subvectors,
            bytes_per_vector=dimension * bytes_per_value,
            code_bytes_per_vector=subvectors * code_bytes,
        )


class GraphNeighborhoodModel(ForwardModel):
    """
    A typed traversal's neighborhood, as an exact tree, against a token allowance.

    Every node has ``c`` children and no node repeats, so depth ``d`` reaches
    ``(c^(d+1) - 1) / (c - 1)`` nodes; each serializes to a fixed size plus one
    envelope for edges. Real code graphs have cycles and hubs, so the count
    bounds distinct nodes from above.
    """
    requires = ()
    produces = GraphNeighborhoodResult

    def solve(self, window: int, output_reserve: int, other_input: int, children: int, depth: int,
              node_tokens: int, envelope_tokens: int) -> GraphNeighborhoodResult:
        allowance = calc_observation_budget(window, output_reserve, other_input)
        by_depth = [calc_full_tree_nodes(children, d) for d in range(depth + 1)]
        tokens = [n * node_tokens + envelope_tokens for n in by_depth]
        fit = (allowance - envelope_tokens) // node_tokens
        return GraphNeighborhoodResult(
            allowance=allowance, nodes_by_depth=by_depth, tokens_by_depth=tokens,
            nodes=by_depth[-1], required=tokens[-1], fit=fit,
            packed=fit * node_tokens + envelope_tokens,
            shallow_nodes=by_depth[-2], shallow_required=tokens[-2],
        )


class IndexMaintenanceModel(ForwardModel):
    """
    Who pays to keep an index current after an edit.

    Synchronous re-indexing parses each changed file, updates postings per
    chunk, and embeds the chunks (in local batches, one remote request per
    batch, or one remote request per file) before the write returns.
    Exclusion records one entry per changed file and adds one lookup per
    candidate at query time. Durable commit and coordination, which both
    strategies share, are excluded.
    """
    requires = ()
    produces = IndexMaintenanceResult

    def solve(self, files: int, chunks_per_file: int, parse_time, postings_time, local_batch_time,
              batch: int, remote_request_time, exclusion_write_time, candidates: int,
              exclusion_check_time) -> IndexMaintenanceResult:
        chunks = files * chunks_per_file
        batches = math.ceil(chunks / batch)
        local = (files * parse_time + chunks * postings_time + batches * local_batch_time).to(millisecond)
        remote = (files * parse_time + chunks * postings_time + batches * remote_request_time).to(millisecond)
        serial = (files * parse_time + chunks * postings_time + files * remote_request_time).to(second)
        write = (files * exclusion_write_time).to(microsecond)
        return IndexMaintenanceResult(
            chunks=chunks, batches=batches, local_time=local, remote_batch_time=remote,
            remote_per_file_time=serial, exclusion_write_time=write,
            exclusion_check_time=(candidates * exclusion_check_time).to(microsecond),
            write_path_ratio=(local / write).to("").magnitude,
        )


class StorageAmplificationModel(ForwardModel):
    """
    Auxiliary bytes and re-index time of a lexical-plus-graph store and a
    lexical-plus-dense store over the same files.

    Both keep postings. The first adds structural edges per file and re-parses
    edited files; the second chunks the files, stores one vector and one set of
    graph-index links per chunk, and re-embeds edited chunks.
    """
    requires = ()
    produces = StorageAmplificationResult

    def solve(self, files: int, tokens_per_file: int, bytes_per_file, postings_per_token,
              edges_per_file: int, edge_bytes, passage_tokens: int, dimension: int, bytes_per_value,
              links_per_node: int, link_bytes, edited_files: int, lexical_rate: float,
              parse_rate: float, embed_rate: float) -> StorageAmplificationResult:
        tokens = files * tokens_per_file
        raw = files * bytes_per_file
        postings = tokens * postings_per_token
        graph = files * edges_per_file * edge_bytes
        chunks = tokens // passage_tokens
        vectors = chunks * dimension * bytes_per_value
        links = chunks * links_per_node * link_bytes
        aux_a = postings + graph
        aux_b = postings + vectors + links
        edited_chunks = edited_files * tokens_per_file // passage_tokens
        t_a = ((edited_files / lexical_rate + edited_files / parse_rate) * second).to(millisecond)
        t_b = ((edited_files / lexical_rate + edited_chunks / embed_rate) * second).to(millisecond)
        return StorageAmplificationResult(
            raw=raw, postings=postings, graph=graph, chunks=chunks, vectors=vectors,
            vector_links=links, aux_lexical_graph=aux_a, aux_lexical_dense=aux_b,
            ratio_lexical_graph=(aux_a / raw).to("").magnitude,
            ratio_lexical_dense=(aux_b / raw).to("").magnitude,
            edited_chunks=edited_chunks, reindex_lexical_graph=t_a, reindex_lexical_dense=t_b,
            bytes_ratio=(aux_b / aux_a).to("").magnitude,
            time_ratio=(t_b / t_a).to("").magnitude,
        )




