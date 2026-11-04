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
from ...physics.context import calc_observation_budget
from ...physics.retrieval import (
    calc_bm25_idf,
    calc_bm25_length_normalizer,
    calc_bm25_term_score,
    calc_full_tree_nodes,
    calc_records_that_fit,
    calc_vectors_in_budget,
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

    def solve(
        self,
        window: int,
        output_reserve: int,
        other_input: int,
        envelope: int,
        body: int,
        metadata: int,
        metadata_scale: Optional[int] = None,
    ) -> RetrievalPayloadResult:
        """Fit serialized retrieval records into one model-call allowance.

        Parameters
        ----------
        window : int
            Positive context-window token count.
        output_reserve : int
            Nonnegative tokens reserved for the generated answer.
        other_input : int
            Nonnegative tokens occupied by other input roles.
        envelope : int
            Nonnegative tokens for shared serialization structure; must fit the remaining allowance.
        body : int
            Positive body tokens per record.
        metadata : int
            Nonnegative metadata tokens per record.
        metadata_scale : int, optional
            Nonnegative metadata multiplier for a counterfactual; None omits it.

        Returns
        -------
        RetrievalPayloadResult
            Integer token counts, whole-record capacity, and overshoot of body-only accounting.

        See Also
        --------
        mlsysim.physics.context.calc_observation_budget : Context left after answer and other-input reserves.
        mlsysim.physics.retrieval.calc_records_that_fit : Whole records admitted after fixed serialization overhead.

        Notes
        -----
        Equation: allowance = window-output_reserve-other_input; records = floor((allowance-envelope)/(body+metadata)).

        Assumptions: Fixed token counts from a caller-selected tokenizer and identical record sizes. Inputs leave room for the shared envelope.

        Excludes: Tokenization, variable record lengths, relevance ranking, and clamping overcommitted envelopes to zero records.

        References
        ----------
        Derivation: reserve fixed context terms, then divide the remaining integer
        token budget by the serialized record size.

        Examples
        --------
        >>> result = RetrievalPayloadModel().solve(100, 20, 10, 5, 10, 5)
        >>> result.used <= result.allowance and result.body_only_used > result.allowance
        True
        """
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
            allowance=allowance,
            record=record,
            records=records,
            used=used,
            remaining=allowance - used,
            body_only=body_only,
            body_only_used=body_only_used,
            overshoot=body_only_used - allowance,
            overcount=body_only - records,
            scaled_metadata=scaled,
            scaled_metadata_records=scaled_records,
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

    def solve(
        self,
        documents: int,
        matching_documents: int,
        average_length: float,
        short_length: float,
        short_frequency: float,
        long_length: float,
        long_frequency: float,
        k1: float,
        b: float,
    ) -> Bm25PairResult:
        """Compare one query term in two documents under the same BM25 parameters.

        Parameters
        ----------
        documents : int
            Positive corpus document count N.
        matching_documents : int
            Documents containing the term, between zero and documents.
        average_length : float
            Positive mean document length in analyzer terms.
        short_length : float
            Positive analyzer-term count in the first document.
        short_frequency : float
            Nonnegative occurrences of the query term in the first document.
        long_length : float
            Positive analyzer-term count in the second document.
        long_frequency : float
            Nonnegative occurrences of the query term in the second document.
        k1 : float
            Positive BM25 term-frequency saturation parameter.
        b : float
            Length-normalization strength in [0, 1].

        Returns
        -------
        Bm25PairResult
            Dimensionless shared inverse document frequency, length normalizers, and term scores.

        See Also
        --------
        mlsysim.physics.retrieval.calc_bm25_term_score : One-term BM25 score under the selected normalizer.

        Notes
        -----
        Equation: score = ``IDF*f*(k1+1)/(f+k1*(1-b+b*length/average_length))``; IDF = log(1+(N-n+0.5)/(n+0.5)).

        Assumptions: Both documents use one corpus and analyzer. The nonnegative IDF variant is an implementation convention.

        Excludes: Multiple query terms, tokenizer differences, learned relevance, and a probability interpretation of scores.

        References
        ----------
        Robertson and Zaragoza (2009), The Probabilistic Relevance Framework:
        BM25 and Beyond. https://doi.org/10.1561/1500000019
        This solver uses the stated nonnegative IDF variant.

        Examples
        --------
        >>> result = Bm25PairModel().solve(100, 10, 50, 25, 2, 100, 2, 1.2, 0.75)
        >>> result.short_score > result.long_score
        True
        """
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

    def solve(
        self,
        host,
        vectors: int,
        dimension: int,
        bytes_per_value,
        searches: Optional[int] = None,
        filtered_vectors: Optional[int] = None,
        budget=None,
    ) -> ExactScanFloorResult:
        """Compute separate bandwidth and arithmetic floors for an exact vector scan.

        Parameters
        ----------
        host : HardwareNode
            Hardware with positive memory bandwidth in bytes/s and compute peak in flop/s.
        vectors : int
            Positive number of eligible corpus vectors.
        dimension : int
            Positive values per vector.
        bytes_per_value : Quantity
            Positive storage bytes per scalar value.
        searches : int, optional
            Nonnegative sequential scans for cumulative stream time; None omits it.
        filtered_vectors : int, optional
            Nonnegative vector count after filtering; None omits the filtered stream floor.
        budget : Quantity, optional
            Nonnegative stream-time allowance in seconds; None omits the vector-budget calculation.

        Returns
        -------
        ExactScanFloorResult
            Storage bytes, FLOPs, FLOP/byte intensity, separate times in milliseconds, and optional aggregate time in seconds.

        Notes
        -----
        Equation: bytes = ``N*d*b``; work = ``2*N*d``; stream_time = bytes/BW; math_time = work/peak.

        Assumptions: Every eligible dense vector is read once, a dot product uses two FLOPs per value, and peak bandwidth/compute rates apply separately.

        Excludes: Query encoding, launch overhead, cache reuse, result fetches, and a claim that stream_time alone predicts total latency.

        References
        ----------
        Derivation: count dense dot-product operations and compulsory input bytes;
        divide each resource demand by its peak service rate.

        Examples
        --------
        >>> from mlsysim import Hardware, Q_
        >>> result = ExactScanFloorModel().solve(Hardware.Cloud.H100, 100, 16, Q_("2 byte"))
        >>> result.flops.magnitude == 2 * 100 * 16
        True
        """
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
            filtered_time=(
                None
                if filtered_vectors is None
                else (filtered_vectors * dimension * bytes_per_value / bw).to(millisecond)
            ),
            budget_vectors=(None if budget is None else calc_vectors_in_budget(budget, bw, dimension, bytes_per_value)),
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

    def solve(
        self,
        vectors: int,
        dimension: int,
        bytes_per_value,
        links_per_node: int,
        link_bytes,
        subvectors: int,
        centroids: int,
        code_bytes,
        table_entry_bytes,
    ) -> AnnFootprintResult:
        """Compare graph-index and product-quantized storage under stated layouts.

        Parameters
        ----------
        vectors : int
            Positive indexed vector count.
        dimension : int
            Positive values per vector; divisible by subvectors for equal subspaces.
        bytes_per_value : Quantity
            Positive bytes per uncompressed scalar.
        links_per_node : int
            Nonnegative stored graph links per vector.
        link_bytes : Quantity
            Nonnegative bytes per stored graph link.
        subvectors : int
            Positive codes per quantized vector.
        centroids : int
            Positive centroid count per subspace.
        code_bytes : Quantity
            Positive bytes per stored subspace code.
        table_entry_bytes : Quantity
            Positive bytes per query-distance lookup entry.

        Returns
        -------
        AnnFootprintResult
            Raw, graph, and code storage; lookup-table KB; dimensionless compression and integer subspace width.

        Notes
        -----
        Equation: raw=``N*d*b``; graph=``N*M*link_bytes``; codes=``N*m*code_bytes``; lookup=``m*centroids*table_entry_bytes``.

        Assumptions: Fixed-degree graph representation and equal-width product-quantization codes. Subspace width uses integer division.

        Excludes: Index headers, graph hierarchy, allocator overhead, codebooks, recall, and query runtime.

        References
        ----------
        Derivation: each layout is a sum of fixed-width fields multiplied by its
        record count. This storage accounting does not model an ANN search algorithm.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> result = AnnFootprintModel().solve(100, 16, Q_("4 byte"), 4, Q_("4 byte"), 4, 16, Q_("1 byte"), Q_("4 byte"))
        >>> result.graph_total > result.raw > result.codes
        True
        """
        raw = vectors * dimension * bytes_per_value
        graph = vectors * links_per_node * link_bytes
        codes = vectors * subvectors * code_bytes
        return AnnFootprintResult(
            raw=raw,
            graph=graph,
            graph_total=raw + graph,
            codes=codes,
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

    def solve(
        self,
        window: int,
        output_reserve: int,
        other_input: int,
        children: int,
        depth: int,
        node_tokens: int,
        envelope_tokens: int,
    ) -> GraphNeighborhoodResult:
        """Compare an exact tree neighborhood with a serialized context budget.

        Parameters
        ----------
        window : int
            Positive context capacity in tokens.
        output_reserve : int
            Nonnegative answer reserve in tokens.
        other_input : int
            Nonnegative tokens consumed by other input roles.
        children : int
            Nonnegative fixed child count per node.
        depth : int
            Traversal depth of at least one, allowing comparison with the preceding depth.
        node_tokens : int
            Positive serialized tokens per node.
        envelope_tokens : int
            Nonnegative shared edge-envelope tokens; must fit the remaining allowance.

        Returns
        -------
        GraphNeighborhoodResult
            Integer node counts, token demands by depth, and token-budget packing capacity.

        See Also
        --------
        mlsysim.physics.retrieval.calc_full_tree_nodes : Geometric count of nodes through a fixed depth.

        Notes
        -----
        Equation: nodes(d)=sum(children^j for j=0..d); required(d)=``nodes(d)*node_tokens+envelope_tokens``.

        Assumptions: No node is revisited and every node has exactly children descendants. A graph upper-bound interpretation additionally requires children to bound out-degree.

        Excludes: Cycles, deduplication, variable degree, and an upper-bound claim when children is merely an average.

        References
        ----------
        Derivation: sum each tree level as a geometric series, then apply fixed
        serialization bytes in token units to each reached node.

        Examples
        --------
        >>> result = GraphNeighborhoodModel().solve(100, 10, 10, 2, 2, 5, 5)
        >>> result.nodes > result.shallow_nodes and result.packed <= result.allowance
        True
        """
        allowance = calc_observation_budget(window, output_reserve, other_input)
        by_depth = [calc_full_tree_nodes(children, d) for d in range(depth + 1)]
        tokens = [n * node_tokens + envelope_tokens for n in by_depth]
        fit = (allowance - envelope_tokens) // node_tokens
        return GraphNeighborhoodResult(
            allowance=allowance,
            nodes_by_depth=by_depth,
            tokens_by_depth=tokens,
            nodes=by_depth[-1],
            required=tokens[-1],
            fit=fit,
            packed=fit * node_tokens + envelope_tokens,
            shallow_nodes=by_depth[-2],
            shallow_required=tokens[-2],
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

    def solve(
        self,
        files: int,
        chunks_per_file: int,
        parse_time,
        postings_time,
        local_batch_time,
        batch: int,
        remote_request_time,
        exclusion_write_time,
        candidates: int,
        exclusion_check_time,
    ) -> IndexMaintenanceResult:
        """Account for synchronous re-indexing versus exclusion-list maintenance.

        Parameters
        ----------
        files : int
            Positive edited-file count.
        chunks_per_file : int
            Positive embedding chunks produced per file.
        parse_time : Quantity
            Nonnegative parsing duration per file in seconds.
        postings_time : Quantity
            Nonnegative lexical update duration per chunk in seconds.
        local_batch_time : Quantity
            Nonnegative embedding duration per local batch in seconds.
        batch : int
            Positive maximum chunks per embedding batch.
        remote_request_time : Quantity
            Nonnegative remote embedding duration per request in seconds.
        exclusion_write_time : Quantity
            Positive exclusion-write duration per edited file in seconds.
        candidates : int
            Nonnegative retrieved candidates to check against exclusions.
        exclusion_check_time : Quantity
            Nonnegative exclusion lookup duration per candidate in seconds.

        Returns
        -------
        IndexMaintenanceResult
            Integer work counts, milliseconds for batched paths, seconds for per-file remote path, and microseconds for exclusion work.

        Notes
        -----
        Equation: chunks=``files*chunks_per_file``; batches=ceil(chunks/batch); reindex=``files*parse+chunks*postings+batches*embed``; exclusion_write=``files*write``.

        Assumptions: Costs add serially. A remote per-file request embeds all chunks of that file; a batched request follows the supplied batch limit.

        Excludes: Durable commit, concurrent workers, request queueing, and retrieval-quality effects of exclusion.

        References
        ----------
        Derivation: enumerate parsing, postings, and embedding operations on the
        write path; exclusion instead charges a per-file write and per-candidate read.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> t = Q_("1 ms")
        >>> result = IndexMaintenanceModel().solve(2, 3, t, t, t, 4, t, t, 5, t)
        >>> result.chunks == 6 and result.batches == 2
        True
        """
        chunks = files * chunks_per_file
        batches = math.ceil(chunks / batch)
        local = (files * parse_time + chunks * postings_time + batches * local_batch_time).to(millisecond)
        remote = (files * parse_time + chunks * postings_time + batches * remote_request_time).to(millisecond)
        serial = (files * parse_time + chunks * postings_time + files * remote_request_time).to(second)
        write = (files * exclusion_write_time).to(microsecond)
        return IndexMaintenanceResult(
            chunks=chunks,
            batches=batches,
            local_time=local,
            remote_batch_time=remote,
            remote_per_file_time=serial,
            exclusion_write_time=write,
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

    def solve(
        self,
        files: int,
        tokens_per_file: int,
        bytes_per_file,
        postings_per_token,
        edges_per_file: int,
        edge_bytes,
        passage_tokens: int,
        dimension: int,
        bytes_per_value,
        links_per_node: int,
        link_bytes,
        edited_files: int,
        lexical_rate: float,
        parse_rate: float,
        embed_rate: float,
    ) -> StorageAmplificationResult:
        """Compare auxiliary storage and serial re-indexing costs for two layouts.

        Parameters
        ----------
        files : int
            Positive corpus file count.
        tokens_per_file : int
            Positive fixed token count per file.
        bytes_per_file : Quantity
            Positive raw storage bytes per file.
        postings_per_token : Quantity
            Nonnegative posting storage bytes per token.
        edges_per_file : int
            Nonnegative structural graph edges per file.
        edge_bytes : Quantity
            Nonnegative storage bytes per structural edge.
        passage_tokens : int
            Positive tokens per dense passage; incomplete trailing passages are dropped.
        dimension : int
            Positive values per dense vector.
        bytes_per_value : Quantity
            Positive bytes per vector value.
        links_per_node : int
            Nonnegative ANN graph links per passage.
        link_bytes : Quantity
            Nonnegative storage bytes per ANN link.
        edited_files : int
            Positive files to re-index.
        lexical_rate : float
            Positive files per second for lexical re-indexing.
        parse_rate : float
            Positive files per second for structural parsing.
        embed_rate : float
            Positive passages per second for dense embedding.

        Returns
        -------
        StorageAmplificationResult
            Storage quantities, integer passage counts, re-indexing milliseconds, and dimensionless storage/time amplification ratios.

        Notes
        -----
        Equation: postings=``files*tokens_per_file*postings_per_token``; chunks=floor(total_tokens/passage_tokens); graph_layout=postings+edges; dense_layout=postings+vectors+links.

        Assumptions: Fixed per-file sizes; serial lexical-plus-parse or lexical-plus-embedding work; positive auxiliary-storage and time denominators. Passage counts are taken over aggregate tokens.

        Excludes: Per-file boundary padding, overlapping passages, compression, concurrency, and retrieval effectiveness.

        References
        ----------
        Derivation: sum stored field widths for each layout and divide changed
        work counts by the caller-supplied serial service rates.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> b = Q_("1 byte")
        >>> result = StorageAmplificationModel().solve(10, 100, 1000*b, b, 2, 4*b, 50, 16, 4*b, 4, 4*b, 2, 10.0, 5.0, 2.0)
        >>> result.chunks == 10 * 100 // 50
        True
        """
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
            raw=raw,
            postings=postings,
            graph=graph,
            chunks=chunks,
            vectors=vectors,
            vector_links=links,
            aux_lexical_graph=aux_a,
            aux_lexical_dense=aux_b,
            ratio_lexical_graph=(aux_a / raw).to("").magnitude,
            ratio_lexical_dense=(aux_b / raw).to("").magnitude,
            edited_chunks=edited_chunks,
            reindex_lexical_graph=t_a,
            reindex_lexical_dense=t_b,
            bytes_ratio=(aux_b / aux_a).to("").magnitude,
            time_ratio=(t_b / t_a).to("").magnitude,
        )
