"""
Retrieval from long-term memory: the equations behind a read and its index.

Pure arithmetic (added 2026-10-20). A read stages whole serialized records
inside a token allowance; a lexical index ranks candidates with BM25; an
exact vector scan streams every eligible vector, so bandwidth sets its floor;
a typed traversal grows with depth; and similarity between two texts counts
shared words, which is all a similarity-first deduplicator sees. The solvers
in engine/solvers/retrieval.py compose these.
"""

import math as _math


def calc_records_that_fit(allowance: int, envelope: int, record: int) -> int:
    """
    Whole serialized records that fit a retrieval allowance after its envelope.

        n = floor((allowance - envelope) / (body + metadata))
    """
    return (allowance - envelope) // record


def calc_bm25_idf(documents: int, matching_documents: int) -> float:
    """
    BM25 inverse document frequency (the non-negative variant).

        IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
    """
    return _math.log(1 + (documents - matching_documents + 0.5) / (matching_documents + 0.5))


def calc_bm25_length_normalizer(length: float, average_length: float, k1: float, b: float) -> float:
    """BM25 length term in the denominator: ``k1 (1 - b + b |D| / avgdl)``."""
    return k1 * (1 - b + b * length / average_length)


def calc_bm25_term_score(idf: float, frequency: float, length: float, average_length: float,
                         k1: float, b: float) -> float:
    """
    One query term's BM25 contribution to a document's score.

        IDF(q) f (k1 + 1) / (f + k1 (1 - b + b |D| / avgdl))
    """
    normalizer = calc_bm25_length_normalizer(length, average_length, k1, b)
    return idf * frequency * (k1 + 1) / (frequency + normalizer)


def calc_full_tree_nodes(children: int, depth: int) -> int:
    """
    Nodes of an exact tree through ``depth`` (the seed is depth zero).

        1 + c + c^2 + ... + c^d = (c^(d+1) - 1) / (c - 1)
    """
    if children == 1:
        return depth + 1
    return (children ** (depth + 1) - 1) // (children - 1)


def calc_vectors_in_budget(budget, bandwidth, dimension: int, bytes_per_value):
    """
    Vectors an exact scan can stream within a time budget: ``t B / (d b)``.

    Quantities in, a dimensionless float out; the scan's arithmetic is ignored
    because it sits far below the host's ridge point.
    """
    return float((budget * bandwidth / (dimension * bytes_per_value)).to("").magnitude)


def calc_bandwidth_fit_through_origin(bytes_streamed, seconds) -> float:
    """
    Least-squares bandwidth of measured scans on the line ``t = bytes / B``.

    Minimizing the squared time residuals over ``1 / B`` gives
    ``B = sum(bytes^2) / sum(bytes t)``, in bytes per second.
    """
    num = sum(b * b for b in bytes_streamed)
    den = sum(b * t for b, t in zip(bytes_streamed, seconds))
    return num / den


def calc_interface_bandwidth(width, transfer_rate):
    """
    Peak bandwidth of a memory interface: ``width x transfers per second``.

    ``width`` in bits and ``transfer_rate`` in bits per second per pin (one bit
    per transfer); returns bytes per second.
    """
    return (width.to("bit").magnitude * transfer_rate).to("byte / second")


def _words(text: str):
    return {w.strip(".,;:!?\"'()").lower() for w in text.split()} - {""}


def calc_word_overlap(first: str, second: str):
    """
    Distinct words two texts share, and the union, after lowercasing and
    stripping punctuation. Returns ``(shared, union, words_in_first,
    words_in_second)``; ``shared / union`` is the Jaccard similarity.
    """
    a, b = _words(first), _words(second)
    return len(a & b), len(a | b), len(a), len(b)
