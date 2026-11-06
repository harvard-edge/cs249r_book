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
    """Count whole serialized records within a token allowance.

    Parameters
    ----------
    allowance : int
        Total nonnegative retrieval allowance, in tokens.
    envelope : int
        Nonnegative fixed framing cost, in tokens; normally at most allowance.
    record : int
        Positive tokens per record, including its body and metadata.

    Returns
    -------
    int
        Whole records that fit. A negative remainder produces a negative count;
        this arithmetic helper does not clamp an infeasible budget to zero.

    Notes
    -----
    Equation: n = floor((allowance - envelope) / record).
    Assumptions: every record has the same serialized size and records are indivisible.
    Excludes: variable record lengths, truncation, tokenizer variation, and admission policy.

    References
    ----------
    Derivation: subtract fixed framing from the allowance, then divide the remaining
    space into equal record-sized slots and discard the fractional slot.

    Examples
    --------
    >>> calc_records_that_fit(100, 10, 20)
    4
    """
    return (allowance - envelope) // record


def calc_bm25_idf(documents: int, matching_documents: int) -> float:
    """Calculate the nonnegative BM25 inverse document frequency variant.

    Parameters
    ----------
    documents : int
        Positive corpus document count N.
    matching_documents : int
        Document count n containing the term, between zero and N inclusive.

    Returns
    -------
    float
        Dimensionless inverse document frequency, using the natural logarithm.

    Notes
    -----
    Equation: IDF = ln(1 + (N - n + 0.5) / (n + 0.5)).
    Assumptions: document-level occurrence counts and a fixed corpus. The outer
    addition of one is the nonnegative variant; it differs from a raw log-odds IDF.
    Excludes: term frequency, query weighting, corpus updates, and learned relevance.

    References
    ----------
    Robertson and Zaragoza (2009), "The Probabilistic Relevance Framework: BM25
    and Beyond", https://doi.org/10.1561/1500000019
    (bibliography key: robertson2009bm25). The nonnegative transform is stated
    explicitly above rather than treated as the only BM25 IDF convention.

    Examples
    --------
    >>> round(calc_bm25_idf(100, 10), 6)
    2.263745
    """
    return _math.log(1 + (documents - matching_documents + 0.5) / (matching_documents + 0.5))


def calc_bm25_length_normalizer(length: float, average_length: float, k1: float, b: float) -> float:
    """Calculate the document-length term in the BM25 denominator.

    Parameters
    ----------
    length : float
        Nonnegative document length, in tokens.
    average_length : float
        Positive corpus mean document length, in the same tokens.
    k1 : float
        Positive dimensionless term-frequency saturation parameter.
    b : float
        Dimensionless length-normalization strength, normally in [0, 1].

    Returns
    -------
    float
        Dimensionless denominator offset before adding term frequency.

    Notes
    -----
    Equation: normalizer = k1 * (1 - b + b * length / average_length).
    Assumptions: document and mean lengths use the same tokenizer and units.
    At the average length the offset equals k1 regardless of b.
    Excludes: query-term frequency, IDF, field weighting, and learned length effects.

    References
    ----------
    Robertson and Zaragoza (2009), "The Probabilistic Relevance Framework: BM25
    and Beyond", https://doi.org/10.1561/1500000019
    (bibliography key: robertson2009bm25).

    Examples
    --------
    >>> calc_bm25_length_normalizer(100, 100, 1.2, 0.75)
    1.2
    """
    return k1 * (1 - b + b * length / average_length)


def calc_bm25_term_score(
    idf: float, frequency: float, length: float, average_length: float, k1: float, b: float
) -> float:
    """Calculate one query term's BM25 contribution to a document score.

    Parameters
    ----------
    idf : float
        Dimensionless inverse document frequency under the caller's convention.
    frequency : float
        Nonnegative term occurrence count in this document.
    length : float
        Nonnegative document token count.
    average_length : float
        Positive corpus mean token count under the same tokenizer.
    k1 : float
        Positive dimensionless term-frequency saturation parameter.
    b : float
        Dimensionless length-normalization strength, normally in [0, 1].

    Returns
    -------
    float
        Dimensionless term score; sum over terms outside this helper if desired.

    Notes
    -----
    Equation: score = IDF * f * (k1 + 1) /
    (f + k1 * (1 - b + b * length / average_length)).
    Assumptions: a fixed corpus, a supplied IDF, and the denominator produced by
    ``calc_bm25_length_normalizer``. A zero frequency contributes zero.
    Excludes: query-term repetition weights, field boosts, and ranking evaluation.

    References
    ----------
    Robertson and Zaragoza (2009), "The Probabilistic Relevance Framework: BM25
    and Beyond", https://doi.org/10.1561/1500000019
    (bibliography key: robertson2009bm25).

    Examples
    --------
    >>> calc_bm25_term_score(1.0, 2, 100, 100, 1.2, 0.75)
    1.375
    """
    normalizer = calc_bm25_length_normalizer(length, average_length, k1, b)
    return idf * frequency * (k1 + 1) / (frequency + normalizer)


def calc_full_tree_nodes(children: int, depth: int) -> int:
    """Count nodes in a full tree through a specified depth.

    Parameters
    ----------
    children : int
        Nonnegative branching factor c, identical at every non-leaf node.
    depth : int
        Nonnegative final depth d; the root has depth zero.

    Returns
    -------
    int
        Number of nodes including the root and the final level.

    Notes
    -----
    Equation: nodes = ``sum(c**j`` for j in range(d + 1)); equivalently
    ``(c**(d`` + 1) - 1) / (c - 1) for c != 1, and d + 1 for c = 1.
    Assumptions: every branch expands fully through the same depth.
    Excludes: shared graph nodes, cycles, deduplication, pruning, and traversal cost.

    References
    ----------
    Derivation: level j contains ``c**j`` nodes; sum the finite geometric series.

    Examples
    --------
    >>> calc_full_tree_nodes(2, 3)
    15
    >>> calc_full_tree_nodes(1, 3)
    4
    """
    if children == 1:
        return depth + 1
    return (children ** (depth + 1) - 1) // (children - 1)


def calc_vectors_in_budget(budget, bandwidth, dimension: int, bytes_per_value):
    """Calculate the bandwidth-only vector count for an exact scan.

    Parameters
    ----------
    budget : Quantity
        Nonnegative scan time allowance, with time units.
    bandwidth : Quantity
        Positive sustained or assumed memory bandwidth, in bytes per time.
    dimension : int
        Positive components per vector.
    bytes_per_value : Quantity
        Positive storage per component, with byte units.

    Returns
    -------
    float
        Dimensionless vector count, including a fractional final vector.
        A caller needing whole records must round down.

    Notes
    -----
    Equation: vectors = budget * bandwidth / (dimension * bytes_per_value).
    Assumptions: every eligible vector is streamed once at the supplied bandwidth.
    Treat this as an achievable count only when scan arithmetic remains below the
    hardware's Roofline ridge point; otherwise it is a traffic-only upper bound.
    Excludes: distance-computation time, metadata, cache effects, and transfer startup.

    References
    ----------
    Derivation: bandwidth times time is the byte budget; dividing by bytes per
    vector yields the number of vector footprints that can cross the interface.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> calc_vectors_in_budget(Q_('1 ms'), Q_('1 GB/s'), 100, Q_('4 byte'))
    2500.0
    """
    return float((budget * bandwidth / (dimension * bytes_per_value)).to("").magnitude)


def calc_bandwidth_fit_through_origin(bytes_streamed, seconds) -> float:
    """Fit bandwidth from paired scan bytes and elapsed times.

    Parameters
    ----------
    bytes_streamed : sequence of float
        Nonnegative bytes for each scan, with at least one positive value.
    seconds : sequence of float
        Corresponding positive elapsed seconds; must have the same length.

    Returns
    -------
    float
        Fitted bandwidth in bytes per second, minimizing squared time residuals.

    Notes
    -----
    Equation: B = ``sum(x_i**2)`` / sum(x_i * t_i), from t_i = x_i / B.
    Assumptions: a zero-intercept latency model and paired sequences of equal length.
    The helper uses zip for cross-products and does not validate equal lengths;
    callers must ensure alignment. Inputs are supplied data, not measurements
    performed by this function.
    Excludes: startup intercept, uncertainty intervals, outlier removal, and weighting.

    References
    ----------
    Derivation: minimize sum((t_i - ``a*x_i)**2)`` over a = 1/B. Setting its
    derivative to zero gives a = ``sum(x_i*t_i)/sum(x_i**2)``.

    Examples
    --------
    >>> calc_bandwidth_fit_through_origin([100, 200], [1, 2])
    100.0
    """
    num = sum(b * b for b in bytes_streamed)
    den = sum(b * t for b, t in zip(bytes_streamed, seconds))
    return num / den


def calc_interface_bandwidth(width, transfer_rate):
    """Calculate peak bandwidth from interface width and per-pin transfer rate.

    Parameters
    ----------
    width : Quantity
        Positive interface width in bits, one bit per pin per transfer.
    transfer_rate : Quantity
        Positive per-pin bit rate, in bits per second.

    Returns
    -------
    Quantity
        Aggregate interface bandwidth in bytes per second.

    Notes
    -----
    Equation: bandwidth = width_in_bits * per_pin_bit_rate / 8 bits_per_byte.
    Assumptions: all data pins transfer simultaneously at the stated rate, which
    already includes any multiple transfers per clock cycle.
    Excludes: command/address pins, refresh, coding overhead, and utilization losses.

    References
    ----------
    Derivation: each data pin contributes its bit rate; summing over the interface
    width gives aggregate bit rate, then unit conversion gives byte rate.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> calc_interface_bandwidth(Q_('64 bit'), Q_('1e9 bit/s')).m_as('GB/s')
    8.0
    """
    return (width.to("bit").magnitude * transfer_rate).to("byte / second")


def _words(text: str):
    return {w.strip(".,;:!?\"'()").lower() for w in text.split()} - {""}


def calc_word_overlap(first: str, second: str):
    """Count distinct-word intersection and union after simple normalization.

    Parameters
    ----------
    first : str
        First text, split on whitespace.
    second : str
        Second text, split on whitespace.

    Returns
    -------
    tuple of int
        Intersection size, union size, first-set size, and second-set size.
        Intersection divided by union is Jaccard similarity when the union is nonempty.

    Notes
    -----
    Equation: result = (|A intersection B|, |A union B|, |A|, |B|).
    Assumptions: words are lowercased and stripped of leading/trailing characters
    in ``.,;:!?"'()``; internal punctuation is retained and duplicates count once.
    Excludes: stemming, stop-word removal, Unicode normalization, semantics, and a
    definition of Jaccard similarity for two empty texts.

    References
    ----------
    Derivation: form a set of normalized tokens for each input, then count its
    intersection and union by the definitions of finite set cardinality.

    Examples
    --------
    >>> calc_word_overlap('Cat, cat dog!', 'dog bird')
    (1, 3, 2, 2)
    """
    a, b = _words(first), _words(second)
    return len(a & b), len(a | b), len(a), len(b)


# --- MinHash banding for near-duplicate search (added 2026-10-20) ---


def calc_lsh_candidate_probability(similarity: float, bands: int, rows: int) -> float:
    """Calculate the ideal MinHash banding candidate probability.

    Parameters
    ----------
    similarity : float
        Jaccard similarity s, a dimensionless probability in [0, 1].
    bands : int
        Positive number b of bands in the signature.
    rows : int
        Positive independent hashes r per band.

    Returns
    -------
    float
        Probability that at least one band matches exactly.

    Notes
    -----
    Equation: candidate_probability = 1 - (1 - ``s**r)**b``.
    Assumptions: independent ideal MinHash collisions, each with probability s;
    any matching band admits a candidate.
    Excludes: hash dependence, finite bucket collisions, candidate verification,
    and guarantees about semantic similarity.

    References
    ----------
    Leskovec, Rajaraman, and Ullman, "Mining of Massive Datasets", Section 3.4.
    https://www.mmds.org/
    Derivation: all r rows of a band match with probability ``s**r``. No band matches
    with probability ``(1-s**r)**b``; take the complement.

    Examples
    --------
    >>> calc_lsh_candidate_probability(0.5, 2, 2)
    0.4375
    """
    return 1 - (1 - similarity**rows) ** bands


def calc_lsh_threshold(bands: int, rows: int) -> float:
    """Estimate the transition similarity of a MinHash banding curve.

    Parameters
    ----------
    bands : int
        Positive number b of signature bands.
    rows : int
        Positive number r of independent hashes per band.

    Returns
    -------
    float
        Dimensionless transition approximation (1/b)**(1/r), between zero and one.

    Notes
    -----
    Equation: threshold = ``b**(-1/r)``.
    Assumptions: ideal independent MinHash banding. This rule sets the expected
    number of matching bands ``b*s**r`` to one; it is not the exact maximum-slope
    location or a deterministic acceptance boundary.
    Excludes: a specified recall target, hash dependence, and candidate verification.

    References
    ----------
    Leskovec, Rajaraman, and Ullman, "Mining of Massive Datasets", Section 3.4.
    https://www.mmds.org/
    Derivation: solve ``b*s**r`` = 1 for s to locate the transition scale.

    Examples
    --------
    >>> calc_lsh_threshold(4, 2)
    0.5
    """
    return (1 / bands) ** (1 / rows)


def calc_minhash_standard_error(similarity: float, hashes: int) -> float:
    """Calculate standard error of an ideal MinHash similarity estimate.

    Parameters
    ----------
    similarity : float
        Population Jaccard similarity J, a dimensionless probability in [0, 1].
    hashes : int
        Positive number k of independent MinHash trials.

    Returns
    -------
    float
        Dimensionless standard deviation of the fraction of agreeing hashes.

    Notes
    -----
    Equation: SE = sqrt(J * (1 - J) / k).
    Assumptions: independent Bernoulli agreement indicators with common probability
    J. The largest standard error occurs at J = 0.5 and equals 0.5/sqrt(k).
    Excludes: confidence-interval construction, hash dependence, and finite-hash bias.

    References
    ----------
    Derivation: a Bernoulli indicator has variance ``J*(1-J)``. The mean of k
    independent indicators has variance ``J*(1-J)/k``; take its square root.

    Examples
    --------
    >>> calc_minhash_standard_error(0.5, 100)
    0.05
    """
    return (similarity * (1 - similarity) / hashes) ** 0.5
