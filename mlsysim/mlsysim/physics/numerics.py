"""Number formats: the range a binary floating-point or two's-complement format holds.

The limits follow from the bit layout alone (IEEE 754-2019 for binary floating
point; two's complement for signed integers), so they are computed here rather
than stored (added 2026-10-20).
"""

from __future__ import annotations

import math

from mlsysim.core._validation import validate_at_least


def calc_float_range(exponent_bits: int, fraction_bits: int):
    """Compute the finite range of a binary floating-point format.

    Parameters
    ----------
    exponent_bits : int
        Exponent-field width in bits, at least two.
    fraction_bits : int
        Stored fraction-field width in bits, at least one.

    Returns
    -------
    tuple of float
        Largest finite, smallest normal, and smallest subnormal positive values.

    Notes
    -----
    Equation: Bias b = ``2**(e-1)-1``; limits are ``(2-2**(-f))*2**b``, ``2**(1-b)``, and ``2**(1-b-f)``.

    Assumptions: Biased exponent, implicit leading bit for normals, reserved top exponent.

    Excludes: Alternative exponent encodings; host overflow for extremely wide formats.

    Largest finite, smallest normal, and smallest subnormal magnitude of a binary float.

        bias              = 2^(e-1) - 1
        largest finite    = (2 - 2^-f) 2^bias
        smallest normal   = 2^(1 - bias)
        smallest subnormal= 2^(1 - bias - f)

    Returns ``(largest, smallest_normal, smallest_subnormal)``.

    References
    ----------
    Derivation: Binary positional notation with reserved zero and top exponents.

    Examples
    --------
    >>> calc_float_range(3, 2)
    (14.0, 0.25, 0.0625)
    """
    validate_at_least(exponent_bits, 2, "exponent_bits")
    validate_at_least(fraction_bits, 1, "fraction_bits")
    bias = 2 ** (exponent_bits - 1) - 1
    largest = (2 - 2.0**-fraction_bits) * 2.0**bias
    return largest, 2.0 ** (1 - bias), 2.0 ** (1 - bias - fraction_bits)


def calc_float_round(value: float, exponent_bits: int, fraction_bits: int) -> float:
    """Round a scalar to the modeled binary floating-point format.

    Parameters
    ----------
    value : float
        Input magnitude in arbitrary consistent scalar units.
    exponent_bits : int
        Exponent-field width in bits, at least two.
    fraction_bits : int
        Stored fraction-field width in bits, at least one.

    Returns
    -------
    float
        Rounded scalar; overflow becomes signed infinity.

    Notes
    -----
    Equation: Round abs(value)/step to even, multiply by step, and restore the sign.

    Assumptions: Subnormal spacing is fixed; Python round supplies ties-to-even.

    Excludes: Exception flags, directed rounding, and exact host-independent arithmetic.

    ``value`` stored in a binary float format, rounded to nearest (ties to even).

    A magnitude below half the smallest subnormal flushes to zero (underflow);
    one that rounds above the largest finite value becomes infinity (overflow).

    References
    ----------
    Derivation: Rounding to the nearest point of the binade grid defined by calc_float_range.

    Examples
    --------
    >>> calc_float_round(1.3, 3, 2)
    1.25
    """
    if value == 0 or math.isnan(value) or math.isinf(value):
        return value
    largest, smallest_normal, _ = calc_float_range(exponent_bits, fraction_bits)
    exponent = max(math.floor(math.log2(abs(value))), math.floor(math.log2(smallest_normal)))
    step = 2.0 ** (exponent - fraction_bits)
    stored = round(abs(value) / step) * step
    if stored > largest:
        return math.copysign(math.inf, value)
    return math.copysign(stored, value)


def calc_int_max(bits: int) -> int:
    """Compute the largest signed integer for a code width.

    Parameters
    ----------
    bits : int
        Integer code width in bits; at least two for signed maxima, otherwise one.

    Returns
    -------
    int
        Largest positive two's-complement code.

    Notes
    -----
    Equation: maximum = ``2**(bits-1)-1``.

    Assumptions: One sign bit and a two's-complement representation.

    Excludes: Saturating arithmetic and unsigned encodings.

    Largest value of a signed two's-complement integer of ``bits`` bits, ``2^(n-1) - 1``.

    References
    ----------
    Derivation: The positive half of the binary code space contains zero and ``2**(bits-1)-1`` positive values.

    Examples
    --------
    >>> calc_int_max(4)
    7
    """
    validate_at_least(bits, 2, "bits")
    return 2 ** (bits - 1) - 1


def calc_float32_bit_flip(value: float, bit: int):
    """Flip one bit of a stored binary32 value.

    Parameters
    ----------
    value : float
        Input magnitude in arbitrary consistent scalar units.
    bit : int
        Zero-based bit index in [0, 31], counting from the least significant bit.

    Returns
    -------
    tuple of float
        Original binary32 stored value and value after the bit flip.

    Notes
    -----
    Equation: corrupted_word = stored_word XOR (1 << bit).

    Assumptions: struct packs the input to binary32 before corruption.

    Excludes: Fault probabilities, error correction, and arithmetic after corruption.

    ``value`` stored as an IEEE 754 binary32 word, read back after bit ``bit`` flips.

    Bit 31 is the sign, bits 30 to 23 the exponent, bits 22 to 0 the fraction.
    Returns ``(stored, corrupted)``: the value as stored, and the value after
    the flip. For a normal value whose encoded exponent is in 1..126,
    flipping bit 30 raises that exponent by 128 and preserves a finite normal
    value with the same sign and fraction. Encoded exponent 127 instead becomes
    the all-ones exponent: a zero fraction produces infinity and a nonzero
    fraction produces NaN. Zero, subnormals, and values whose exponent already
    has its high bit set require interpreting the resulting bits directly;
    there is no universal finite-multiplier rule.

    References
    ----------
    Derivation: Exclusive-or toggles exactly the selected bit of the packed word.

    Examples
    --------
    >>> calc_float32_bit_flip(1.0, 31)
    (1.0, -1.0)
    >>> import math
    >>> math.isfinite(calc_float32_bit_flip(0.5, 30)[1])
    True
    >>> math.isinf(calc_float32_bit_flip(1.0, 30)[1])
    True
    >>> math.isnan(calc_float32_bit_flip(1.5, 30)[1])
    True
    """
    import struct

    if not 0 <= bit < 32:
        raise ValueError("bit must be in 0..31 for a binary32 word.")
    word = struct.unpack(">I", struct.pack(">f", value))[0]
    stored = struct.unpack(">f", struct.pack(">I", word))[0]
    corrupted = struct.unpack(">f", struct.pack(">I", word ^ (1 << bit)))[0]
    return stored, corrupted


def calc_float_spacing(value: float, exponent_bits: int, fraction_bits: int) -> float:
    """Compute the magnitude-binade spacing of a binary float.

    Parameters
    ----------
    value : float
        Input magnitude in arbitrary consistent scalar units.
    exponent_bits : int
        Exponent-field width in bits, at least two.
    fraction_bits : int
        Stored fraction-field width in bits, at least one.

    Returns
    -------
    float
        Positive grid spacing in the input scalar units.

    Notes
    -----
    Equation: step = ``2**(max(floor(log2(abs(value)))``, emin)-fraction_bits).

    Assumptions: Finite nonzero input; fixed spacing below the smallest normal.

    Excludes: Zero input and directional neighbor differences at binade boundaries
    for either sign.

    A number in the binade ``[2^E, 2^(E+1))`` has neighbors ``2^(E - f)`` apart;
    below the smallest normal the spacing stays at the subnormal step. A
    perturbation smaller than half the distance to the neighbor in that
    direction rounds back to an exactly representable value under nearest
    rounding, away from midpoint ties. At a binade boundary the upward and
    downward distances differ; the spacing returned here must not be used as
    a direction-independent rounding threshold.

    References
    ----------
    Derivation: Adjacent significands differ by one stored fraction unit within a binade.

    Examples
    --------
    >>> calc_float_spacing(1.0, 3, 2)
    0.25
    """
    _, smallest_normal, _ = calc_float_range(exponent_bits, fraction_bits)
    exponent = max(math.floor(math.log2(abs(value))), math.floor(math.log2(smallest_normal)))
    return 2.0 ** (exponent - fraction_bits)


def calc_repeated_add(start: float, update: float, repeats: int, exponent_bits: int, fraction_bits: int):
    """Trace repeated additions with format rounding after each step.

    Parameters
    ----------
    start : float
        Initial scalar value, rounded before the first update.
    update : float
        Scalar increment in the same units as start; rounded before reuse.
    repeats : int
        Nonnegative number of update operations.
    exponent_bits : int
        Exponent-field width in bits, at least two.
    fraction_bits : int
        Stored fraction-field width in bits, at least one.

    Returns
    -------
    list of float
        Initial rounded value followed by each rounded update result.

    Notes
    -----
    Equation: x[0]=round_format(start); x[i+1]=round_format(x[i]+round_format(update)).

    Assumptions: Each update reuses the same rounded increment.

    Excludes: Fused or higher-precision accumulation and stochastic rounding.

    ``start`` after adding ``update`` ``repeats`` times, rounding to the format after every add.

    Returns the stored value after each add, starting with ``start`` itself
    (``repeats + 1`` values). This is how a weight kept in a narrow format
    accumulates small optimizer updates (added 2026-10-20).

    References
    ----------
    Derivation: Apply the scalar rounding operation after each addition in sequence.

    Examples
    --------
    >>> calc_repeated_add(1.0, 0.25, 2, 3, 2)
    [1.0, 1.25, 1.5]
    """
    validate_at_least(repeats, 0, "repeats")
    stored = calc_float_round(start, exponent_bits, fraction_bits)
    step = calc_float_round(update, exponent_bits, fraction_bits)
    trace = [stored]
    for _ in range(repeats):
        stored = calc_float_round(stored + step, exponent_bits, fraction_bits)
        trace.append(stored)
    return trace


def calc_float_decimal_digits(fraction_bits: int) -> float:
    """Estimate decimal precision from binary significand width.

    Parameters
    ----------
    fraction_bits : int
        Stored fraction-field width in bits, at least one.

    Returns
    -------
    float
        Equivalent decimal digits, not an integer round-trip guarantee.

    Notes
    -----
    Equation: digits = ``(fraction_bits+1)*log10(2)``.

    Assumptions: Normal values include one implicit leading bit.

    Excludes: Decimal serialization guarantees and subnormal precision loss.

    Decimal digits a binary float keeps: ``(f + 1) log10 2`` for ``f`` stored fraction bits
    plus the implicit leading bit (about 7.2 for FP32, 3.3 for FP16, 2.4 for BF16; added 2026-10-20).

    References
    ----------
    Derivation: Equate ``2**precision_bits`` with ``10**decimal_digits`` and take logarithms.

    Examples
    --------
    >>> round(calc_float_decimal_digits(3), 3)
    1.204
    """
    validate_at_least(fraction_bits, 1, "fraction_bits")
    return (fraction_bits + 1) * math.log10(2)


# --- Quantization: integer grids, scales, and stored-value grids (added 2026-10-20) ---


def calc_float_values(exponent_bits: int, fraction_bits: int, ieee: bool = True):
    """Enumerate nonnegative finite values of a binary format.

    Parameters
    ----------
    exponent_bits : int
        Exponent-field width in bits, at least two.
    fraction_bits : int
        Stored fraction-field width in bits, at least one.
    ieee : bool, optional
        Reserve the entire top exponent for special values if True (default).

    Returns
    -------
    list of float
        Sorted represented values, including zero.

    Notes
    -----
    Equation: Normals are ``(1+f/2**F)*2**(e-bias)``; subnormals are ``(f/2**F)*2**(1-bias)``.

    Assumptions: Small field widths suitable for exhaustive enumeration; ieee selects the special-value encoding.

    Excludes: Negative values, infinities, NaNs, and practical enumeration of wide formats.

    Every non-negative finite value of a binary float format, sorted (zero included).

    ``ieee=True`` reserves the top exponent for infinities and NaNs (IEEE 754,
    and FP8 E5M2). ``ieee=False`` is the "finite" layout of FP8 E4M3
    (Micikevicius et al. 2022): the top exponent holds normal values and only
    its all-ones fraction is NaN, which lifts the largest value from 240 to 448.

    References
    ----------
    Derivation: Enumerate each exponent/fraction code under the selected reserved-code convention.

    Examples
    --------
    >>> calc_float_values(2, 1)
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    """
    validate_at_least(exponent_bits, 2, "exponent_bits")
    validate_at_least(fraction_bits, 1, "fraction_bits")
    bias = 2 ** (exponent_bits - 1) - 1
    top = 2**exponent_bits - 1
    values = []
    for e in range(2**exponent_bits):
        for f in range(2**fraction_bits):
            if e == top and (ieee or f == 2**fraction_bits - 1):
                continue
            frac = f / 2**fraction_bits
            values.append(frac * 2.0 ** (1 - bias) if e == 0 else (1 + frac) * 2.0 ** (e - bias))
    return sorted(values)


def calc_int_levels(bits: int) -> int:
    """Count distinct integer codes of a fixed width.

    Parameters
    ----------
    bits : int
        Integer code width in bits; at least two for signed maxima, otherwise one.

    Returns
    -------
    int
        Number of bit patterns.

    Notes
    -----
    Equation: levels = ``2**bits``.

    Assumptions: Each bit can independently take two values.

    Excludes: Reserved codes and the interpretation of sign.

    Distinct codes of a ``bits``-bit integer, ``2^b``.

    References
    ----------
    Derivation: Multiply two choices for each bit.

    Examples
    --------
    >>> calc_int_levels(3)
    8
    """
    validate_at_least(bits, 1, "bits")
    return 2**bits


def calc_affine_scale(alpha: float, beta: float, bits: int) -> float:
    """Compute the step of an affine integer quantization grid.

    Parameters
    ----------
    alpha : float
        Lower clipping endpoint in the same units as the modeled values.
    beta : float
        Upper clipping endpoint, greater than alpha.
    bits : int
        Integer code width in bits; at least two for signed maxima, otherwise one.

    Returns
    -------
    float
        Value-unit spacing between neighboring codes.

    Notes
    -----
    Equation: scale = ``(beta-alpha)/(2**bits-1)``.

    Assumptions: Both clipping endpoints are grid endpoints.

    Excludes: Calibration of clipping thresholds and quantization error distributions.

    Step of an asymmetric (affine) integer grid over the clipping range ``[alpha, beta]``:
    ``s = (beta - alpha) / (2^b - 1)``.

    References
    ----------
    Derivation: Divide endpoint span by the number of intervals between ``2**bits`` codes.

    Examples
    --------
    >>> calc_affine_scale(-1.0, 2.0, 2)
    1.0
    """
    validate_at_least(bits, 1, "bits")
    if beta <= alpha:
        raise ValueError("beta must exceed alpha.")
    return (beta - alpha) / (2**bits - 1)


def calc_symmetric_scale(max_abs: float, bits: int) -> float:
    """Compute the step of a symmetric signed quantization grid.

    Parameters
    ----------
    max_abs : float
        Positive largest absolute represented value.
    bits : int
        Integer code width in bits; at least two for signed maxima, otherwise one.

    Returns
    -------
    float
        Value-unit spacing per code.

    Notes
    -----
    Equation: scale = ``max_abs/(2**(bits-1)-1)``.

    Assumptions: Zero point is zero; the most negative signed code is unused.

    Excludes: Asymmetric clipping and selection of max_abs from data.

    Step of a symmetric signed grid that reaches ``max_abs``: ``s = max_abs / (2^(b-1) - 1)``
    (zero-point 0; the code ``-2^(b-1)`` goes unused).

    References
    ----------
    Derivation: Map the largest positive signed code to max_abs.

    Examples
    --------
    >>> calc_symmetric_scale(7.0, 4)
    1.0
    """
    return max_abs / calc_int_max(bits)


def calc_zero_point(alpha: float, scale: float, qmin: int = 0):
    """Locate the integer code representing zero on an affine grid.

    Parameters
    ----------
    alpha : float
        Lower clipping endpoint in the same units as the modeled values.
    scale : float
        Positive quantization step in value units per integer code.
    qmin : int, optional
        Lowest code; defaults to zero when omitted.

    Returns
    -------
    tuple of float and int
        Unrounded offset -alpha/scale and rounded zero code including qmin.

    Notes
    -----
    Equation: offset=-alpha/scale; zero_code=qmin+round(offset).

    Assumptions: Positive scale; nearest rounding with ties to even.

    Excludes: Clamping the zero point to a valid code interval.

    Integer that stands for real zero on an affine grid: ``qmin + round(-alpha / s)``.

    Returns ``(exact, zero_point)``: the unrounded ``-alpha / s`` and the stored
    integer. Rounding is to nearest, ties to even, as Python and PyTorch round.

    References
    ----------
    Derivation: Solve ``alpha+scale*(code-qmin)``=0 and round the resulting offset.

    Examples
    --------
    >>> calc_zero_point(-2.0, 0.5)
    (4.0, 4)
    """
    exact = -alpha / scale
    return exact, qmin + round(exact)


def calc_quantize(x: float, scale: float, zero_point: int, qmin: int, qmax: int) -> int:
    """Encode a scalar by rounding and clipping to integer codes.

    Parameters
    ----------
    x : float
        Scalar value to encode, in the same units as scale.
    scale : float
        Positive quantization step in value units per integer code.
    zero_point : int
        Integer code representing zero.
    qmin : int, optional
        Lowest code; defaults to zero when omitted.
    qmax : int
        Highest code, at least qmin.

    Returns
    -------
    int
        Clipped integer code.

    Notes
    -----
    Equation: code = max(qmin, min(qmax, round(x/scale+zero_point))).

    Assumptions: Positive scale and ordered code bounds; ties round to even.

    Excludes: Scale calibration, stochastic rounding, and tensor broadcasting.

    Code for ``x``: ``clamp(round(x / s + z), qmin, qmax)`` (ties to even).

    References
    ----------
    Derivation: Invert the affine decoding map and project onto the allowed integer interval.

    Examples
    --------
    >>> calc_quantize(1.2, 0.5, 0, -7, 7)
    2
    """
    return max(qmin, min(qmax, round(x / scale + zero_point)))


def calc_dequantize(code: int, scale: float, zero_point: int) -> float:
    """Decode an integer code with an affine quantization map.

    Parameters
    ----------
    code : int
        Stored integer code to decode.
    scale : float
        Positive quantization step in value units per integer code.
    zero_point : int
        Integer code representing zero.

    Returns
    -------
    float
        Reconstructed scalar in scale units.

    Notes
    -----
    Equation: value = ``scale*(code-zero_point)``.

    Assumptions: The supplied scale and zero point match the encoding.

    Excludes: Recovering information lost through rounding or clipping.

    Real value a code reads back as: ``s (q - z)``.

    References
    ----------
    Derivation: Translate the zero code to zero and multiply code distance by grid spacing.

    Examples
    --------
    >>> calc_dequantize(2, 0.5, 0)
    1.0
    """
    return scale * (code - zero_point)


def calc_block_bits_per_weight(block_weights: int, code_bits: int, scale_bits: int) -> float:
    """Amortize a shared scale over a block of weight codes.

    Parameters
    ----------
    block_weights : int
        Positive count of weights sharing one scale.
    code_bits : int
        Nonnegative stored bits per weight code.
    scale_bits : int
        Nonnegative stored bits for the shared scale.

    Returns
    -------
    float
        Stored bits per weight including its share of the scale.

    Notes
    -----
    Equation: bits_per_weight = code_bits + scale_bits/block_weights.

    Assumptions: Every block contains the stated number of weights and one scale.

    Excludes: Padding, alignment, headers, and incomplete final blocks.

    Stored bits per weight of a block format: ``(scale_bits + n code_bits) / n``.

    One shared scale per block of ``n`` codes (llama.cpp's Q4_0: 32 four-bit
    codes and one 16-bit scale, 4.5 bits per weight).

    References
    ----------
    Derivation: Divide total code and scale storage by the count of weights.

    Examples
    --------
    >>> calc_block_bits_per_weight(32, 4, 16)
    4.5
    """
    validate_at_least(block_weights, 1, "block_weights")
    return (scale_bits + block_weights * code_bits) / block_weights


def calc_accumulator_headroom(operand_bits: int, accumulator_bits: int) -> int:
    """Estimate worst-case positive-product accumulator headroom.

    Parameters
    ----------
    operand_bits : int
        Signed operand width in bits, at least two.
    accumulator_bits : int
        Signed accumulation register width in bits, at least two.

    Returns
    -------
    int
        Approximate number of positive-maximum products fitting the sign-bit budget.

    Notes
    -----
    Equation: headroom = ``2**(accumulator_bits-1)`` // ``(2**(operand_bits-1)-1)**2``.

    Assumptions: Products use the largest positive signed operands; numerator uses the power-of-two boundary.

    Excludes: Negative minimum operands, bias terms, and an exact overflow-safe count at the signed maximum.

    About how many worst-case products of two signed ``operand_bits`` codes fit in a
    signed ``accumulator_bits`` register: ``2^(a-1) // (2^(b-1) - 1)^2`` (127 times 127
    at INT8; about 133,000 in INT32).

    References
    ----------
    Derivation: Divide the modeled accumulation budget by one maximum positive product.

    Examples
    --------
    >>> calc_accumulator_headroom(4, 8)
    2
    """
    return 2 ** (accumulator_bits - 1) // calc_int_max(operand_bits) ** 2
