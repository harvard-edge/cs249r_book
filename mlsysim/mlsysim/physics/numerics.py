"""Number formats: the range a binary floating-point or two's-complement format holds.

The limits follow from the bit layout alone (IEEE 754-2019 for binary floating
point; two's complement for signed integers), so they are computed here rather
than stored (added 2026-10-20).
"""

from __future__ import annotations

import math

from mlsysim.core._validation import validate_at_least


def calc_float_range(exponent_bits: int, fraction_bits: int):
    """
    Largest finite, smallest normal, and smallest subnormal magnitude of a binary float.

        bias              = 2^(e-1) - 1
        largest finite    = (2 - 2^-f) 2^bias
        smallest normal   = 2^(1 - bias)
        smallest subnormal= 2^(1 - bias - f)

    Returns ``(largest, smallest_normal, smallest_subnormal)``.
    """
    validate_at_least(exponent_bits, 2, "exponent_bits")
    validate_at_least(fraction_bits, 1, "fraction_bits")
    bias = 2 ** (exponent_bits - 1) - 1
    largest = (2 - 2.0 ** -fraction_bits) * 2.0 ** bias
    return largest, 2.0 ** (1 - bias), 2.0 ** (1 - bias - fraction_bits)


def calc_float_round(value: float, exponent_bits: int, fraction_bits: int) -> float:
    """
    ``value`` stored in a binary float format, rounded to nearest (ties to even).

    A magnitude below half the smallest subnormal flushes to zero (underflow);
    one that rounds above the largest finite value becomes infinity (overflow).
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
    """Largest value of a signed two's-complement integer of ``bits`` bits, ``2^(n-1) - 1``."""
    validate_at_least(bits, 2, "bits")
    return 2 ** (bits - 1) - 1


def calc_float32_bit_flip(value: float, bit: int):
    """
    ``value`` stored as an IEEE 754 binary32 word, read back after bit ``bit`` flips.

    Bit 31 is the sign, bits 30 to 23 the exponent, bits 22 to 0 the fraction.
    Returns ``(stored, corrupted)``: the value as stored, and the value after
    the flip. Flipping bit 30 of a weight below 2 multiplies it by 2^128 and
    leaves it finite (added 2026-10-20).
    """
    import struct
    if not 0 <= bit < 32:
        raise ValueError("bit must be in 0..31 for a binary32 word.")
    word = struct.unpack(">I", struct.pack(">f", value))[0]
    stored = struct.unpack(">f", struct.pack(">I", word))[0]
    corrupted = struct.unpack(">f", struct.pack(">I", word ^ (1 << bit)))[0]
    return stored, corrupted


def calc_float_spacing(value: float, exponent_bits: int, fraction_bits: int) -> float:
    """
    Gap between ``value`` and the next larger number of a binary float format (one ulp).

    A number in the binade ``[2^E, 2^(E+1))`` has neighbors ``2^(E - f)`` apart;
    below the smallest normal the spacing stays at the subnormal step. An
    update smaller than half this spacing rounds away when added to ``value``
    (added 2026-10-20).
    """
    _, smallest_normal, _ = calc_float_range(exponent_bits, fraction_bits)
    exponent = max(math.floor(math.log2(abs(value))), math.floor(math.log2(smallest_normal)))
    return 2.0 ** (exponent - fraction_bits)


def calc_repeated_add(start: float, update: float, repeats: int, exponent_bits: int, fraction_bits: int):
    """
    ``start`` after adding ``update`` ``repeats`` times, rounding to the format after every add.

    Returns the stored value after each add, starting with ``start`` itself
    (``repeats + 1`` values). This is how a weight kept in a narrow format
    accumulates small optimizer updates (added 2026-10-20).
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
    """
    Decimal digits a binary float keeps: ``(f + 1) log10 2`` for ``f`` stored fraction bits
    plus the implicit leading bit (about 7.2 for FP32, 3.3 for FP16, 2.4 for BF16; added 2026-10-20).
    """
    validate_at_least(fraction_bits, 1, "fraction_bits")
    return (fraction_bits + 1) * math.log10(2)


# --- Quantization: integer grids, scales, and stored-value grids (added 2026-10-20) ---

def calc_float_values(exponent_bits: int, fraction_bits: int, ieee: bool = True):
    """
    Every non-negative finite value of a binary float format, sorted (zero included).

    ``ieee=True`` reserves the top exponent for infinities and NaNs (IEEE 754,
    and FP8 E5M2). ``ieee=False`` is the "finite" layout of FP8 E4M3
    (Micikevicius et al. 2022): the top exponent holds normal values and only
    its all-ones fraction is NaN, which lifts the largest value from 240 to 448.
    """
    validate_at_least(exponent_bits, 2, "exponent_bits")
    validate_at_least(fraction_bits, 1, "fraction_bits")
    bias = 2 ** (exponent_bits - 1) - 1
    top = 2 ** exponent_bits - 1
    values = []
    for e in range(2 ** exponent_bits):
        for f in range(2 ** fraction_bits):
            if e == top and (ieee or f == 2 ** fraction_bits - 1):
                continue
            frac = f / 2 ** fraction_bits
            values.append(frac * 2.0 ** (1 - bias) if e == 0 else (1 + frac) * 2.0 ** (e - bias))
    return sorted(values)


def calc_int_levels(bits: int) -> int:
    """Distinct codes of a ``bits``-bit integer, ``2^b``."""
    validate_at_least(bits, 1, "bits")
    return 2 ** bits


def calc_affine_scale(alpha: float, beta: float, bits: int) -> float:
    """
    Step of an asymmetric (affine) integer grid over the clipping range ``[alpha, beta]``:
    ``s = (beta - alpha) / (2^b - 1)``.
    """
    validate_at_least(bits, 1, "bits")
    if beta <= alpha:
        raise ValueError("beta must exceed alpha.")
    return (beta - alpha) / (2 ** bits - 1)


def calc_symmetric_scale(max_abs: float, bits: int) -> float:
    """
    Step of a symmetric signed grid that reaches ``max_abs``: ``s = max_abs / (2^(b-1) - 1)``
    (zero-point 0; the code ``-2^(b-1)`` goes unused).
    """
    return max_abs / calc_int_max(bits)


def calc_zero_point(alpha: float, scale: float, qmin: int = 0):
    """
    Integer that stands for real zero on an affine grid: ``qmin + round(-alpha / s)``.

    Returns ``(exact, zero_point)``: the unrounded ``-alpha / s`` and the stored
    integer. Rounding is to nearest, ties to even, as Python and PyTorch round.
    """
    exact = -alpha / scale
    return exact, qmin + round(exact)


def calc_quantize(x: float, scale: float, zero_point: int, qmin: int, qmax: int) -> int:
    """Code for ``x``: ``clamp(round(x / s + z), qmin, qmax)`` (ties to even)."""
    return max(qmin, min(qmax, round(x / scale + zero_point)))


def calc_dequantize(code: int, scale: float, zero_point: int) -> float:
    """Real value a code reads back as: ``s (q - z)``."""
    return scale * (code - zero_point)


def calc_block_bits_per_weight(block_weights: int, code_bits: int, scale_bits: int) -> float:
    """
    Stored bits per weight of a block format: ``(scale_bits + n code_bits) / n``.

    One shared scale per block of ``n`` codes (llama.cpp's Q4_0: 32 four-bit
    codes and one 16-bit scale, 4.5 bits per weight).
    """
    validate_at_least(block_weights, 1, "block_weights")
    return (scale_bits + block_weights * code_bits) / block_weights


def calc_accumulator_headroom(operand_bits: int, accumulator_bits: int) -> int:
    """
    About how many worst-case products of two signed ``operand_bits`` codes fit in a
    signed ``accumulator_bits`` register: ``2^(a-1) // (2^(b-1) - 1)^2`` (127 times 127
    at INT8; about 133,000 in INT32).
    """
    return 2 ** (accumulator_bits - 1) // calc_int_max(operand_bits) ** 2
