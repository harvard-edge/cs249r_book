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
