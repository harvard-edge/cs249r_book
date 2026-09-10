"""Input validation helpers for mlsysim formulas and solvers.

These guards catch common student mistakes (zero bandwidth, negative efficiency,
n_gpus=0) before they produce confusing inf/nan results or crash with unhelpful
error messages.
"""


def validate_positive(val, name: str):
    """Ensure a numeric value is strictly positive (> 0)."""
    mag = getattr(val, 'magnitude', val)
    if mag <= 0:
        raise ValueError(f"{name} must be positive, got {val}")


def validate_nonnegative(val, name: str):
    """Ensure a numeric value is non-negative (>= 0)."""
    mag = getattr(val, 'magnitude', val)
    if mag < 0:
        raise ValueError(f"{name} must be non-negative, got {val}")


def validate_range(val, lo, hi, name: str):
    """Ensure a numeric value is within [lo, hi]."""
    mag = getattr(val, 'magnitude', val)
    if mag < lo or mag > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {val}")


def validate_at_least(val, minimum, name: str):
    """Ensure a numeric value is >= minimum."""
    mag = getattr(val, 'magnitude', val)
    if mag < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {val}")
