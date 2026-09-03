from typing import Any, Annotated, Optional
from pydantic import AfterValidator, PlainSerializer, BaseModel, ConfigDict
from .units import Q_
from .provenance import Provenance

def validate_quantity(v: Any) -> Q_:
    if isinstance(v, Q_):
        return v
    if isinstance(v, (int, float, str)):
        try:
            return Q_(v)
        except Exception as e:
            raise ValueError(f"Could not parse Quantity from {v}: {e}")
    raise ValueError(f"Expected Quantity, got {type(v)}")

def serialize_quantity(v: Q_) -> str:
    # Use compact format for serialization
    return f"{v:~P}"


def require_dimensionality(v: Any, expected_unit, field_name: str):
    """Parse a value as a Quantity and validate its physical dimension.

    This catches obvious unit errors such as watts in a latency field. The
    unit-family helpers below add the stricter semantic checks for Pint units
    that are dimensionless by construction, such as bytes, FLOPs, and dollars.
    """
    if v is None:
        return None
    q = validate_quantity(v)
    if not q.check(expected_unit):
        raise ValueError(
            f"{field_name} must have units compatible with "
            f"{expected_unit.dimensionality}; got {q.units}"
        )
    return q


# Pint represents byte, flop, count, parameter, and currency units as
# dimensionless aliases. MLSysIM treats those aliases as distinct modeling
# families so registry typos fail before reaching solver math.
_DATA_UNIT_NAMES = {
    "B",
    "KB", "MB", "GB", "TB", "PB",
    "KiB", "MiB", "GiB", "TiB",
    "Gbps",
}
_OP_UNIT_NAMES = {
    "OPS", "KOPS", "MOPS", "GOPS", "TOPS",
    "KFLOPs", "MFLOPs", "GFLOP", "GFLOPs", "TFLOP", "TFLOPs",
    "PFLOPs", "EFLOP", "EFLOPs", "ZFLOPs",
    "count",
}
_COUNT_UNIT_NAMES = {"count", "param", "Kparam", "Mparam", "Bparam", "Tparam"}
_CURRENCY_UNIT_NAMES = {"dollar", "USD", "EUR"}


def _unit_names(q: Q_) -> set[str]:
    return set(q.units._units)


def _has_unit_family(q: Q_, family: str) -> bool:
    names = _unit_names(q)
    lowered = {name.lower() for name in names}
    if family == "data":
        return bool(names & _DATA_UNIT_NAMES) or any("byte" in name or "bit" in name for name in lowered)
    if family == "operation":
        return bool(names & _OP_UNIT_NAMES) or any("flop" in name or "ops" in name for name in lowered)
    if family == "count":
        return bool(names & _COUNT_UNIT_NAMES)
    if family == "currency":
        return bool(names & _CURRENCY_UNIT_NAMES)
    raise ValueError(f"Unknown unit family: {family}")


def require_unit_families(v: Any, expected_unit, field_name: str, families: tuple[str, ...]):
    """Validate dimension first, then require explicit semantic unit families."""
    q = require_dimensionality(v, expected_unit, field_name)
    if q is None:
        return None
    # Step 1: reject bare numbers for fields that should carry units.
    if not _unit_names(q):
        required = ", ".join(families)
        raise ValueError(f"{field_name} must include explicit {required} units; got a bare number")
    # Step 2: reject dimensionally-compatible but semantically-wrong aliases,
    # e.g. "GB/s" as compute throughput or "TFLOP/s" as memory bandwidth.
    missing = [family for family in families if not _has_unit_family(q, family)]
    if missing:
        required = ", ".join(families)
        raise ValueError(f"{field_name} must use {required} units; got {q.units}")
    return q


def require_unit_family(v: Any, expected_unit, field_name: str, family: str):
    return require_unit_families(v, expected_unit, field_name, (family,))

Quantity = Annotated[
    Any,
    AfterValidator(validate_quantity),
    PlainSerializer(serialize_quantity, return_type=str)
]

class Metadata(BaseModel):
    """Provenance for registry entries (hardware, models, fabrics)."""

    model_config = ConfigDict(extra="forbid")

    provenance: Optional[Provenance] = None
    description: Optional[str] = None
    last_verified: Optional[str] = None  # YYYY-MM-DD
    version: Optional[str] = None

    @property
    def source_label(self) -> Optional[str]:
        """Human-readable source label from provenance."""
        if self.provenance is not None:
            return self.provenance.ref
        return None
