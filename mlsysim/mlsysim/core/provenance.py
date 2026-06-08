"""Provenance types for registry entries and public sourced scalars."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

Scalar = Union[int, float]


class ProvenanceKind(str, Enum):
    """How a registry value is known. Evidence-backed kinds (datasheet,
    literature, industry_report) require a URL; estimate/derived require
    explanatory notes (enforced in ``Provenance``)."""

    DATASHEET = "datasheet"
    LITERATURE = "literature"
    INDUSTRY_REPORT = "industry_report"
    CONVENTION = "convention"
    ESTIMATE = "estimate"
    DERIVED = "derived"
    ILLUSTRATIVE = "illustrative"
    HEURISTIC = "heuristic"


class Provenance(BaseModel):
    """How we know a numeric value (package audit trail; not BibTeX)."""

    model_config = ConfigDict(extra="forbid")

    kind: ProvenanceKind
    ref: str = Field(min_length=1)
    url: Optional[str] = None
    verified: Optional[str] = None  # YYYY-MM-DD
    notes: Optional[str] = None
    id: Optional[str] = None  # stable slug, e.g. prov:iea-weo-2023-carbon

    @model_validator(mode="after")
    def _validate_kind_rules(self) -> Provenance:
        """
        Enforces validation rules based on the ProvenanceKind.
        
        Requires source URLs and verification dates for evidence-backed records,
        and notes for estimates and derived values. This keeps public numbers
        traceable and makes intentionally illustrative numbers explicit.
        """
        if not self.verified:
            raise ValueError(f"provenance requires verified date: {self.ref!r}")
        try:
            date.fromisoformat(self.verified)
        except ValueError as e:
            raise ValueError(f"provenance verified date must be YYYY-MM-DD: {self.ref!r}") from e

        if self.kind in {
            ProvenanceKind.DATASHEET,
            ProvenanceKind.LITERATURE,
            ProvenanceKind.INDUSTRY_REPORT,
        } and not self.url:
            raise ValueError(f"{self.kind.value} provenance requires url: {self.ref!r}")
        if self.kind in {ProvenanceKind.ESTIMATE, ProvenanceKind.DERIVED} and not self.notes:
            raise ValueError(f"{self.kind.value} provenance requires notes: {self.ref!r}")
        return self


class Sourced(float):
    """
    Scalar with mandatory ``Provenance``. Subclasses ``float`` so notebooks and
    generated calculations can divide and format values without extra coercion.

    .. warning:: Provenance does NOT survive arithmetic.
        Any operation on a ``Sourced`` (``x * 2``, ``x / y``, ``-x``,
        ``round(x)`` ...) returns a **plain float** — Python numeric dunders
        construct ``float`` results, and nothing re-attaches
        ``.provenance``/``.name``/``.description``. (``copy.copy`` raises
        ``TypeError`` because ``__new__`` requires a provenance argument.)
        The audit trail therefore lives only on the registry leaf value
        itself. Derived quantities that need their own audit trail must be
        wrapped explicitly with
        ``sourced(derived_value, Provenance(kind=DERIVED, ...))``.
    """

    def __new__(
        cls,
        value: Scalar,
        provenance: Provenance,
        *,
        name: str = "",
        description: str = "",
    ):
        obj = super().__new__(cls, float(value))
        obj.provenance = provenance
        obj.name = name
        obj.description = description
        return obj

    @property
    def source(self) -> str:
        """Returns the reference string of the provenance source."""
        return self.provenance.ref

    @property
    def url(self) -> Optional[str]:
        """Returns the URL of the provenance source, if any."""
        return self.provenance.url

    @property
    def value(self) -> float:
        """Returns the underlying float value of the sourced scalar."""
        return float(self)

    def render_markdown(self) -> str:
        """
        Renders a Markdown representation of the sourced scalar.
        
        This includes the value, description, provenance kind, source reference, 
        and URL. It is used primarily by CLI commands to inspect values.
        """
        lines = [
            f"**Assumption: {self.name}** = `{float(self)}`",
            "",
            f"_{self.description}_",
            "",
            f"> **Kind:** {self.provenance.kind.value}",
        ]
        if self.url:
            lines.append(f"> **Source:** [{self.source}]({self.url})")
        else:
            lines.append(f"> **Source:** {self.source}")
        if self.provenance.notes:
            lines.append(f"> **Notes:** {self.provenance.notes}")
        return "\n".join(lines)


def sourced(
    value: Scalar,
    provenance: Provenance,
    *,
    name: str = "",
    description: str = "",
) -> Sourced:
    """Attach provenance to a scalar used in registries or analyses."""
    return Sourced(value, provenance, name=name, description=description)


def sourced_qty(quantity, provenance: Provenance, *, name: str = "", description: str = ""):
    """Attach provenance to a unit-bearing pint Quantity (the Quantity analogue of
    ``sourced``). The Quantity is returned unchanged for arithmetic/``.m_as`` use, with
    ``.provenance``/``.name``/``.description`` attached for the audit trail. Use for
    registry reference values that carry units (e.g. data rates, latencies, energies).

    .. warning:: The attached attributes do NOT survive ``.to()``, arithmetic,
        or copying. They are set on this one Quantity *instance*; pint builds
        fresh Quantity objects for every conversion and operation
        (``q.to('ms')``, ``q * 2``, ``q + other``, ``copy.copy(q)`` ...), and
        those results carry no ``.provenance``/``.name``/``.description``.
        Read provenance off the registry leaf before transforming the value,
        and re-attach explicitly (kind=DERIVED) if a derived quantity needs
        its own audit trail.
    """
    quantity.provenance = provenance
    quantity.name = name
    quantity.description = description
    return quantity


def scalar_value(x: Scalar | Sourced) -> float:
    """Coerce a scalar (plain or ``Sourced``) to a plain ``float``.

    ``Sourced`` already subclasses ``float``, so ``float(x)`` handles both
    cases identically; this helper exists so call sites in generated
    calculations can state the intent ("drop provenance, keep the number")
    explicitly. Note that the returned float carries no ``.provenance`` —
    see the ``Sourced`` docstring for how provenance is lost in arithmetic.
    """
    return float(x)


def fleet_mttf_hours(
    hours: float,
    *,
    component: str,
    failure_mode: str = "",
) -> Sourced:
    """MTTF anchor for reliability registry entries."""
    from .provenance_catalog import RELIABILITY_MTTF_LITERATURE

    desc = f"Steady-state MTTF for {component} in continuous datacenter operation."
    if failure_mode:
        desc += f" Typical mode: {failure_mode}."
    return sourced(
        hours,
        RELIABILITY_MTTF_LITERATURE,
        name=f"{component} MTTF (hours)",
        description=desc,
    )
