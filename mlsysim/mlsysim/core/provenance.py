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

    provenance: Provenance
    name: str
    description: str

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

    @property
    def magnitude(self) -> float:
        """Quantity-compatible alias returning the underlying float value."""
        return float(self)

    def to(self, unit: str = "dimensionless"):
        """Convert the sourced scalar into a dimensionless Pint quantity.

        Parameters
        ----------
        unit : str, optional
            Target unit compatible with dimensionless quantities; defaults to
            dimensionless. Percent is supported; physical units are not.

        Returns
        -------
        Quantity
            Converted value in the requested unit, without attached provenance.

        Notes
        -----
        The stored scalar is dimensionless. Conversion delegates to
        Pint and raises its dimensionality error for incompatible target units.

        Conversion does not assign physical dimensions, preserve provenance, or
        infer units from the scalar's name or description.

        Examples
        --------
        >>> p = Provenance(kind=ProvenanceKind.ILLUSTRATIVE, ref="Example", verified="2026-11-04")
        >>> Sourced(0.5, p).to("percent").magnitude
        50.0
        """
        from .units import Q_
        return Q_(float(self), "dimensionless").to(unit)

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
    """Extract a plain float from a sourced or ordinary scalar.

    Parameters
    ----------
    x : int, float, or Sourced
        Numeric scalar whose magnitude is to be retained.

    Returns
    -------
    float
        Numeric value without provenance attributes or units.

    Notes
    -----
    The input must support Python float conversion. Sourced subclasses
    float, so conversion follows the same path as for an ordinary scalar.

    The conversion does not propagate provenance or perform unit conversion.
    Use the registry leaf to inspect evidence before transforming its value.

    Examples
    --------
    >>> scalar_value(3)
    3.0
    """
    return float(x)


def fleet_mttf_hours(
    hours: float,
    *,
    component: str,
    failure_mode: str = "",
) -> Sourced:
    """Attach the reliability catalog provenance to an MTTF anchor.

    Parameters
    ----------
    hours : float
        Supplied mean time to failure in hours. Callers are responsible for
        selecting a physically meaningful positive value.
    component : str
        Component name used in the anchor name and description.
    failure_mode : str, optional
        Typical failure mode appended to the description; empty by default.

    Returns
    -------
    Sourced
        The supplied numeric hours with catalog provenance and descriptive
        labels. This scalar is not a unit-bearing Pint quantity.

    Notes
    -----
    The caller must select a steady-state MTTF appropriate to
    continuous datacenter operation. The catalog supplies the evidence label.

    The helper does not estimate MTTF, check a value against the cited literature,
    aggregate component hazards, or validate a reliability distribution.

    Examples
    --------
    >>> anchor = fleet_mttf_hours(1000, component="Illustrative component")
    >>> float(anchor)
    1000.0
    >>> anchor.name
    'Illustrative component MTTF (hours)'
    """
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
