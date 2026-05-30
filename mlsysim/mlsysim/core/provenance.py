"""Provenance types for registry entries and book-facing sourced scalars."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator

Scalar = Union[int, float]


class ProvenanceKind(str, Enum):
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
        
        Requires URLs for datasheets, and notes for estimates and derived values,
        ensuring proper traceability and justification for textbook numbers.
        """
        if self.kind == ProvenanceKind.DATASHEET and not self.url:
            raise ValueError(f"datasheet provenance requires url: {self.ref!r}")
        if self.kind == ProvenanceKind.ESTIMATE and not self.notes:
            raise ValueError(f"estimate provenance requires notes: {self.ref!r}")
        if self.kind == ProvenanceKind.DERIVED and not self.notes:
            raise ValueError(f"derived provenance requires notes: {self.ref!r}")
        return self


class Sourced(float):
    """
    Scalar with mandatory ``Provenance``. Subclasses ``float`` so appendix
    LEGO cells can divide and format values without extra coercion.
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
    """Attach provenance to a scalar used in registries or appendices."""
    return Sourced(value, provenance, name=name, description=description)


def sourced_qty(quantity, provenance: Provenance, *, name: str = "", description: str = ""):
    """Attach provenance to a unit-bearing pint Quantity (the Quantity analogue of
    ``sourced``). The Quantity is returned unchanged for arithmetic/``.m_as`` use, with
    ``.provenance``/``.name``/``.description`` attached for the audit trail. Use for
    registry reference values that carry units (e.g. data rates, latencies, energies)."""
    quantity.provenance = provenance
    quantity.name = name
    quantity.description = description
    return quantity


def scalar_value(x: Scalar | Sourced) -> float:
    """Plain float for arithmetic and Quarto ``{python}`` cells."""
    if isinstance(x, Sourced):
        return float(x)
    return float(x)


def fleet_mttf_hours(
    hours: float,
    *,
    component: str,
    failure_mode: str = "",
) -> Sourced:
    """MTTF anchor aligned with appendix_reliability @tbl-component-fit."""
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
