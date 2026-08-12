"""Style compatibility exports plus the new academic lab CSS."""

from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme, confidence_widget, concept_section_header, progress_bar

from .ui import ACADEMIC_LAB_CSS

__all__ = [
    "COLORS",
    "LAB_CSS",
    "ACADEMIC_LAB_CSS",
    "apply_plotly_theme",
    "confidence_widget",
    "concept_section_header",
    "progress_bar",
]
