"""Check registration registry for Physical AI CLI."""

from .base import BaseCheck, CheckRegistry
from .citations import CitationBibTeXCheck
from .cross_references import CrossReferenceCheck
from .orphans import OrphanArtifactCheck
from .formatting import CalloutListFormatCheck
from .editorial import BannedTerminologyCheck
from .structure import ChapterOpenerStructureCheck
from .latex_log import LaTeXLogDiagnosticsCheck
from .layout_margins import VisualMarginBoundingBoxCheck
from .playwright_visual import PlaywrightVisualCheck

__all__ = [
    "BaseCheck",
    "CheckRegistry",
    "CitationBibTeXCheck",
    "CrossReferenceCheck",
    "OrphanArtifactCheck",
    "CalloutListFormatCheck",
    "BannedTerminologyCheck",
    "ChapterOpenerStructureCheck",
    "LaTeXLogDiagnosticsCheck",
    "VisualMarginBoundingBoxCheck",
    "PlaywrightVisualCheck",
]
