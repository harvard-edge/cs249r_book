# mlsysim/labs/__init__.py
"""
Lab UI toolkit for the MLSys Co-Labs curriculum.

Provides the Design Ledger (persistent student state), visual style system,
and reusable UI components. Every Marimo lab notebook imports from here.
"""

from .state import DesignLedger, LedgerState
from .style import (
    COLORS,
    LAB_CSS,
    apply_plotly_theme,
    progress_bar,
    concept_section_header,
    confidence_widget,
)
from .components import (
    Card,
    MetricRow,
    ComparisonRow,
    PredictionLock,
    StakeholderMessage,
    RooflineVisualizer,
    LatencyWaterfall,
    MathPeek,
)

__all__ = [
    "DesignLedger",
    "LedgerState",
    "COLORS",
    "LAB_CSS",
    "apply_plotly_theme",
    "progress_bar",
    "concept_section_header",
    "confidence_widget",
    "Card",
    "MetricRow",
    "ComparisonRow",
    "PredictionLock",
    "StakeholderMessage",
    "RooflineVisualizer",
    "LatencyWaterfall",
    "MathPeek",
]
