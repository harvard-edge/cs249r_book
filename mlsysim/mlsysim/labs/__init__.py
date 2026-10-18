# mlsysim/labs/__init__.py
"""
Lab UI toolkit for the MLSys Co-Labs curriculum.

Provides the Design Ledger (persistent student state), visual style system,
and reusable UI components. Every Marimo lab notebook imports from here.
"""

from .state import DesignLedger, LedgerState

_UI_EXPORTS = {
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
    "DecisionLog",
    "FailureBanner",
}

try:
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
        DecisionLog,
        FailureBanner,
    )
    _UI_AVAILABLE = True
except ImportError:
    _UI_AVAILABLE = False


def __getattr__(name: str):
    if name == "experiments":
        import importlib
        return importlib.import_module(".experiments", __name__)
    if not _UI_AVAILABLE and name in _UI_EXPORTS:
        raise ImportError(
            f"'{name}' requires the 'labs' extra (marimo). "
            f"Install it with: pip install 'mlsysim[labs]'"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DesignLedger",
    "LedgerState",
    "experiments",
    *sorted(_UI_EXPORTS),
]

