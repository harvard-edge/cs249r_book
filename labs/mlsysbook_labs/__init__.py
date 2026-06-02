"""Shared helpers for MLSysBook interactive trade-off labs."""

from .legacy_components import (
    Card,
    ComparisonRow,
    DecisionLog,
    FailureBanner,
    LatencyWaterfall,
    MathPeek,
    MetricRow,
    PredictionLock,
    RooflineVisualizer,
    StakeholderMessage,
)
from .catalog import LAB_CATALOG, get_lab_metadata
from .reports import build_lab_report, report_export
from .schemas import (
    DEFAULT_TRACKS,
    ChapterRecap,
    InstructorMetadata,
    LabMetadata,
    LabReport,
    LedgerEntry,
    NuggetSpec,
    TrackSpec,
)
from .state import DesignLedger, LedgerState
from .style import ACADEMIC_LAB_CSS, COLORS, LAB_CSS, apply_plotly_theme
from .ui import (
    advanced_knob_drawer,
    chapter_recap,
    decision_card,
    instructor_adoption_card,
    lab_header,
    nugget_shell,
    reflection_card,
    scenario_brief,
    track_selector,
)
from .versions import LEDGER_SCHEMA_VERSION, MLSYSBOOK_LABS_VERSION, REPORT_SCHEMA_VERSION, STABLE_RELEASE_CHANNEL

__all__ = [
    "ACADEMIC_LAB_CSS",
    "COLORS",
    "LAB_CSS",
    "apply_plotly_theme",
    "DesignLedger",
    "LedgerState",
    "Card",
    "ComparisonRow",
    "DecisionLog",
    "FailureBanner",
    "LatencyWaterfall",
    "MathPeek",
    "MetricRow",
    "PredictionLock",
    "RooflineVisualizer",
    "StakeholderMessage",
    "LabMetadata",
    "ChapterRecap",
    "TrackSpec",
    "NuggetSpec",
    "InstructorMetadata",
    "LedgerEntry",
    "LabReport",
    "LAB_CATALOG",
    "DEFAULT_TRACKS",
    "MLSYSBOOK_LABS_VERSION",
    "REPORT_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "STABLE_RELEASE_CHANNEL",
    "lab_header",
    "chapter_recap",
    "scenario_brief",
    "track_selector",
    "nugget_shell",
    "advanced_knob_drawer",
    "reflection_card",
    "decision_card",
    "instructor_adoption_card",
    "build_lab_report",
    "report_export",
    "get_lab_metadata",
]
