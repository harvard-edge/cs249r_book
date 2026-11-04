"""The published calibration table must be generated from executed anchors."""
from pathlib import Path
import runpy


def test_calibration_report_is_current():
    root = Path(__file__).resolve().parents[1]
    script = runpy.run_path(str(root / "scripts" / "calibration_report.py"))
    assert script["DESTINATION"].read_text() == script["render"](), (
        "Run python scripts/calibration_report.py to refresh the report."
    )
