import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import measure_course_budgets as budgets  # noqa: E402


def test_directory_bytes_and_report_discovery(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "data.bin").write_bytes(b"12345")
    report = output / "image-classification_min_report.json"
    report.write_text(json.dumps({"status": "passed"}) + "\n")

    assert budgets.directory_bytes(output) == 5 + report.stat().st_size
    assert budgets.find_workload_report(output, "image-classification") == {
        "status": "passed"
    }


def test_run_with_peak_rss_records_child_process(tmp_path):
    returncode, wall_seconds, peak_rss_bytes, output = budgets.run_with_peak_rss(
        [
            sys.executable,
            "-c",
            "import time; payload = bytearray(2000000); print(len(payload)); time.sleep(0.05)",
        ],
        cwd=tmp_path,
    )

    assert returncode == 0
    assert wall_seconds >= 0.05
    assert peak_rss_bytes > 0
    assert output.strip() == "2000000"
