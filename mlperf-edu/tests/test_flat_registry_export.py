"""The packaged flat registry must track the source registry.

`workloads.yaml` and its packaged copy are what an installed wheel reads. When
they drift from `registry/`, the installed suite runs a different contract than
the repository describes, silently. That happened: after recommendation moved
from DLRM on Criteo Terabyte to NCF on MovieLens-20M, the export was never
rerun, so the packaged registry still dispatched to the DLRM runner and still
described a Criteo-gated 256 GB host.

The exporter has always had a --check mode. Nothing called it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_flat_registry_is_current():
    result = subprocess.run(
        [sys.executable, "tools/export_flat_registry.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "the exported flat registry is stale; an installed wheel would run a "
        f"different contract than registry/ describes.\n{result.stdout}{result.stderr}"
    )


def test_recommendation_dispatches_to_the_ncf_runner():
    """A regression guard on the specific drift that motivated this file."""
    import yaml

    for path in (ROOT / "workloads.yaml", ROOT / "src/mlperf_edu/workloads.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "mlperf.runners.ncf:run_recommendation_max" in text, path
        assert "mlperf.runners.recommendation:run_recommendation_max" not in text, path
        yaml.safe_load(text)
