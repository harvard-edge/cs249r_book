from mlsysim.core.series_loader import load_series
"""The two Waymo Safety Impact tables must agree wherever they overlap.

``load_series("vol4/16_release/data/waymo_safety_impact_headline.csv")`` (reduction by data cut) and
``load_series("vol2/16_responsible_ai/data/waymo_rider_only_miles.csv")`` / ``WaymoVsHumanBenchmark`` (mileage
by release; latest cut against every benchmark variant) are built separately
from the same Waymo releases. Added 2026-10-13: the benchmark table had been
built from the superseded 2026-09-24 CSV3, which Waymo's 2026-10-02 release
notes replace with a corrected _v2 file (Atlanta dynamic benchmark), so the two
tables disagreed at the 202606 cut.
"""

import pytest

from mlsysim.literature.registry import Literature

HEADLINE = load_series("vol4/16_release/data/waymo_safety_impact_headline.csv").entries
MILES = load_series("vol2/16_responsible_ai/data/waymo_rider_only_miles.csv").entries
BENCH = load_series("vol2/16_responsible_ai/data/waymo_vs_human_benchmark.csv").entries

# Headline outcome name -> benchmark-table row using the same dynamic benchmark.
DYNAMIC_ROW = {
    "Any Airbag Deployment": "Any Airbag Deployment",
    "Any Fatal": "Any Fatal (Dynamic)",
    "Any-injury-reported": "Any-injury-reported",
    "Ego Vehicle Airbag Deployment": "Ego Vehicle Airbag Deployment",
    "Police-reported": "Police-reported",
    "Suspected Serious Injury+": "Suspected Serious Injury+",
}


def test_cumulative_miles_agree_at_shared_cuts():
    by_cut = {r["data_through"].replace("-", ""): r["cumulative_ro_miles_millions"] for r in MILES}
    shared = 0
    for r in HEADLINE:
        cut = str(int(r["data_cut"]))
        if cut in by_cut:
            shared += 1
            assert r["ro_miles_millions"] == pytest.approx(by_cut[cut], abs=1e-3), cut
    assert shared > 0


@pytest.mark.parametrize("outcome", sorted(DYNAMIC_ROW))
def test_latest_cut_rates_agree(outcome):
    latest = max(int(r["data_cut"]) for r in HEADLINE)
    head = [r for r in HEADLINE if int(r["data_cut"]) == latest and r["outcome"] == outcome]
    bench = [r for r in BENCH if r["benchmark"] == DYNAMIC_ROW[outcome]]
    assert len(head) == 1 and len(bench) == 1
    h, b = head[0], bench[0]
    assert h["waymo_count"] == b["waymo_count"]
    assert h["waymo_ipmm"] == pytest.approx(b["waymo_ipmm"], rel=1e-6)
    assert h["benchmark_ipmm"] == pytest.approx(b["bench_ipmm"], rel=1e-6)
    assert h["pct_reduction"] == pytest.approx(b["reduction"], rel=1e-6)
