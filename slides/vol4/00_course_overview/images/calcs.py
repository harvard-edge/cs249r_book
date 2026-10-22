#!/usr/bin/env python3
"""Every number on the Volume IV Lecture 0 slides and notes, read from MLSysIM.

Lecture 0 previews the course, so each number it shares with a later lecture is
read from the same entry that lecture's deck reads: the latch campaign's clean
runs (Lecture 7, Scenarios.LatchEvaluationCampaign) and the Challenger flight
record (Lecture 16, Literature.SpaceflightAnomalies).

Sources behind the values
  Waymo miles       vol2/16_responsible_ai/data/waymo_rider_only_miles.csv (Safety Impact Data Hub)
  Robot fleets      Literature.RobotDeployments (IFR World Robotics 2025; Amazon, July 2025)
  ER-FLEX body      Literature.PlatformConfigurations (UR16e datasheet; MiR250 Specifications)

This file only selects values, converts units, and formats them; it types no
numbers. Run from this folder: python3 calcs.py (writes images/numbers.tex).
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios
WAYMO = load_series("vol2/16_responsible_ai/data/waymo_rider_only_miles.csv").entries
FLEETS, BODY = Literature.RobotDeployments, Literature.PlatformConfigurations
CAMPAIGN = Scenarios.LatchEvaluationCampaign
YEAR_START, YEAR_END = "2025-06", "2026-06"   # the twelve months the vote asks about


def waymo_rows():
    """(data-through month, cumulative rider-only miles in millions), in release order."""
    return [(r["data_through"], r["cumulative_ro_miles_millions"]) for r in WAYMO]


def month(ym):
    return date.fromisoformat(ym + "-01")


miles = dict(waymo_rows())
first_ym, last_ym = waymo_rows()[0][0], waymo_rows()[-1][0]
year = miles[YEAR_END] - miles[YEAR_START]
year_days = (month(YEAR_END) - month(YEAR_START)).days

print("== Waymo rider-only miles")
v("wm_year", year, "{:.0f}", "miles in the twelve months, millions")
v("wm_year_1d", year, "{:.1f}", "the same, one decimal")
v("wm_ch_a", year / 100, "{:.2f}", "vote choice A, millions (a hundredth)")
v("wm_ch_b", year / 10, "{:.1f}", "vote choice B, millions (a tenth)")
v("wm_ch_d", year * 10 / 10 ** 3, "{:.2f}", "vote choice D, billions (ten times)")
v("wm_per_day", round(year * 10 ** 6 / year_days, -4), "{:,.0f}", "miles a day over the year, nearest 10,000")
v("wm_first", miles[first_ym], "{:.1f}", "June 2024 cumulative, millions")
v("wm_start", miles[YEAR_START], "{:.1f}", "June 2025 cumulative, millions")
v("wm_dec25", miles["2025-12"], "{:.1f}", "December 2025 cumulative, millions")
v("wm_last", miles[last_ym], "{:.1f}", "June 2026 cumulative, millions")
v("wm_growth", miles[last_ym] / miles[first_ym], "{:.1f}", "growth over two years, x")

print("== Robot fleets")
stock, installed = FLEETS.IfrRobotsInOperation2024, FLEETS.IfrRobotsInstalled2024
v("ifr_stock_m", stock / 10 ** 6, "{:.2f}", "industrial robots in operation, millions")
v("ifr_installed", installed, "{:,.0f}", "installed in 2024")
v("amazon_robots", FLEETS.AmazonRobotsDeployed2025, "{:,.0f}", "Amazon robots deployed")
v("amazon_gain_pct", FLEETS.AmazonDeepFleetTravelGainPct, "{:.0f}", "DeepFleet travel gain Amazon expects, %")

print("== The ER-FLEX body (datasheets)")
v("mir_mass_kg", BODY.Mir250MassKg, "{:.0f}", "MiR250 weight, kg")
v("mir_payload_kg", BODY.Mir250PayloadKg, "{:.0f}", "MiR250 payload, kg")
v("mir_scanners_words", BODY.Mir250SafetyLaserScanners, "words", "safety laser scanners, spelled")
v("mir_cameras_words", BODY.Mir250DepthCameras, "words", "3D cameras, spelled")
v("ur_joints", BODY.Ur16eJoints, "{:.0f}", "UR16e joints")
v("ur_joints_words", BODY.Ur16eJoints, "words", "the same, spelled")
v("ur_payload_kg", BODY.Ur16ePayloadKg, "{:.0f}", "UR16e payload, kg")
v("ur_reach_mm", BODY.Ur16eReachMm, "{:.0f}", "UR16e reach, mm")

print("== Previews of later lectures")
v("latch_clean", CAMPAIGN.clean_runs, "{:.0f}", "clean door runs (Lecture 7)")
v("latch_clean_words", CAMPAIGN.clean_runs, "words", "the same, spelled")
v("latch_next", CAMPAIGN.clean_runs + 1, "{:.0f}", "the next run")
v("challenger_prior", Literature.SpaceflightAnomalies.ChallengerPriorJointFlights, "{:.0f}",
  "flights before Challenger (Lecture 16)")

if __name__ == "__main__":
    write(__file__)
