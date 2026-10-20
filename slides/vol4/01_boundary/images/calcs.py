#!/usr/bin/env python3
"""Every number on the Volume IV Lecture 1 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (the running machine in Embodied.*, the Tempe and
Cruise records in Literature.IncidentReports, the Epoch AI series in
Literature.ModelsByDomain) and the calculations (Scenarios.*, each backed by a
solver); the chapter, The Causal Boundary, reads the same scenarios. This file
only selects values, converts units, and formats them; it types no numbers.
Each value is recorded with v() under the key the deck prints as \\val{key};
running this file writes images/numbers.tex. make_figs.py imports it.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

from mlsysim.embodied.datasheet import machine_card  # noqa: E402

Embodied, Literature, Scenarios = mlsysim.Embodied, mlsysim.Literature, mlsysim.Scenarios
Q_ = mlsysim.core.units.Q_
RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
IR = Literature.IncidentReports
G0 = Q_(1, "standard_gravity")

STOP, HALF = Scenarios.WarehouseAisleStopAtDriveLimit, Scenarios.WarehouseAisleStopHalfSpeed
LEASE2 = Scenarios.WarehouseAisleStopDoubledLease
SCANNER, STALL = Scenarios.WarehouseScannerSkippedDelay, Scenarios.WarehouseInferenceHiccup
SUPP, SUPP_CAP = Scenarios.TempeActionSuppression, Scenarios.TempeActionSuppressionAdsCap
FIRST = Scenarios.TempeFirstDetectionStop
SLICK = Scenarios.HydraulicFluidSlickTraction
VLA, GR00T = Scenarios.OpenVlaComputeGap, Scenarios.Gr00tN1ComputeGap


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


print("== Hook: Tempe at first detection (NTSB HAR-19/03)")
f = FIRST.result
v("tempe_impact_mph", IR.TempeImpactSpeedMph, "{:.0f}", "impact speed, mph")
v("tempe_detect_s", IR.TempeFirstDetectionS, "{:.1f}", "first detection, s before impact")
v("tempe_recog_s", IR.TempeEmergencyRecognitionS, "{:.1f}", "emergency recognized, s before impact")
v("tempe_supp_end_s", IR.TempeSuppressionEndS, "{:.1f}", "suppression ends, s before impact")
v("tempe_detect_ft", m(FIRST.clear_distance, "ft"), "{:.0f}", "distance at first detection, ft")
v("tempe_detect_m", m(FIRST.clear_distance, "m"), "{:.0f}", "the same, m (derived)")
v("tempe_init_mph", m(FIRST.speed, "mph"), "{:.0f}", "initial speed in footnote 91, mph")
v("tempe_stop_ntsb_s", IR.TempeFullStopTimeS, "{:.1f}", "NTSB stopping time from 44 mph at 0.92 g, s")
v("tempe_stop_m", m(f.braking_travel, "m"), "{:.1f}", "full stop from 44 mph at 0.92 g, m (derived)")
v("tempe_stop_m_round", m(f.braking_travel, "m"), "{:.0f}", "the same, whole meters")
v("tempe_stops_words", f.clear_over_stop, "words", "detection distance over full stop")
v("tempe_relabel_s", IR.TempeFirstDetectionS - IR.TempeEmergencyRecognitionS, "{:.1f}",
  "detection to emergency recognition, s")

print("== Running machine: card and Napkin Math 1.1 (stopping-budget scenario)")
r = STOP.result
v("card_rows", len(machine_card()), "words", "parameters on the machine card")
v("v_drive", m(STOP.speed, "m/s"), "{:.1f}", "drive limit, m/s")
v("a_brake", m(STOP.deceleration, "m/s**2"), "{:.0f}", "credible deceleration, m/s^2")
v("d_clear_m", m(STOP.clear_distance, "m"), "{:.2f}", "rack-end clear distance, m")
v("d_clear_mm", m(STOP.clear_distance, "mm"), "{:,.0f}", "the same, mm")
v("tau_ms", m(STOP.delay, "ms"), "{:.1f}", "pre-brake delay, ms")
v("tau_s", m(STOP.delay, "s"), "{:.4f}", "pre-brake delay, s")
v("t_age_ms", m(AISLE.perception_age, "ms"), "{:.1f}", "observation age, ms")
v("t_backbone_ms", m(AISLE.t_backbone, "ms"), "{:.0f}", "perception backbone, ms")
v("t_rest_age_ms", m(AISLE.perception_age - AISLE.t_backbone, "ms"), "{:.1f}", "rest of the observation age, ms")
v("t_lease_ms", m(AISLE.t_lease, "ms"), "{:.0f}", "chunk lease, ms")
v("t_tick_ms", m(AISLE.t_tick, "ms"), "{:.0f}", "permission tick, ms")
v("t_bus_ms", m(AISLE.t_bus, "ms"), "{:.0f}", "bus cycle, ms")
v("t_onset_ms", m(AISLE.t_brake_onset, "ms"), "{:.0f}", "brake onset, ms")
v("d_delay_mm", m(r.delay_travel, "mm"), "{:.1f}", "delay travel, mm")
v("d_brake_mm", m(r.braking_travel, "mm"), "{:.1f}", "braking travel, mm")
v("d_stop_mm", m(r.stop_distance, "mm"), "{:.1f}", "elementary stop, mm")
v("d_spare_mm", m(r.spare, "mm"), "{:.1f}", "spare inside the clear distance, mm")
v("d_per_ms", m(r.travel_over(Q_(1, "ms")), "mm"), "{:.1f}", "travel per millisecond of delay, mm")
v("d_backbone_mm", m(r.travel_over(AISLE.t_backbone), "mm"), "{:.0f}", "backbone travel, mm")
v("d_lease_mm", m(r.travel_over(AISLE.t_lease), "mm"), "{:.0f}", "lease travel, mm")

print("== Crossover")
v("v_star", m(r.crossover_speed, "m/s"), "{:.2f}", "crossover speed 2 a tau, m/s")
v("d_at_vstar_mm", m(STOP.variant(speed=r.crossover_speed).result.delay_travel, "mm"), "{:.0f}",
  "each term at the crossover, mm")
v("brake_over_delay", r.braking_over_delay, "{:.1f}", "braking over delay travel at the drive limit")
v("delay_share_pct", 100 * r.delay_share_of_braking, "{:.1f}", "delay travel added to braking, %")

print("== Estimate: half speed and a doubled lease (Checkpoint 1.1)")
h, d2 = HALF.result, LEASE2.result
v("v_half", m(HALF.speed, "m/s"), "{:.2f}", "half the drive limit, m/s")
v("two_a", m(r.crossover_speed / r.delay, "m/s**2"), "{:.0f}", "2 a_brake, the braking-term divisor, m/s^2")
v("d_delay_half_mm", m(h.delay_travel, "mm"), "{:.1f}", "half-speed delay travel, mm")
v("d_brake_half_mm", m(h.braking_travel, "mm"), "{:.1f}", "half-speed braking travel, mm")
v("d_stop_half_mm", m(h.stop_distance, "mm"), "{:.1f}", "half-speed stop, mm")
v("t_lease2_ms", m(LEASE2.delay - STOP.delay + AISLE.t_lease, "ms"), "{:.0f}", "doubled lease, ms")
v("tau2_ms", m(LEASE2.delay, "ms"), "{:.1f}", "pre-brake delay with the doubled lease, ms")
v("tau2_s", m(LEASE2.delay, "s"), "{:.4f}", "the same, s")
v("d_delay2_mm", m(d2.delay_travel, "mm"), "{:.1f}", "delay travel with the doubled lease, mm")
v("d_stop2_mm", m(d2.stop_distance, "mm"), "{:.1f}", "stop with the doubled lease, mm")
v("d_spare2_mm", m(d2.spare, "mm"), "{:.1f}", "spare with the doubled lease, mm")

print("== Tempe through the budget (suppression scenario)")
s, sc = SUPP.result, SUPP_CAP.result
v("tempe_recog_mph", m(SUPP.speed, "mph"), "{:.1f}", "speed at emergency recognition, mph")
v("tempe_supp_end_mph", m(SUPP.speed_at_delay_end, "mph"), "{:.1f}", "speed when suppression ended, mph")
v("tempe_recog_mps", m(SUPP.speed, "m/s"), "{:.1f}", "speed at emergency recognition, m/s")
v("tempe_supp_s", m(SUPP.delay, "s"), "{:.0f}", "action suppression, s")
v("tempe_supp_m", m(s.delay_travel, "m"), "{:.1f}", "travel during the suppression, m (derived)")
v("tempe_road_g", IR.TempeRoadDecelerationG, "{:.2f}", "road deceleration capability, g")
v("tempe_road_mps2", m(SUPP.deceleration, "m/s**2"), "{:.1f}", "the same, m/s^2")
v("tempe_cap_g", m(SUPP_CAP.deceleration / G0, ""), "{:.2f}", "ADS braking cap, g")
v("tempe_brake_road_m", m(s.braking_travel, "m"), "{:.1f}", "full stop at the road's capability, m")
v("tempe_brake_cap_m", m(sc.braking_travel, "m"), "{:.1f}", "full stop at the ADS cap, m")
v("tempe_vstar_mps", m(s.crossover_speed, "m/s"), "{:.1f}", "crossover for the 1 s delay, m/s")
v("tempe_vstar_mph", m(s.crossover_speed, "mph"), "{:.0f}", "the same, mph")

print("== Scanner path")
v("scanner_ms", m(SCANNER.delay, "ms"), "{:.0f}", "backbone plus lease, ms")
v("scanner_mm", m(SCANNER.result.delay_travel, "mm"), "{:.0f}", "their travel at the drive limit, mm")

print("== Five levels (running machine's rates)")
v("f_intent_hz", m(RM.intent_rate, "Hz"), "{:.0f}", "intent model, Hz")
v("f_chunk_hz", m(RM.chunk_rate, "Hz"), "{:.0f}", "chunk renewal, Hz")
v("f_tick_khz", m(RM.permission_rate, "kHz"), "{:.0f}", "permission loop, kHz")
v("f_current_khz", m(RM.current_loop_rate, "kHz"), "{:.0f}", "current loop, kHz")
v("rate_span", math.log10(m(RM.current_loop_rate / RM.intent_rate, "")), "{:.1f}",
  "intent to current loop, orders of magnitude")

print("== Safety vote: an inference stall")
v("stall_ms", m(STALL.delay, "ms"), "{:.0f}", "inference stall, ms")
v("stall_mm", m(STALL.result.delay_travel, "mm"), "{:.0f}", "travel during the stall at the drive limit, mm")

print("== Convergence: slick floor and Cruise")
k = SLICK.result
assert not k.premise_holds  # the notes say the slick floor falls below the premise (the chapter guards "quadruple")
v("mu_dry", SLICK.mu_before, "{:.1f}", "illustrative dry friction")
v("mu_slick", SLICK.mu_after, "{:.1f}", "illustrative friction with hydraulic fluid")
v("mu_required", k.required_friction, "{:.3f}", "friction the card's deceleration needs")
v("cruise_m", m(Q_(float(IR.CruisePulloverDistanceFt), "ft"), "m"), "{:.1f}", "Cruise pullover distance, m")
v("cruise_mph", IR.CruisePulloverMaxSpeedMph, "{:.1f}", "Cruise pullover top speed, mph")

print("== Epoch AI: robot policies against the language frontier")
gv, gg = VLA.result, GR00T.result
assert (gv.year, gv.reference_model, gg.year, gg.reference_model) == (
    "2024", "Llama 3.1-405B", "2025", "Grok 4")  # the slide names these models
v("gap_vla", round(gv.gap, -1), "{:,.0f}", "OpenVLA gap, as the chapter rounds it")
v("gap_gr00t", round(gg.gap, -2), "{:,.0f}", "GR00T N1 gap, as the chapter rounds it")
v("gap_vla_exact", gv.gap, "{:,.0f}", "OpenVLA gap, whole times")
v("gap_gr00t_exact", gg.gap, "{:,.0f}", "GR00T N1 gap, whole times")
v("n_robotics", gg.domain_models_with_compute, "{:.0f}", "robotics models with a compute estimate")

if __name__ == "__main__":
    write(__file__)
