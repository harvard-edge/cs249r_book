#!/usr/bin/env python3
"""Every number on the Lecture 15 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature) and the calculations
(Scenarios.*, each backed by a solver in engine/solvers/sustainability.py);
the chapter reads the same scenarios. This file only selects values,
converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex.

Running case: Scenarios.Llama31_405BTrainingFootprint (Llama 3.1 model card
GPU-hours on up to 16,384 H100s, DGX H100 rack ratio, PUE 1.12, nine grids,
three-year life). Outside tables: Literature.Series (Patterson Table 4,
BLOOM Table 5, CSO Ireland Table 1, Energy-Charts EU week 47 of 2023,
Elsworth et al. Gemini prompt energy and serving factors) and Literature
scalars (Llama 3.1 card, BLOOM Table 3, TPU v6e lifecycle, LBNL).

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Infrastructure, Systems = mlsysim.Hardware, mlsysim.Infrastructure, mlsysim.Systems
Literature, Scenarios, Models = mlsysim.Literature, mlsysim.Scenarios, mlsysim.Models
SERIES, RUNS, LIFE = Literature.Series, Literature.TrainingRuns, Literature.AcceleratorLifecycle
H100, RACK, COOL = Hardware.Cloud.H100, Systems.Racks.DGX_H100_4Node, Infrastructure.FacilityCooling

RUN = Scenarios.Llama31_405BTrainingFootprint
R = RUN.result
LIGHT = Scenarios.Reference25KFacilityPower
FEED14K = Scenarios.FeedSizing14KGpus
CAP = Scenarios.FeedCappedTrainingRun
RETRO = Scenarios.PueRetrofit2MW
ROLL = Scenarios.Training10KCoolingRollback
SITES = Scenarios.PueVersusGridSites
DECODE = Scenarios.BatchOneDecodeWeightEnergy
PAY = Scenarios.H100ReplacementPayback
LIFEX = Scenarios.AcceleratorLifeExtension
SWEEP = Scenarios.CandidateSweep50
SERVE = Scenarios.GeminiPromptsAgainstLlama31Training
LAB = Scenarios.TenMillionQueriesAgainstLab64Training
CASCADE = Scenarios.EdgeFirstImageCascade
WAKE = Scenarios.WakeWordHierarchyPower
GEO = Scenarios.QuebecVersusPolandPlacement
PRICED = Scenarios.CarbonPricedRegionPlacement
EU = Scenarios.EuropeWeek47FossilShare
WATER = Scenarios.ReferenceItLoadAnnualWater
JEV, JEV3 = Scenarios.HalvedQueryEnergyRebound, Scenarios.TokenPriceRebound
KAPLAN = Scenarios.KaplanTenPercentLowerLoss
PGD = Scenarios.PgdSevenTraining
SYNC = Scenarios.Gpt3GradientSync


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def row(series, key, value):
    """The entry of a Literature series whose ``key`` equals ``value``."""
    return next(r for r in series.entries if r[key] == value)


def factor(name):
    """A Google serving factor (Elsworth et al. 2025) by parameter name."""
    return row(SERIES.GoogleServingFactors, "parameter", name)["value"]


def prompt(prefix, column):
    """One component of the median Gemini prompt's energy (Wh)."""
    return next(r[column] for r in SERIES.GeminiPromptEnergy.entries if r["component"].startswith(prefix))


print("== Running case: Llama 3.1 405B")
v("gpus", RUN.devices, "{:,.0f}", "H100s in the run")
v("tdp_w", m(H100.tdp, "W"), "{:,.0f}", "H100 TDP, W")
v("card_gpuh_m", m(RUN.device_hours, "hour") / 10 ** 6, "{:.2f}", "model card GPU-hours, millions")
v("card_lb_t", float(RUNS.Llama31_405B_LocationBasedTco2e), "{:,.0f}", "card location-based t CO2eq")
v("card_mb_t", float(RUNS.Llama31_405B_MarketBasedTco2e), "{:.0f}", "card market-based t CO2eq")
v("p_acc_mw", m(R.power.accelerator_power, "MW"), "{:.2f}", "accelerator power, MW")
v("p_it_mw", m(R.power.it_power, "MW"), "{:.2f}", "IT power, MW")
v("p_fac_mw", m(R.power.facility_power, "MW"), "{:.2f}", "facility power, MW")
v("p_fac_mw1", m(R.power.facility_power, "MW"), "{:.1f}", "facility power, MW (one decimal)")
v("it_ratio", RUN.it_ratio, "{:.3f}", "IT / accelerator power, DGX H100 rack")
v("it_ratio2", RUN.it_ratio, "{:.2f}", "the same, two decimals")
v("pue_run", RUN.pue, "{:.2f}", "run PUE (best-in-class air)")
v("run_hours", m(R.duration, "hour"), "{:,.0f}", "hours if every GPU ran throughout")
v("run_days", m(R.duration, "day"), "{:.0f}", "the same, days")
v("e_acc_mwh", m(R.accelerator_energy, "MWh"), "{:,.0f}", "accelerator energy, MWh")
v("e_it_mwh", m(R.it_energy, "MWh"), "{:,.0f}", "IT energy, MWh")
v("e_fac_mwh", m(R.facility_energy, "MWh"), "{:,.0f}", "facility energy, MWh")
v("e_acc_gwh", m(R.accelerator_energy, "GWh"), "{:.1f}", "accelerator energy, GWh")
v("e_it_gwh", m(R.it_energy, "GWh"), "{:.1f}", "IT energy, GWh")
v("e_fac_gwh", m(R.facility_energy, "GWh"), "{:.1f}", "facility energy, GWh")
GRID_KEYS = {"Norway": "norway", "Quebec": "quebec", "France": "france", "EU average": "eu",
             "Germany": "germany", "US average": "us", "China": "china", "India": "india", "Poland": "poland"}
for name, key in GRID_KEYS.items():
    v(f"g_{key}", RUN.grids[name].carbon_intensity_g_kwh, "{:.0f}", f"{name} grid, g/kWh")
    v(f"co2_{key}", m(R.carbon[name], "metric_ton"), "{:,.0f}", f"run on {name}'s grid, t CO2")
v("region_spread", Scenarios.RegionalGridPlacement.result.spread, "{:.0f}", "Poland / Norway grid factor")
v("card_implied_g", m(Literature.TrainingRuns.Llama31_405B_LocationBasedTco2e * mlsysim.core.units.Q_(1, "metric_ton")
                      / R.accelerator_energy, "g/kWh"), "{:.0f}", "card t over GPU-h x TDP, g/kWh")
v("fac_over_card", m(R.carbon["US average"], "metric_ton") / float(RUNS.Llama31_405B_LocationBasedTco2e),
  "{:.2f}", "facility account / model card")
v("emb_kg", m(H100.embodied_carbon, "kg"), "{:.0f}", "H100 embodied carbon, kg")
v("hgx_kg", m(H100.embodied_carbon * Systems.Nodes.DGX_H100.accelerators_per_node, "kg"), "{:,.0f}",
  "eight-GPU HGX baseboard, kg")
v("emb_made_t", m(R.embodied_manufactured, "metric_ton"), "{:,.0f}", "fleet embodied at manufacture, t")
v("emb_alloc_t", m(R.embodied_allocated, "metric_ton"), "{:.0f}", "run's time share, t")
v("life_h", m(RUN.service_life, "hour"), "{:,.0f}", "service life, h")
v("life_y", Infrastructure.Pricing.Capital.AcceleratorServiceLifeYears.rate, "{:.0f}", "service life, years")
v("emb_share_us", 100 * R.embodied_share["US average"], "{:.1f}", "embodied share on US grid, %")
v("emb_share_qc", 100 * R.embodied_share["Quebec"], "{:.0f}", "embodied share on Quebec grid, %")
v("water_lo_ml", m(R.water["Google fleet"], "liter") / 10 ** 6, "{:.0f}", "run water at Google WUE, million L")
v("water_hi_ml", m(R.water["evaporative"], "liter") / 10 ** 6, "{:.0f}", "run water at evaporative WUE, million L")
v("wue_google", m(RUN.wue["Google fleet"], "liter/kWh"), "{:.2f}", "Google fleet WUE, L/kWh")
v("wue_evap", m(RUN.wue["evaporative"], "liter/kWh"), "{:.1f}", "evaporative WUE, L/kWh")

print("== Ceiling")
v("lh_gpus", LIGHT.devices, "{:,.0f}", "Lighthouse fleet GPUs")
v("lh_mw", m(LIGHT.result.accelerator_power, "MW"), "{:.1f}", "Lighthouse accelerator power, MW")
v("rack_gpus", RACK.node.accelerators_per_node * RACK.nodes_per_rack, "{:.0f}", "GPUs per rack")
v("rack_gpu_kw", m(RACK.accelerator_power, "kW"), "{:.1f}", "rack GPU power, kW")
v("rack_host_kw", m(RACK.host_power, "kW"), "{:.1f}", "hosts and DRAM, kW")
v("rack_nvs_kw", m(RACK.switch_power, "kW"), "{:.1f}", "NVSwitch, kW")
v("rack_ib_kw", m(RACK.network_power, "kW"), "{:.1f}", "InfiniBand adapters, kW")
v("rack_conv_kw", m(RACK.conversion_loss_power, "kW"), "{:.1f}", "conversion inside IT, kW")
v("rack_support_kw", m(RACK.it_power - RACK.accelerator_power, "kW"), "{:.1f}", "support above the GPUs, kW")
v("rack_it_kw", m(RACK.it_power, "kW"), "{:.1f}", "rack IT load, kW")
v("rack_cool_kw", m(RACK.cooling_allocation_power, "kW"), "{:.1f}", "allocated facility overhead, kW")
v("rack_total_kw", m(RACK.total_power, "kW"), "{:.1f}", "rack total, kW")
v("rack_pue", ratio(RACK.total_power, RACK.it_power), "{:.2f}", "PUE implied by the allocation")
v("rack_total_ratio", ratio(RACK.total_power, RACK.accelerator_power), "{:.2f}", "rack total / GPUs")
v("rack_total_ratio1", ratio(RACK.total_power, RACK.accelerator_power), "{:.1f}", "the same, one decimal")
v("rack_gpu_share", 100 * ratio(RACK.accelerator_power, RACK.total_power), "{:.0f}", "GPU share of the rack, %")
v("feed_gpus", FEED14K.devices, "{:,.0f}", "GPUs a planner fits under the feed")
v("feed_acc_mw", m(FEED14K.result.accelerator_power, "MW"), "{:.1f}", "their accelerator power, MW")
v("feed_it_mw", m(FEED14K.result.it_power, "MW"), "{:.1f}", "their IT load, MW")
v("substation_months", Infrastructure.Capacity.SubstationLeadTimeMonths, "{:.0f}", "substation lead time, months")
v("gpu_lead_months", Infrastructure.Capacity.GpuLeadTimeMonths, "{:.0f}", "GPU lead time, months")
v("cluster10k_gpus", ROLL.fleet.total_accelerators, "{:,.0f}", "10K training cluster GPUs")
v("cap_req_mw", m(CAP.required_power, "MW"), "{:.0f}", "run power, MW")
v("cap_feed_mw", m(CAP.feed_power, "MW"), "{:.0f}", "grid feed, MW")
v("cap_pct", 100 * CAP.result.concurrency, "{:.0f}", "concurrency the feed allows, %")
v("cap_days", m(CAP.result.stretched_duration, "day"), "{:.1f}", "stretched run, days")
v("cap_plan_days", m(CAP.planned_duration, "day"), "{:.0f}", "planned run, days")
v("cap_plan_days_w", m(CAP.planned_duration, "day"), "words", "the same, in prose")

print("== Account")
v("pue_it_mw", m(RETRO.it_power, "MW"), "{:.0f}", "PUE example IT load, MW")
v("pue_legacy", RETRO.pue_before, "{:.2f}", "legacy air-cooled PUE")
v("pue_survey", COOL.Typical.pue, "{:.2f}", "Uptime 2022 survey average PUE")
v("pue_soa", RETRO.pue_after, "{:.2f}", "state-of-the-art PUE")
v("pue_google", factor("fleet_pue"), "{:.2f}", "Google fleet PUE")
PUE_ROWS = {"legacy": RETRO.pue_before, "survey": COOL.Typical.pue, "soa": RETRO.pue_after,
            "google": factor("fleet_pue")}
for key, pue in PUE_ROWS.items():
    v(f"pue_fac_{key}", m(RETRO.variant(pue_before=pue).result.facility_power_before, "MW"), "{:.2f}",
      f"facility MW at the {key} PUE")
v("pue_saved_mw", m(RETRO.result.saved_power, "MW"), "{:.2f}", "power saved, MW")
v("hours_year", m(RETRO.period, "hour"), "{:,.0f}", "hours in a year")
v("pue_saved_mwh", m(RETRO.result.saved_energy, "MWh"), "{:,.1f}", "energy saved a year, MWh")
v("pue_saved_mwh0", m(RETRO.result.saved_energy, "MWh"), "{:,.0f}", "the same, whole MWh")
v("pue_saved_usd", m(RETRO.result.saved_cost, "USD"), "{:,.0f}", "dollars saved a year")
v("price_mwh", m(RETRO.electricity_price, "USD/MWh"), "{:.0f}", "electricity, $/MWh")
v("rb_min_lo", m(ROLL.rollbacks["low"], "minute"), "{:.0f}", "rollback, low, min")
v("rb_min_hi", m(ROLL.rollbacks["high"], "minute"), "{:.0f}", "rollback, high, min")
v("rb_mwh_lo", m(ROLL.result.energy["low"], "MWh"), "{:.1f}", "repeated energy, low, MWh")
v("rb_mwh_hi", m(ROLL.result.energy["high"], "MWh"), "{:.0f}", "repeated energy, high, MWh")
full = sum(r["comprehensive_wh"] for r in SERIES.GeminiPromptEnergy.entries if r["comprehensive_counted"] == 1)
narrow = sum(r["existing_wh"] for r in SERIES.GeminiPromptEnergy.entries if r["existing_counted"] == 1)
v("prompt_narrow_wh", narrow, "{:.2f}", "active accelerators only, Wh")
v("prompt_full_wh", full, "{:.2f}", "full serving stack, Wh")
for key, prefix in (("accel", "Active AI"), ("host", "Active host"), ("idle", "Idle"), ("ovh", "Data center")):
    v(f"prompt_{key}_wh", prompt(prefix, "comprehensive_wh"), "{:.2f}", f"{prefix}, full stack, Wh")
    v(f"prompt_{key}_x_wh", prompt(prefix, "existing_wh"), "{:.2f}", f"{prefix}, narrow table, Wh")
v("prompt_ratio", full / narrow, "{:.1f}", "full / narrow boundary")
v("prompt_accel_pct", 100 * prompt("Active AI", "comprehensive_wh") / full, "{:.0f}", "accelerator share, %")
v("lb_g", factor("lb_emission_factor_2024"), "{:.0f}", "Google 2024 location-based, g/kWh")
v("mb_g", factor("mb_emission_factor_2024"), "{:.0f}", "Google 2024 market-based, g/kWh")
v("lb_mb_ratio", factor("lb_emission_factor_2024") / factor("mb_emission_factor_2024"), "{:.1f}", "LB / MB")
v("prompt_water_ml", factor("prompt_water"), "{:.2f}", "water per prompt, mL")
v("prompt_cut", factor("energy_reduction_12_months"), "{:.0f}", "one-year prompt energy reduction")
d = DECODE.result
v("dram_pj", m(DECODE.energy_per_byte, "pJ/byte"), "{:.0f}", "DRAM energy, pJ/byte")
v("flop_pj", m(DECODE.energy_per_flop, "pJ/flop"), "{:.0f}", "GPU FLOP energy, pJ")
v("operand_bytes", m(DECODE.operand_bytes, "byte"), "{:.0f}", "FP32 operand, bytes")
v("operand_pj", m(d.operand_energy, "pJ"), "{:.0f}", "operand fetch, pJ")
v("operand_ratio", d.operand_over_flop, "{:.0f}", "operand fetch / FLOP")
v("crossover_fpb", d.crossover_intensity, "{:.0f}", "energy crossover, FLOP/byte")
v("fp16_bytes", m(DECODE.bytes_per_param, "byte"), "{:.0f}", "FP16 weight, bytes")
v("decode_flops_w", DECODE.flops_per_param, "words", "FLOPs per weight per token")
v("decode_fpb", d.intensity, "{:.0f}", "batch-1 decode, FLOP/byte")
v("move_pj", m(d.movement_energy, "pJ"), "{:.0f}", "movement per weight, pJ")
v("compute_pj", m(d.compute_energy, "pJ"), "{:.0f}", "arithmetic per weight, pJ")
v("move_pct", 100 * d.movement_share, "{:.0f}", "movement share of modeled energy, %")
v("ib_pj_bit", m(SYNC.energy_per_bit, "pJ/bit"), "{:.0f}", "fabric energy, pJ/bit")
v("sync_j", m(SYNC.result.energy_per_step, "J"), "{:.0f}", "fabric energy per worker per step, J")
TPU3 = [r for r in SERIES.PattersonTrainingFootprints.entries if r["processor"] == "TPU v3"]
meas = [r["measured_w"] for r in TPU3]
v("tpu3_tdp", TPU3[0]["tdp_w"], "{:.0f}", "TPU v3 TDP, W")
v("tpu3_meas_lo", min(meas), "{:.0f}", "TPU v3 measured power, low, W")
v("tpu3_meas_hi", max(meas), "{:.0f}", "TPU v3 measured power, high, W")
v("tpu3_lo_pct", 100 * min(meas) / TPU3[0]["tdp_w"], "{:.0f}", "low as % of TDP")
v("tpu3_hi_pct", 100 * max(meas) / TPU3[0]["tdp_w"], "{:.0f}", "high as % of TDP")
v("tpu3_runs_w", len(TPU3), "words", "TPU v3 runs in Table 4")
rates = SITES.result.rates
v("site_it_mw", m(SITES.it_power, "MW"), "{:.0f}", "self-check IT load, MW")
v("site_a_pue", SITES.pues["A"], "{:.2f}", "site A PUE")
v("site_b_pue", SITES.pues["B"], "{:.2f}", "site B PUE")
v("site_a_g", m(SITES.intensities["A"], "g/kWh"), "{:.0f}", "site A grid, g/kWh")
v("site_b_g", m(SITES.intensities["B"], "g/kWh"), "{:.0f}", "site B grid, g/kWh")
v("site_a_t", m(rates["A"], "metric_ton/hour"), "{:.2f}", "site A, t/h")
v("site_b_t", m(rates["B"], "metric_ton/hour"), "{:.3f}", "site B, t/h")
v("site_ratio", ratio(rates["A"], rates["B"]), "{:.1f}", "A / B")
PAT = {r["model"]: r for r in SERIES.PattersonTrainingFootprints.entries}
v("gshard_t", PAT["GShard-600B"]["net_tco2e"], "{:.1f}", "GShard-600B net t CO2e")
v("gshard_act_pct", PAT["GShard-600B"]["activated_pct"], "{:.2f}", "GShard weights active per token, %")
v("t5_t", PAT["T5"]["net_tco2e"], "{:.1f}", "T5 net t CO2e")
v("t5_params_b", PAT["T5"]["params_b"], "{:.0f}", "T5 parameters, billions")
v("gpt3_t", PAT["GPT-3"]["net_tco2e"], "{:.1f}", "GPT-3 net t CO2e")

print("== Lifecycle")
v("v6e_emb_kg", float(LIFE.TpuV6eEmbodiedKg), "{:,.0f}", "TPU v6e embodied, kg")
v("v6e_lb_kg", float(LIFE.TpuV6eOperationalLocationBasedKg), "{:,.0f}", "TPU v6e operational LB, kg")
v("v6e_mb_kg", float(LIFE.TpuV6eOperationalMarketBasedKg), "{:,.0f}", "TPU v6e operational MB, kg")
v("v6e_years_w", float(LIFE.TpuV6eLifetimeYears), "words", "TPU v6e study life, years")
v("v6e_years", float(LIFE.TpuV6eLifetimeYears), "{:.0f}", "TPU v6e study life, years (figure)")
V6E_LB_SHARE = 100 * LIFE.TpuV6eEmbodiedKg / (LIFE.TpuV6eEmbodiedKg + LIFE.TpuV6eOperationalLocationBasedKg)
V6E_MB_SHARE = 100 * LIFE.TpuV6eEmbodiedKg / (LIFE.TpuV6eEmbodiedKg + LIFE.TpuV6eOperationalMarketBasedKg)
v("v6e_lb_pct", V6E_LB_SHARE, "{:.0f}", "embodied share, location-based, %")
v("v6e_mb_pct", V6E_MB_SHARE, "{:.0f}", "embodied share, market-based, %")
v("bloom_g", float(RUNS.BloomGridGPerKwh), "{:.0f}", "BLOOM grid, g/kWh")
v("bloom_emb_t", float(RUNS.BloomEmbodiedTco2e), "{:.1f}", "BLOOM embodied, t")
v("bloom_total_t", float(RUNS.BloomTotalTco2e), "{:.1f}", "BLOOM final run total, t")
v("bloom_emb_pct", float(RUNS.BloomEmbodiedSharePct), "{:.1f}", "BLOOM embodied share, %")
pr = PAY.result
v("saved_kw", m(pr.saved_power, "kW"), "{:.2f}", "power saved, kW")
v("hours_day", mlsysim.core.units.HOURS_PER_DAY, "{:.0f}", "hours per day")
v("saved_kwh_day", m(pr.saved_energy_per_day, "kWh"), "{:.1f}", "energy saved per day, kWh")
v("avoid_us_kg", m(pr.avoided_per_day["US average"], "kg"), "{:.2f}", "avoided per day, US, kg")
v("avoid_qc_kg", m(pr.avoided_per_day["Quebec"], "kg"), "{:.3f}", "avoided per day, Quebec, kg")
v("pay_us_weeks", m(pr.payback["US average"], "week"), "{:.1f}", "payback, US, weeks")
v("pay_qc_years", m(pr.payback["Quebec"], "year"), "{:.1f}", "payback, Quebec, years")
v("life_long_y", m(LIFEX.long_life, "year"), "{:.0f}", "extended life, years")
v("life_cut_pct", 100 * LIFEX.result.annual_charge_reduction, "{:.0f}", "embodied charged per year cut, %")
BLOOM = SERIES.BloomProjectEmissions.entries
final = BLOOM[0]["tco2e"]
project = float(RUNS.BloomProjectTotalTco2e)
v("bloom_final_t", final, "{:.2f}", "BLOOM-176B final run, t")
v("bloom_project_t", project, "{:.2f}", "whole BigScience project, t")
v("bloom_final_pct", 100 * final / project, "{:.0f}", "final run share of the project, %")
v("bloom_project_mult", project / final, "{:.1f}", "project / final run")
v("bloom_models_t", sum(r["tco2e"] for r in BLOOM[1:] if r["process"].startswith(("BLOOM", "Other"))),
  "{:.1f}", "models other than the final run, t")
v("sweep_n", SWEEP.candidates, "{:.0f}", "candidates in the sweep")
v("sweep_pct", 100 * SWEEP.relative_cost, "{:.0f}", "each candidate's scale, % of final")
v("sweep_runs", SWEEP.result.final_run_equivalents, "{:.0f}", "final-run equivalents")
v("prompts_b", SERVE.requests_per_day / 10 ** 9, "{:.0f}", "prompts a day, billions")
v("prompts_b_w", SERVE.requests_per_day / 10 ** 9, "words", "the same, in prose")
v("serve_mwh_day", m(SERVE.result.daily_energy, "MWh"), "{:.0f}", "serving energy per day, MWh")
v("serve_cross_day", SERVE.result.crossover_days, "{:.0f}", "day serving passes training")
v("lab_cross_days", LAB.result.crossover_days, "{:.1f}", "chapter example crossover, days")
radio_lo, radio_hi = Systems.NetworkEnergy.RadioBitLow, Systems.NetworkEnergy.RadioBitHigh
fp32 = Hardware.Tech.Op.FlopFp32.energy
v("radio_lo_pj", m(radio_lo, "pJ/bit"), "{:,.0f}", "radio bit, low, pJ")
v("radio_hi_pj", m(radio_hi, "pJ/bit"), "{:,.0f}", "radio bit, high, pJ")
v("fp32_pj", m(fp32, "pJ/flop"), "{:.1f}", "FP32 multiply, pJ")
v("int32_add_pj", m(Hardware.Tech.Op.AddInt32.energy, "pJ/flop"), "{:.1f}", "INT32 add, pJ")
lo_x = m(radio_lo, "pJ/bit") / m(fp32, "pJ/flop")
v("radio_x_lo", lo_x, "{:,.0f}", "radio bit / FP32 multiply, low")
v("radio_x_hi", m(radio_hi, "pJ/bit") / m(fp32, "pJ/flop"), "{:,.0f}", "radio bit / FP32 multiply, high")
v("radio_x_floor", math.floor(lo_x / 10 ** 3) * 10 ** 3, "{:,.0f}", "low ratio, whole thousands below")
v("wake_vad_uw", m(WAKE.always_on_power, "uW"), "{:.0f}", "voice-activity detector, uW")
v("wake_stage_uw", m(WAKE.stage_power, "uW"), "{:.0f}", "small network, uW")
v("wake_full_mw", m(WAKE.full_power, "mW"), "{:.0f}", "full model, mW")
v("wake_full_ms", m(WAKE.full_duration, "ms"), "{:.0f}", "full model per wake, ms")
v("wake_avg_uw", m(WAKE.result.average_power, "uW"), "{:.1f}", "average power, uW")
v("wake_x", WAKE.result.reduction, "{:.0f}", "always-on / average")
c = CASCADE.result
v("cas_edge_mj", m(CASCADE.edge_energy, "mJ"), "{:.1f}", "edge model, mJ")
v("cas_esc", CASCADE.escalation, "{:.2f}", "escalated share")
v("cas_esc_pct", 100 * CASCADE.escalation, "{:.0f}", "the same, %")
v("cas_tx_mj", m(CASCADE.transmit_energy, "mJ"), "{:.0f}", "uplink, mJ")
v("cas_cloud_mj", m(CASCADE.cloud_energy, "mJ"), "{:.0f}", "cloud model, mJ")
v("cas_mean_mj", m(c.mean_energy, "mJ"), "{:.1f}", "cascade mean, mJ")
v("cas_only_mj", m(c.cloud_only_energy, "mJ"), "{:.0f}", "always cloud, mJ")
v("cas_save_pct", 100 * c.saving, "{:.0f}", "saving, %")

print("== Placement")
g = GEO.result
v("geo_mwh", m(GEO.energy, "MWh"), "{:,.0f}", "placement run, MWh")
v("geo_cut_pct", 100 * GEO.energy_cut, "{:.0f}", "algorithmic cut, %")
v("geo_pl_t", m(g.emissions["Poland"], "metric_ton"), "{:,.0f}", "Poland, t")
v("geo_pl_cut_t", m(g.emissions_after_cut["Poland"], "metric_ton"), "{:,.0f}", "Poland after the cut, t")
v("geo_qc_t", m(g.emissions["Quebec"], "metric_ton"), "{:,.0f}", "Quebec, t")
v("geo_ratio", ratio(g.emissions["Poland"], g.emissions["Quebec"]), "{:.0f}", "Poland / Quebec")
p = PRICED.result
v("carbon_price", PRICED.carbon_price_per_ton, "{:.0f}", "internal carbon price, $/t")
v("priced_tco_musd", p.objective_value / 10 ** 6, "{:.1f}", "Quebec total cost, $M")
v("priced_gap_musd", (p.objective["US_Avg"] - p.objective["Quebec"]) / 10 ** 6, "{:.1f}", "US-average gap, $M")
e = EU.result
shares = EU.shares
wind = [r["wind"] for r in EU.rows]
v("eu_min_pct", min(shares), "{:.0f}", "hourly fossil share, low, %")
v("eu_max_pct", max(shares), "{:.0f}", "hourly fossil share, high, %")
v("eu_hi_pct", e.high, "{:.0f}", "dirtiest 24-h window, %")
v("eu_lo_pct", e.low, "{:.0f}", "cleanest 24-h window, %")
v("eu_ratio", e.ratio, "{:.2f}", "dirtiest / cleanest window")
v("wind_min_gw", min(wind) / 10 ** 3, "{:.0f}", "wind, low, GW")
v("wind_max_gw", max(wind) / 10 ** 3, "{:.0f}", "wind, high, GW")
v("water_10mw", m(WATER.it_power, "MW"), "{:.0f}", "reference IT load, MW")
v("water_10mw_ml", m(WATER.result.water["evaporative"], "liter") / 10 ** 6, "{:.0f}", "a year at 1.8 L/kWh, million L")
v("latent_kj", m(mlsysim.physics.constants.WATER_LATENT_HEAT_VAPORIZATION, "kJ/kg"), "{:,.0f}",
  "latent heat of water, kJ/kg")

print("== Demand")
v("jev_unit_pct", 100 * (1 - JEV.unit_factor), "{:.0f}", "energy per query cut, %")
v("jev_unit", JEV.unit_factor, "{:.1f}", "energy per query factor")
v("jev_vol", JEV.volume_factor, "{:.0f}", "query volume factor")
v("jev_vol_pct", 100 * (JEV.volume_factor - 1), "{:.0f}", "query growth, %")
v("jev_net", JEV.result.net_factor, "{:.0f}", "fleet energy factor")
v("jev3_unit_x", 1 / JEV3.unit_factor, "{:.0f}", "fallacy: cheaper per query")
v("jev3_vol", JEV3.volume_factor, "{:.0f}", "fallacy: volume")
v("jev3_net", JEV3.result.net_factor, "{:.1f}", "fallacy: fleet energy")
v("kaplan_alpha", KAPLAN.exponent, "{:.2f}", "Kaplan compute exponent")
v("kaplan_pct", 100 * KAPLAN.loss_reduction, "{:.0f}", "loss reduction, %")
v("kaplan_x", KAPLAN.reduction_multiple, "{:.1f}", "compute multiple for 10% lower loss")
v("kaplan_x2", KAPLAN.reduction_multiple * KAPLAN.reduction_multiple, "{:.0f}", "two steps")
DC = Literature.DatacenterElectricity
v("us_2014_twh", float(DC.Us2014Twh), "{:.0f}", "US data centers 2014, TWh")
v("us_2024_twh", float(DC.Us2024Twh), "{:.0f}", "US data centers 2024, TWh")
v("us_2030_twh", float(DC.Us2030ReferenceTwh), "{:.0f}", "2030 reference, TWh")
v("us_2030_lo", float(DC.Us2030LowTwh), "{:.0f}", "2030 low, TWh")
v("us_2030_hi", float(DC.Us2030HighTwh), "{:.0f}", "2030 high, TWh")
v("us_growth", DC.Us2024Twh / DC.Us2014Twh, "{:.1f}", "2024 / 2014")

print("== Hook and recall")
CSO = SERIES.CsoIrelandDatacenterElectricity.entries
v("cso_2015_pct", CSO[0]["share_pct"], "{:.0f}", "Irish data centers 2015, % of metered")
v("cso_2023_pct", CSO[-1]["share_pct"], "{:.0f}", "Irish data centers 2023, % of metered")
v("cso_2015_gwh", CSO[0]["gwh"], "{:,.0f}", "2015, GWh")
v("cso_2023_gwh", CSO[-1]["gwh"], "{:,.0f}", "2023, GWh")
v("cso_growth", CSO[-1]["gwh"] / CSO[0]["gwh"], "{:.1f}", "2023 / 2015 GWh")
v("pgd_passes", PGD.passes, "{:.0f}", "PGD-7 passes per batch")
v("pgd_attack", PGD.attack_passes, "{:.0f}", "attack passes")
v("pgd_update", PGD.update_passes, "{:.0f}", "update passes")
v("pgd_attack_w", PGD.attack_passes, "words", "attack passes, in prose")
v("pgd_update_w", PGD.update_passes, "words", "update passes, in prose")

if __name__ == "__main__":
    write(__file__)
