#!/usr/bin/env python3
"""Every number on the Lecture 15b (Responsible Engineering, Part 2) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the cited results in
Literature.TrainingRuns (Strubell et al. 2019; Patterson et al. 2021, Table 1,
Figure 1, Sec. 4.1, Appendices C and D; Luccioni et al. 2023 on BLOOM),
Literature.ServingResults (Elsworth et al. 2025, the BLOOM API, fleet shares,
the CodeCarbon run), Literature.DifferentialPrivacy (De et al. 2022),
Literature.ExtractionAttacks (Carlini et al. 2021), Literature.MachineUnlearning
(SISA), Literature.GovernanceEnforcement (DPC, ProPublica, Regulation B), and
the Scenarios.* the Responsible Engineering chapter reads: the ranking
service's three-year cost and carbon, GPT-3's and BLOOM's training carbon, the
measured architecture search, the embodied share of a GPU-hour, the credit
service's explanation load, appeal review, and audit log, and per-example
DP-SGD gradients. This file only selects values, converts units, and formats
them; it types no numbers. Each value is recorded with v() under the key the
deck prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports this module, so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, val, num, write, mlsysim  # noqa: E402,F401

Scenarios, Literature, Hardware = mlsysim.Scenarios, mlsysim.Literature, mlsysim.Hardware
ureg = mlsysim.core.units.ureg
RUNS, SERVE = Literature.TrainingRuns, Literature.ServingResults
DP, GOV = Literature.DifferentialPrivacy, Literature.GovernanceEnforcement

TCO = Scenarios.RankingServiceLifecycleCost
GPT3 = Scenarios.Gpt3TrainingCarbon
BLOOM = Scenarios.BloomTrainingCarbon
NAS = Scenarios.EvolvedTransformerSearchCarbon
EMB = Scenarios.GpuHourEmbodiedShare
EXPL = Scenarios.CreditDenialExplanationLoad
REVIEW = Scenarios.CreditAppealReview
LOG = Scenarios.CreditDecisionAuditLog
DLOG = Scenarios.RankingServiceAuditLog
GRADS = Scenarios.DpSgdPerExampleGradients
LOOP = Scenarios.CreditServiceBiasFeedback
CREDIT = Scenarios.CreditServiceTwoGroups


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


def row(series, key, value):
    """One row of a Literature series, selected by its first column."""
    rows = [r for r in series.entries if r[key] == value]
    assert len(rows) == 1, (key, value)
    return rows[0]


def mag(q, unit):
    return q.to(unit).magnitude


print("== Hook: architecture search (Strubell 2019; Patterson et al. 2021, Sec. 4.1, App. C and D)")
NAS_EST_LB, CAR_LB = lit(RUNS.StrubellNasTransformerCo2eLb), lit(RUNS.StrubellCarLifetimeCo2eLb)
NAS_EST_T = mag(NAS_EST_LB * ureg.lb, "t")
CAR_T = mag(CAR_LB * ureg.lb, "t")
NAS_PRINTED_T = lit(RUNS.EvolvedTransformerNasTco2e)
v("nas_est_lb", NAS_EST_LB, "{:,.0f}", "Strubell estimate, lb CO2e")
v("car_lb", CAR_LB, "{:,.0f}", "one car's lifetime, lb CO2e")
v("nas_est_t", NAS_EST_T, "{:.0f}", "Strubell estimate, t")
v("car_t", CAR_T, "{:.0f}", "car lifetime, t")
v("nas_cars", NAS_EST_T / CAR_T, "{:.0f}", "estimate / car lifetimes")
v("nas_meas_t", NAS_PRINTED_T, "{:.1f}", "measured as printed, t")
v("nas_meas_t0", NAS_PRINTED_T, "{:.0f}", "the same, whole tonnes (vote answer)")
v("nas_err", lit(RUNS.EvolvedTransformerNasEstimateErrorGoogle), "{:.0f}", "estimate / measurement, printed")
v("nas_err_calc", NAS_EST_T / NAS_PRINTED_T, "{:.0f}", "estimate / 3.2 t, computed")
PROXY = lit(RUNS.EvolvedTransformerNasProxyOverestimate)
TPU_PERF_W = (lit(RUNS.EvolvedTransformerNasP100Hours) / lit(RUNS.EvolvedTransformerNasTpuV2Hours)) * (
    lit(RUNS.P100MeasuredPowerW) / lit(RUNS.TpuV2MeasuredPowerW))
PUE_US, PUE_G = lit(RUNS.UsAveragePue2018), lit(RUNS.EvolvedTransformerNasPue)
GRID_G, GRID_US = lit(RUNS.EvolvedTransformerNasGridKgPerKwh), lit(RUNS.UsAverageGridKgPerKwhAppendixD)
v("proxy", PROXY, "{:.1f}", "proxy-task factor, Appendix C")
v("tpu_perf_w", TPU_PERF_W, "{:.1f}", "TPU v2 over P100, performance per watt")
v("pue_g", PUE_G, "{:.2f}", "search datacenter PUE")
v("pue_us", PUE_US, "{:.2f}", "US average PUE 2018")
v("pue_ratio", PUE_US / PUE_G, "{:.2f}", "PUE factor")
v("grid_g", GRID_G, "{:.3f}", "search grid, kg/kWh")
v("grid_us", GRID_US, "{:.3f}", "US average grid, kg/kWh")
v("factors", PROXY * TPU_PERF_W * PUE_US / PUE_G, "{:.0f}", "proxy x hardware x PUE")
v("proxy_batch", lit(RUNS.EvolvedTransformerNasProxyBatch), "{:,.0f}", "proxy batch, Transformer (Base)")
v("full_batch", lit(RUNS.EvolvedTransformerNasFullBatch), "{:,.0f}", "Transformer (Big) batch")
v("nas_iowa_t", lit(RUNS.EvolvedTransformerNasIowaNetTco2e), "{:.1f}", "same search in Iowa, net, t")
v("nas_days", mag(NAS.duration, "day"), "{:.1f}", "search days")
v("nas_chips", NAS.devices, "{:.0f}", "TPU v2 chips")
v("nas_w", mag(NAS.device_power, "W"), "{:.0f}", "measured W per chip")
v("nas_kwh", mag(NAS.energy, "kWh"), "{:,.0f}", "facility energy, kWh")
v("nas_calc_t", mag(NAS.carbon, "t"), "{:.2f}", "computed carbon, t")
MEENA = lit(RUNS.MeenaSavedTco2e)
v("meena_t", MEENA, "{:.1f}", "Meena saved, t")
v("payback", MEENA / NAS_PRINTED_T, "{:.0f}", "payback, whole")
v("payback_1", MEENA / NAS_PRINTED_T, "{:.1f}", "payback, one decimal")

print("== Recall (Literature.Fairness; Scenarios.CreditServiceBiasFeedback)")
F = Literature.Fairness
v("hook_cost", lit(F.ObermeyerBlackShareBeforePct), "{:.1f}", "Black share ranked by cost, %")
v("hook_health", lit(F.ObermeyerBlackShareAfterPct), "{:.1f}", "Black share ranked by health, %")
v("fb_gap0", LOOP.launch_gap, "{:.0f}", "gap at launch, points")
v("fb_alpha", LOOP.factor, "{:.2f}", "feedback factor")
v("fb_alpha_1", LOOP.factor, "{:.1f}", "the same, spoken")
v("fb_gate", LOOP.gate, "{:.0f}", "gate, points")
v("fb_cross", LOOP.crossing, "{:.1f}", "crossing k")
v("fb_fail_q", LOOP.breach_cycle, "{:.0f}", "quarter the gate fails")
v("fb_gap_fail", LOOP.gap_at_breach, "{:.1f}", "gap in that quarter")
v("credit_all_n", sum(CREDIT.size(g) for g in CREDIT.groups), "{:,.0f}", "credit applicants in the worked example")
v("credit_appr_n", sum(CREDIT.positives(g) for g in CREDIT.groups), "{:,.0f}", "approved in the worked example")

print("== Gemini prompt (Literature.Series.GeminiPromptEnergy, GoogleServingFactors; Elsworth et al. 2025)")
GEM = Literature.Series.GeminiPromptEnergy.entries
GEM_WH = sum(r["comprehensive_wh"] for r in GEM)
GEM_NARROW = sum(r["existing_wh"] for r in GEM if r["existing_counted"])
SF = Literature.Series.GoogleServingFactors


def factor(name):
    return row(SF, "parameter", name)["value"]


GEM_S2, GEM_S13 = factor("prompt_emissions_scope2_mb"), factor("prompt_emissions_scope13")
v("gem_wh", GEM_WH, "{:.2f}", "energy at the full boundary, Wh")
v("gem_narrow_wh", GEM_NARROW, "{:.2f}", "accelerators alone, Wh")
v("gem_pue", factor("fleet_pue"), "{:.2f}", "fleet PUE")
v("gem_s2", GEM_S2, "{:.3f}", "Scope 2 market-based, g")
v("gem_s13", GEM_S13, "{:.3f}", "Scopes 1 and 3, g")
v("gem_intensity", GEM_S2 / GEM_WH, "{:.3f}", "implied intensity, kg/kWh")
v("gem_total", GEM_S2 + GEM_S13, "{:.2f}", "total, g")
v("gem_energy_x", lit(SERVE.GooglePromptEnergyReduction), "{:.0f}", "energy reduction in a year")
v("gem_carbon_x", lit(SERVE.GooglePromptEmissionsReduction), "{:.0f}", "emissions reduction in a year")
v("gem_model_x", lit(SERVE.GooglePromptEnergyModelReduction), "{:.0f}", "from model changes")
v("gem_util_x", lit(SERVE.GooglePromptEnergyUtilizationReduction), "{:.1f}", "from utilization")

print("== Four factors (Patterson et al. 2021, Table 1, Figure 1, Sec. 2.3 and 2.4)")
CUM = {"model": lit(RUNS.FourFactorsCumulativeGainModel), "processor": lit(RUNS.FourFactorsCumulativeGainProcessor),
       "datacenter": lit(RUNS.FourFactorsCumulativeGainDatacenter), "grid": lit(RUNS.FourFactorsCumulativeGainGrid)}
v("ff_base_t", lit(RUNS.TransformerBigP100UsAvgTco2e), "{:.4f}", "baseline, t")
v("ff_base_kwh", lit(RUNS.TransformerBigP100UsAvgKwh), "{:.0f}", "baseline, kWh")
v("ff_best_t", lit(RUNS.EvolvedMediumTpuV2IowaNetTco2e), "{:.4f}", "best case, t")
v("ff_best_kwh", lit(RUNS.EvolvedMediumTpuV2IowaKwh), "{:.0f}", "best case, kWh")
v("ff_printed", lit(RUNS.FourFactorsPrintedGain), "{:.0f}", "the paper's rounded gain")
v("ff_ops", lit(RUNS.EvolvedMediumOperationsReduction), "{:.1f}", "fewer operations")
for k in CUM:
    v(f"ff_{k}", CUM[k], "{:.1f}", f"cumulative after {k}")
v("ff_proc_step", CUM["processor"] / CUM["model"], "{:.1f}", "processor step")
PUE_A, PUE_B = lit(RUNS.UsAverageDatacenterPue), lit(RUNS.GoogleIowaPue)
GR_A, GR_B = lit(RUNS.UsAverageGridKgPerKwh), lit(RUNS.GoogleIowaNetGridKgPerKwh)
v("ff_pue_a", PUE_A, "{:.2f}", "average US PUE")
v("ff_pue_b", PUE_B, "{:.2f}", "Iowa PUE")
v("ff_pue_step", PUE_A / PUE_B, "{:.1f}", "PUE step")
v("ff_grid_a", GR_A, "{:.3f}", "US average grid, kg/kWh")
v("ff_grid_b", GR_B, "{:.3f}", "Iowa net grid, kg/kWh")
v("ff_grid_step", GR_A / GR_B, "{:.1f}", "grid step")
v("ff_iowa_local", lit(RUNS.IowaLocalGridKgPerKwh), "{:.3f}", "Iowa local grid, kg/kWh")
v("ff_cfe_pct", lit(RUNS.GoogleIowaHourlyCfePct2019), "{:.0f}", "hourly carbon-free share, %")

print("== Estimate: GPT-3 against BLOOM (Scenarios.Gpt3TrainingCarbon, BloomTrainingCarbon)")
G3 = row(Literature.Series.PattersonTrainingFootprints, "model", "GPT-3")
GPT3_KWH, GPT3_GRID = mag(GPT3.energy, "kWh"), mag(GPT3.intensity, "kg/kWh")
BLOOM_GPU_KWH, BLOOM_GRID = mag(BLOOM.it_energy_total, "kWh"), mag(BLOOM.intensity, "kg/kWh")
BLOOM_KWH, BLOOM_T, GPT3_T = mag(BLOOM.energy, "kWh"), mag(BLOOM.carbon, "t"), mag(GPT3.carbon, "t")
v("gpt3_mwh", mag(GPT3.energy, "MWh"), "{:,.0f}", "GPT-3 energy, MWh")
v("gpt3_kwh", GPT3_KWH, "{:,.0f}", "GPT-3 energy, kWh")
v("gpt3_pue", G3["pue"], "{:.1f}", "GPT-3 PUE, included")
v("gpt3_grid", GPT3_GRID, "{:.3f}", "US grid, kg/kWh")
v("gpt3_t", GPT3_T, "{:.0f}", "GPT-3, t")
v("car_year_t", mag(GPT3.comparison, "t"), "{:.1f}", "passenger car year, t")
v("gpt3_cars", GPT3.comparisons, "{:.0f}", "car-years")
v("bloom_mwh", BLOOM_GPU_KWH / 10 ** 3, "{:.0f}", "BLOOM GPU energy, MWh")
v("bloom_gpu_kwh", BLOOM_GPU_KWH, "{:,.0f}", "BLOOM GPU energy, kWh")
v("bloom_pue", BLOOM.pue, "{:.1f}", "BLOOM PUE")
v("bloom_grid", BLOOM_GRID, "{:.3f}", "French grid, kg/kWh")
v("bloom_grid_g", mag(BLOOM.intensity, "g/kWh"), "{:.0f}", "French grid, g/kWh")
v("bloom_kwh", round(BLOOM_KWH, -3), "{:,.0f}", "BLOOM energy with PUE, kWh (to the thousand)")
v("bloom_kmwh", round(BLOOM_KWH, -3) / 10 ** 3, "{:,.0f}", "the same, MWh")
v("bloom_t", BLOOM_T, "{:.1f}", "BLOOM, t")
ENERGY_X, GRID_X = GPT3_KWH / BLOOM_KWH, GPT3_GRID / BLOOM_GRID
v("energy_x", ENERGY_X, "{:.2f}", "energy ratio")
v("energy_x3", ENERGY_X, "{:.3f}", "energy ratio, three decimals")
v("grid_x", GRID_X, "{:.2f}", "grid ratio")
v("grid_x3", GRID_X, "{:.3f}", "grid ratio, three decimals")
v("carbon_x", GPT3_T / BLOOM_T, "{:.1f}", "carbon ratio")
v("naive_energy_x", GPT3_KWH / BLOOM_GPU_KWH, "{:.1f}", "energy ratio without PUE")
v("naive_carbon_x", GPT3_T / (BLOOM_GPU_KWH * BLOOM_GRID / 10 ** 3), "{:.0f}", "carbon ratio without PUE")
v("bloom_t4", lit(RUNS.BloomTable4Tco2e), "{:.0f}", "BLOOM, Table 4, t")
v("llm_gap_t4", GPT3_T / lit(RUNS.BloomTable4Tco2e), "{:.0f}", "GPT-3 / BLOOM with Table 4's rounding")

print("== Four large models (Literature.Series.LuccioniLlmTrainingComparison; PattersonTrainingFootprints)")
LLM = {r["model"]: r for r in Literature.Series.LuccioniLlmTrainingComparison.entries}
v("llm_gpt3", LLM["GPT-3"]["tco2e_pue"], "{:.0f}", "GPT-3, t")
v("llm_gopher", LLM["Gopher"]["tco2e_pue"], "{:.0f}", "Gopher, t")
v("llm_opt", LLM["OPT"]["tco2e_pue"], "{:.1f}", "OPT, t")
v("llm_bloom", LLM["BLOOM"]["tco2e_pue"], "{:.0f}", "BLOOM, t")
v("llm_span", LLM["GPT-3"]["tco2e_pue"] / LLM["BLOOM"]["tco2e_pue"], "{:.0f}", "span")
PT = Literature.Series.PattersonTrainingFootprints.entries
v("pt_min_t", min(r["net_tco2e"] for r in PT), "{:.1f}", "smallest Table 4 net emissions, t")
v("pt_max_t", max(r["net_tco2e"] for r in PT), "{:.0f}", "largest, t")
v("pt_min_grid", min(r["net_kg_per_kwh"] for r in PT), "{:.3f}", "lowest net grid, kg/kWh")
v("pt_max_grid", max(r["net_kg_per_kwh"] for r in PT), "{:.3f}", "highest net grid, kg/kWh")

print("== BLOOM life cycle (Luccioni et al. 2023, Tables 2 and 3, Sec. 4.1 and 4.3)")
B_DYN, B_IDLE, B_EMB = lit(RUNS.BloomDynamicTco2e), lit(RUNS.BloomIdleTco2e), lit(RUNS.BloomEmbodiedTco2e)
B_TOTAL = lit(RUNS.BloomTotalTco2e)
v("b_dyn", B_DYN, "{:.1f}", "dynamic, t")
v("b_idle", B_IDLE, "{:.1f}", "idle, t")
v("b_emb", B_EMB, "{:.1f}", "embodied, t")
v("b_total", B_TOTAL, "{:.1f}", "total, t")
v("b_dyn_share", 100 * B_DYN / (B_DYN + B_IDLE + B_EMB), "{:.1f}", "dynamic share, %")
v("b_dyn_share0", 100 * B_DYN / B_TOTAL, "{:.0f}", "dynamic share of 50.5 t, whole %")
OFF, IDLE, TRAIN = (lit(RUNS.BloomNodesOffKwhPerInterval), lit(RUNS.BloomIdleKwhPerInterval),
                    lit(RUNS.BloomTrainingKwhPerInterval))
v("b_off_kwh", OFF, "{:.0f}", "nodes off, kWh per interval")
v("b_idle_kwh", IDLE, "{:.0f}", "idle, kWh per interval")
v("b_train_kwh", TRAIN, "{:.0f}", "training, kWh per interval")
v("b_code_pct", 100 * (TRAIN - IDLE) / TRAIN, "{:.0f}", "code's share of training power, %")
v("b_idle_total_kwh", lit(RUNS.BloomIdleKwh), "{:,.0f}", "idle energy over the run, kWh")
v("b_servers_t", lit(RUNS.BloomServersEmbodiedTco2e), "{:.2f}", "servers embodied, t")
v("b_gpus_t", lit(RUNS.BloomGpusEmbodiedTco2e), "{:.2f}", "GPUs embodied, t")
v("server_kg", mag(EMB.server_embodied, "kg"), "{:,.0f}", "one server, kg")
v("gpu_kg", mag(EMB.gpu_embodied, "kg"), "{:.0f}", "one GPU, kg")
v("life_years", mag(EMB.lifetime, "year"), "{:.0f}", "replacement interval, years")
v("use_pct", 100 * EMB.utilization, "{:.0f}", "use, %")
v("server_kg_h", mag(EMB.server_per_hour, "kg"), "{:.3f}", "server, kg per hour")
v("gpu_kg_h", mag(EMB.gpu_per_hour, "kg"), "{:.4f}", "GPU, kg per hour")
CARD_T = mag(lit(RUNS.BloomModelCardEmissionsG) * ureg.g, "t")
v("card_g", lit(RUNS.BloomModelCardEmissionsG), "{:,.0f}", "model card, g")
v("card_t", CARD_T, "{:.1f}", "model card, t")
v("card_share", 100 * CARD_T / B_TOTAL, "{:.0f}", "card share of the life cycle, %")

print("== Ranking service (Scenarios.RankingServiceLifecycleCost)")
v("users_m", TCO.daily_users / 10 ** 6, "{:.0f}", "daily users, millions")
v("items", TCO.items_per_user, "{:.0f}", "items per user")
v("inf_day_m", TCO.inferences_per_day / 10 ** 6, "{:.0f}", "inferences a day, millions")
v("service_ms", mag(TCO.service_time, "ms"), "{:.0f}", "GPU time per request, ms")
v("service_s", mag(TCO.service_time, "s"), "{:.2f}", "the same, seconds")
v("gpu_s_day_m", mag(TCO.inferences_per_day * TCO.service_time, "s") / 10 ** 6, "{:.0f}", "GPU-seconds a day, millions")
v("gpu_h_day", mag(TCO.gpu_hours_per_day, "hour"), "{:.0f}", "GPU-hours a day")
v("gpu_h_day_1", mag(TCO.gpu_hours_per_day, "hour"), "{:.1f}", "the same, one decimal")
v("gpus", TCO.gpus_busy, "{:.1f}", "GPUs busy")
v("gpus_0", TCO.gpus_busy, "{:.0f}", "GPUs busy, whole")
v("servers", TCO.gpus_busy / EMB.gpus_per_server, "{:.1f}", "8-GPU servers")
v("gpus_per_server", EMB.gpus_per_server, "{:.0f}", "GPUs per server")
v("servers_w_words", round(TCO.gpus_busy / EMB.gpus_per_server), "words", "8-GPU servers, whole, in words")
v("inf_rate", mag(TCO.inference_rate, "dollar/hour"), "{:.2f}", "reserved rate, $/GPU-hour")
v("train_rate", mag(TCO.training_rate, "dollar/hour"), "{:.0f}", "on-demand rate, $/GPU-hour")
v("serve_year", mag(TCO.inference_cost_per_year, "dollar"), "{:,.0f}", "serving a year, $")
v("serve_year_k", mag(TCO.inference_cost_per_year, "dollar") / 10 ** 3, "{:.0f}", "the same, $K")
v("cycle", mag(TCO.cycle_cost, "dollar"), "{:,.0f}", "training cycle, $")
v("cycle_prep", mag(TCO.data_prep_cost, "dollar"), "{:,.0f}", "data preparation, $")
v("cycle_search", mag(TCO.search_cost, "dollar"), "{:,.0f}", "search, $")
v("cycle_final", mag(TCO.final_run_cost, "dollar"), "{:,.0f}", "final run, $")
v("cycles", TCO.cycles, "words", "training cycles in three years")
v("years", TCO.years, "{:.0f}", "years")
v("days", TCO.service_days, "{:,.0f}", "service days")
v("train_3y_k", mag(TCO.training_cost, "dollar") / 10 ** 3, "{:.0f}", "training, $K")
v("train_3y_k1", mag(TCO.training_cost, "dollar") / 10 ** 3, "{:.1f}", "training, $K, one decimal")
v("inf_3y_m", mag(TCO.inference_cost, "dollar") / 10 ** 6, "{:.2f}", "inference, $M")
v("ops_year_k", mag(TCO.operations_per_year, "dollar") / 10 ** 3, "{:.0f}", "operations a year, $K")
v("ops_3y_k", mag(TCO.operations_cost, "dollar") / 10 ** 3, "{:.0f}", "operations, $K")
v("tco_m", mag(TCO.total_cost, "dollar") / 10 ** 6, "{:.2f}", "total, $M")
v("train_share", 100 * TCO.shares["training"], "{:.1f}", "training share, %")
v("inf_share", 100 * TCO.shares["inference"], "{:.1f}", "inference share, %")
v("cost_x", TCO.cost_ratio, "{:.0f}", "inference / training cost")
GPU_W = mag(EMB.gpu_power, "W")
FLEET_GRID = mag(EMB.intensities["fleet"], "kg/kWh")
v("gpu_w", GPU_W, "{:.0f}", "GPU power, W")
v("gpu_kw", mag(EMB.gpu_power, "kW"), "{:.1f}", "GPU power, kW")
v("fleet_grid", FLEET_GRID, "{:.1f}", "fleet grid, kg/kWh")
v("kg_gpu_h", mag(TCO.carbon_per_gpu_hour, "kg"), "{:.2f}", "kg per GPU-hour")
v("inf_gpu_h", mag(TCO.inference_gpu_hours, "hour"), "{:,.0f}", "inference GPU-hours over three years")
v("inf_t", mag(TCO.inference_carbon, "t"), "{:.1f}", "inference carbon, t")
v("inf_t0", mag(TCO.inference_carbon, "t"), "{:.0f}", "the same, whole")
v("train_t", mag(TCO.training_carbon, "t"), "{:.1f}", "training carbon, t")
v("carbon_x_dlrm", TCO.carbon_ratio, "{:.0f}", "inference / training carbon")
v("inf_t_year", mag(TCO.inference_carbon, "t") / TCO.years, "{:.0f}", "inference carbon a year, t")

print("== Estimate: cut serving or halve training")
v("serve_cut_pct", 100 * TCO.serving_cut, "{:.0f}", "serving cut, %")
v("serve_cut_frac", TCO.serving_cut, "{:.1f}", "the same as a fraction")
v("train_cut_frac", TCO.training_cut, "{:.1f}", "training cut as a fraction")
v("save_serve_k", mag(TCO.serving_saving, "dollar") / 10 ** 3, "{:.0f}", "serving cut, $K")
v("save_serve_t", mag(TCO.serving_saving_carbon, "t"), "{:.0f}", "serving cut, t")
v("save_train_k", mag(TCO.training_saving, "dollar") / 10 ** 3, "{:.0f}", "halving training, $K")
v("save_train_t", mag(TCO.training_saving_carbon, "t"), "{:.1f}", "halving training, t")
v("save_serve_pct", 100 * (TCO.serving_saving / TCO.total_cost).to("").magnitude, "{:.0f}", "serving cut, % of total")
v("save_train_pct", 100 * (TCO.training_saving / TCO.total_cost).to("").magnitude, "{:.0f}", "halving training, % of total")
v("halvings", (TCO.serving_saving / TCO.training_saving).to("").magnitude, "{:.0f}", "halvings the serving cut is worth")
v("final_half", mag(TCO.final_run_cost, "dollar") * TCO.training_cut, "{:,.0f}", "half the final run, $")
TREND = Scenarios.FrontierTrainingCostTrend
v("frontier_growth", TREND.growth_per_year, "{:.1f}", "frontier training cost growth a year")
v("priciest_m", TREND.priciest["training_cost_2023usd"] / 10 ** 6, "{:.0f}", "priciest frontier run, $M (2023 US$)")

print("== Real fleets (Patterson 2022; Wu 2022; BLOOM API)")
v("api_req", lit(SERVE.BloomApiRequests), "{:,.0f}", "BLOOM API requests")
v("api_kwh", lit(SERVE.BloomApiKwh), "{:.0f}", "BLOOM API energy, kWh")
v("api_gpus", lit(SERVE.BloomApiGpus), "{:.0f}", "A100 GPUs")
v("api_days", lit(SERVE.BloomApiDays), "{:.0f}", "days")
v("api_wh", lit(SERVE.BloomApiKwh) * 10 ** 3 / lit(SERVE.BloomApiRequests), "{:.2f}", "Wh per request")
v("api_idle_kwh", lit(SERVE.BloomApiIdleKwhPer10Min), "{:.2f}", "kWh every 10 minutes when idle")
v("api_loaded_pct", lit(SERVE.BloomApiLoadedSharePct), "{:.0f}", "share to keep the model loaded, %")
v("g_inf_share", 100 * lit(SERVE.GoogleInferenceShareOfMlEnergy), "{:.0f}", "Google inference share, %")
v("g_train_share", 100 * (1 - lit(SERVE.GoogleInferenceShareOfMlEnergy)), "{:.0f}", "Google training share, %")
v("g_ml_lo", lit(SERVE.GoogleMlShareOfEnergyLowPct), "{:.0f}", "ML share of Google energy, low, %")
v("g_ml_hi", lit(SERVE.GoogleMlShareOfEnergyHighPct), "{:.0f}", "high, %")
v("m_exp", lit(SERVE.MetaAiPowerExperimentationPct), "{:.0f}", "Meta experimentation, %")
v("m_train", lit(SERVE.MetaAiPowerTrainingPct), "{:.0f}", "Meta training, %")
v("m_inf", lit(SERVE.MetaAiPowerInferencePct), "{:.0f}", "Meta inference, %")
v("m_emb", lit(SERVE.MetaEmbodiedSharePct), "{:.0f}", "Meta embodied share, %")

print("== Embodied share of one GPU-hour (Scenarios.GpuHourEmbodiedShare)")
v("op_dirty_g", mag(EMB.operational_per_gpu_hour("fleet"), "g"), "{:.0f}", "operational, fleet grid, g")
v("op_clean_g", mag(EMB.operational_per_gpu_hour("France"), "g"), "{:.0f}", "operational, French grid, g")
v("emb_g", mag(EMB.embodied_per_gpu_hour, "g"), "{:.1f}", "embodied per GPU-hour, g")
v("gpu_g_h", mag(EMB.gpu_per_hour, "g"), "{:.1f}", "GPU, g per hour")
v("server_g_h", mag(EMB.server_per_hour, "g"), "{:.0f}", "server, g per hour")
v("emb_dirty_pct", 100 * EMB.embodied_share("fleet"), "{:.0f}", "embodied share, fleet grid, %")
v("emb_clean_pct", 100 * EMB.embodied_share("France"), "{:.0f}", "embodied share, French grid, %")
DLRM_EMB_T = mag(TCO.inference_gpu_hours.magnitude * EMB.embodied_per_gpu_hour, "t")
v("dlrm_emb_t", DLRM_EMB_T, "{:.1f}", "DLRM embodied over three years, t")
v("dlrm_emb_t_year", DLRM_EMB_T / TCO.years, "{:.0f}", "the same a year, t")

print("== Duties (Literature.GovernanceEnforcement; Scenarios.CreditDenialExplanationLoad, CreditAppealReview)")
FB, IG = lit(GOV.DpcMetaFacebookFineMeur), lit(GOV.DpcMetaInstagramFineMeur)
v("dpc_fb", FB, "{:.0f}", "Facebook fine, EUR M")
v("dpc_ig", IG, "{:.0f}", "Instagram fine, EUR M")
v("dpc_total", FB + IG, "{:.0f}", "total, EUR M")
v("decisions_m", EXPL.decisions_per_day / 10 ** 6, "{:.0f}", "decisions a day, millions")
v("deny_pct", 100 * EXPL.denial_share, "{:.1f}", "denied, %")
v("deny_frac", EXPL.denial_share, "{:.3f}", "denied as a fraction")
v("samples", EXPL.samples, "{:,.0f}", "sampled evaluations per explanation")
v("features", EXPL.features, "{:.0f}", "features")
v("notices", EXPL.notices_per_day, "{:,.0f}", "notices a day")
v("sampled_m", EXPL.sampled_evaluations / 10 ** 6, "{:.0f}", "sampled evaluations a day, millions")
v("sampled_x", EXPL.sampled_over_decisions, "{:.0f}", "over the decisions")
v("exact_per", EXPL.exact_per_explanation, "{:,.0f}", "2^20")
EXP11 = math.floor(math.log10(EXPL.exact_evaluations))
v("exact_mant", EXPL.exact_evaluations / 10 ** EXP11, "{:.0f}", "exact evaluations a day, mantissa")
v("exact_exp", EXP11, "{:.0f}", "exact evaluations a day, exponent")
v("all_decisions_b", EXPL.decisions_per_day * EXPL.samples / 10 ** 9, "{:.0f}", "explaining every decision, billions")
v("appeal_pct", 100 * REVIEW.appeal_rate, "{:.1f}", "appealed, %")
v("appeals", REVIEW.appeals_per_day, "{:,.0f}", "appeals a day")
v("case_min", mag(REVIEW.time_per_case, "minute"), "{:.0f}", "minutes per case")
v("review_h", mag(REVIEW.reviewer_hours_per_day, "hour"), "{:.0f}", "reviewer-hours a day")
v("reviewers", REVIEW.reviewers, "{:.1f}", "reviewers")
v("shift_h", mag(REVIEW.shift, "hour"), "{:.0f}", "shift, hours")
v("cigna_cases", REVIEW.compare_cases, "{:,.0f}", "Cigna denials")
v("cigna_s", mag(REVIEW.compare_time_per_case, "s"), "{:.1f}", "Cigna seconds each")
v("cigna_h", mag(REVIEW.compare_hours, "hour"), "{:.0f}", "Cigna hours")
v("cigna_months", lit(GOV.CignaWindowMonths), "words", "months of Cigna denials")
v("credit_per_h", REVIEW.cases_per_hour, "{:.0f}", "credit cases per reviewer-hour")
v("cigna_per_h", REVIEW.compare_cases_per_hour, "{:,.0f}", "Cigna cases per hour")

print("== Governance (Literature.ExtractionAttacks, MachineUnlearning, DifferentialPrivacy; Scenarios)")
EA, MU = Literature.ExtractionAttacks, Literature.MachineUnlearning
v("carlini_mem", lit(EA.Gpt2SamplesMemorized), "{:.0f}", "memorized samples")
v("carlini_cand", lit(EA.Gpt2SamplesInspected), "{:,.0f}", "candidates inspected")
v("carlini_names", lit(EA.Gpt2MemorizedWithNames), "{:.0f}", "with names")
v("carlini_contact", lit(EA.Gpt2MemorizedWithContact), "{:.0f}", "with contact information")
v("cycle_days", TCO.service_days / TCO.cycles, "{:.0f}", "days between cycles")
v("cycle_days_r", round(TCO.service_days / TCO.cycles, -1), "{:.0f}", "the same, to tens")
v("sisa_purchase", lit(MU.SisaSpeedupPurchase), "{:.2f}", "SISA, Purchase")
v("sisa_svhn", lit(MU.SisaSpeedupSvhn), "{:.2f}", "SISA, SVHN")
v("sisa_imagenet", lit(MU.SisaSpeedupImagenet), "{:.2f}", "SISA, ImageNet")
EPS = lit(DP.Cifar10ComparisonEpsilon)
v("eps", EPS, "{:.0f}", "epsilon of the comparison")
v("e_eps", round(math.exp(EPS), -3), "{:,.0f}", "e^epsilon, to the thousand")
v("kws_params", GRADS.models["KWS"].parameters.to("count").magnitude, "{:,.0f}", "KWS parameters")
v("kws_params_k", GRADS.models["KWS"].parameters.to("count").magnitude / 10 ** 3, "{:.0f}", "KWS parameters, K")
v("fp32_b", mag(GRADS.bytes_per_param, "byte"), "{:.0f}", "bytes per FP32 parameter")
v("kws_grad_kb", mag(GRADS.per_example("KWS"), "kB"), "{:.0f}", "KWS gradient, KB")
v("batch", GRADS.batch_size, "{:.0f}", "batch size")
v("kws_batch_mb", mag(GRADS.per_batch("KWS"), "MB"), "{:.1f}", "KWS batch, MB")
v("gpt2_params_b", GRADS.models["GPT-2"].parameters.to("count").magnitude / 10 ** 9, "{:.1f}", "GPT-2 parameters, billions")
v("gpt2_grad_gb", mag(GRADS.per_example("GPT-2"), "GB"), "{:.0f}", "GPT-2 gradient, GB")
v("gpt2_batch_tb", mag(GRADS.per_batch("GPT-2"), "TB"), "{:.1f}", "GPT-2 batch, TB")
v("gpt2_batch_tb3", mag(GRADS.per_batch("GPT-2"), "TB"), "{:.3f}", "the same, three decimals")
v("h100_gb", mag(Hardware.Cloud.H100.memory.capacity, "GiB"), "{:.0f}", "H100 memory as labeled, GB")
CIFAR_DP, CIFAR_PREV = lit(DP.Cifar10DeAccuracyAtEps8Pct), lit(DP.Cifar10PreviousBestAtEps8Pct)
CIFAR_NP = lit(DP.Cifar10Wrn40NonprivateLowerBoundPct)
IM_DP, IM_SAME, IM_BEST = (lit(DP.ImagenetDeAccuracyAtEps8Pct), lit(DP.ImagenetDeNonprivateSameModelPct),
                           lit(DP.ImagenetBestNonprivatePct))
v("cifar_dp", CIFAR_DP, "{:.1f}", "CIFAR-10 at epsilon 8, %")
v("cifar_prev", CIFAR_PREV, "{:.1f}", "previous best, %")
v("cifar_gain", CIFAR_DP - CIFAR_PREV, "{:.1f}", "gain, points")
v("cifar_np", CIFAR_NP, "{:.0f}", "non-private exceeds, %")
v("cifar_gap", math.floor(CIFAR_NP - CIFAR_DP), "{:.0f}", "privacy costs over, points")
v("im_dp", IM_DP, "{:.1f}", "ImageNet at epsilon 8, %")
v("im_same", IM_SAME, "{:.1f}", "same model, no privacy, %")
v("im_gap_same", IM_SAME - IM_DP, "{:.1f}", "gap, same model")
v("im_best", IM_BEST, "{:.1f}", "best non-private, %")
v("im_gap_best", IM_BEST - IM_DP, "{:.1f}", "gap, best model")
v("im_eps05", lit(DP.ImagenetDeAccuracyAtEps05Pct), "{:.1f}", "ImageNet at epsilon 0.5, %")
AUDIO = Scenarios.VoiceAssistantBillionDevices
AUDIO_HOUR = AUDIO.stream_rate * ureg.hour
v("audio_khz", mag(AUDIO.audio_sample_rate, "kHz"), "{:.0f}", "audio sample rate, kHz")
v("audio_bits", mag(AUDIO.audio_sample_bits, "bit"), "{:.0f}", "bits per sample")
v("audio_bps", mag(AUDIO.stream_rate, "byte/s"), "{:,.0f}", "audio bytes a second")
v("audio_mb_h", mag(AUDIO_HOUR, "MB"), "{:.1f}", "one hour of audio, MB")
v("update_mb", mag(GRADS.per_example("KWS"), "MB"), "{:.1f}", "one update, MB")
v("audio_x", (AUDIO_HOUR / GRADS.per_example("KWS")).to("").magnitude, "{:.0f}", "audio / update")

print("== Audit log (Scenarios.CreditDecisionAuditLog, RankingServiceAuditLog)")
v("rec_b", mag(LOG.record_size, "byte"), "{:.0f}", "bytes per record")
v("ret_months", mag(LOG.retention, "month"), "{:.0f}", "retention, months")
v("ret_days", round(LOG.retention_days, -1), "{:.0f}", "the same in days, about")
v("log_mb_day", mag(LOG.per_day, "MB"), "{:.0f}", "MB a day")
v("log_gb", mag(LOG.total, "GB"), "{:.0f}", "log over the retention, GB")
v("dlrm_req_m", DLOG.events_per_day / 10 ** 6, "{:.0f}", "ranking requests a day, millions")
v("dlrm_days", DLOG.retention_days, "{:.0f}", "days in a year")
v("dlrm_manifest_tb", mag(DLOG.total, "TB"), "{:.1f}", "manifests, TB")
v("dense", mlsysim.Datasets.CriteoKaggle.dense_features, "{:.0f}", "dense features")
v("sparse", mlsysim.Datasets.CriteoKaggle.categorical_features, "{:.0f}", "categorical features")
v("emb_dim", (mag(DLOG.payload_size, "byte") / mag(GRADS.bytes_per_param, "byte")
              - mlsysim.Datasets.CriteoKaggle.dense_features) / mlsysim.Datasets.CriteoKaggle.categorical_features,
  "{:.0f}", "embedding width")
v("input_b", mag(DLOG.payload_size, "byte"), "{:,.0f}", "full input, bytes")
v("dlrm_input_tb", mag(DLOG.payload_total, "TB"), "{:.0f}", "full inputs, TB")
v("payload_x", DLOG.payload_ratio, "{:.0f}", "full input / manifest")

print("== CodeCarbon run (Literature.ServingResults)")
CC_KWH, CC_KG = lit(SERVE.CodeCarbonRunKwh), lit(SERVE.CodeCarbonRunKg)
v("cc_wh", CC_KWH * 10 ** 3, "{:.3f}", "energy, Wh")
v("cc_mg", CC_KG * 10 ** 6, "{:.1f}", "emissions, mg")
v("cc_intensity", CC_KG / CC_KWH, "{:.3f}", "implied intensity, kg/kWh")
v("cc_cpu_pct", 100 * lit(SERVE.CodeCarbonRunCpuKwh) / CC_KWH, "{:.0f}", "CPU share, %")


if __name__ == "__main__":
    write(__file__)
