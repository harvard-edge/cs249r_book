#!/usr/bin/env python3
"""Every derived number in Lecture 15 (Sustainable AI), computed from sourced inputs.

Run from this folder: python3 calcs.py
Each printed line is "<slide label>: <value>"; the deck and notes copy these values.
make_figs.py imports the same constants, so figures and slides agree.

Sources (inputs only; nothing below is typed from mental arithmetic):
  SA      the rendered chapter, Sustainable AI (section, table, notebook named per input)
  MLSysIM registry: grid factors, PUE 1.12, H100 TDP and embodied carbon, DGX H100 rack
  LLAMA3  Dubey et al. 2024, "The Llama 3 Herd of Models", Sec. 3.3.1 (16K H100, 700 W TDP)
  CARD    Llama 3.1 model card (meta-llama/llama-models, MODEL_CARD.md), emissions table
  ELS     Elsworth et al. 2025, arXiv 2508.15734, Tables 1 and 2, Sec. 3.3 and 4.3
  PAT     Patterson et al. 2021, arXiv 2104.10350, Table 4
  TPU     Schneider et al. 2025, arXiv 2502.01671, Table 1
  BLOOM   Luccioni et al. 2023, arXiv 2211.02001, Tables 3 and 5
  CSO     CSO Ireland, Data Centres Metered Electricity Consumption 2023, Table 1
  EC      Energy-Charts (Fraunhofer ISE), EU generation, week 47 2023 (SA figure 19 data)
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Hardware, Infrastructure, Systems  # noqa: E402

# ------------------------------------------------------------------ registry
GRID = {name: getattr(Infrastructure.Grids, key).carbon_intensity_g_kwh
        for key, name in [("Norway", "Norway"), ("Quebec", "Quebec"),
                          ("France", "France"), ("EU_Avg", "EU average"),
                          ("Germany", "Germany"), ("US_Avg", "US average"),
                          ("China_Avg", "China"), ("India_Avg", "India"),
                          ("Poland", "Poland")]}           # g/kWh, IEA WEO 2023 rounded
PUE = Infrastructure.FacilityCooling.BestAir.pue            # 1.12 (SA Napkin Math 15.6)
TDP_KW = Hardware.Cloud.H100.tdp.m_as("kW")                 # 0.7
EMBODIED_KG = Hardware.Cloud.H100.embodied_carbon_kg        # 164 (SA 15.5.5, NVIDIA PCF / 8)
RACK = Systems.Racks.DGX_H100_4Node                         # SA table 7
RACK_GPU_KW = RACK.accelerator_power.m_as("kW")             # 22.4
RACK_IT_KW = RACK.it_power.m_as("kW")                       # 30.8
SUPPORT = RACK_IT_KW / RACK_GPU_KW                          # IT / GPU ratio
RACK_TOTAL_KW = 33.5                                        # SA table 7, with allocated non-IT
SERVICE_LIFE_H = 3 * 8760                                   # SA: 3-year standard life

# ------------------------------------------------------------ running case
GPUS = 16_384                 # LLAMA3 Sec. 3.3.1 ("up to 16K H100")
GPU_HOURS = 30.84e6           # CARD, Llama 3.1 405B
CARD_LB_T = 8_930             # CARD, location-based t CO2eq
CARD_MB_T = 0                 # CARD, market-based

P_ACC = GPUS * TDP_KW / 1000          # MW
P_IT = P_ACC * SUPPORT
P_FAC = P_IT * PUE
E_ACC = GPU_HOURS * TDP_KW / 1000     # MWh
E_IT = E_ACC * SUPPORT
E_FAC = E_IT * PUE
HOURS = GPU_HOURS / GPUS
DAYS = HOURS / 24
CARD_IMPLIED_G = CARD_LB_T / E_ACC * 1000   # g/kWh if card energy = GPU-h x TDP


def tco2(mwh, g):
    return mwh * g / 1000.0


OP_US = tco2(E_FAC, GRID["US average"])
OP_QC = tco2(E_FAC, GRID["Quebec"])
OP_PL = tco2(E_FAC, GRID["Poland"])
OP_NO = tco2(E_FAC, GRID["Norway"])
EMB_TOTAL_T = GPUS * EMBODIED_KG / 1000
EMB_T = GPU_HOURS * EMBODIED_KG / SERVICE_LIFE_H / 1000
EMB_SHARE_US = EMB_T / (OP_US + EMB_T) * 100
EMB_SHARE_QC = EMB_T / (OP_QC + EMB_T) * 100
WUE_GOOGLE = 1.15             # ELS Sec. 3.3, L/kWh (IT boundary)
WUE_EVAP = 1.8                # SA Sec. 15.4.1, evaporative example
WATER_LO = E_IT * 1000 * WUE_GOOGLE / 1e6   # million liters
WATER_HI = E_IT * 1000 * WUE_EVAP / 1e6
PROMPT_WH = 0.24              # ELS Table 1, full stack incl. PUE 1.09
PROMPTS_PER_DAY = 1e9         # illustrative volume
SERVE_MWH_DAY = PROMPTS_PER_DAY * PROMPT_WH / 1e6
CROSSOVER_DAYS = E_FAC / SERVE_MWH_DAY

# ----------------------------------------------------------- chapter scenarios
# SA Napkin Math 15.3: 2 MW IT, PUE 1.58 -> 1.10, $70/MWh
PUE_SAVE_MW = 2.0 * (1.58 - 1.10)
PUE_SAVE_MWH = PUE_SAVE_MW * 8760
PUE_SAVE_USD = PUE_SAVE_MWH * 70
# SA 15.6 self-check 3: 10 MW IT; A PUE 1.08 at 450 g; B PUE 1.35 at 50 g
A_T_PER_H = 10 * 1.08 * 450 / 1000
B_T_PER_H = 10 * 1.35 * 50 / 1000
# SA 15.8.6: replacement with 50% less energy than a 700 W GPU, continuous
SAVED_KWH_DAY = 0.5 * TDP_KW * 24
BE_US_WEEKS = EMBODIED_KG / (SAVED_KWH_DAY * GRID["US average"] / 1000) / 7
BE_QC_YEARS = EMBODIED_KG / (SAVED_KWH_DAY * GRID["Quebec"] / 1000) / 365
# SA Checkpoint 15.5: energy per query -50%, queries +300%
JEV_TOTAL = 0.5 * 4.0
# SA 15.3 self-check: energy per task /4, volume x20
JEV2_TOTAL = 20 / 4
# SA fallacy: 30x less energy per query, 100x volume
JEV3_TOTAL = 100 / 30
# SA 15.3.3: Kaplan compute exponent 0.05, 10% lower loss
KAPLAN_MULT = 0.9 ** (-1 / 0.05)
KAPLAN_TWO = KAPLAN_MULT ** 2
# SA 15.2: Quebec vs Poland, 10,000 MWh; self-check: -15% energy in Poland
GEO_QC_T = tco2(10_000, GRID["Quebec"])
GEO_PL_T = tco2(10_000, GRID["Poland"])
GEO_PL_ALGO_T = GEO_PL_T * 0.85
GEO_RATIO = GRID["Poland"] / GRID["Quebec"]
REGION_RATIO = GRID["Poland"] / GRID["Norway"]
# SA 15.4.2: DRAM FP32 operand vs GPU FLOP; crossover; batch-1 decode
DRAM_OPERAND_PJ = 4 * 160
DRAM_RATIO = DRAM_OPERAND_PJ / 10
CROSSOVER_FPB = 160 / 10
DECODE_MOVE_SHARE = 320 / (320 + 20) * 100
# SA 15.7.4: cascade 0.5 + 0.10 x (10 + 50) mJ vs 60 mJ always-cloud
CASCADE_MJ = 0.5 + 0.10 * (10 + 50)
CASCADE_SAVE = (1 - CASCADE_MJ / 60) * 100
# SA 15.3 self-check 2: 120 MW run, 80 MW feed, 3 days
FEED_FRACTION = 80 / 120
FEED_DAYS = 3 / FEED_FRACTION
# Lighthouse 15.1
LIGHTHOUSE_MW = 25_000 * TDP_KW / 1000
# SA 15.6.3: 10 MW at WUE 1.8 for a year
WATER_10MW_ML = 10 * 1000 * 8760 * 1.8 / 1e6
# SA 15.8.6: 3-year to 5-year life
LIFE_CUT = (1 - 3 / 5) * 100

# ------------------------------------------------------------ outside tables
ELS_EXISTING = 0.10
ELS_FULL = 0.14 + 0.06 + 0.02 + 0.02
ELS_RATIO = ELS_FULL / ELS_EXISTING
ELS_ACCEL_SHARE = 0.14 / ELS_FULL * 100
ELS_LB, ELS_MB = 345, 94                    # g/kWh, ELS Sec. 3.3
ELS_FACTOR_RATIO = ELS_LB / ELS_MB
# PAT Table 4 (net tCO2e; params B)
PAT_GSHARD_T5 = 4.3 / 46.7
PAT_GPT3_CHECK = 1287 * 0.429               # MWh x t/MWh
TPU_V3_TDP = 450
TPU_V3_MEAS = (245, 310)                    # PAT Table 4, incl. host
TPU_V3_MEAS_SHARE = (245 / 450 * 100, 310 / 450 * 100)
# TPU Table 1: v6e, 6 years, kg
V6E_EMB, V6E_MB, V6E_LB = 692, 2141, 5759
V6E_SHARE_LB = V6E_EMB / (V6E_EMB + V6E_LB) * 100
V6E_SHARE_MB = V6E_EMB / (V6E_EMB + V6E_MB) * 100
# BLOOM Table 3 and 5
BLOOM_EMB_SHARE = 22.2
BLOOM_FINAL_T, BLOOM_PROJECT_T = 24.69, 66.29
BLOOM_FINAL_SHARE = BLOOM_FINAL_T / BLOOM_PROJECT_T * 100
BLOOM_PROJECT_MULT = BLOOM_PROJECT_T / BLOOM_FINAL_T
# CSO Table 1
CSO = [(2015, 5), (2016, 6), (2017, 7), (2018, 8), (2019, 9), (2020, 11),
       (2021, 14), (2022, 18), (2023, 21)]
CSO_GWH_2015, CSO_GWH_2023 = 1_238, 6_334
CSO_GROWTH = CSO_GWH_2023 / CSO_GWH_2015
# LBNL via SA 15.6.1
US_2014, US_2024 = 58, 192
US_GROWTH = US_2024 / US_2014


def eu_week():
    rows = list(csv.DictReader(open(HERE / "europe_generation_2023w47.csv")))
    cols = ["nuclear", "hydro", "biomass_other", "wind", "solar", "gas", "coal_oil"]
    t = [datetime.fromisoformat(r["time_cet"]) for r in rows]
    fos = [(float(r["gas"]) + float(r["coal_oil"])) /
           sum(float(r[c]) for c in cols) * 100 for r in rows]
    roll = [sum(fos[i:i + 24]) / 24 for i in range(len(fos) - 23)]
    lo = min(range(len(roll)), key=roll.__getitem__)
    hi = max(range(len(roll)), key=roll.__getitem__)
    wind = [float(r["wind"]) / 1000 for r in rows]
    return t, fos, roll, lo, hi, min(fos), max(fos), min(wind), max(wind)


EU_T, EU_FOS, EU_ROLL, EU_LO, EU_HI, EU_MIN, EU_MAX, WIND_MIN, WIND_MAX = eu_week()
EU_RATIO = EU_ROLL[EU_HI] / EU_ROLL[EU_LO]


def report():
    p = print
    p("--- running case: Llama 3.1 405B")
    p(f"accelerator power (16,384 x 700 W): {P_ACC:.2f} MW")
    p(f"IT/GPU ratio (DGX H100 rack): {SUPPORT:.3f}")
    p(f"IT power: {P_IT:.2f} MW")
    p(f"facility power at PUE {PUE}: {P_FAC:.2f} MW")
    p(f"GPU energy (30.84M GPU-h x 700 W): {E_ACC:,.0f} MWh")
    p(f"IT energy: {E_IT:,.0f} MWh")
    p(f"facility energy: {E_FAC:,.0f} MWh")
    p(f"duration if all GPUs ran: {HOURS:,.0f} h = {DAYS:.0f} days")
    p(f"card implied factor (8,930 t / GPU energy): {CARD_IMPLIED_G:.0f} g/kWh")
    p(f"operational US 429: {OP_US:,.0f} t; Quebec 20: {OP_QC:,.0f} t; "
      f"Poland 820: {OP_PL:,.0f} t; Norway 10: {OP_NO:,.0f} t")
    p(f"facility account / card: {OP_US / CARD_LB_T:.2f}x")
    p(f"embodied manufactured (16,384 x 164 kg): {EMB_TOTAL_T:,.0f} t")
    p(f"embodied allocated (3-year life): {EMB_T:.0f} t")
    p(f"embodied share US: {EMB_SHARE_US:.1f}%; Quebec: {EMB_SHARE_QC:.0f}%")
    p(f"water at WUE 1.15 / 1.8: {WATER_LO:.0f} / {WATER_HI:.0f} million L")
    p(f"serving 1B prompts/day x 0.24 Wh: {SERVE_MWH_DAY:.0f} MWh/day; "
      f"crossover day {CROSSOVER_DAYS:.0f}")
    for n, g in GRID.items():
        p(f"  region {n}: {g:g} g/kWh -> {tco2(E_FAC, g):,.0f} t")
    p("--- chapter scenarios")
    p(f"rack-note example: 14,000 GPUs x 700 W = {14_000 * TDP_KW / 1000:.1f} MW; "
      f"IT {14_000 * TDP_KW / 1000 * SUPPORT:.1f} MW")
    p(f"3-year service life: {SERVICE_LIFE_H:,} h")
    p(f"Lighthouse 25,000 x 700 W: {LIGHTHOUSE_MW:.1f} MW")
    p(f"feed 80 of 120 MW: {FEED_FRACTION * 100:.0f}% concurrency, {FEED_DAYS:.1f} days")
    p(f"rack IT/GPU {SUPPORT:.2f}x, total/GPU {RACK_TOTAL_KW / RACK_GPU_KW:.2f}x, "
      f"GPU share of rack {RACK_GPU_KW / RACK_TOTAL_KW * 100:.0f}%, support {RACK_IT_KW - RACK_GPU_KW:.1f} kW")
    p(f"facility MW for 2 MW IT at PUE 1.58/1.55/1.10/1.09: "
      f"{2 * 1.58:.2f}/{2 * 1.55:.2f}/{2 * 1.10:.2f}/{2 * 1.09:.2f}")
    p(f"PUE 1.58->1.10 on 2 MW: {PUE_SAVE_MW:.2f} MW, {PUE_SAVE_MWH:,.1f} MWh/yr, ${PUE_SAVE_USD:,.0f}")
    p(f"PUE A vs B: {A_T_PER_H:.2f} vs {B_T_PER_H:.3f} t/h, ratio {A_T_PER_H / B_T_PER_H:.1f}x")
    p(f"saved power {0.5 * TDP_KW:.2f} kW; saved per day US: {SAVED_KWH_DAY * GRID['US average'] / 1000:.2f} kg; "
      f"Quebec: {SAVED_KWH_DAY * GRID['Quebec'] / 1000:.3f} kg")
    p(f"radio bit vs FP32 multiply: {100_000 / 3.7:,.0f}x to {500_000 / 3.7:,.0f}x")
    p(f"wake-word 10 mW / 0.0155 mW: {10 / 0.0155:.0f}x")
    p(f"break-even US: {BE_US_WEEKS:.1f} weeks; Quebec: {BE_QC_YEARS:.1f} years "
      f"({SAVED_KWH_DAY:.1f} kWh/day saved)")
    p(f"geography 10,000 MWh: Quebec {GEO_QC_T:.0f} t, Poland {GEO_PL_T:,.0f} t, "
      f"Poland -15% {GEO_PL_ALGO_T:,.0f} t, ratio {GEO_RATIO:.0f}x; regions {REGION_RATIO:.0f}x")
    p(f"Jevons: 0.5 x 4 = {JEV_TOTAL:g}x; 20/4 = {JEV2_TOTAL:g}x; 100/30 = {JEV3_TOTAL:.1f}x")
    p(f"Kaplan: 10% lower loss = {KAPLAN_MULT:.1f}x compute; twice = {KAPLAN_TWO:.0f}x")
    p(f"DRAM operand {DRAM_OPERAND_PJ} pJ = {DRAM_RATIO:.0f}x FLOP; crossover {CROSSOVER_FPB:.0f} "
      f"FLOP/byte; batch-1 decode movement share {DECODE_MOVE_SHARE:.0f}%")
    p(f"cascade {CASCADE_MJ:.1f} mJ vs 60 mJ: {CASCADE_SAVE:.0f}% less")
    p(f"10 MW evaporative water: {WATER_10MW_ML:.0f} million L/yr; life 3->5 yr cut {LIFE_CUT:.0f}%")
    p("--- outside tables")
    p(f"Gemini prompt: {ELS_EXISTING} vs {ELS_FULL:.2f} Wh, {ELS_RATIO:.1f}x; "
      f"accelerators {ELS_ACCEL_SHARE:.0f}% of full stack; LB/MB {ELS_FACTOR_RATIO:.1f}x")
    p(f"Patterson: GShard/T5 emissions {PAT_GSHARD_T5:.2f}; GPT-3 check {PAT_GPT3_CHECK:.1f} t; "
      f"TPU v3 measured {TPU_V3_MEAS_SHARE[0]:.0f}-{TPU_V3_MEAS_SHARE[1]:.0f}% of TDP")
    p(f"TPU v6e embodied share LB {V6E_SHARE_LB:.0f}%, MB {V6E_SHARE_MB:.0f}%")
    p(f"BLOOM final run {BLOOM_FINAL_SHARE:.0f}% of project; project {BLOOM_PROJECT_MULT:.1f}x final")
    p(f"CSO data-center GWh 2015->2023: {CSO_GROWTH:.1f}x")
    p(f"LBNL US 2014->2024: {US_GROWTH:.1f}x")
    p(f"EU week fossil share hourly {EU_MIN:.0f}-{EU_MAX:.0f}%; 24-h windows "
      f"{EU_ROLL[EU_LO]:.0f}% ({EU_T[EU_LO]:%a %H:%M} start) to {EU_ROLL[EU_HI]:.0f}% "
      f"({EU_T[EU_HI]:%a %H:%M} start), ratio {EU_RATIO:.2f}x; wind {WIND_MIN:.0f}-{WIND_MAX:.0f} GW")


if __name__ == "__main__":
    report()
