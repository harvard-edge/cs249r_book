#!/usr/bin/env python3
"""Every number on the Lecture 0 (Course Overview) slides and notes, read from MLSysIM.

Lecture 0 previews later lectures, so each previewed number is read from the
same MLSysIM object the later deck or chapter reads (named in each block), not
retyped: the H100 chat energy scenario of Model Serving (Lecture 13), the
measured Gemini prompt (Lectures 13b, 15b, 16), the IEA datacenter figures
(Lecture 16), the power tiers (Lectures 2 and 16), GPT-3's training energy
(Lecture 15b), and Google's search volume (Introduction). One scenario is this
deck's own, Scenarios.ChatGptDayAtGeminiMedianPrompt. This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex. make_figs.py imports this module, so figures
and slides share one computation.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

from mlsysim.core.units import Q_  # noqa: E402

Hardware, Models, Systems, Platforms = mlsysim.Hardware, mlsysim.Models, mlsysim.Systems, mlsysim.Platforms
Scenarios, Literature, ReferenceStats = mlsysim.Scenarios, mlsysim.Literature, mlsysim.ReferenceStats
DCE = Literature.DatacenterElectricity

ENERGY = Scenarios.H100ChatEnergyPerToken                    # Lecture 13 (Model Serving)
CHAT = Scenarios.ChatGptDayAtGeminiMedianPrompt               # this deck
POD, SPARK = Systems.Pods.TPUv4_4096, Hardware.Workstation.DGX_Spark   # Lecture 2


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def lit(entry):
    """A registry value as a plain number."""
    return float(entry)


print("== Predict: one answer (Scenarios.H100ChatEnergyPerToken; OpenAI tokenizer rule of thumb)")
v("ans_tokens", ENERGY.response_tokens, "{:.0f}", "tokens in one answer")
v("ans_words", ENERGY.response_tokens * lit(ReferenceStats.TextTokenization.WordsPerToken), "{:.0f}",
  "the same answer in words")

print("== Reveal: the measured median prompt (Literature.Series.GeminiPromptEnergy, as in Lecture 13)")
phone_j = m(ReferenceStats.EnergyAnchors.SmartphoneCharge, "J")
v("prompt_wh", m(CHAT.energy_per_prompt, "Wh"), "{:.2f}", "median Gemini prompt, Wh")
v("prompt_j", m(CHAT.energy_per_prompt, "J"), "{:,.0f}", "the same, J")
v("phone_j", phone_j, "{:,.0f}", "charging a phone, J (Model Serving)")
v("phone_prompts", phone_j / m(CHAT.energy_per_prompt, "J"), "{:.0f}", "prompts in one phone charge")

print("== Fleet scale (Scenarios.ChatGptDayAtGeminiMedianPrompt; Models.Language.GPT3, as in Lecture 15b)")
v("prompts_bn", CHAT.prompts_per_day / 10 ** 9, "{:.1f}", "ChatGPT prompts per day, billions")
v("fleet_mwh", m(CHAT.daily_energy, "MWh"), "{:,.0f}", "one day of prompts, MWh")
v("gpt3_mwh", m(CHAT.reference_energy, "MWh"), "{:,.0f}", "training GPT-3 once, MWh")
v("gpt3_share", CHAT.reference_share, "{:.2f}", "one day over one GPT-3 run (calcs only)")
v("gpt3_days_word", round(CHAT.days_per_reference), "words", "days of prompts per GPT-3 run, whole, in prose")
v("search_bn", lit(ReferenceStats.Workloads.GoogleSearchesPerDay) / 10 ** 9, "{:.1f}",
  "Google searches per day, billions (Introduction)")

print("== Datacenter electricity (Literature.DatacenterElectricity, IEA; as in Lecture 16)")
v("dc_share_pct", 100 * lit(DCE.Global2024ShareOfElectricity), "{:.1f}", "datacenters' share of world electricity, 2024, %")
v("dc_twh_2024", lit(DCE.Global2024Twh), "{:.0f}", "datacenters, 2024, TWh")
v("dc_twh_2030", lit(DCE.Global2030Twh), "{:.0f}", "datacenters, 2030 projection, TWh")
v("dc_growth", lit(DCE.Global2030Twh) / lit(DCE.Global2024Twh), "{:.1f}", "2030 over 2024")

print("== Three Mile Island Unit 1 (Literature.DatacenterElectricity, Constellation 2024)")
v("tmi_mw", lit(DCE.CraneUnit1CapacityMw), "{:.0f}", "Unit 1 capacity, MW")
v("tmi_retired", lit(DCE.CraneUnit1RetiredYear), "{:.0f}", "retired, year")
v("tmi_online", lit(DCE.CraneUnit1ExpectedOnlineYear), "{:.0f}", "expected online, year")
v("tmi_ppa_years", lit(DCE.CraneMicrosoftPpaYears), "{:.0f}", "Microsoft agreement, years")
v("tmi_gap_word", lit(DCE.CraneUnit1ExpectedOnlineYear) - lit(DCE.CraneUnit1RetiredYear), "words",
  "years from retirement to restart, in prose")

print("== Power tiers (Lecture 2's ladder, as in Lecture 16)")
cloud_w, edge_w = m(POD.power, "W"), m(SPARK.tdp, "W")
mobile_w = float(Platforms.Mobile.tdp_range_w.split("-")[1])
tiny_w = m(Platforms.Tiny.power_threshold, "W")
v("cloud_mw", cloud_w / 10 ** 6, "{:.0f}", "cloud rung, MW")
v("edge_w", edge_w, "{:.0f}", "edge rung, W")
v("mobile_w", mobile_w, "{:.0f}", "mobile rung, W")
v("tiny_mw", tiny_w * 10 ** 3, "{:.0f}", "TinyML rung, mW")
v("power_orders_word", math.floor(math.log10(cloud_w / tiny_w)), "words", "whole orders of magnitude, cloud to TinyML")
TIERS = [("Cloud", cloud_w), ("Edge", edge_w), ("Mobile", mobile_w), ("TinyML", tiny_w)]

print("== Estimate: accelerators for one day of prompts (Scenarios.ChatGptDayAtGeminiMedianPrompt)")
v("day_s", m(Q_(1, "day"), "s"), "{:,.0f}", "seconds in a day")
v("fleet_tok_m", m(CHAT.token_rate, "1/s") / 10 ** 6, "{:.1f}", "the day's tokens per second, millions")
v("h100_users", ENERGY.concurrent_requests, "{:.0f}", "users one H100 serves at once (Lecture 13)")
v("h100_tok_s", m(ENERGY.token_rate, "1/s"), "{:.0f}", "one H100's tokens per second (Lecture 13)")
v("gpus_exact", CHAT.accelerators_average, "{:,.0f}", "accelerators at average load (calcs only)")
v("gpus_k", round(CHAT.accelerators_average, -3), "{:,.0f}", "the same, to the thousand")

if __name__ == "__main__":
    write(__file__)
