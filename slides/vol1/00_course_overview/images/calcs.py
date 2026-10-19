#!/usr/bin/env python3
"""Every derived number Lecture 0 shows, computed from sourced inputs.

Run from this folder: python3 calcs.py
Each line prints the value and the slide (or note) that shows it. Copy the
printed values onto the slides; never compute a slide number by hand.
"""

import math

# --- One chat answer ----------------------------------------------------------
ANSWER_TOK = 500        # tokens in one answer (the predict slide's setup)
WORDS_PER_TOKEN = 0.75  # OpenAI tokenizer guide: one token is about 3/4 of a word
GOOGLE_WH = 0.24        # Elsworth et al. 2025: median Gemini Apps text prompt
PHONE_J = 40_000        # charging a smartphone (Model Serving, Napkin Math 13.5)
google_j = GOOGLE_WH * 3600

# --- Fleet scale -----------------------------------------------------------------
PROMPTS_PER_DAY = 2.5e9    # ChatGPT, OpenAI to Axios, July 2025
GPT3_MWH = 1287            # Patterson et al. 2021, GPT-3 training energy
fleet_mwh = PROMPTS_PER_DAY * GOOGLE_WH / 1e6

# --- Accelerators to serve one day of prompts (estimate, capability slide) -----
GPU_TOK_PER_S = 114 * 8    # one H100 serving 114 users at 8 tokens/s each
                           # (Model Serving, Napkin Math 13.5)
SECONDS_PER_DAY = 86_400
tok_per_s_fleet = PROMPTS_PER_DAY * ANSWER_TOK / SECONDS_PER_DAY
gpus_avg = tok_per_s_fleet / GPU_TOK_PER_S

# --- Datacenter electricity (IEA, Energy and AI, 2025) ---------------------------
DC_TWH_2024, DC_TWH_2030 = 415, 945

# --- Three Mile Island Unit 1 (Constellation press release, 20 Sept 2024) --------
TMI_MW = 837
TMI_RETIRED, TMI_ONLINE = 2019, 2028

# --- Power tiers (System Paradigms, Sec. 2.1) -------------------------------------
CLOUD_W, TINY_W = 3e6, 1e-3

rows = [
    ("Predict: 500 tokens in words", f"{ANSWER_TOK * WORDS_PER_TOKEN:.0f} words"),
    ("Reveal: Google median prompt in joules", f"{google_j:.0f} J"),
    ("Reveal: phone charge in median prompts", f"{PHONE_J / google_j:.0f} prompts"),
    ("Fleet: prompts x 0.24 Wh", f"{fleet_mwh:.0f} MWh per day"),
    ("Fleet: GPT-3 runs per day", f"{fleet_mwh / GPT3_MWH:.2f} (one run every {GPT3_MWH / fleet_mwh:.1f} days)"),
    ("Electricity: 2030 over 2024", f"{DC_TWH_2030 / DC_TWH_2024:.2f}x"),
    ("TMI: years from retirement to restart", f"{TMI_ONLINE - TMI_RETIRED} years"),
    ("Accelerators: fleet tokens per second", f"{tok_per_s_fleet:.3g} tokens/s"),
    ("Accelerators: one GPU's token rate", f"{GPU_TOK_PER_S} tokens/s"),
    ("Accelerators: GPUs at average load", f"{gpus_avg:,.0f}"),
    ("Power tiers: cloud over TinyML", f"10^{math.log10(CLOUD_W / TINY_W):.1f}"),
]

if __name__ == "__main__":
    for label, value in rows:
        print(f"{label:46s} {value}")
