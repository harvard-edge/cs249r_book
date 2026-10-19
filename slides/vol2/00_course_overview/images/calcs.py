#!/usr/bin/env python3
"""Every derived number in Volume II, Lecture 0, computed from sourced inputs.

Each line prints the value as the slide or note shows it, with its label.
Run from this folder: python3 calcs.py
"""

import math
from datetime import date

SEC_PER_DAY, HOURS_PER_YEAR = 86400, 24 * 365


def show(label, value):
    print(f"{label:<66} {value}")


# --- Hook: GPT-3 (Introduction Sec. 1.1 and footnote 6; Brown et al. 2020;
#     Patterson et al. 2021 for the 10,000 V100s at about 24.6 TFLOP/s each)
gpt3_flops = 3.14e23
a100_peak = 312e12
v100_count, v100_delivered = 10_000, 24.6e12
t_one = gpt3_flops / a100_peak
t_fleet = gpt3_flops / (v100_count * v100_delivered)
show("One A100 at peak: years (slide: about 32 years)", f"{t_one / SEC_PER_DAY / 365:.1f}")
show("10,000 V100s: days (slide: about 15 days)", f"{t_fleet / SEC_PER_DAY:.1f}")
show("Ratio one-A100 time / fleet time (note: about 790x)", f"{t_one / t_fleet:.0f}x")

# --- Landmark accelerator counts (MLSysIM Literature.TrainingRuns; Introduction figure 2)
verified = [(2012, 2), (2019, 512), (2022, 6144), (2023, 2048), (2024, 2048), (2024, 16384)]
n = len(verified)
xs = [y for y, _ in verified]
ys = [math.log10(g) for _, g in verified]
mx, my = sum(xs) / n, sum(ys) / n
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
show("Fitted growth in accelerator count per year (plot: about 2x)", f"{10 ** slope:.2f}x")
show("AlexNet 2 GPUs to Llama 3 16,384: ratio (note: 8,192x)", f"{16384 / 2:,.0f}x")

# --- Alps (ETH Zurich press release, 14 Sep 2024: 10,752 Grace Hopper superchips)
alps_chips = 10_752
show("Alps superchips against GPT-3's 10,000 V100s (note: about 1.08x)", f"{alps_chips / 10_000:.2f}x")

# --- Serving (Altman, OpenAI DevDay, 6 Oct 2025, as reported by TechCrunch:
#     'More than 800 million people use ChatGPT every week, and we process over
#     6 billion tokens per minute on the API.')
api_tokens_per_min = 6e9
show("API tokens per second (slide: about 100 million)", f"{api_tokens_per_min / 60:.1e}")
show("API tokens per day (note: about 8.6 trillion)", f"{api_tokens_per_min * 60 * 24:.2e}")

# --- BLOOM-176B (bigscience-workshop/bigscience, train/tr11-176B-ml/README.md:
#     started 11 March 2022, epoch one finished 28 June 2022; 416 A100 80 GB in
#     52 nodes, 384 used in 48 nodes, 32 (4 nodes) in reserve. Slurm script at
#     commit ed196cd (19 June 2022): --nodes=48, --gres=gpu:8, --time 100:00:00,
#     TP_SIZE=4, PP_SIZE=12.)
start, epoch_end = date(2022, 3, 11), date(2022, 6, 28)
days = (epoch_end - start).days
gpus_used, gpus_total, nodes, per_node = 384, 416, 48, 8
show("BLOOM: days from start to end of epoch one (slide: 109 days)", f"{days}")
show("BLOOM: nodes x GPUs per node (slide: 384)", f"{nodes * per_node}")
show("BLOOM: reserve GPUs (slide: 32)", f"{gpus_total - gpus_used}")
show("BLOOM: reserve share of the allocation (note: 7.7 percent)", f"{100 * (gpus_total - gpus_used) / gpus_total:.1f}%")
gpu_hours = gpus_used * days * 24
show("BLOOM: GPU-hours, 384 x 109 days x 24 (slide: about 1.0 million)", f"{gpu_hours:,}")
show("BLOOM: one GPU alone, years (note: about 115 years)", f"{gpu_hours / HOURS_PER_YEAR:.0f}")
show("BLOOM: relaunches forced by a 100 h limit, at least (note: 27)", f"{math.ceil(days * 24 / 100)}")
show("BLOOM: GPUs per model copy, TP x PP (note: 48)", f"{4 * 12}")

# --- Running case A (Introduction Sec. 1.6.1)
show("175B x 16 bytes (slide: 2.8 TB)", f"{175e9 * 16 / 1e12:.1f} TB")

# --- Take-home self-check (Appendix A): H100 ridge point, dense FP16
show("H100 ridge: 989 TFLOP/s / 3.35 TB/s (note: about 295 FLOP/byte)", f"{989 / 3.35:.0f}")
