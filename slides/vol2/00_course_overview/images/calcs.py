#!/usr/bin/env python3
"""Every derived number in Volume II, Lecture 0, computed from sourced inputs.

Each line prints the value as the slide or note shows it, with its label.
Run from this folder: python3 calcs.py
"""

import math

SEC_PER_DAY, HOURS_PER_YEAR = 86400, 24 * 365


def show(label, value):
    print(f"{label:<62} {value}")


# --- GPT-3 (Introduction Sec. 1.1, footnote 6; Brown et al. 2020; Patterson et al. 2021)
gpt3_flops = 3.14e23          # total training work, FLOPs
a100_peak = 312e12            # FLOP/s, A100 peak (Introduction footnote 6)
v100_count = 10_000           # V100s in the run (Patterson et al. 2021)
v100_delivered = 24.6e12      # FLOP/s delivered per V100 (Patterson et al. 2021)
t_one = gpt3_flops / a100_peak
t_fleet = gpt3_flops / (v100_count * v100_delivered)
show("One A100: seconds", f"{t_one:.1e} s")
show("One A100: years (slide: about 32 years)", f"{t_one / SEC_PER_DAY / 365:.1f}")
show("One A100: days (plot bar)", f"{t_one / SEC_PER_DAY:,.0f}")
show("Fleet: seconds", f"{t_fleet:.1e} s")
show("Fleet: days (slide: about 15 days)", f"{t_fleet / SEC_PER_DAY:.1f}")
show("Orders of magnitude between the two (note: almost three)", f"{math.log10(t_one / t_fleet):.2f}")

# --- Landmark accelerator counts (MLSysIM Literature.TrainingRuns, model registry)
verified = [(2012, 2), (2019, 512), (2022, 6144), (2023, 2048), (2024, 2048), (2024, 16384)]
n = len(verified)
xs = [y for y, _ in verified]
ys = [math.log10(g) for _, g in verified]
mx, my = sum(xs) / n, sum(ys) / n
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
show("Fitted growth in accelerator count per year (slide: about 2x)", f"{10 ** slope:.2f}x")
show("Orders of magnitude, AlexNet to Llama 3 (note: four)", f"{math.log10(16384 / 2):.1f}")

# --- Growth decomposition (Introduction Sec. 1.1, figure 1b; Epoch AI 2026)
count, duration, rate, compute = 1.6, 1.3, 2.0, 3.9
show("Product count x duration x rate (note: about 4.2)", f"{count * duration * rate:.2f}")
show("Count x duration (note: fleet's share)", f"{count * duration:.2f}x")

# --- Scaling price (Kaplan et al. 2020 compute fit, exponent 0.050; Conclusion Sec. 17)
alpha = 0.050
show("Loss multiple per doubling (slide: 0.966)", f"{2 ** -alpha:.3f}")
show("Loss reduction per doubling (slide: about 3.4 percent)", f"{100 * (1 - 2 ** -alpha):.1f}%")
show("Loss multiple at 1,024x compute (slide: 0.71)", f"{1024 ** -alpha:.2f}")

# --- Serving cost (Serving at Scale, Napkin Math 10.1; $0.001/request is a scenario input)
users, req_per_day, days, usd_per_req, training_usd = 1e6, 50, 365, 0.001, 2.0e6
requests = users * req_per_day * days
serving = requests * usd_per_req
show("Requests per year (note: 18.25 billion)", f"{requests / 1e9:.2f} B")
show("First-year serving cost (slide: $18.25M)", f"${serving / 1e6:.2f}M")
show("Serving / training (slide: 9.1x)", f"{serving / training_usd:.1f}x")

# --- Bandwidth staircase (Compute Infrastructure, Napkin Math 2.3), 10 GB buffer
buf = 10.0  # GB
for name, bw in (("HBM 3,350 GB/s", 3350.0), ("NVLink 450 GB/s", 450.0), ("InfiniBand 50 GB/s", 50.0)):
    show(f"10 GB over {name} (ms)", f"{1000 * buf / bw:.1f} ms")
show("InfiniBand / NVLink time (slide: 9x)", f"{450.0 / 50.0:.0f}x")

# --- Failure clock (Introduction Sec. 1.1, assumed MTTF; Fault Tolerance War Story 7.2)
mttf, gpus = 50_000, 16_384
show("50,000 h in years (slide: about 5.7 years)", f"{mttf / HOURS_PER_YEAR:.1f}")
show("MTBF = MTTF / N, 16,384 GPUs (slide: 3.05 h)", f"{mttf / gpus:.2f} h")
show("Llama 3 measured: 54 days x 24 h / 419 (slide: 3.1 h)", f"{54 * 24 / 419:.1f} h")

# --- Llama 3 interruption causes (Dubey et al. 2024, table 5, via Fault Tolerance figure 2)
causes = {"Faulty GPU": 148, "GPU HBM3": 72, "Software bug": 54, "Network": 35,
          "Host maintenance": 32, "GPU SRAM": 19, "GPU system processor": 17}
show("Faulty GPU + HBM3 (slide title: 220)", f"{148 + 72}")
show("Remaining 11 causes (plot: 42)", f"{419 - sum(causes.values())}")

# --- Reliability gap (Introduction Sec. 1.4.1)
show("0.999^1,000 (slide: 36.8%)", f"{100 * 0.999 ** 1000:.1f}%")
show("0.999^10,000 (credit: 0.0045%)", f"{100 * 0.999 ** 10000:.4f}%")
show("0.9999^10,000 (plot: 36.8%)", f"{100 * 0.9999 ** 10000:.1f}%")

# --- Rack power (Compute Infrastructure Sec. 2.6.2) and the grid (Sustainable AI, Lighthouse 15.1)
show("100 kW / 12 kW (title: about eight times)", f"{100 / 12:.1f}x")
show("25,000 GPUs x 700 W (slide: 17.5 MW)", f"{25_000 * 700 / 1e6:.1f} MW")
show("16,384 GPUs x 700 W (note: about 11.5 MW)", f"{16_384 * 700 / 1e6:.1f} MW")

# --- The Fleet Law step (Introduction Sec. 1.5 self-check 1)
comp, comm, sync = 0.9, 0.4, 0.2
step = comp + comm + sync
fast = comp / 2 + comm + sync
show("Measured step (slide: 1.5 s)", f"{step:.1f} s")
show("Compute halved (slide: 1.05 s)", f"{fast:.2f} s")
show("Speedup (slide: 1.43x)", f"{step / fast:.2f}x")
show("Half of the old step (plot mark: 0.75 s)", f"{step / 2:.2f} s")
# Chapter fallacy (Introduction Sec. 1.8): 1.2 s compute, 28 s exposed transfer
show("2x faster on 1.2 s + 28 s (note: 2.1 percent shorter)", f"{100 * (0.6 / 29.2):.1f}%")

# --- Archetype A (Introduction Sec. 1.6.1) and the self-check
show("175B x 16 bytes (slide: 2.8 TB)", f"{175e9 * 16 / 1e12:.1f} TB")
show("H100 ridge: 989 TFLOP/s / 3.35 TB/s (note: about 295 FLOP/byte)", f"{989 / 3.35:.0f}")
