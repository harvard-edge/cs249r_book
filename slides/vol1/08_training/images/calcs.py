#!/usr/bin/env python3
"""Every derived number in Lecture 8 (Model Training, Part 1).

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered chapter, Model Training); each printed line is a value
the slide or its note shows, under the label used there. make_figs.py
imports from this file so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
P_GPT2XL = 1.5e9            # ch Lighthouse 8.1: GPT-2 XL parameters
FLOP_PER_TOKEN_FWD = 3.0e9  # ch Sec. 8.1: one generated GPT-2 token
RUN_FLOP = 1.5e21           # ch Sec. 8.1: GPT-2 training run
BATCH, SEQ = 32, 1024       # ch Sec. 8.6.6: 32 sequences of 1,024 tokens
LAYERS, HIDDEN, HEADS = 48, 1600, 25   # ch Napkin Math 8.3
PER_LAYER_STEP_TFLOP = 320.8  # ch: per-layer count incl. attention scores
H100_PEAK16 = 989e12        # ch Sec. 8.2: dense 16-bit peak, FLOP/s
H100_HBM_BW = 3.35e12       # ch Sec. 8.3.2: HBM bandwidth, B/s
H100_GIB = 80               # H100 capacity, GiB (sold as "80 GB")
H100_TDP_W = 700            # ch Sec. 8.2: thermal design power
UTIL = 0.50                 # ch Sec. 8.2: assumed utilization
PRICE_PER_HOUR = 4.0        # ch Napkin Math 8.7: dollars per H100-hour
MP_BYTES = (2, 2, 4, 8)     # ch Sec. 8.3.2: FP16 W, FP16 G, FP32 master, Adam m+v
CKPT_BYTES_PER_PARAM = 14   # ch Sec. 8.4.5: 2 + 4 + 8 (no gradients)
NVME_BW = 7e9               # ch Sec. 8.4.2: local NVMe, B/s

# Napkin Math 8.3 (ch): per-layer line items in MB, FP16, batch 32, S 1,024
ATTN_ACT_MB = 419.4
FFN_ACT_MB = 419.4
LN_MB = 10
LIB_TOTAL_ACT_GB = 286.9    # ch: "library estimate" of saved activations
FP32_STEP_GB = 598          # ch Napkin Math 8.6 / 8.10 (Part 2 opens with it)

# Llama 3 8B fine-tune (ch Napkin Math 8.4)
P_LLAMA8B = 8.0e9
LLAMA_ACT_GB = 36.5
LORA_PARAMS = 6.8e6

# ResNet-50 input pipeline (ch Sec. 8.4.2)
RESNET_DEMAND_A100 = 3200
RESNET_DEMAND_H100 = 10100
HOST_WORKERS, PER_WORKER = 14, 375

# Data-bound trace (ch Example 8.2)
PREP_MS, GPU_MS = 250, 12

# PaLM (ch War Story 8.1; Chowdhery et al. 2022)
P_PALM = 540e9
PALM_CHIPS = 6144

# PaLM Table 3 (Chowdhery et al. 2022, arXiv:2204.02311v5): MFU, one definition
PALM_TABLE3 = [("GPT-3 175B, V100", 21.3), ("Gopher 280B, 4,096 TPU v3", 32.5),
               ("Megatron-Turing NLG 530B, 2,240 A100", 30.2),
               ("PaLM 540B, 6,144 TPU v4", 46.2)]
PALM_HFU = 57.8             # same paper: hardware FLOPs utilization

# Llama 3.1 model card (github.com/meta-llama/llama-models, models/llama3_1/
# MODEL_CARD.md at commit 0e0b8c5, "Training Energy Use" table)
LLAMA31_GPU_HOURS = {"8B": 1.46e6, "70B": 7.0e6, "405B": 30.84e6}
LLAMA31_W = 700
# Dubey et al. 2024 (arXiv:2407.21783v3): 405B parameters, 15.6T tokens,
# 3.8e25 FLOPs (Sec. 3.2), BF16 MFU 38 to 43 percent (Table 4)
P_405B, D_405B = 405e9, 15.6e12

# PyTorch blog "Understanding GPU Memory 1" (Nov 2024), Memory Profiler figure:
# ResNet-50, SGD with momentum, FP32. Parameter count: torchvision docs
# (ResNet50_Weights.IMAGENET1K_V1, num_params 25557032).
RESNET50_PARAMS = 25_557_032

# OPT-175B logbook (metaseq projects/OPT/chronicles/images/logbook_screenshot.jpeg,
# 27_percent_update.md): entries run 12.17 (2021-11-24) to 12.38 (2021-12-01)
OPT_RUN_FIRST, OPT_RUN_LAST = 17, 38
OPT_DAY_FIRST, OPT_DAY_LAST = (2021, 11, 24), (2021, 12, 1)
OPT_NOV28_FIRST, OPT_NOV28_LAST = 27, 33     # entries dated 2021-11-28
OPT_GPUS = 992                              # Zhang et al. 2022, Sec. 2.4

# ZeRO (Rajbhandari et al. 2020, arXiv:1910.02054v3), Sec. 3.2 footnote 3:
# activations about 12 x hidden x batch x seq x layers (FP16, 2 bytes)
ZERO_ACT_COEFF = 12

GB = 1e9
DAY = 86400


# ---------------------------------------------------------------------------
def estimate_tokens():
    head("Estimate: tokens per training run")
    show("ratio run/token", f"{RUN_FLOP / FLOP_PER_TOKEN_FWD:.1e}")


def flops_per_token():
    head("6P against 2P")
    ratio = PER_LAYER_STEP_TFLOP / (step_flop() / 1e12)
    show("per-layer count over 6P count", f"{ratio:.3f} (about {100 * (ratio - 1):.0f} percent more)")


def step_flop():
    return 6 * P_GPT2XL * BATCH * SEQ


def one_step():
    head("One GPT-2 XL step")
    tokens = BATCH * SEQ
    t = step_flop() / (H100_PEAK16 * UTIL)
    show("tokens per step", f"{tokens:,}")
    show("O_step", f"{step_flop():.3e} FLOP = {step_flop() / 1e12:.0f} TFLOP")
    show("T_step", f"{t:.3f} s")
    show("steps", f"{RUN_FLOP / step_flop() / 1e6:.2f} million")
    show("D tokens", f"{RUN_FLOP / (6 * P_GPT2XL):.2e}")
    show("T_step at 25% utilization", f"{step_flop() / (H100_PEAK16 * 0.25):.2f} s")


def run_time():
    head("35 days and 590 kWh")
    t = RUN_FLOP / (H100_PEAK16 * UTIL)
    show("T", f"{t:.2e} s = {t / DAY:.1f} days")
    show("E", f"{H100_TDP_W * t / 3.6e6:.0f} kWh")
    show("cost", f"${t / 3600 * PRICE_PER_HOUR:,.0f}")
    show("one utilization point (51 vs 50%)", f"{(t - RUN_FLOP / (H100_PEAK16 * 0.51)) / 3600:.1f} hours")
    show("8 H100s, perfect scaling", f"{t / 8 / DAY:.1f} days")
    show("H100 80 GiB in GB", f"{H100_GIB * 2**30 / GB:.1f}")


def llama_card():
    head("Llama 3.1 model card")
    o = 6 * P_405B * D_405B
    hrs = LLAMA31_GPU_HOURS["405B"]
    rate = o / (hrs * 3600)
    show("6PD, 405B on 15.6T tokens", f"{o:.2e} FLOP")
    show("delivered per GPU", f"{rate / 1e12:.0f} TFLOP/s")
    show("implied utilization over all card hours", f"{100 * rate / H100_PEAK16:.1f}%")
    e = hrs * LLAMA31_W / 1e9   # Wh -> GWh
    show("energy at 700 W, 405B", f"{e:.1f} GWh")
    show("ratio to GPT-2 XL run (590 kWh)", f"{e * 1e6 / 590:,.0f}x")
    tot = sum(LLAMA31_GPU_HOURS.values())
    show("sum of card hours", f"{tot / 1e6:.2f} M (card total 39.3M)")
    gpt2_h = RUN_FLOP / (H100_PEAK16 * UTIL) / 3600
    show("405B hours over GPT-2 XL hours", f"{hrs / gpt2_h:,.0f}x")


def opt_logbook():
    head("OPT-175B logbook")
    show("runs in the screenshot", OPT_RUN_LAST - OPT_RUN_FIRST + 1)
    import datetime as dt
    days = (dt.date(*OPT_DAY_LAST) - dt.date(*OPT_DAY_FIRST)).days + 1
    show("calendar days spanned, inclusive", days)
    show("runs launched on 2021-11-28", OPT_NOV28_LAST - OPT_NOV28_FIRST + 1)


def zero_activations():
    head("ZeRO footnote 3 activation estimate")
    elems = ZERO_ACT_COEFF * HIDDEN * BATCH * SEQ * LAYERS
    show("12 x h x b x s x l x 2 B", f"{elems * 2 / GB:.1f} GB")


def static_bytes():
    head("16 bytes; 7B; GPT-2 XL static")
    b = sum(MP_BYTES)
    show("bytes per parameter", b)
    show("over 2-byte inference", f"{b / 2:.0f}x")
    p7 = 7e9
    parts = [p7 * x / GB for x in MP_BYTES]
    show("7B parts (W, G, master, m+v) GB", [f"{v:.0f}" for v in parts])
    show("7B total", f"{sum(parts):.0f} GB")
    show("7B serving FP16", f"{p7 * 2 / GB:.0f} GB")
    fp32 = [P_GPT2XL * 4 / GB, P_GPT2XL * 4 / GB, P_GPT2XL * 8 / GB]
    mixed = [P_GPT2XL * x / GB for x in MP_BYTES]
    show("GPT-2 XL FP32 W, G, m+v", fp32, )
    show("GPT-2 XL FP32 total", f"{sum(fp32):.0f} GB")
    show("GPT-2 XL mixed W, G, master, m+v", mixed)
    show("GPT-2 XL mixed total", f"{sum(mixed):.0f} GB")
    cap = H100_GIB * 2**30 / GB
    show("share of H100", f"{100 * sum(mixed) / cap:.0f}%")
    show("ZeRO 16 x 1.5B", f"{16 * P_GPT2XL / GB:.0f} GB; FP16 weights {2 * P_GPT2XL / GB:.0f} GB")


def adam_update():
    head("Adam update traffic")
    traffic = P_GPT2XL * 28
    t = traffic / H100_HBM_BW
    show("traffic", f"{traffic / GB:.0f} GB")
    show("time", f"{t * 1e3:.1f} ms")
    show("share of step", f"{100 * t / (step_flop() / (H100_PEAK16 * UTIL)):.1f}%")


def scores_gb(s):
    """Attention scores, all layers: 5 bytes x heads x S^2 x batch (ch 8.3)."""
    return LAYERS * 5 * HEADS * s ** 2 * BATCH / GB


def linear_gb(s):
    """Per-token activations at sequence length s: the library total minus
    the scores at 1,024, scaled linearly (every omitted tensor the chapter
    names, layer inputs, GELU outputs, dropout masks, is per token)."""
    return (LIB_TOTAL_ACT_GB - scores_gb(SEQ)) * s / SEQ


def activations():
    head("Activations against sequence length; 311 GB")
    line_items_linear = LAYERS * (ATTN_ACT_MB + FFN_ACT_MB + LN_MB) / 1000
    per_layer_scores = 5 * HEADS * SEQ ** 2 * BATCH / 1e6
    per_layer = ATTN_ACT_MB + FFN_ACT_MB + LN_MB + per_layer_scores
    show("scores per layer", f"{per_layer_scores:.1f} MB")
    show("per layer line items", f"{per_layer:.1f} MB")
    show("scores share of layer", f"{100 * per_layer_scores / per_layer:.0f}%")
    show("line items, linear part, 48 layers", f"{line_items_linear:.1f} GB")
    show("line items total", f"{per_layer * LAYERS / 1000:.1f} GB")
    show("scores, 48 layers", f"{scores_gb(SEQ):.1f} GB")
    show("per-token part (library total minus scores)", f"{linear_gb(SEQ):.1f} GB")
    show("activations at 1,024", f"{linear_gb(SEQ) + scores_gb(SEQ):.1f} GB")
    show("at 2,048: per-token, scores", f"{linear_gb(2048):.0f}, {scores_gb(2048):.0f} GB")
    static = P_GPT2XL * sum(MP_BYTES) / GB
    total = static + LIB_TOTAL_ACT_GB
    cap = H100_GIB * 2**30 / GB
    show("step total", f"{total:.1f} GB")
    show("over H100", f"{total / cap:.1f}x")
    show("FP32 step over H100", f"{FP32_STEP_GB / cap:.1f}x")


def llama_finetune():
    head("Llama 3 8B fine-tune")
    parts = [P_LLAMA8B * x / GB for x in MP_BYTES]
    full = sum(parts) + LLAMA_ACT_GB
    lora_state = LORA_PARAMS * 16 / GB
    lora = P_LLAMA8B * 2 / GB + lora_state + LLAMA_ACT_GB
    cap = H100_GIB * 2**30 / GB
    show("full parts W, G, master, m+v", [f"{v:.0f}" for v in parts])
    show("full state", f"{sum(parts):.0f} GB")
    show("full + activations", f"{full:.1f} GB (rounded half up: {int(full + 0.5)})")
    show("LoRA trainable share", f"{100 * LORA_PARAMS / P_LLAMA8B:.2f}%")
    show("LoRA adapter state", f"{lora_state:.2f} GB")
    show("LoRA total", f"{lora:.1f} GB")
    show("LoRA headroom", f"{cap - lora:.1f} GB")
    show("grads+master+moments (figure block)", f"{sum(parts[1:]):.0f} GB")


def resnet_memory():
    head("PyTorch memory timeline (ResNet-50, SGD momentum, FP32)")
    show("parameters", f"{RESNET50_PARAMS * 4 / GB:.2f} GB")
    show("momentum state", f"{RESNET50_PARAMS * 4 / GB:.2f} GB")


def mfu():
    head("Published MFU (PaLM Table 3)")
    for name, v in PALM_TABLE3:
        show(name, f"{v}%")
    show("GPT-2 XL MFU at 0.60 s", f"{100 * step_flop() / (H100_PEAK16 * step_flop() / (H100_PEAK16 * UTIL)):.0f}%")


def resnet_feed():
    head("ResNet-50 feeding")
    supply = HOST_WORKERS * PER_WORKER
    show("host supply", f"{supply:,} images/s")
    show("H100 busy", f"{100 * supply / RESNET_DEMAND_H100:.0f}%")
    show("H100 demand over A100", f"{RESNET_DEMAND_H100 / RESNET_DEMAND_A100:.1f}x")


def trace():
    head("Data-bound trace")
    show("busy", f"{100 * GPU_MS / (PREP_MS + GPU_MS):.1f}%")
    show("step now / with 2x GPU", f"{PREP_MS + GPU_MS} ms / {PREP_MS + GPU_MS / 2:.0f} ms")


def checkpoint():
    head("Checkpoint cadence")
    size = P_GPT2XL * CKPT_BYTES_PER_PARAM
    t_save = size / NVME_BW
    t_step = step_flop() / (H100_PEAK16 * UTIL)
    show("checkpoint", f"{size / GB:.0f} GB")
    show("save time", f"{t_save:.0f} s")
    show("every 100 steps: work", f"{100 * t_step:.0f} s")
    show("stall added to run time (save / work)", f"{100 * t_save / (100 * t_step):.1f}%")
    show("every 10,000 steps: work", f"{10000 * t_step:,.0f} s")
    show("every 10,000 steps: replay", f"{10000 * t_step / 60:.0f} min")


def palm():
    head("Back to PaLM")
    state = P_PALM * 16
    cap = H100_GIB * 2**30
    show("state", f"{state / 1e12:.2f} TB")
    show("H100s of memory", f"{state / cap:.0f}")
    show("checkpoint", f"{P_PALM * CKPT_BYTES_PER_PARAM / 1e12:.2f} TB")


if __name__ == "__main__":
    estimate_tokens()
    flops_per_token()
    one_step()
    run_time()
    llama_card()
    opt_logbook()
    static_bytes()
    zero_activations()
    adam_update()
    activations()
    llama_finetune()
    resnet_memory()
    mfu()
    resnet_feed()
    trace()
    checkpoint()
    palm()
