#!/usr/bin/env python3
"""Every derived number in Lecture 9 (Data Selection).

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered Data Selection chapter); each printed line is a value
the slide or its note shows, under the label used there. make_figs.py
imports from this file so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""

import math


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
# GPT-2 XL on WebText (ch Sec. 9.1)
P_GPT2 = 1.5e9              # parameters
WEBTEXT_TOKENS = 10e9       # about 10 billion unique tokens (40 GB at ~4 B/token)
RUN_FLOP = 1.5e21           # the original run
FLOP_PER_PARAM_TOKEN = 6    # training costs ~6 FLOPs per parameter per token
CHINCHILLA_TPP = 20         # Hoffmann et al. 2022 reference, tokens per parameter
H100_PEAK16 = 989e12        # ch / Lecture 8: dense 16-bit peak, FLOP/s
H100_W = 700                # board power, W
GPT2_GPUS, GPT2_MFU = 8, 0.50   # ch Sec. 9.1: one 8-H100 machine at 50% MFU

# Tokens per parameter (ch footnote 4; the papers)
GPT3_P, GPT3_TOKENS = 175e9, 300e9      # Brown et al. 2020
LLAMA2_P, LLAMA2_TOKENS = 70e9, 2.0e12  # Touvron et al. 2023 (2T tokens)

# Hoffmann et al. 2022 (arXiv:2203.15556v1): Chinchilla 70B on 1.4T tokens,
# Gopher 280B on 300B tokens, same compute budget (Sec. 1, Sec. 4);
# Table 2 approach 1: N_opt ~ C^0.50, D_opt ~ C^0.50.
CHIN_P, CHIN_TOKENS = 70e9, 1.4e12
GOPHER_P, GOPHER_TOKENS = 280e9, 300e9
EXP_B = 0.50

# ResNet-50 on ImageNet, one H100 (ch Napkin Math 9.1, 9.4, 9.5)
IMAGENET = 1_281_167        # ImageNet-1k training images ("1.28M")
GFLOP_PER_IMAGE = 24.6e9    # forward + backward
UTIL_RESNET = 0.25          # 25% of peak
GAIN_FULL, GAIN_HALF = 5.0, 4.5   # assumed epoch-1 gains, points
IMAGENET_GB = 148           # archive size
NVME_GBS = 7                # local NVMe, GB/s
PCIE_GBS = 64               # PCIe, GB/s
DEMAND_IPS = 10_100         # images/s the H100 consumes
HOST_IPS = 5_250            # images/s the host supplies
EPOCHS_90 = 90
CH_EPOCH_S = 126            # ch prints "about 126 seconds" per epoch

# Example 9.2 (ch): 100 epochs, 10% coreset
BASE_MIN = 211.0
SAVE_A_MIN, SCORE_A_S = 188.8, 42
SAVE_B_MIN, SCORE_B_S = 189.1, 18
EPOCHS_100 = 100

# MinHash (ch Sec. 9.3.2; FineWeb card and datatrove examples/fineweb.py)
WEBTEXT_DOCS = 8e6
WEBTEXT_GB = 40
HASHES_CH = 128
BYTES_PER_HASH = 4
FW_BUCKETS, FW_PER_BUCKET, FW_NGRAM = 14, 8, 5

# Lee et al. 2022: near-duplicate shares
DUP_C4, DUP_REALNEWS = 3.04, 13.63

# Coverage (ch Sec. 9.3.4)
POOL, RARE_FRAC, CORESET_FRAC, FLOOR = 1_000_000, 0.001, 0.10, 150

# Curriculum (ch Sec. 9.4.1), W illustrative
W = 10

# Active learning exercise (ch Sec. 9.8.2)
LABEL_USD, RANDOM_LABELS, AL_LABELS, ROUNDS, ROUND_USD = 10, 5000, 2000, 10, 50
START_LABELS, PER_ROUND = 1000, 100
HOSP_SCANS, HOSP_USD, HOSP_BUDGET, HOSP_AL = 1_000_000, 5, 500_000, 50_000

# FixMatch (Sohn et al. 2020), foundation-model amortization (ch)
CIFAR_TRAIN, FM_LABELS = 50_000, 4000
PRETRAIN_GPUH, FT_GPUH_BASE, FT_GPUH_FM = 10_000, 1000, 50

# Cost breakdown (ch Napkin Math 9.6, recomputed at 1.28M images, 148 GB)
ACQ_USD, LABEL_PRICE, STORE_USD_GB_MO, MONTHS = 50_000, 0.05, 0.02, 12
COMPUTE_USD, GPU_HOUR_USD = 421, 4   # ch table: compute line $421 at $4/GPU-h

# ROI (ch tables 13, 14)
BUILD, OPERATE, SAVE_PER_RUN = 50_000, 5_000, 10_000


# ---------------------------------------------------------------------------
# Derived values, one block per slide
# ---------------------------------------------------------------------------
def gpt2():
    tokens = RUN_FLOP / (FLOP_PER_PARAM_TOKEN * P_GPT2)
    passes = tokens / WEBTEXT_TOKENS
    optimal = CHINCHILLA_TPP * P_GPT2
    secs = RUN_FLOP / (GPT2_GPUS * H100_PEAK16 * GPT2_MFU)
    days = secs / 86400
    hours = secs / 3600
    kwh = GPT2_GPUS * H100_W * hours / 1000
    one_h100_days = RUN_FLOP / (H100_PEAK16 * GPT2_MFU) / 86400
    return dict(tokens=tokens, passes=passes, optimal=optimal, days=days,
                hours=hours, kwh=kwh, one_h100_days=one_h100_days)


def tokens_per_param():
    g = gpt2()
    return dict(gpt3=GPT3_TOKENS / GPT3_P,
                gpt2_unique=WEBTEXT_TOKENS / P_GPT2,
                gpt2_trained=g["tokens"] / P_GPT2,
                llama2=LLAMA2_TOKENS / LLAMA2_P,
                share=(WEBTEXT_TOKENS / P_GPT2) / CHINCHILLA_TPP)


def chinchilla():
    c_chin = FLOP_PER_PARAM_TOKEN * CHIN_P * CHIN_TOKENS
    c_goph = FLOP_PER_PARAM_TOKEN * GOPHER_P * GOPHER_TOKENS
    # With D = 20 N and C = 6 N D: N = sqrt(C / 120), D = 20 N
    n_opt = math.sqrt(RUN_FLOP / (FLOP_PER_PARAM_TOKEN * CHINCHILLA_TPP))
    return dict(chin_tpp=CHIN_TOKENS / CHIN_P, goph_tpp=GOPHER_TOKENS / GOPHER_P,
                c_chin=c_chin, c_goph=c_goph, c_ratio=c_chin / c_goph,
                p_ratio=GOPHER_P / CHIN_P, d_ratio=CHIN_TOKENS / GOPHER_TOKENS,
                double=(2 ** EXP_B - 1) * 100,
                gpt2_n_opt=n_opt, gpt2_d_opt=CHINCHILLA_TPP * n_opt)


def icr():
    flops = IMAGENET * GFLOP_PER_IMAGE
    full = GAIN_FULL / flops
    half = GAIN_HALF / (flops / 2)
    return dict(flops=flops, full=full, half=half, ratio=half / full)


def scoreboard():
    rescore_min = EPOCHS_100 * SCORE_A_S / 60
    sub_min = BASE_MIN * 0.10
    return dict(save_a_pct=SAVE_A_MIN / BASE_MIN * 100,
                save_b_pct=SAVE_B_MIN / BASE_MIN * 100,
                extra_s=(SAVE_B_MIN - SAVE_A_MIN) * 60,
                score_diff_s=SCORE_A_S - SCORE_B_S,
                rescore_min=rescore_min, sub_min=sub_min,
                rescore_save_pct=(BASE_MIN - rescore_min - sub_min) / BASE_MIN * 100,
                rescore_ratio=rescore_min / sub_min,
                lost_share=(SAVE_A_MIN / BASE_MIN * 100
                            - (BASE_MIN - rescore_min - sub_min) / BASE_MIN * 100)
                / (SAVE_A_MIN / BASE_MIN * 100) * 100)


def estimate_12h():
    return dict(total=12 + 1, breakeven=10 - 1)


def minhash():
    pairs = WEBTEXT_DOCS ** 2 / 2
    read_s = WEBTEXT_GB / NVME_GBS
    sig_gb = WEBTEXT_DOCS * HASHES_CH * BYTES_PER_HASH / 1e9
    return dict(pairs=pairs, read_s=read_s, sig_gb=sig_gb)


def fineweb():
    hashes = FW_BUCKETS * FW_PER_BUCKET
    thresh = (1 / FW_BUCKETS) ** (1 / FW_PER_BUCKET)

    def p(s):
        return 1 - (1 - s ** FW_PER_BUCKET) ** FW_BUCKETS
    return dict(hashes=hashes, thresh=thresh, p50=p(0.5), p70=p(0.7),
                p80=p(0.8), p90=p(0.9))


def dedup():
    g = gpt2()
    return dict(c4_h=g["hours"] * DUP_C4 / 100, rn_h=g["hours"] * DUP_REALNEWS / 100,
                c4_kwh=g["kwh"] * DUP_C4 / 100, rn_kwh=g["kwh"] * DUP_REALNEWS / 100,
                run_h=g["hours"])


def coverage():
    rare = POOL * RARE_FRAC
    kept = rare * CORESET_FRAC
    return dict(rare=rare, kept=kept, extra=FLOOR - kept)


def warmup():
    return dict(served=(W + 1) / 2, full=W)


def active_learning():
    rnd = RANDOM_LABELS * LABEL_USD
    al = AL_LABELS * LABEL_USD + ROUNDS * ROUND_USD
    return dict(rnd=rnd, al=al, roi=(rnd - al) / al * 100,
                labels_after=START_LABELS + ROUNDS * PER_ROUND,
                hosp_all=HOSP_SCANS * HOSP_USD,
                hosp_random=HOSP_BUDGET / HOSP_USD,
                hosp_mult=(HOSP_BUDGET / HOSP_USD) / HOSP_AL)


def fixmatch():
    return dict(share=FM_LABELS / CIFAR_TRAIN * 100,
                tasks=PRETRAIN_GPUH / (FT_GPUH_BASE - FT_GPUH_FM))


def data_path():
    flops = IMAGENET * GFLOP_PER_IMAGE
    epoch_s = flops / (H100_PEAK16 * UTIL_RESNET)
    pruned_s = CH_EPOCH_S / 2
    epoch_kj = CH_EPOCH_S * H100_W / 1000
    score_s = epoch_s / 3
    saved_h = EPOCHS_90 * pruned_s / 3600
    saved_mj = EPOCHS_90 * pruned_s * H100_W / 1e6
    score_kj = SCORE_A_S * H100_W / 1000
    decoded_gb = IMAGENET * 224 * 224 * 3 / 1e9
    return dict(epoch_s=epoch_s, demand_epoch_s=IMAGENET / DEMAND_IPS,
                pruned_s=pruned_s, epoch_kj=epoch_kj, pruned_kj=epoch_kj / 2,
                score_s=score_s, score_kj=score_kj, saved_h=saved_h,
                saved_mj=saved_mj, energy_ratio=saved_mj * 1000 / score_kj,
                read_s=IMAGENET_GB / NVME_GBS, half_gb=IMAGENET_GB / 2,
                decoded_gb=decoded_gb, pcie_s=decoded_gb / PCIE_GBS,
                host_epoch_s=IMAGENET / HOST_IPS)


def echo():
    r = DEMAND_IPS / HOST_IPS
    return dict(r=r, busy=HOST_IPS / DEMAND_IPS * 100,
                h_no=EPOCHS_90 * IMAGENET / HOST_IPS / 3600,
                h_e2=EPOCHS_90 * IMAGENET / DEMAND_IPS / 3600)


def cost():
    labels = 1.28e6 * LABEL_PRICE
    storage = IMAGENET_GB * STORE_USD_GB_MO * MONTHS
    total = ACQ_USD + labels + storage + COMPUTE_USD
    return dict(labels=labels, storage=storage, compute=COMPUTE_USD,
                gpu_h=COMPUTE_USD / GPU_HOUR_USD, total=total,
                compute_pct=COMPUTE_USD / total * 100,
                data_pct=(total - COMPUTE_USD) / total * 100)


def roi():
    invest = BUILD + OPERATE
    vals = {n: (SAVE_PER_RUN * n - invest) / invest * 100 for n in (1, 5, 10, 50)}
    # Panel fix L10: the slide shows savings as a multiple of the investment
    # (savings / investment) instead of a percentage return such as +809%.
    mult = {n: SAVE_PER_RUN * n / invest for n in (1, 5, 10, 50)}
    return dict(invest=invest, vals=vals, mult=mult, breakeven=invest / SAVE_PER_RUN)


if __name__ == "__main__":
    g = gpt2()
    head("Slide: Predict, how often did GPT-2 see each WebText token?")
    show("tokens processed (billions)", f"{g['tokens'] / 1e9:.0f}")
    show("passes over WebText", f"{g['passes']:.1f}")
    show("compute-optimal tokens, 20 x 1.5B (billions)", f"{g['optimal'] / 1e9:.0f}")
    show("days on 8 H100s at 50% MFU", f"{g['days']:.1f}")
    show("run hours", f"{g['hours']:.1f}")
    show("energy (kWh)", f"{g['kwh']:.0f}")
    show("days on one H100 at 50% (Lecture 8)", f"{g['one_h100_days']:.0f}")

    t = tokens_per_param()
    head("Slide: WebText held a third of the unique tokens GPT-2's size calls for")
    show("GPT-3 tokens per parameter", f"{t['gpt3']:.1f}")
    show("GPT-2 XL unique tokens per parameter", f"{t['gpt2_unique']:.1f}")
    show("GPT-2 XL training tokens per parameter", f"{t['gpt2_trained']:.0f}")
    show("Llama 2 70B tokens per parameter", f"{t['llama2']:.1f}")
    show("GPT-2 unique share of 20", f"{t['share']:.2f}")
    show("processed / compute-optimal tokens", f"{g['tokens'] / g['optimal']:.1f}x")

    c = chinchilla()
    head("Slide: Chinchilla, the compute-optimal frontier")
    show("Chinchilla tokens per parameter", f"{c['chin_tpp']:.0f}")
    show("Gopher tokens per parameter", f"{c['goph_tpp']:.2f}")
    show("Gopher / Chinchilla parameters", f"{c['p_ratio']:.0f}x")
    show("Chinchilla / Gopher tokens", f"{c['d_ratio']:.1f}x")
    show("6ND Chinchilla (FLOPs)", f"{c['c_chin']:.2e}")
    show("6ND Gopher (FLOPs)", f"{c['c_goph']:.2e}")
    show("6ND ratio Chinchilla/Gopher", f"{c['c_ratio']:.2f}")
    show("doubling compute: more tokens (%)", f"{c['double']:.0f}")
    show("GPT-2 budget: N_opt (billions)", f"{c['gpt2_n_opt'] / 1e9:.1f}")
    show("GPT-2 budget: D_opt (billions)", f"{c['gpt2_d_opt'] / 1e9:.0f}")
    show("D_opt / WebText unique tokens", f"{c['gpt2_d_opt'] / WEBTEXT_TOKENS:.1f}x")

    i = icr()
    head("Slide: Half of ImageNet returned 1.8x the learning per FLOP")
    show("FLOPs per epoch", f"{i['flops']:.2e}")
    show("full ICR (per FLOP)", f"{i['full']:.2e}")
    show("half FLOPs", f"{i['flops'] / 2:.2e}")
    show("half ICR (per FLOP)", f"{i['half']:.2e}")
    show("ratio", f"{i['ratio']:.1f}x")

    s = scoreboard()
    head("Slide: Rescoring every epoch eats a third of the saving")
    show("Option A saving (%)", f"{s['save_a_pct']:.1f}")
    show("Option B saving (%)", f"{s['save_b_pct']:.1f}")
    show("B over A, from rounded minutes (s)", f"{s['extra_s']:.0f}")
    show("scoring time difference, 42 - 18 (s)", f"{s['score_diff_s']:.0f}")
    show("rescoring 100 x 42 s (min)", f"{s['rescore_min']:.0f}")
    show("subset training (min)", f"{s['sub_min']:.0f}")
    show("rescoring / subset training", f"{s['rescore_ratio']:.1f}x")
    show("saving with rescoring (%)", f"{s['rescore_save_pct']:.0f}")
    show("share of the one-pass saving lost (%)", f"{s['lost_share']:.0f}")

    e = estimate_12h()
    head("Slide: Estimate, 12 hours of scoring")
    show("scoring + subset (h)", e["total"])
    show("scoring must fall below (h)", e["breakeven"])

    m = minhash()
    head("Slide: MinHash")
    show("all pairs", f"{m['pairs']:.1e}")
    show("40 GB read (s)", f"{m['read_s']:.1f}")
    show("signatures (GB)", f"{m['sig_gb']:.1f}")

    f = fineweb()
    head("Slide: FineWeb card and MinHash config")
    show("hashes per document, 14 x 8", f["hashes"])
    show("candidate threshold (1/b)^(1/r)", f"{f['thresh']:.2f}")
    for k in ("p50", "p70", "p80", "p90"):
        show(f"P(candidate) at Jaccard 0.{k[1]}", f"{f[k] * 100:.1f}%")

    d = dedup()
    head("Slide: Deduplication returns 3 to 14 hours")
    show("run hours", f"{d['run_h']:.1f}")
    show("C4 share: hours, kWh", f"{d['c4_h']:.1f} h, {d['c4_kwh']:.0f} kWh")
    show("RealNews share: hours, kWh", f"{d['rn_h']:.1f} h, {d['rn_kwh']:.0f} kWh")

    v = coverage()
    head("Slide: Predict, rare examples in a 10% coreset")
    show("rare in pool", f"{v['rare']:.0f}")
    show("kept unstratified", f"{v['kept']:.0f}")
    show("extra to reach floor", f"{v['extra']:.0f}")

    w = warmup()
    head("Slide: Curriculum warmup")
    show("served (D)", w["served"])
    show("full (D)", w["full"])

    a = active_learning()
    head("Slide: Estimate, active learning")
    show("labels after 10 rounds", a["labels_after"])
    show("random cost ($)", f"{a['rnd']:,.0f}")
    show("active cost ($)", f"{a['al']:,.0f}")
    show("return (%)", f"{a['roi']:.1f}")
    show("hospital: all labels ($)", f"{a['hosp_all']:,.0f}")
    show("hospital: random labels for budget", f"{a['hosp_random']:,.0f}")
    show("hospital: multiplier", f"{a['hosp_mult']:.0f}x")

    x = fixmatch()
    head("Slide: FixMatch")
    show("4,000 of 50,000 (%)", f"{x['share']:.0f}")
    show("pretraining pays back after (tasks)", f"{x['tasks']:.1f}")

    p = data_path()
    head("Slide: Pruning halves the arithmetic, but not always the reads")
    show("epoch at 25% of peak, computed (s)", f"{p['epoch_s']:.1f}")
    show("epoch at 10,100 images/s (s)", f"{p['demand_epoch_s']:.1f}")
    show("chapter epoch (s), pruned (s)", f"{CH_EPOCH_S}, {p['pruned_s']:.0f}")
    show("epoch energy, pruned (kJ)", f"{p['epoch_kj']:.0f}, {p['pruned_kj']:.0f}")
    show("scoring = 1/3 epoch (s)", f"{p['score_s']:.0f}")
    show("scoring energy (kJ)", f"{p['score_kj']:.0f}")
    show("90 epochs saved (h)", f"{p['saved_h']:.2f}")
    show("90 epochs saved (MJ)", f"{p['saved_mj']:.2f}")
    show("saved / scoring energy", f"{p['energy_ratio']:.0f}x")
    show("NVMe read (s)", f"{p['read_s']:.0f}")
    show("subset shards (GB)", f"{p['half_gb']:.0f}")
    show("decoded tensors (GB)", f"{p['decoded_gb']:.0f}")
    show("PCIe copy (s)", f"{p['pcie_s']:.1f}")
    show("epoch at host rate (s)", f"{p['host_epoch_s']:.0f}")

    h = echo()
    head("Slide: Predict, data echoing")
    show("r", f"{h['r']:.2f}")
    show("busy without echo (%)", f"{h['busy']:.0f}")
    show("waiting on input without echo (%)", f"{100 - h['busy']:.0f}")
    show("90 epochs, no echo (h)", f"{h['h_no']:.1f}")
    show("90 epochs, e = 2 (h)", f"{h['h_e2']:.1f}")

    k = cost()
    head("Slide: Predict, share of the bill")
    show("labels ($)", f"{k['labels']:,.0f}")
    show("storage ($)", f"{k['storage']:.2f}")
    show("compute GPU-hours", f"{k['gpu_h']:.0f}")
    show("total ($)", f"{k['total']:,.0f}")
    show("compute share (%)", f"{k['compute_pct']:.1f}")
    show("data share (%)", f"{k['data_pct']:.1f}")

    o = roi()
    head("Slide: Deduplication pipeline ROI")
    show("investment ($)", f"{o['invest']:,}")
    for n, val in o["vals"].items():
        show(f"return after {n} runs (%)", f"{val:+.0f}")
    for n, m in o["mult"].items():
        show(f"savings / investment after {n} runs (x)", f"{m:.1f}")
    show("break-even (runs)", f"{o['breakeven']:.1f}")
