"""Numbers for the Model Training study sheet (Volume I, Chapter 8).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to training_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Derived results are computed from the rounded values the sheet prints beside them, so a
student who redoes the arithmetic from the page gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Hardware, Models, Systems
from mlsysim.core.units import BYTES_FP16, BYTES_FP32
from mlsysim.engine.scenarios import Scenarios
from mlsysim.physics.communication import calc_ring_allreduce_data_factor


def c(x):
    return f"{x:,.0f}"


def r(x, nd=1):
    """Round to nd decimals and return the float the sheet prints."""
    return round(x, nd)


def numbers() -> dict:
    n = {}
    h100 = Hardware.Cloud.H100
    gpt2 = Models.Language.GPT2

    # ---- Reference machine (chapter: ReferenceMachine, MixedPrecisionThroughputCost)
    peak = h100.compute.peak_flops.to("TFLOP/s").magnitude                      # dense 16-bit
    fp32 = h100.compute.precision_flops["fp32"].to("TFLOP/s").magnitude
    cap_gib = h100.memory.capacity.to("GiB").magnitude
    cap_gb = h100.memory.capacity.to("GB").magnitude
    tdp = h100.tdp.to("W").magnitude
    n.update(hpeak=c(peak), hfp32=c(fp32), hcapgib=c(cap_gib), hcapgb=f"{cap_gb:.1f}", htdp=c(tdp),
             hbw=f"{h100.memory.bandwidth.to('TB/s').magnitude:.2f}", bfx=f"{peak / fp32:.0f}")

    # ---- Work per token (chapter lighthouse: 2 forward + 4 backward = 6 FLOPs per parameter per token)
    fwd, bwd = 2, 4
    n.update(ffwd=c(fwd), fbwd=c(bwd), fpp=c(fwd + bwd))

    # ---- 16 bytes per parameter, mixed-precision Adam (chapter: GPT2Optimizer, tbl-optimizer-properties)
    b16, b32 = BYTES_FP16.to("byte").magnitude, BYTES_FP32.to("byte").magnitude
    bw, bg, bm, bmom = b16, b16, b32, 2 * b32
    bstate = bw + bg + bm + bmom
    n.update(bw=c(bw), bg=c(bg), bmaster=c(bm), bmom=c(bmom), bstate=c(bstate), b32=c(b32),
             b32state=c(b32 + b32 + 2 * b32), xinfer=c(bstate / b16))

    # ---- Worked example: GPT-2 XL, batch 32 x 1,024 tokens, mixed precision, one H100 (chapter walkthrough)
    amp = Scenarios.Gpt2MixedPrecisionAdamOnH100
    flash = Scenarios.Gpt2MixedPrecisionFlashAttentionOnH100
    micro = Scenarios.Gpt2MixedPrecisionTwoMicroBatchesOnH100
    acts = Scenarios.Gpt2XlBatch32Activations
    p_b = gpt2.parameters.to("param").magnitude / 1e9
    static = r(p_b * bstate, 0)                                      # 1.5 B x 16 bytes
    a_std = r(acts.activations("none").to("GB").magnitude)
    a_fl = r(acts.activations("selective").to("GB").magnitude)
    a_score = r(a_std - a_fl)
    a_mic = r(micro.activations.to("GB").magnitude)
    a_fp32 = r(Scenarios.Gpt2XlBatch32ActivationsFp32.activations("none").to("GB").magnitude)
    t_std, t_fl, t_mic = r(static + a_std), r(static + a_fl), r(static + a_mic)
    capr = r(cap_gb)
    n.update(gpb=f"{p_b:.1f}", gbatch=c(amp.batch_size), gmicro=c(micro.batch_size), gseq=c(acts.sequence),
             gstatic=f"{static:.0f}", gact=f"{a_std:.1f}", gtot=f"{t_std:.1f}", gover=f"{t_std / capr:.1f}",
             gscore=f"{a_score:.1f}", gactfl=f"{a_fl:.1f}", gtotfl=f"{t_fl:.1f}", gexcess=f"{t_fl - capr:.1f}",
             gactmic=f"{a_mic:.1f}", gmom=f"{p_b * bmom:.0f}", gtotmic=f"{t_mic:.1f}", ghead=f"{capr - t_mic:.1f}", gact32=f"{a_fp32:.1f}")
    # check the scenario totals against the printed sums
    assert abs(flash.training_total.to("GB").magnitude - t_fl) < 0.1
    assert abs(micro.training_total.to("GB").magnitude - t_mic) < 0.1

    # ---- Your turn: Llama 3 8B with rank-16 adapters (chapter: LlamaFinetuneMemory); ASSUMED batch 8
    lora = Scenarios.Llama3_8BLoraOnH100
    lp = Models.Language.Llama3_8B.parameters.to("param").magnitude / 1e9
    lw = r(lp * bw)
    lad = round(lora.adapter_state.to("GB").magnitude, 2)
    la4 = r(lora.activations.to("GB").magnitude)
    batch_new = 8                                                   # ASSUMED
    la8 = r(la4 * batch_new / lora.batch_size)
    assert abs(lora.activations_at(batch_new).to("GB").magnitude - la8) < 0.1
    l8 = r(lw + lad + la8)
    l4 = r(lw + lad + la4)
    n.update(lpb=f"{lp:.2f}", lw=f"{lw:.1f}", lad=f"{lad:.2f}", lbatch=c(lora.batch_size), lseq=c(Scenarios.Llama3_8BBatch4Activations.sequence),
             lact4=f"{la4:.1f}", lbatchnew=c(batch_new), lact8=f"{la8:.1f}", ltot8=f"{l8:.1f}", ltot4=f"{l4:.1f}",
             lsteps=c(batch_new // lora.batch_size), lfull=f"{r(lp * bstate, 0):.0f}", lhead=f"{capr - l4:.1f}",
             lrank=c(16), ladm=f"{lora.adapter_parameters / 1e6:.1f}")

    # ---- Q1: ASSUMED GPT-2 XL trained on 20B more tokens on one H100 at 40 percent, rented at 4 dollars/hour
    tokens, eta, price = 20e9, 0.40, 4.0
    work = (fwd + bwd) * p_b * 1e9 * tokens
    hours = r(work / (peak * 1e12 * eta) / 3600)
    kwh = r(hours * tdp / 1e3)
    n.update(q1tok=f"{tokens / 1e9:.0f}", q1eta=f"{eta * 100:.0f}", q1price=f"{price:.0f}",
             q1work=f"{work / 10 ** int(math.log10(work)):.1f}", q1pow=f"10^{{{int(math.log10(work))}}}", q1h=f"{hours:.1f}", q1d=f"{hours / 24:.1f}", q1kwh=f"{kwh:.1f}",
             q1usd=c(hours * price))

    # ---- Q2: ASSUMED measured step of 1.5 s for GPT-2 XL, 32 x 1,024 tokens (chapter: MFUExampleCalc)
    step_tokens = amp.batch_size * acts.sequence
    step_tflop = r((fwd + bwd) * p_b * 1e9 * step_tokens / 1e12, 0)
    t_step, busy = 1.5, 98
    n.update(q2tflop=f"{step_tflop:.0f}", q2t=f"{t_step:.1f}", q2busy=c(busy),
             q2mfu=f"{step_tflop / (peak * t_step) * 100:.0f}")

    # ---- Q3: tokenization feed (chapter: GPT2DataPipeline); ASSUMED four workers instead of eight
    feed = Scenarios.Gpt2XlSingleCoreTokenizerFeed
    tc = r(feed.compute_time.to("ms").magnitude, 0)
    tp = r(feed.tokenization_time.to("ms").magnitude)
    workers = 4
    n.update(q3comp=f"{tc:.0f}", q3prep=f"{tp:.1f}", q3w=c(workers), q3int=f"{tp / workers:.1f}",
             q3serial=f"{tc / (tc + tp) * 100:.0f}", q3rate=f"{amp.batch_size / (tc / 1e3):.0f}")

    # ---- Q4: ring AllReduce inside one server (chapter: GPT2NodeAllReduce); ASSUMED four accelerators
    node = Systems.Nodes.DGX_H100
    link = node.intra_node_bw.to("GB/s").magnitude
    nacc = 4
    grad = r(gpt2.size_in_bytes(BYTES_FP16).to("GB").magnitude)
    factor = calc_ring_allreduce_data_factor(nacc)
    t_ar = r(factor * grad / link * 1e3)
    per_rep = amp.batch_size // nacc
    eta_ch = Scenarios.Gpt2XlRunOnOneH100.efficiency
    t_c = r((fwd + bwd) * p_b * 1e9 * per_rep * acts.sequence / (peak * 1e12 * eta_ch) * 1e3)
    n.update(q4n=c(nacc), q4link=c(link), q4grad=f"{grad:.0f}", q4factor=f"{factor:.1f}", q4ar=f"{t_ar:.1f}",
             q4per=c(per_rep), q4eta=f"{eta_ch * 100:.0f}", q4comp=f"{t_c:.0f}", q4share=f"{t_ar / (t_c + t_ar) * 100:.1f}")

    # ---- Mistakes
    # 2: linear scaling rule, chapter's illustrative batch 512 -> 4,096 at learning rate 0.1 (FallaciesPitfallsSetup)
    bs0, bs1, lr0 = 512, 4096, 0.1
    n.update(m2b0=c(bs0), m2b1=c(bs1), m2lr0=f"{lr0:.1f}", m2lr1=f"{lr0 * bs1 / bs0:.1f}", m2k=c(bs1 / bs0))
    # 3: ResNet-50 input feed on the H100 (ResNetFeedH100)
    rf = Scenarios.ResNet50H100InputFeed
    dem = round(rf.sustained_rate.to("count/second").magnitude, -2)
    sup = rf.host_supply.to("count/second").magnitude
    n.update(m3dem=c(dem), m3sup=c(sup), m3busy=f"{sup / dem * 100:.0f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
