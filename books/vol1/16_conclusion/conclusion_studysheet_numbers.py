"""Numbers for the Conclusion study sheet (Volume I, Chapter 16).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to conclusion_studysheet_numbers.tex next to this file.
Values the chapter states in prose are loaded from the same scenario objects where one exists.
Inputs the sheet chooses for its own questions are marked ASSUMED.
"""
from pathlib import Path

from mlsysim import Hardware, Models
from mlsysim.core.units import BYTES_FP16, BYTES_INT8, HOURS_PER_DAY, DAYS_PER_YEAR
from mlsysim.engine.scenarios import Scenarios as SC


def g(x, nd=3):
    return f"{x:.{nd}g}"


def c(x):
    return f"{x:,.0f}"


def numbers() -> dict:
    n = {}

    # ---- Worked example: Llama 2 70B batch-one decode on two H100s (chapter: conclusion-roofline-setup)
    ll = SC.Llama2_70B_DecodeTwoH100
    shards = ll.tp_size
    d = ll.result
    w_gb = (d.weight_bytes * shards).to("GB").magnitude            # total FP16 weights read per token
    flop_g = (d.flops_per_step * shards).to("GFLOP").magnitude      # two FLOPs per parameter
    bw = ll.hardware.memory.bandwidth.to("TB/s").magnitude * shards
    peak = ll.hardware.compute.peak_flops.to("TFLOP/s").magnitude * shards
    t_mem = d.memory_time.to("ms").magnitude
    t_cmp = d.compute_time.to("ms").magnitude
    ridge = peak / bw
    intensity = flop_g / w_gb                                       # FLOP per byte
    n.update(llparams=f"{ll.workload.parameters.to('param').magnitude / 1e9:.0f}", llshards=c(shards),
             llgb=f"{w_gb:.0f}", llgflop=f"{flop_g:.0f}", llbw=f"{bw:.1f}", llpeak=c(peak),
             lltmem=f"{t_mem:.1f}", lltcmp=f"{t_cmp:.3f}", llratio=f"{d.bound_ratio:.0f}",
             ridge=f"{ridge:.0f}", llint=f"{intensity:.0f}",
             fpp=f"{flop_g / w_gb * BYTES_FP16.to('byte').magnitude:.0f}", bpp=f"{BYTES_FP16.to('byte').magnitude:.0f}")
    # printed result reproduces from printed inputs
    assert abs(w_gb / bw - t_mem) / t_mem < 0.01 and abs(flop_g / peak - t_cmp) / t_cmp < 0.01

    # ---- Your turn: GPT-2 XL batch-one FP16 decode on one H100 (chapter: conclusion-voice-request-trace)
    gx = SC.Gpt2XlFp16DecodeOnH100
    r = gx.result
    g_gb = r.weight_bytes.to("GB").magnitude
    g_gflop = r.flops_per_step.to("GFLOP").magnitude
    g_bw = gx.hardware.memory.bandwidth.to("TB/s").magnitude
    g_peak = gx.hardware.compute.peak_flops.to("TFLOP/s").magnitude
    n.update(gxparams=f"{gx.workload.parameters.to('param').magnitude / 1e9:.1f}", gxgb=f"{g_gb:.0f}",
             gxgflop=f"{g_gflop:.0f}", gxbw=f"{g_bw:.2f}", gxpeak=c(g_peak),
             gxtmem=f"{r.memory_time.to('ms').magnitude:.3f}", gxtcmp=f"{r.compute_time.to('ms').magnitude:.5f}",
             gxratio=f"{r.bound_ratio:.0f}", gxridge=f"{g_peak / g_bw:.0f}",
             scale=f"{(ll.workload.parameters / gx.workload.parameters).to('').magnitude:.0f}")

    # ---- Energy-movement ratio (chapter: conclusion-energy-movement-math)
    dram = Hardware.Tech.Memory.DRAM.energy_per_access.to("pJ").magnitude
    fp32 = Hardware.Tech.Op.FlopFp32.energy.to("pJ/flop").magnitude
    fp16 = Hardware.Tech.Op.FlopFp16.energy.to("pJ/flop").magnitude
    n.update(elow=f"{dram / fp32:.0f}", ehigh=f"{dram / fp16:.0f}")

    # ---- Mistake: Amdahl pitfall (chapter: conclusion-amdahl-pitfall)
    am = SC.HostPreprocessingRequestPath
    frac = am.result.stage_fraction
    n.update(amlocal=f"{am.speedup:.0f}", amfrac=f"{frac * 100:.0f}", amserial=f"{(1 - frac) * 100:.0f}",
             amsys=f"{am.result.overall_speedup:.1f}")

    # ---- Mistake: tail latency (chapter: conclusion-tail-latency-ratio)
    tl = SC.ChatEndpointTtftMeanVersusP99
    n.update(tlmean=c(tl.mean.to("ms").magnitude), tlp99=c(tl.tail_high.to("ms").magnitude), tlx=f"{tl.gap_high:.0f}", tlp99raw=f"{tl.tail_high.to('ms').magnitude:.0f}")

    # ---- Mistake: one accuracy metric (chapter prose, Fallacies; scenario values stated in the chapter)
    n.update(fleetdrop="4", slo="100", accA="95", latA="500", accB="93", latB="50")

    # ---- Q1: DLRM tables against one H100 (chapter: conclusion-voice-request-trace)
    h100 = Hardware.Cloud.H100
    n.update(dlrmgb=f"{Models.Recommendation.DLRM.model_size.to('GB').magnitude:.0f}",
             gpugib=f"{h100.memory.capacity.to('GiB').magnitude:.0f}",
             gpugb=f"{h100.memory.capacity.to('GB').magnitude:.0f}")

    # ---- Fleet note (chapter: conclusion-fleet-failure-math) and Q4 (ASSUMED larger pool, run length)
    fl = SC.GpuFailuresTraining1K
    mttf_h = fl.component_mtbf.to("hour").magnitude
    n_gpu = fl.n_components
    cl_h = fl.result.interval.to("hour").magnitude
    assert abs(mttf_h / n_gpu - cl_h) < 1e-6
    n.update(mttfh=c(mttf_h), mttfy=f"{mttf_h / (HOURS_PER_DAY * DAYS_PER_YEAR):.1f}", fleetn=c(n_gpu),
             fleeth=f"{cl_h:.1f}")
    q4n, q4days = 4096, 14                                               # ASSUMED
    q4h = mttf_h / q4n
    n.update(q4n=c(q4n), q4days=c(q4days), q4h=f"{q4h:.1f}", q4runh=c(q4days * 24),
             q4fail=f"{q4days * 24 / q4h:.0f}")

    # ---- Q2: composed request critical path (ASSUMED stage times, ms)
    retr, plan, tool_a, tool_b, gen, ver, orch, deadline = 40, 120, 150, 90, 400, 60, 30, 1000
    path = retr + plan + max(tool_a, tool_b) + gen + ver + orch
    n.update(q2retr=c(retr), q2plan=c(plan), q2ta=c(tool_a), q2tb=c(tool_b), q2gen=c(gen), q2ver=c(ver),
             q2orch=c(orch), q2dead=c(deadline), q2path=c(path), q2slack=c(deadline - path),
             q2seq=c(retr + plan + tool_a + tool_b + gen + ver + orch))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
