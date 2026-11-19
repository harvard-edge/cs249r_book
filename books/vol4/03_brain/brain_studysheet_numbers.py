"""Numbers for The Cognitive Brain study sheet (Volume IV, Chapter 3).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(IntentModelLatency, IntentQuantization, ActionChunkingLPDDR5, ChunkSchedule, ConveyorChunkSizing),
and written as \\setnum{key}{value} to brain_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Embodied, Scenarios as SC


def fx(x, nd):
    """Round half up, as a reader would, and format with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP))


def r(x, nd):
    return float(fx(x, nd))


def pct(x, nd=0):
    return fx(x * 100, nd)


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def mag(q, unit):
    return q.to(unit).magnitude


def numbers() -> dict:
    n = {}
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    soc = rm.brain_soc

    # ---- The running machine's processor (registry): peak and sustained bandwidth, FP16 peak, memory
    bw_peak = mag(soc.memory.bandwidth, "GB/s")
    eta_bus = rm.dram_efficiency
    bw = r(bw_peak * eta_bus, 1)                                     # printed 142.8 GB/s
    peak = mag(soc.compute.precision_flops["fp16"], "TFLOP/s")
    mem_gib = mag(soc.memory.capacity, "GiB")
    mem_gb = mag(soc.memory.capacity, "GB")                          # GiB rule: print 64 GiB (68.7 GB)
    ridge = peak * 1e12 / (bw_peak * 1e9)
    n.update(bwpeak=fx(bw_peak, 0), buspct=pct(eta_bus), bw=fx(bw, 1), peak=fx(peak, 0),
             ridge=fx(ridge, 0), memgib=fx(mem_gib, 0), memgb=fx(mem_gb, 1))

    # ---- Worked example: one fresh intent-model evaluation (chapter: IntentModelLatency, nbk-brain-intent-model-latency)
    fp16, int4 = SC.IntentModelEvaluationFp16.result, SC.IntentModelEvaluationInt4.result
    model = rm.intent_model
    params = model.parameters.magnitude
    s_in = rm.intent_input_tokens
    d_out = model.action_dim
    steps = d_out - 1
    assert steps == fp16.decode_steps
    util = rm.intent_compute_utilization
    w_fp16 = params * 2 / 1e9
    w_int4 = params * 0.5 / 1e9
    near(w_fp16, mag(fp16.weight_memory, "GB"), 1e-9)
    near(w_int4, mag(rm.intent_weight_memory, "GB"), 1e-9)
    flops = mag(fp16.prefill_flops, "flop")
    t_pre = r(r(flops / 1e12, 3) / (util * peak) * 1e3, 0)        # printed 189 ms, from the printed 4.054e12 FLOPs
    near(r(flops / 1e12, 3) / (util * peak) * 1e3, mag(fp16.t_prefill, "ms"), 0.5)
    sweep16 = r(w_fp16 / bw * 1e3, 1)                                # 98.0 ms
    sweep4 = r(w_int4 / bw * 1e3, 1)                                 # 24.5 ms
    near(sweep16, mag(fp16.t_weight_sweep, "ms"), 0.05)
    near(sweep4, mag(int4.t_weight_sweep, "ms"), 0.05)
    dec16, dec4 = r(steps * sweep16, 0), r(steps * sweep4, 0)
    tot16, tot4 = t_pre + dec16, t_pre + dec4
    near(tot16, mag(fp16.t_total, "ms"), 0.5)
    near(tot4, mag(int4.t_total, "ms"), 0.5)
    wall16 = 1000 / sweep16
    near(wall16, mag(fp16.weight_sweep_rate, "Hz"), 0.05)
    rate = mag(rm.intent_rate, "Hz")
    period = 1000 / rate
    assert 1000 / (rate + 1) < tot4 <= period                       # 2 Hz is the largest whole-hertz rate
    n.update(params=f"{params / 1e9:.0f}", tokensin=f"{s_in}", tokensout=f"{d_out}", decodesteps=f"{steps}",
             util=fx(util, 1), flops=fx(flops / 1e12, 3), wfp=fx(w_fp16, 0), wint=fx(w_int4, 1),
             tpre=fx(t_pre, 0), sweepfp=fx(sweep16, 1), sweepint=fx(sweep4, 1), wallfp=fx(wall16, 1),
             decfp=fx(dec16, 0), decint=fx(dec4, 0), totfp=fx(tot16, 0), totint=fx(tot4, 0),
             fmaxint=fx(1000 / tot4, 2), rate=fx(rate, 0), period=fx(period, 0),
             speedup=fx(tot16 / tot4, 1), lesstime=pct(1 - tot4 / tot16), preshare=pct(t_pre / tot4),
             ceiling=fx(1000 / t_pre, 1))

    # ---- Your turn: 8-bit weights (chapter: IntentQuantization, the 8-bit row of tbl-03-intent-quantization)
    q8 = SC.OpenVlaOrinQuantization.formats["8-bit"].result
    w8 = params * 1 / 1e9
    near(w8, mag(q8.weight_memory, "GB"), 1e-9)
    sweep8 = r(w8 / bw * 1e3, 1)                                     # 49.0 ms
    dec8 = r(steps * sweep8, 0)
    tot8 = t_pre + dec8
    near(tot8, mag(q8.t_total, "ms"), 0.5)
    assert tot8 < period
    n.update(weight=fx(w8, 0), sweepeight=fx(sweep8, 1), deceight=fx(dec8, 0), toteight=fx(tot8, 0),
             slackeight=fx(period - tot8, 0), fmaxeight=fx(1000 / tot8, 2))

    # ---- Front strip and mistakes: chunk supply against the evidence horizon (chapter: ConveyorChunkSizing, ChunkSchedule)
    ev = SC.ConveyorMugEvidenceHorizon.result
    aisle = Embodied.Scenario.WarehouseAisle
    tol, e0 = mag(aisle.grasp_tolerance, "mm"), mag(aisle.grasp_initial_error, "mm")
    slip = mag(aisle.a_conveyor_slip, "m/s**2")
    vconv = mag(aisle.v_conveyor, "m/s")
    step = mag(rm.chunk_step, "ms")
    horizon = rm.chunk_horizon
    renew = mag(ev.renewal_period, "ms")
    infer = mag(rm.chunk_inference_latency, "ms")
    tau = math.sqrt(2 * (tol - e0) / 1000 / slip) * 1000
    near(tau, mag(ev.horizon, "ms"), 1e-6)
    age = renew + infer
    near(age, mag(ev.age_at_replacement, "ms"), 1e-9)
    hmin = math.ceil(age / step)
    assert hmin == ev.min_supply_setpoints
    span = horizon * step
    near(span, mag(ev.chunk_span, "ms"), 1e-9)
    drift_end = 0.5 * slip * (span / 1000) ** 2 * 1000
    near(drift_end, mag(ev.drift_at_chunk_end, "mm"), 1e-9)
    near((tol - e0) / vconv, mag(ev.untracked_deadline, "ms"), 1e-9)
    n.update(tol=fx(tol, 0), errzero=fx(e0, 0), margin=fx(tol - e0, 0), slip=fx(slip, 1), conv=fx(vconv, 2),
             untracked=fx((tol - e0) / vconv, 0),
             step=fx(step, 0), horizon=f"{horizon}", renew=fx(renew, 0), chunkrate=fx(mag(rm.chunk_rate, "Hz"), 0),
             infer=fx(infer, 0), age=fx(age, 0), hmin=f"{hmin}", tau=fx(tau, 0), span=fx(span, 0),
             driftend=fx(drift_end, 0), errend=fx(e0 + drift_end, 0),
             chunkparams=fx(rm.chunk_model.parameters.magnitude / 1e6, 0))

    # ---- Mistake: chunking does not let the intent model drive the arm (chapter: ActionChunkingLPDDR5)
    traffic = SC.WarehouseProposerTraffic.result
    amort = sweep16 / horizon
    near(amort, mag(traffic.amortized_sweep, "ms"), 0.01)
    gap = tot4 / renew
    near(gap, traffic.renewal_gap, 0.02)
    n.update(amort=fx(amort, 1), gap=fx(gap, 1))

    # ---- Q1. ASSUMED: residual slip rises to 1.5 m/s^2; tolerance, error, step and inference as the chapter
    q_slip = 1.5
    q_tau = math.sqrt(2 * (tol - e0) / 1000 / q_slip) * 1000
    q_renew_slow = 2 * renew                                         # renewal halved to 10 Hz
    q_age_slow = q_renew_slow + infer
    assert age < q_tau < q_age_slow
    n.update(qslip=fx(q_slip, 1), qtau=fx(q_tau, 0), qrateslow=fx(1000 / q_renew_slow, 0),
             qrenewslow=fx(q_renew_slow, 0), qageslow=fx(q_age_slow, 0), qhminslow=f"{math.ceil(q_age_slow / step)}",
             qtauinner=fx(2 * (tol - e0) / 1000 / q_slip, 3))

    # ---- Q2. ASSUMED: a retained history of 3 s at 10 Hz on the chapter's three-view step (KV per token from eq-brain-kv-cache)
    hist = SC.ThreeViewKvHistory
    layers, width = model.layers, model.hidden_dim
    per_token = 2 * layers * width * 2                               # bytes, FP16 keys and values
    near(per_token, mag(hist.tokens.kv_per_token, "B"), 1e-9)
    ctx = hist.tokens.context_tokens
    per_step = r(per_token * ctx / 1e6, 0)                           # MB per observation step
    near(per_token * ctx / 1e6, mag(hist.kv_per_step, "MB"), 1e-6)
    q_window, q_hz = 3, 10
    q_steps = q_window * q_hz
    q_cache = r(q_steps * per_step / 1000, 1)                        # GB
    q_total = q_cache + w_int4
    q_share = q_total / r(mem_gb, 1)
    q_reread = q_cache / bw * 1000
    near(q_cache, q_steps * per_token * ctx / 1e9, 0.05)
    assert q_reread > renew
    n.update(layers=f"{layers}", width=f"{width}", ctx=f"{ctx}", pertoken=f"{per_token:,}",
             perstep=fx(per_step, 0), qwindow=f"{q_window}", qhz=f"{q_hz}", qsteps=f"{q_steps}",
             qcache=fx(q_cache, 1), qtotal=fx(q_total, 1), qshare=pct(q_share), qreread=fx(q_reread, 0))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
