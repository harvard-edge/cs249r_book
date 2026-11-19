"""Numbers for the Model Compression study sheet (Volume I, Chapter 10).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to model_compression_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
"""
from pathlib import Path

from mlsysim.core.units import BYTES_FP16, BYTES_INT4, BYTES_INT8, BYTES_INT16
from mlsysim.engine.scenarios import Scenarios


def c(x):
    return f"{x:,.0f}"


def numbers() -> dict:
    n = {}

    # ---- Phone decode of GPT-2 XL (chapter: GPT2DecodeQuantization)
    fp16 = Scenarios.Gpt2XlFp16DecodeFullContextPhone
    int4 = Scenarios.Gpt2XlInt4DecodeFullContextPhone
    model = fp16.workload
    params = model.parameters.to("param").magnitude
    bw = fp16.hardware.memory.bandwidth.to("GB/s").magnitude
    kv = fp16.kv_cache_bytes.to("GB").magnitude
    w16 = fp16.weight_bytes.to("GB").magnitude
    w4 = int4.weight_bytes.to("GB").magnitude
    t16 = fp16.token_time.to("ms").magnitude
    t4 = int4.token_time.to("ms").magnitude
    b16 = BYTES_FP16.to("byte").magnitude
    b4 = BYTES_INT4.to("byte").magnitude
    b8 = BYTES_INT8.to("byte").magnitude
    # check against the scenario's own byte counts
    assert abs(params * b16 / 1e9 - w16) < 1e-9 and abs(params * b4 / 1e9 - w4) < 1e-9
    n.update(gptparams=f"{params / 1e9:.1f}", phonebw=f"{bw:.1f}", context=c(fp16.context_tokens),
             layers=c(model.layers), width=c(model.hidden_dim),
             kvmb=c(kv * 1e3), weightsixteen=c(w16 * 1e3), weightfour=c(w4 * 1e3),
             totalsixteen=c((w16 + kv) * 1e3), totalfour=c((w4 + kv) * 1e3),
             timesixteen=f"{t16:.1f}", timefour=f"{t4:.1f}", ratesixteen=f"{1000 / t16:.0f}",
             ratefour=f"{1000 / t4:.0f}", speedfour=f"{t16 / t4:.1f}",
             weightcut=f"{w16 / w4:.0f}",
             weightreadsixteen=f"{w16 / bw * 1e3:.1f}", kvread=f"{kv / bw * 1e3:.1f}",
             weightreadfour=f"{w4 / bw * 1e3:.1f}",
             # comma-free copies for TikZ lengths (MB)
             drawsixteen=f"{w16 * 1e3:.0f}", drawfour=f"{w4 * 1e3:.0f}",
             drawtotalsixteen=f"{(w16 + kv) * 1e3:.0f}", drawtotalfour=f"{(w4 + kv) * 1e3:.0f}")
    # KV bytes as the chapter computes them: keys and values, every layer, full width, 16-bit, every token
    kv_check = 2 * model.layers * model.hidden_dim * b16 * fp16.context_tokens / 1e9
    assert abs(kv_check - kv) < 1e-6

    # ---- Your turn: same decode with INT8 weights (ASSUMED variant; same phone, same FP16 KV cache)
    w8 = params * b8 / 1e9
    t8 = (w8 + kv) / bw * 1e3
    n.update(weighteight=c(w8 * 1e3), totaleight=c((w8 + kv) * 1e3), timeeight=f"{t8:.1f}",
             rateeight=f"{1000 / t8:.0f}", speedeight=f"{t16 / t8:.1f}")

    # ---- Front picture: the camera frame vs its weight reads (chapter: ModelDeviceComparison, CameraRunningCase)
    mread = Scenarios.MobileNetV2Fp32WeightReadPhone
    cam = Scenarios.MobileNetV2BlurPipelinePhone
    budget = cam.frame_budget.to("ms").magnitude
    mob_read = mread.read_time.to("ms").magnitude
    n.update(framerate=c(cam.frame_rate.to("1/second").magnitude), framebudget=f"{budget:.0f}",
             mobilemb=f"{mread.weight_bytes.to('MB').magnitude:.0f}", mobileread=f"{mob_read:.2f}",
             mobileshare=f"{mob_read / budget * 100:.0f}",
             mobilemflop=f"{mread.workload.inference_flops.to('MFLOPs').magnitude:.0f}",
             cpuframe=f"{cam.cpu_frame.to('ms').magnitude:.0f}",
             npuframe=f"{cam.accelerator_frame.to('ms').magnitude:.0f}")

    # ---- Mistake 2: paper vs measured speedup (chapter: TheoryPracticeSpeedup, an illustrative scenario)
    paper = Scenarios.PrunedInt8PaperVsMeasured
    n.update(prunex=f"{paper.factors['pruning']:.0f}", quantx=f"{paper.factors['quantization']:.0f}",
             paperx=f"{paper.paper:.0f}", measuredx=f"{paper.measured:.1f}")

    # ---- Key idea 3: break-even for 16-bit values with 16-bit indices (chapter: MobileNetCompressionAnchor)
    i16 = BYTES_INT16.to("byte").magnitude
    n.update(breakevensixteen=f"{i16 / (b16 + i16) * 100:.0f}")

    # ---- Q1: affine UINT8 quantization, ASSUMED range [-2, 6], value 1.0
    lo, hi, bits, x = -2.0, 6.0, 8, 1.0
    steps = 2 ** bits - 1
    s = (hi - lo) / steps
    z = round(-lo / s)
    q = min(max(round(x / s + z), 0), steps)
    back = s * (q - z)
    n.update(qlo=f"{lo:.0f}", qhi=f"{hi:.0f}", qx=f"{x:.1f}", qsteps=c(steps), qrange=f"{hi - lo:.0f}",
             qscale=f"{s:.5f}", qzexact=f"{-lo / round(s, 5):.2f}", qzero=c(z), qdiv=f"{x / s:.2f}", qcode=c(q),
             qback=f"{back:.3f}", qerr=f"{abs(back - x):.3f}", qhalf=f"{s / 2:.3f}")

    # ---- Q3: ASSUMED INT8 values with 16-bit column indices, 60% unstructured sparsity
    sparsity = 0.60
    per_kept = b8 + i16
    stored = (1 - sparsity) * per_kept / b8
    n.update(sparsepct=f"{sparsity * 100:.0f}", valuebytes=c(b8), indexbytes=c(i16), keptbytes=c(per_kept),
             keptshare=f"{1 - sparsity:.1f}", storedratio=f"{stored:.1f}",
             breakeveneight=f"{i16 / per_kept * 100:.0f}")

    # ---- Q4: model is a fifth of request latency (chapter: AmdahlCompression); ASSUMED model made 3x faster
    frac = Scenarios.ModelFifthOfRequestLatency.result.stage_fraction
    g = 3
    overall = 1 / ((1 - frac) + frac / g)
    n.update(modelpct=f"{frac * 100:.0f}", modelfrac=f"{frac:.1f}", restfrac=f"{1 - frac:.1f}", modelgain=c(g),
             overall=f"{overall:.2f}", unity=c(1), ceiling=f"{1 / (1 - frac):.2f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
