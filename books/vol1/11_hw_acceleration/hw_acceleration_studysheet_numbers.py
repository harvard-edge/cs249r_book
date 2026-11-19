"""Numbers for the Hardware Acceleration study sheet (Volume I, Chapter 11).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to hw_acceleration_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
"""
from pathlib import Path

from mlsysim import Hardware, Models
from mlsysim.core.units import BYTES_FP16, BYTES_INT8
from mlsysim.engine.scenarios import Scenarios


def c(x):
    return f"{x:,.0f}"


def numbers() -> dict:
    n = {}
    h100, a100, t4 = Hardware.Cloud.H100, Hardware.Cloud.A100, Hardware.Cloud.T4

    # ---- H100 paths (chapter: Gpt2ThroughputCalc, ResNet50Int8Crossing)
    peak16 = h100.compute.precision_flops["fp16_dense"].to("flop/s").magnitude
    peak8 = h100.compute.precision_flops["int8"].to("flop/s").magnitude
    bw = h100.memory.bandwidth.to("byte/s").magnitude
    ridge16, ridge8 = peak16 / bw, peak8 / bw
    n.update(hpeaksixteen=c(peak16 / 1e12), hpeakeight=c(peak8 / 1e12), hbw=f"{bw / 1e12:.2f}",
             ridgesixteen=f"{ridge16:.0f}", ridgeeight=f"{ridge8:.0f}")

    # ---- Energy ladder (chapter: EnergyGapAnchor, fig-energy-hierarchy; Horowitz 45 nm)
    T = Hardware.Tech
    e_mul32 = T.Op.FlopFp32.energy.to("pJ/flop").magnitude
    e_sram = T.Memory.SRAM_8KB.energy_per_access.to("pJ").magnitude
    e_dram = T.Memory.DRAM.energy_per_access.to("pJ").magnitude
    n.update(emulfp=f"{e_mul32:g}", esram=f"{e_sram:g}", edram=f"{e_dram:g}",
             dramovermul=f"{e_dram / e_mul32:.0f}", dramoversram=f"{e_dram / e_sram:.0f}")

    # ---- Worked example: GPT-2 XL decoding one token at batch 1, FP16, on the H100
    gpt2 = Models.Language.GPT2
    params = gpt2.parameters.to("param").magnitude
    wbytes = gpt2.size_in_bytes(BYTES_FP16).to("byte").magnitude
    ops = 2 * params                      # two operations per parameter per token
    inten = ops / wbytes
    attain = min(peak16, bw * inten)
    floor_s = wbytes / bw
    roof = Scenarios.Gpt2XlDecodeH100Roofline
    assert abs(roof.point_intensity.to("flop/byte").magnitude - inten) < 1e-9
    n.update(gparams=f"{params / 1e9:.1f}", gweightgb=f"{wbytes / 1e9:.1f}", gops=f"{ops / 1e9:.1f}",
             gintensity=f"{inten:.1f}", gattain=f"{attain / 1e12:.2f}", gshare=f"{attain / peak16 * 100:.2f}",
             gfloorms=f"{floor_s * 1e3:.2f}", gtokens=c(round(1 / floor_s, -2)))

    # ---- Your turn: same decode, INT8 weights on the INT8 path
    wbytes8 = gpt2.size_in_bytes(BYTES_INT8).to("byte").magnitude
    inten8 = ops / wbytes8
    floor8 = wbytes8 / bw
    n.update(yweightgb=f"{wbytes8 / 1e9:.1f}", yintensity=f"{inten8:.1f}",
             yfloorms=f"{floor8 * 1e3:.2f}", yspeedup=f"{floor_s / floor8:.1f}", ycut=f"{(1 - floor8 / floor_s) * 100:.0f}")

    # ---- Mistake 1: stage-3 3x3 conv, batch 32 (chapter: ResNet50Int8Crossing)
    s3 = Scenarios.ResNet50Stage3ConvBatch32
    s3w = Scenarios.ResNet50Stage3ConvBatch32Int8Weights
    s3all = Scenarios.ResNet50Stage3ConvBatch32Int8
    act_mb = (s3.batch_input_bytes + s3.batch_output_bytes).to("MB").magnitude
    n.update(convsixteen=f"{s3.traffic_intensity:.0f}", convmixed=f"{s3w.traffic_intensity:.0f}",
             convall=c(s3all.traffic_intensity), convact=f"{act_mb:.1f}",
             convweight=f"{s3.weight_bytes.to('MB').magnitude:.1f}")

    # ---- Mistake 2: batch-one 2048 x 2048 dense layer, A100 vs T4 (chapter: FpSmallBatchCalc)
    bw_a, bw_t = a100.memory.bandwidth.to("GB/s").magnitude, t4.memory.bandwidth.to("GB/s").magnitude
    pk_a, pk_t = a100.compute.peak_flops.to("TFLOP/s").magnitude, t4.compute.peak_flops.to("TFLOP/s").magnitude
    cost_a, cost_t = a100.unit_cost.to("USD").magnitude, t4.unit_cost.to("USD").magnitude
    n.update(abw=c(bw_a), tbw=c(bw_t), apeak=c(pk_a), tpeak=c(pk_t), bwratio=f"{bw_a / bw_t:.1f}",
             peakratio=f"{pk_a / pk_t:.1f}", costratio=f"{cost_a / cost_t:.1f}")

    # ---- Q1: ASSUMED 80% of baseline time accelerated, 40x path gain
    p, gain = 0.8, 40
    n.update(qonefrac=c(p * 100), qoneserialfrac=f"{1 - p:.1f}", qonegain=c(gain), qoneserial=c((1 - p) * 100), qonepg=f"{p / gain:.2f}",
             qonespeed=f"{1 / ((1 - p) + p / gain):.1f}", qoneceiling=f"{1 / (1 - p):.0f}")

    # ---- Q2: the chapter's DRAM-streaming MAC model (Scenarios.SystolicArrayReuseEnergy: 4 accesses per MAC,
    # 2 edge loads per array width, 1 pJ per MAC) against an ASSUMED 64-wide systolic array
    reuse = Scenarios.SystolicArrayReuseEnergy
    width = 64
    e_access = reuse.access_energy.to("pJ").magnitude
    e_mac = reuse.mac_energy.to("pJ").magnitude
    naive_acc, edge = reuse.naive_accesses, reuse.edge_loads
    e_naive = naive_acc * e_access + e_mac
    e_sys = edge / width * e_access + e_mac
    n.update(qtwowidth=c(width), qtwonaiveacc=c(naive_acc), qtwosysacc=c(edge), qtwomac=f"{e_mac:g}",
             qtwonaive=c(e_naive), qtwodram=f"{edge / width * e_access:.0f}", qtwosys=f"{e_sys:.0f}",
             qtworatio=f"{e_naive / e_sys:.0f}")

    # ---- Q3: three kernels on the H100 FP16 roofline (chapter scenarios)
    ln = Scenarios.Gpt2XlLayerNormH100Roofline
    head = Scenarios.ResNet50ClassifierHeadH100Roofline
    i_ln = ln.point_intensity.to("flop/byte").magnitude
    i_head = head.point_intensity.to("flop/byte").magnitude
    i_conv = s3.traffic_intensity
    n.update(qthreeln=f"{i_ln:.1f}", qthreehead=f"{i_head:.1f}", qthreeconv=f"{i_conv:.0f}",
             qthreelnrate=f"{bw * i_ln / 1e12:.1f}", qthreeheadrate=c(bw * i_head / 1e12),
             qthreeconvrate=c(min(peak16, bw * i_conv) / 1e12),
             qthreebatch=c(Scenarios.ResNet50ClassifierHeadBatch32.batch_size))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
