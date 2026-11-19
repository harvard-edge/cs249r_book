"""Numbers for the Network Architectures study sheet (Volume I, Chapter 6).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to nn_architectures_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
"""
import math
from pathlib import Path

from mlsysim import Hardware, Models
from mlsysim.core.units import BYTES_FP32
from mlsysim.engine.scenarios import Scenarios


def g(x, nd=3):
    return f"{x:.{nd}g}"


def c(x):
    return f"{x:,.0f}"


def numbers() -> dict:
    n = {}
    resnet, gpt2 = Models.Vision.ResNet50, Models.Language.GPT2
    mobilenet, kws = Models.Vision.MobileNetV2, Models.Tiny.DS_CNN

    # Weight-only FP32 intensity proxy per inference (chapter: WorkloadSignatures)
    def intensity(m):
        return m.inference_flops.to("flop").magnitude / m.size_in_bytes(BYTES_FP32).to("byte").magnitude
    i_r, i_g, i_m, i_k = (intensity(m) for m in (resnet, gpt2, mobilenet, kws))
    n.update(iresnet=f"{i_r:.0f}", igpt=f"{i_g:.1f}", imobile=f"{i_m:.0f}", ikws=f"{i_k:.0f}",
             igap=f"{i_r / i_g:.0f}")
    n.update(resnetgflop=f"{resnet.inference_flops.to('GFLOPs').magnitude:.1f}",
             mobilemflop=f"{mobilenet.inference_flops.to('MFLOPs').magnitude:.0f}",
             flopratio=f"{(resnet.inference_flops / mobilenet.inference_flops).to('').magnitude:.1f}")

    # Weight sharing: 3x3 conv, 64 -> 64, vs dense from a 224x224x64 map to 64 outputs (chapter: CNNDefinitionSharing)
    conv = Scenarios.ImageNet64ChannelConv
    side = int(math.isqrt(int(conv.positions)))
    dense_w = side * side * int(conv.in_channels) * int(conv.out_channels)
    n.update(convw=c(conv.weights), densew=f"{dense_w / 1e6:.0f}M", shareratio=c(dense_w / conv.weights))

    # Attention crossover S = 6d for GPT-2 XL (chapter: AttentionCrossover)
    blk = Scenarios.Gpt2XlBlockFullWindow
    n.update(gptwidth=c(blk.width), crossover=c(blk.crossover), crossk=f"{blk.crossover / 1000:.1f}")

    # ---- Worked example: ASSUMED 4096 x 4096 dense layer, FP16 weights, batch one, on an A100
    a100 = Hardware.Cloud.A100
    peak = a100.compute.precision_flops["fp16_dense"].to("flop/s").magnitude
    bw = a100.memory.bandwidth.to("byte/s").magnitude
    width, bytes_w = 4096, 2
    w = width * width
    wbytes = w * bytes_w
    flops1 = 2 * w
    i1 = flops1 / wbytes
    bal = peak / bw
    t_mem, t_cmp = wbytes / bw, flops1 / peak
    n.update(wxw=c(width), wcount=f"{w / 1e6:.1f}M", wmb=f"{wbytes / 1e6:.1f}", wflop=f"{flops1 / 1e6:.1f}M",
             wi=g(i1), apeak=f"{peak / 1e12:.0f}", abw=c(bw / 1e9), abal=f"{bal:.0f}",
             tmem=f"{t_mem * 1e6:.1f}", tcmp=f"{t_cmp * 1e6:.2f}", tratio=f"{t_mem / t_cmp:.0f}",
             wbatch=f"{math.ceil(bal / i1):.0f}")

    # ---- Your turn: ASSUMED 4096 -> 16384 feed-forward layer, 16-bit weights, batch 64, same accelerator
    din, dout, batch = 4096, 16384, 64
    yw = din * dout
    ybytes = yw * bytes_w
    yflop = 2 * yw * batch
    yi = yflop / ybytes
    n.update(ydin=c(din), ydout=c(dout), ybatch=c(batch), yw=f"{yw / 1e6:.1f}M", ymb=f"{ybytes / 1e6:.1f}",
             yflop=f"{yflop / 1e9:.2f}", yi=f"{yi:.0f}", ytmem=f"{ybytes / bw * 1e6:.1f}",
             ytcmp=f"{yflop / peak * 1e6:.1f}", ybal=f"{bal:.0f}")

    # ---- Q3 uses the chapter's KV-cache scenario: Llama 2 7B, 16-bit (Scenarios.Llama2_7BKvAt2048Tokens)
    kv = Scenarios.Llama2_7BKvAt2048Tokens
    per_tok = 2 * kv.workload.layers * kv.workload.heads * kv.workload.head_width * kv.bytes_per_value.to("byte").magnitude
    weights = kv.weights.to("GB").magnitude
    n.update(kvlayers=c(kv.workload.layers), kvheads=c(kv.workload.heads), kvhd=c(kv.workload.head_width),
             kvwidth=c(kv.workload.heads * kv.workload.head_width), kvpertokbytes=c(per_tok), kvweights=f"{weights:.0f}")

    # ---- Check yourself (ASSUMED inputs)
    # Q1: ASSUMED camera board, 8-bit, sustains 5 GOPS, 250 ms budget, 256 MB RAM
    gops, budget_ms, ram_mb = 5, 250, 256
    rg = resnet.inference_flops.to("GFLOPs").magnitude
    mg = mobilenet.inference_flops.to("GFLOPs").magnitude
    n.update(q1gops=c(gops), q1ms=c(budget_ms), q1ram=c(ram_mb), q1rg=f"{rg:.1f}", q1mg=f"{mg:.1f}",
             q1rmb=f"{resnet.parameters.to('param').magnitude / 1e6:.1f}", q1mmb=f"{mobilenet.parameters.to('param').magnitude / 1e6:.1f}",
             q1rt=f"{rg / gops * 1e3:,.0f}", q1mt=f"{mg / gops * 1e3:.0f}")
    # Q2: 3x3 conv, 32 -> 64 channels, 112 x 112 output, FP16
    k, cin, cout, hw, b16 = 3, 32, 64, 112, 2
    q2w = k * k * cin * cout
    q2mac = hw * hw * q2w
    q2act = hw * hw * cout
    n.update(q2k=c(k), q2cin=c(cin), q2cout=c(cout), q2hw=c(hw), q2w=c(q2w), q2wkb=f"{q2w * b16 / 1e3:.1f}",
             q2mac=f"{q2mac / 1e6:.0f}M", q2act=c(q2act), q2actmb=f"{q2act * b16 / 1e6:.1f}",
             q2ratio=f"{q2act / q2w:.0f}")
    # Front picture: what doubling the context does (weights fixed, KV cache linear, scores quadratic)
    grow = 2
    n.update(dgrow=c(grow), dw=c(1), dkv=c(grow), dsc=c(grow ** 2))
    # Q3: context grows 4K -> 16K tokens
    s0, s1 = 4096, 16384
    n.update(q3s0=f"{s0 // 1024}K", q3s1=f"{s1 // 1024}K", q3score=f"{(s1 / s0) ** 2:.0f}", q3kv=f"{s1 / s0:.0f}",
             q3tok=c(s1), q3gb=f"{per_tok * s1 / 1e9:.1f}", q3users=f"{weights / (per_tok * s1 / 1e9):.1f}")
    # Q4: embedding table 500M rows x 64 values, FP32, against one 80 GiB accelerator
    rows, dim = 500_000_000, 64
    t32 = rows * dim * 4
    t8 = rows * dim * 1
    cap = a100.memory.capacity.to("byte").magnitude
    n.update(q4bytes=c(4), q4rows=f"{rows / 1e6:.0f}M", q4dim=c(dim), q4gb=f"{t32 / 1e9:.0f}", q4cap=f"{a100.memory.capacity.to('GiB').magnitude:.0f}",
             q4capgb=f"{cap / 1e9:.1f}", q4int8=f"{t8 / 1e9:.0f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
