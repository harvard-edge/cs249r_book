"""Numbers for the Model Serving study sheet (Volume I, Chapter 13).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to model_serving_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Where a question gives rounded inputs, its answers are computed from those rounded inputs,
so a student who works from the printed values gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Hardware, Models
from mlsysim.core.units import BYTES_FP16
from mlsysim.engine.scenarios import Scenarios
from mlsysim.engine.serving_profile import resnet50_h100_profile


def c(x):
    return f"{x:,.0f}"


def ms(q):
    return q.to("ms").magnitude


def numbers() -> dict:
    n = {}
    prof = resnet50_h100_profile()

    # ---- Front strip: the reference ResNet-50 request on one H100 (LatencyTableCalc, AmdahlServingCalc)
    path = Scenarios.ResNet50RequestPathH100
    st = {k: ms(v) for k, v in path.stages.items()}
    total, fast_total = ms(path.total), ms(path.accelerated_total)
    n.update(jpeg=f"{st['JPEG decode']:.1f}", resize=f"{st['resize']:.1f}", normalize=f"{st['normalize']:.1f}",
             copy=f"{st['copy to GPU']:.1f}", forward=f"{st['forward pass']:.1f}", topfive=f"{st['top-5']:.1f}",
             total=f"{total:.1f}", speedup=f"{path.speedup:.0f}", forwardfast=f"{ms(path.accelerated_stage):.2f}",
             totalfast=f"{fast_total:.1f}", amdahl=f"{path.overall_speedup:.1f}",
             hostshare=f"{path.share('JPEG decode', 'resize', 'normalize', 'copy to GPU') * 100:.0f}")
    assert abs(round(total, 1) / round(fast_total, 1) - path.overall_speedup) < 0.05
    # strip geometry (exact values, drawn to scale)
    n.update(sjpeg=f"{st['JPEG decode']:.4f}", sresize=f"{st['resize']:.4f}", snorm=f"{st['normalize']:.4f}",
             scopy=f"{st['copy to GPU']:.4f}", sfwd=f"{st['forward pass']:.4f}", stop=f"{st['top-5']:.4f}",
             sfwdfast=f"{ms(path.accelerated_stage):.4f}")

    # ---- Queueing card: M/M/1 (eq-mm1-wait, eq-p99-latency) and the chapter's fleet plan
    plan = Scenarios.ResNet50FleetPlanH100
    factor = plan.tail_factor
    n.update(tailfactor=f"{factor:.1f}", lnhundred=f"{math.log(100):.2f}")
    assert abs(round(math.log(100), 1) - factor) < 1e-9

    # ---- Worked example: the chapter's capacity plan (CapacityPlanningCalc), printed values reproduce
    svc = round(ms(plan.service_time), 2)                 # 3.53 ms batch-one service time
    budget = ms(plan.tail_budget)                          # 50 ms queue-plus-GPU p99 subbudget
    peak = plan.peak_rate.to("1/s").magnitude              # 5,000 req/s
    head = plan.headroom                                   # 1.3
    rho = round(1 - factor * svc / budget, 3)
    need = round(peak / rho, -1)
    per_gpu = round(1000 / svc)
    raw = need / per_gpu
    gpus = math.ceil(raw)
    final = math.ceil(round(gpus * head, 1))
    assert gpus == plan.accelerators and final == plan.accelerators_planned
    n.update(wservice=f"{svc:.2f}", wbudget=c(budget), wpeak=c(peak), wrho=f"{rho:.3f}", wrhopct=f"{rho * 100:.1f}",
             wneed=c(need), wpergpu=c(per_gpu), wraw=f"{raw:.1f}", wgpus=c(gpus), whead=f"{head:.1f}",
             wheadpct=f"{(head - 1) * 100:.0f}", wheadraw=f"{gpus * head:.1f}", wfinal=c(final),
             wbudgetmult=f"{budget / svc:.1f}")
    # Curve picture: p99 in service times = 4.6/(1-rho); budget line = budget/svc service times
    n.update(cbudget=f"{budget / svc:.4f}", crho=f"{rho:.4f}")

    # ---- Your turn: the TorchScript engine of the runtime ladder (RuntimeComparisonCalc), same plan
    ts = round(ms(prof.torchscript_fp16(1)), 2)            # 1.79 ms
    trho = round(1 - factor * ts / budget, 3)
    tneed = round(peak / trho, -1)
    tper = round(1000 / ts)
    traw = tneed / tper
    tgpus = math.ceil(traw)
    tfinal = math.ceil(round(tgpus * head, 1))
    n.update(tservice=f"{ts:.2f}", trho=f"{trho:.3f}", tneed=c(tneed), tpergpu=c(tper), traw=f"{traw:.1f}",
             tgpus=c(tgpus), theadraw=f"{tgpus * head:.1f}", tfinal=c(tfinal),
             tfaster=f"{svc / ts:.1f}", tfewer=f"{final / tfinal:.1f}")

    # ---- Mistake: cold start (ColdStartCalc, Scenarios.ResNet50ReplicaStartup; FallacyColdstartCalc's point)
    boot = Scenarios.ResNet50ReplicaStartup
    cold = boot.cold_total.to("s").magnitude               # 35 s first deploy
    warm = boot.warm_total.to("s").magnitude               # 1.5 s precompiled, local SSD
    build = boot.phases["TensorRT engine build"].to("s").magnitude
    assert abs(sum(boot.phases[k].to("s").magnitude for k in boot.cold_path) - cold) < 1e-9
    ph = {k: v.to("s").magnitude for k, v in boot.phases.items()}
    # front bars, drawn to scale: cold path = object-store weights, CUDA context, engine build, warmup + runtime
    n.update(bstore=f"{ph['weights from object storage']:.4f}", bcuda=f"{ph['CUDA context']:.4f}",
             bbuild=f"{ph['TensorRT engine build']:.4f}", btail=f"{ph['warmup'] + ph['runtime']:.4f}", bwarm=f"{warm:.4f}")
    n.update(cscold=f"{cold:.0f}", cswarm=f"{warm:.1f}", csbuild=f"{build:.0f}", csratio=f"{round(cold) / round(warm, 1):.0f}")
    # host stages before the forward pass (Q3's idle gap)
    n.update(hostpre=f"{st['JPEG decode'] + st['resize'] + st['normalize'] + st['copy to GPU']:.1f}")

    # ---- Mistake: precision at batch one (RuntimeComparisonCalc)
    eager = ms(prof.eager_fp16(1))
    t32, t8 = ms(prof.tensorrt("fp32", 1)), ms(prof.tensorrt("int8", 1))
    n.update(rteager=f"{eager:.2f}", rtfp32=f"{t32:.2f}", rtint8=f"{t8:.2f}",
             fourx=f"{32 / 8:.0f}", rtcompile=f"{round(eager, 2) / round(t32, 2):.1f}", rtprecision=f"{round(t32, 2) / round(t8, 2):.1f}")

    # ---- Table: ResNet-50 affine service model (ResNetH100Profile.affine_math) and GPT-2 XL decode floor
    fixed = round(ms(prof.eager_fp16(0)), 2)               # 3.50 ms
    per = round(prof.per_image_fp16.to("ms").magnitude, 3)  # 0.033 ms
    n.update(tfixed=f"{fixed:.2f}", tper=f"{per:.3f}")
    h100 = Hardware.Cloud.H100
    bw = h100.memory.bandwidth.to("TB/s").magnitude
    gpt2 = Models.Language.GPT2.size_in_bytes(BYTES_FP16).to("GB").magnitude
    n.update(hbw=f"{bw:.2f}", gptweights=f"{gpt2:.1f}", gptfloor=f"{gpt2 / bw:.2f}")

    # ---- Q1: ASSUMED 2,000 req/s, 8 ms window, the affine model with its rounded inputs
    lam, win = 2000, 8.0
    def tb(b):
        return fixed + per * b
    cap = {b: b / tb(b) * 1000 for b in (1, 8, 16)}
    lat16 = win / 2 + tb(16)
    n.update(qrate=c(lam), qwin=f"{win:.0f}", qhalf=f"{win / 2:.0f}", qavg=c(lam * win / 1000),
             qtone=f"{tb(1):.2f}", qteight=f"{tb(8):.2f}", qtsixteen=f"{tb(16):.2f}",
             qcapone=c(cap[1]), qcapeight=c(cap[8]), qcapsixteen=c(cap[16]),
             qrhoeight=f"{lam / round(cap[8]):.2f}", qrhosixteen=f"{lam / round(cap[16]):.2f}",
             qlat=f"{lat16:.2f}", qinflight=f"{lam * round(lat16, 2) / 1000:.0f}")
    qmax = 0.7  # ASSUMED utilization cap
    n.update(bone=c(1), beight=c(8), bsixteen=c(16), qmax=f"{qmax:.1f}")
    assert lam / cap[8] > qmax > lam / cap[16]

    # ---- Q2: Llama 3 8B on one H100 (KvBudget), ASSUMED 4,096-token conversations
    kv = Scenarios.Llama3_8BFp16ConversationsH100
    w = kv.workload
    hd = int(w.head_width)
    per_tok = 2 * w.layers * w.kv_heads * hd * BYTES_FP16.to("byte").magnitude
    assert per_tok == kv.result.kv_per_token.to("byte").magnitude == 131072
    ctx = 4096
    conv_gb = per_tok * ctx / 1e9
    cap_gb = kv.hbm_per_accelerator.to("GB").magnitude
    cap_gib = kv.hbm_per_accelerator.to("GiB").magnitude
    weights = kv.result.weights.to("GB").magnitude
    free = cap_gb - weights
    n.update(llayers=c(w.layers), lkvheads=c(w.kv_heads), lheaddim=c(hd), lpertok=c(per_tok),
             lctx=c(ctx), lconv=f"{conv_gb:.3f}", lcapgib=f"{cap_gib:.0f}", lcapgb=f"{cap_gb:.1f}", gibgb=f"{2**30 / 1e9:.3f}", restprep=f"{(st['resize'] + st['normalize'] + st['copy to GPU']):.1f}",
             lweights=f"{weights:.1f}", lfree=f"{round(cap_gb, 1) - round(weights, 1):.1f}",
             lconvs=c(round(round(free, 1) / round(conv_gb, 3))), lfloor=f"{round(weights, 1) / bw:.1f}",
             ltokrate=c(1000 / round(round(weights, 1) / bw, 1)))
    assert int(free / conv_gb) in (129, 130)

    # ---- Q4: cost per million images (CostAnalysisCalc), ASSUMED 1,000 img/s and a 70 % utilization cap
    cost = Scenarios.ResNet50CloudInstanceCost
    p_t4, p_h = cost.prices["T4"].to("dollar/hour").magnitude, cost.prices["H100"].to("dollar/hour").magnitude
    r_t4 = cost.rates["T4"].to("1/s").magnitude
    r_h = round(cost.rates["H100"].to("1/s").magnitude, -1)
    load, cap_util = 1000, 0.7
    t4_use = r_t4 * cap_util
    t4_n = math.ceil(load / t4_use)
    t4_hour = t4_n * p_t4
    n.update(ct4price=f"{p_t4:.2f}", chprice=f"{p_h:.2f}", ct4rate=c(r_t4), chrate=c(r_h), cload=c(load),
             cutil=f"{cap_util * 100:.0f}", ct4use=c(t4_use), ct4n=c(t4_n), ct4ratio=f"{load / t4_use:.2f}",
             ct4hour=f"{t4_hour:.2f}", ct4million=f"{t4_hour / (load * 3600) * 1e6:.2f}",
             chmillion=f"{p_h / (load * 3600) * 1e6:.2f}", chbusy=f"{load / r_h * 100:.0f}",
             chsat=f"{p_h / (r_h * 3600) * 1e6:.2f}", cbreak=c(round(cost.breakeven_rate.to('1/s').magnitude, -2)))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
