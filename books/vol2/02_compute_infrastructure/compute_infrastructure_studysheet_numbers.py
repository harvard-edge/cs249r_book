"""Numbers for the Compute Infrastructure study sheet (Volume II, Chapter 2).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to compute_infrastructure_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
Capacities: the H100 is 80 GiB, which is 85.9 GB in decimal bytes; every GB on the sheet is decimal.
"""
import math
from pathlib import Path

from mlsysim import Hardware, Infrastructure, Models, ReferenceStats, Systems
from mlsysim.engine.scenarios import Scenarios

BYTES_PER_PARAM_ADAM = 16  # FP16 weights 2 + FP16 gradients 2 + FP32 master 4 + two FP32 moments 8 (chapter tbl-memory-breakdown)


def c(x, nd=0):
    return f"{x:,.{nd}f}"


def numbers() -> dict:
    n = {}
    h100 = Hardware.Cloud.H100
    peak = h100.compute.peak_flops.to("flop/s").magnitude          # dense FP16/BF16
    bw = h100.memory.bandwidth.to("byte/s").magnitude
    cap_b = h100.memory.capacity.to("byte").magnitude
    ridge = h100.ridge_point().to("flop/byte").magnitude
    n.update(peaktflops=c(peak / 1e12), fpeight=c(h100.compute.precision_flops["fp8"].to("TFLOP/s").magnitude),
             bandwidthtb=f"{bw / 1e12:.2f}", ridge=c(round(ridge)),
             capgib=c(h100.memory.capacity.to("GiB").magnitude), capgb=f"{cap_b / 1e9:.1f}")

    # Bandwidth ladder (chapter: BandwidthStaircase / BufferTransfer10GB, fig-infra-bandwidth-hierarchy)
    st = Scenarios.BufferTransfer10GB
    links = {k: v.to("GB/s").magnitude for k, v in st.links.items()}
    n.update(ladderhbm=c(links["HBM"]), laddernvlink=c(links["NVLink"]), ladderib=c(links["InfiniBand"]),
             nvlinkgap=c(links["NVLink"] / links["InfiniBand"]))
    # numeric positions for the TikZ ladder (unformatted)
    n.update(posHBM=f"{links['HBM']:.0f}", posNVL=f"{links['NVLink']:.0f}", posIB=f"{links['InfiniBand']:.0f}")

    # Rack power (chapter: RackPowerRackRecap)
    rack = Systems.Racks.DGX_H100_4Node
    rack_kw = rack.total_power.to("kW").magnitude
    trad_kw = Infrastructure.Racks.Traditional.power_kw
    n.update(rackkw=f"{rack_kw:.1f}", nodekw=f"{rack_kw / rack.nodes_per_rack:.1f}", tradkw=c(trad_kw),
             liquidkw=c(Infrastructure.Racks.AI_High.power_kw), rackratio=f"{rack_kw / trad_kw:.1f}")

    # Reliability (chapter: WscThroughput, fig-three-walls-grid)
    mttf = Systems.Reliability.Gpu.mttf_hours
    n.update(mttf=c(mttf), bytesparam=c(BYTES_PER_PARAM_ADAM))

    # Running model's per-GPU state on the reference layout (chapter: NodeCapacityTrainMemRecap, TP8 x PP8)
    step0 = Scenarios.DenseLLMPretrainingStep
    n.update(reftp=c(step0.tensor_parallel), refpp=c(step0.pipeline_depth),
             refstate=f"{step0.state_per_gpu.to('GB').magnitude:.1f}")

    # ---- Worked example: Llama-3 70B batch-one decode, one H100-equivalent path (Scenarios.Llama3_70B_DecodeH100)
    dec = Scenarios.Llama3_70B_DecodeH100
    r = dec.result
    wgb = r.weight_bytes.to("GB").magnitude
    flops = r.flops_per_step.to("flop").magnitude
    n.update(wparams=f"{dec.workload.parameters.to('param').magnitude / 1e9:.1f}", wgb=f"{wgb:.1f}",
             wgflop=c(flops / 1e9), wtcomp=f"{r.compute_time.to('ms').magnitude:.2f}",
             wtmem=f"{r.memory_time.to('ms').magnitude:.1f}", wshare=f"{r.memory_share * 100:.1f}",
             wintensity=c(flops / r.weight_bytes.to("byte").magnitude),
             wratio=c(r.memory_time.to("ms").magnitude / r.compute_time.to("ms").magnitude))
    # Roofline points (chapter: RooflineLandscapeMath)
    n.update(rtraini=c(2048 * 1.0), posTRAIN=f"{2048 * 1.0:.0f}")  # ASSUMED-by-chapter: 2,048 local tokens share each weight read

    # ---- Your turn: the same 70.6B path, batch 32 (one condition changed; chapter: RooflineLandscapeMath decode batch)
    batch = 32                                                     # ASSUMED-by-chapter: decode batch of 32
    wtmem_r = round(r.memory_time.to("ms").magnitude, 1)          # printed step floor, ms
    n.update(ybatch=c(batch), ytok=c(round(batch / (wtmem_r / 1e3), -1)), yint=c(batch),
             yone=c(round(1 / (wtmem_r / 1e3))))
    assert batch < ridge                                           # batch 32 is still below the ridge

    # ---- Mistake 1: scaling efficiency tiers (chapter: InfraFrontierStragglerRecap23)
    e1k = round(Scenarios.DenseLLMPretrainingStep.result.scaling_efficiency, 2)
    e8k = round(Scenarios.DenseLLMPretrainingStep8K.result.scaling_efficiency, 2)  # as the chapter: the NDR 8,192-GPU step
    hw = Systems.Clusters.Frontier_8K.total_accelerators / Scenarios.DenseLLMPretrainingStep.fleet.total_accelerators
    n.update(effone=f"{e1k:.2f}", effeight=f"{e8k:.2f}", hwratio=c(hw), speedup=f"{hw * e8k / e1k:.1f}",
             lost=c((1 - e8k) * 100), gpusone=c(Scenarios.DenseLLMPretrainingStep.fleet.total_accelerators),
             gpuseight=c(Systems.Clusters.Frontier_8K.total_accelerators))

    # ---- Q1: compute-only time on the 1,024-GPU reference cluster (chapter: PodTrainingMath, ReferenceClusterTrainingTime)
    step = Scenarios.DenseLLMPretrainingStep
    gpt3 = Models.Language.GPT3
    P = gpt3.parameters.to("param").magnitude
    D = gpt3.training_tokens.to("count").magnitude
    N = step.fleet.total_accelerators
    eta = step.local_efficiency
    total = 6 * P * D
    rate = N * peak * eta
    days = total / rate / 86400
    n.update(qparams=c(P / 1e9), qtokens=c(D / 1e9), qgpus=c(N), qeta=c(eta * 100), qtotal=f"{total / 1e23:.2f}", qtotalsci=f"{total / 1e23:.2f}\\times 10^{{23}}",
             qrate=f"{rate / 1e17:.2f}", qratesci=f"{rate / 1e17:.2f}\\times 10^{{17}}", qhours=c(total / rate / 3600), qdays=f"{days:.1f}",
             qdaysscaled=f"{days / e1k:.1f}")

    # ---- Q2: ASSUMED 70B model, mixed-precision Adam, TP-8, H100 80 GiB
    p70, tp = 70e9, 8
    state = BYTES_PER_PARAM_ADAM * p70
    per_tp = state / tp
    pp = 1
    while per_tp / pp > cap_b:
        pp *= 2
    n.update(stateparams=c(p70 / 1e9), statetotal=c(state / 1e9), statetp=c(per_tp / 1e9), statepp=c(pp),
             stategpu=c(per_tp / pp / 1e9), stateroom=f"{(cap_b - per_tp / pp) / 1e9:.1f}", stategpus=c(tp * pp))

    # ---- Q3: 10 GB buffer on NVLink vs one InfiniBand port (chapter: BufferTransfer10GB)
    pay = st.payload.to("GB").magnitude
    n.update(buffer=c(pay), bufnvlink=f"{pay / links['NVLink'] * 1e3:.1f}", bufib=c(pay / links["InfiniBand"] * 1e3))

    # ---- Q4: ASSUMED 16,384-GPU job, independent failures
    big = 16384
    n.update(fleetgpus=c(big), fleetmtbf=f"{mttf / big:.1f}", fleetmtbfmin=c(mttf / big * 60))
    # Q4(b): its power on four-node DGX H100 racks vs a hall built for traditional racks (chapter: RackPowerRackRecap)
    rack_gpus = rack.accelerator_count
    node_gpus = rack.node.accelerators_per_node
    racks_big = big / rack_gpus
    rack_kw_r = round(rack_kw, 1)
    node_kw_r = round(rack_kw / rack.nodes_per_rack, 1)
    trad_nodes = math.floor(trad_kw / node_kw_r)                  # whole nodes a traditional rack can feed
    assert trad_nodes >= 1
    n.update(nodegpus=c(node_gpus), rackgpus=c(rack_gpus), nodespr=c(rack.nodes_per_rack), bigracks=c(racks_big),
             bigmw=f"{racks_big * rack_kw_r / 1e3:.1f}", tradnodes=c(trad_nodes),
             tradracks=c(big / (trad_nodes * node_gpus)))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
