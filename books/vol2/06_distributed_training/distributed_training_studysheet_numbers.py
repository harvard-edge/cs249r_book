"""Numbers for the Distributed Training study sheet (Volume II, Chapter 6).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to distributed_training_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Units: GB is decimal. Accelerator HBM is the registry capacity, 80 GiB = 85.9 GB; every fit is judged
against that byte value (the chapter's "80 GB" budget is a logged book defect, not copied). Link rates quoted in Gb/s are
bits and are converted to GB/s (bytes) here.
"""
import math
from pathlib import Path

from mlsysim import ReferenceStats, Systems
from mlsysim.core.units import BYTES_FP16, GB, GiB, byte, param
from mlsysim.engine.scenarios import Scenarios


def c(x):
    return f"{x:,.0f}"


def numbers() -> dict:
    n = {}
    adam = ReferenceStats.MixedPrecisionAdam
    b_state = adam.TrainingStateBytesPerParameter.to(byte / param).magnitude      # 16
    b_opt = adam.OptimizerStateBytesPerParameter.to(byte / param).magnitude       # 12
    b_half = BYTES_FP16.to(byte).magnitude                                         # 2 (weights; gradients)
    assert b_half + b_half + b_opt == b_state
    n.update(statebytes=f"{b_state:.0f}", optbytes=f"{b_opt:.0f}", halfbytes=f"{b_half:.0f}",
             gradoptbytes=f"{b_half + b_opt:.0f}", weightgradbytes=f"{2 * b_half:.0f}")

    # ---- Bandwidth tiers (chapter: TensorParallelBandwidth / BandwidthFeasibilityNapkin)
    tpl = Scenarios.Gpt3TensorParallelLayerTp8
    nvl = tpl.links["NVLink"].to("GB/s").magnitude                    # 450 GB/s per direction
    ndr = tpl.links["InfiniBand NDR"].to("GB/s").magnitude            # 400 Gb/s = 50 GB/s
    n.update(nvlink=f"{nvl:.0f}", ndr=f"{ndr:.0f}", ndrgbit=f"{Systems.Fabrics.InfiniBand_NDR.bandwidth.to('Gbps').magnitude:.0f}",
             linkratio=f"{nvl / ndr:.0f}")

    # ---- Front picture: ZeRO stages for the chapter's 7B model on 64 GPUs (Scenarios.Llama2_7B_ZeroStages64)
    z7 = Scenarios.Llama2_7B_ZeroStages64
    st = {k: v.to(GB).magnitude for k, v in z7.result.stage_state.items()}
    _cap = Scenarios.DenseLLMPretrainingStep.fleet.node.accelerator.memory.capacity      # 80 GiB (H100)
    hbm_label = _cap.to(GB).magnitude                                                     # 85.9 GB, the budget
    n.update(zgpus=c(z7.n_devices), zparams=f"{z7.workload.parameters.to('Bparam').magnitude:.0f}",
             zddp=f"{st['DDP']:.0f}", zone=f"{st['ZeRO-1']:.1f}", ztwo=f"{st['ZeRO-2']:.1f}", zthree=f"{st['ZeRO-3']:.2f}",
             hbm=f"{hbm_label:.1f}", hbmgib=f"{_cap.to(GiB).magnitude:.0f}", zthreemult=f"{z7.result.traffic_multiple['ZeRO-3']:.1f}")
    # bar lengths in pt: 112 GB -> 150 pt
    scale = 150.0 / st["DDP"]
    n.update(xddp=f"{st['DDP'] * scale:.1f}", xone=f"{st['ZeRO-1'] * scale:.1f}", xtwo=f"{st['ZeRO-2'] * scale:.1f}",
             xthree=f"{max(st['ZeRO-3'] * scale, 1.2):.1f}", xhbm=f"{hbm_label * scale:.1f}")
    # the stage formulas on the front reproduce the registry's stage totals
    P7, N7 = z7.workload.parameters.magnitude, z7.n_devices
    assert abs(P7 * (2 * b_half + b_opt / N7) / 1e9 - st["ZeRO-1"]) < 1e-6
    assert abs(P7 * (b_half + (b_half + b_opt) / N7) / 1e9 - st["ZeRO-2"]) < 1e-6
    assert abs(P7 * b_state / N7 / 1e9 - st["ZeRO-3"]) < 1e-6

    # ---- Worked example: the chapter's reference job, configuration D (Scenarios.DenseLLMPretrainingStep)
    step = Scenarios.DenseLLMPretrainingStep
    r = step.result
    tp, pp, dp = step.tensor_parallel, step.pipeline_depth, step.data_parallel
    P = step.replica.workload.parameters.magnitude
    state_total = P * b_state / 1e9                                         # 2,800 GB
    per_gpu = state_total / (tp * pp)                                      # 43.75 GB
    assert abs(per_gpu - step.state_per_gpu.to(GB).magnitude) < 1e-9
    m_d = step.microbatches_per_replica                                    # 96
    assert m_d == step.global_batch_sequences // dp
    t_comp = r.node_profile.latency.to("s").magnitude                      # per-GPU computation
    t_tp = r.tp_communication_latency.to("s").magnitude
    t_dp = r.dp_communication_latency.to("s").magnitude
    t_bub = r.pipeline_bubble_latency.to("s").magnitude
    t_step = r.step_latency_total.to("s").magnitude
    t_dp_exposed = t_step - t_comp - t_tp - t_bub
    overlap = 1 - t_dp_exposed / t_dp
    assert abs(t_bub - t_comp * (pp - 1) / m_d) < 1e-6                       # idle = busy x (p-1)/m
    assert abs(r.bubble_fraction - (pp - 1) / (m_d + pp - 1)) < 1e-12
    n.update(refparams=f"{P / 1e9:.0f}", refgpus=c(step.fleet.total_accelerators),
             refnodes=c(step.fleet.total_accelerators // step.fleet.node.accelerators_per_node),
             pernode=c(step.fleet.node.accelerators_per_node), reftp=c(tp), refpp=c(pp), refdp=c(dp),
             refmodel=c(tp * pp), statetotal=c(state_total), refstate=f"{per_gpu:.2f}",
             refhead=f"{float(f'{hbm_label:.1f}') - per_gpu:.2f}", batch=c(step.global_batch_sequences), refmicro=c(m_d),
             refppminus=c(pp - 1), refbubbleden=c(m_d + pp - 1), refbubble=f"{r.bubble_fraction * 100:.1f}",
             tcomp=f"{t_comp:.2f}", ttp=f"{t_tp:.2f}", tdp=f"{t_dp:.2f}", tdpexp=f"{t_dp_exposed:.2f}",
             overlap=f"{overlap * 100:.0f}", exposedfrac=f"{1 - overlap:.2f}", overlapfrac=f"{overlap:.2f}",
             tbub=f"{t_bub:.2f}", tstep=f"{t_step:.2f}",
             refeta=f"{r.scaling_efficiency * 100:.0f}")
    assert abs(float(f"{t_comp:.2f}") + float(f"{t_tp:.2f}") + float(f"{t_dp_exposed:.2f}") + float(f"{t_bub:.2f}") - float(f"{t_step:.2f}")) < 0.011
    # printed exposed DP reproduces from printed reduction time and printed overlap
    assert abs(float(f"{t_dp:.2f}") * (1 - float(f"{overlap:.2f}")) - float(f"{t_dp_exposed:.2f}")) < 0.006
    # ---- Mistake (MFU x eta double counting; chapter's MfuScalingPitfall on the same reference job)
    local_mfu = step.local_efficiency
    eta = r.scaling_efficiency
    fleet_mfu = local_mfu * eta
    n.update(mfulocal=f"{local_mfu * 100:.0f}", mfufleet=f"{fleet_mfu * 100:.0f}",
             mfutwice=f"{fleet_mfu * eta * 100:.0f}")
    assert round(local_mfu * 100) == 45 and round(fleet_mfu * 100) == 37 and round(fleet_mfu * eta * 100) == 31
    # printed fleet MFU reproduces from the printed local MFU and printed eta
    assert abs(round(local_mfu * 100) * round(eta * 100) / 100 - fleet_mfu * 100) < 0.5
    assert abs(round(fleet_mfu * 100) * round(eta * 100) / 100 - fleet_mfu * eta * 100) < 0.5
    # step bar on the back, to scale: 8.71 s -> 160 pt
    sc = 160.0 / t_step
    xa = t_comp * sc
    xb = xa + t_tp * sc
    xc = xb + t_dp_exposed * sc
    xd = xc + t_bub * sc
    n.update(barcomp=f"{xa:.1f}", bartp=f"{xb:.1f}", bardp=f"{xc:.1f}", barbub=f"{xd:.1f}")

    # ---- Your turn: configuration C of the chapter (TP 8, PP 4, DP 32) with ZeRO-1 over its DP group
    pp_c = 4
    dp_c = step.fleet.total_accelerators // (tp * pp_c)
    m_c = step.global_batch_sequences // dp_c
    c_per = state_total / (tp * pp_c)                                       # 87.5 GB
    wg = P * 2 * b_half / 1e9 / (tp * pp_c)                                 # weights + gradients per GPU
    opt = P * b_opt / 1e9 / (tp * pp_c * dp_c)                              # optimizer state shard
    c_bub = (pp_c - 1) / (m_c + pp_c - 1)
    n.update(cpp=c(pp_c), cdp=c(dp_c), cmicro=c(m_c), cstate=f"{c_per:.1f}", cmodel=c(tp * pp_c),
             cwg=f"{wg:.2f}", copt=f"{opt:.2f}", czero=f"{wg + opt:.2f}", cbubble=f"{c_bub * 100:.1f}",
             cppminus=c(pp_c - 1), cbubbleden=c(m_c + pp_c - 1), cbub=f"{t_comp * (pp_c - 1) / m_c:.2f}",
             cgroup=c(tp * pp_c * dp_c))
    assert c_per > hbm_label and wg + opt < hbm_label and per_gpu < hbm_label and st['DDP'] > hbm_label

    # ---- Mistake 1: GPT-2 data parallelism, one NVLink node vs four nodes on 10 Gb Ethernet
    one, four = Scenarios.Gpt2DataParallelOneNode, Scenarios.Gpt2DataParallelFourNodes10GbE
    r1, r4 = one.result, four.result
    eth = four.inter_bandwidth.to("GB/s").magnitude                         # 10 Gb/s = 1.25 GB/s
    ring = 2 * (four.nodes - 1) / four.nodes
    assert abs(ring * r4.gradient_bytes.to(GB).magnitude - r4.inter_bytes.to(GB).magnitude) < 1e-9
    n.update(gptgrad=f"{r4.gradient_bytes.to(GB).magnitude:.0f}", gptgpus=c(r4.n_devices), gptnodes=c(four.nodes), gptsmall=c(r1.n_devices),
             ethgbit=f"{four.inter_bandwidth.to('Gbps').magnitude:.0f}", ethbytes=f"{eth:.2f}",
             gptring=f"{r4.inter_bytes.to(GB).magnitude:.2f}", gptinter=f"{r4.inter_time.to('s').magnitude:.2f}",
             gptcomp=f"{r4.compute_time.to('s').magnitude:.1f}", gptcomm=f"{r4.communication_time.to('s').magnitude:.2f}",
             gpteffone=f"{r1.efficiency * 100:.0f}", gptefffour=f"{r4.efficiency * 100:.0f}",
             gptspone=f"{r1.speedup:.1f}", gptspfour=f"{r4.speedup:.1f}")

    # ---- Mistake 2: ZeRO-3 of the 175B model over 64 GPUs regathers every layer (Scenarios.Gpt3Zero3On64)
    z3 = Scenarios.Gpt3Zero3On64
    gt = z3.result.gather_time.to("s").magnitude
    n.update(fsgpus=c(z3.n_devices), fsstate=f"{z3.result.stage_state['ZeRO-3'].to(GB).magnitude:.2f}",
             fsweights=c(z3.result.weights.to(GB).magnitude), fstraffic=c(z3.result.gather_traffic.to(GB).magnitude),
             fstime=f"{gt:.0f}", fsratio=f"{gt / t_comp:.1f}")

    # ---- Mistake 3: SSP vs BSP at 64 workers (chapter's illustrative ScalingWorkers scenario)
    bsp, ssp = Scenarios.TransformerBsp64Workers, Scenarios.TransformerSsp64Workers
    n.update(syncworkers=c(bsp.workers), bspiters=c(bsp.iterations), sspiters=c(ssp.iterations),
             bspcomm=f"{bsp.communication_overhead * 100:.0f}", sspcomm=f"{ssp.communication_overhead * 100:.0f}",
             bspspeed=f"{bsp.speedup:.0f}", sspspeed=f"{ssp.speedup:.0f}", sspbound=c(ssp.staleness),
             bspmult=f"{1 + bsp.communication_overhead:.2f}", sspmult=f"{1 + ssp.communication_overhead:.2f}",
             bspwork=c(bsp.iterations * (1 + bsp.communication_overhead)),
             sspwork=c(ssp.iterations * (1 + ssp.communication_overhead)))
    assert bsp.iterations * (1 + bsp.communication_overhead) < ssp.iterations * (1 + ssp.communication_overhead)

    # ---- Q1: ASSUMED 13B model, 32-GPU data-parallel group, 80 GB GPUs, at least 20 GB kept for activations
    q1p, q1n, q1keep = 13e9, 32, 20.0
    ddp = q1p * b_state / 1e9
    z1 = q1p * (2 * b_half + b_opt / q1n) / 1e9
    z2 = q1p * (b_half + (b_half + b_opt) / q1n) / 1e9
    z3v = q1p * b_state / q1n / 1e9
    assert ddp > hbm_label and hbm_label - z1 >= q1keep
    n.update(qoneparams=f"{q1p / 1e9:.0f}", qonegpus=c(q1n), qonekeep=f"{q1keep:.0f}", qoneddp=c(ddp),
             qoneone=f"{z1:.1f}", qonetwo=f"{z2:.1f}", qonethree=f"{z3v:.1f}", qoneleft=f"{hbm_label - z1:.1f}",
             qoneoptshard=f"{b_opt / q1n:.3f}", qoneper=f"{2 * b_half + b_opt / q1n:.3f}")

    # ---- Q2: TP across two nodes for one GPT-3 layer (chapter's Gpt3TensorParallelLayerTp8 numbers)
    tr = tpl.result
    t_nv = tr.transfer_times["NVLink"].to("ms").magnitude
    t_ib = tr.transfer_times["InfiniBand NDR"].to("ms").magnitude
    assert abs(t_ib / t_nv - nvl / ndr) < 1e-9
    n.update(qtwocomp=f"{tr.compute_time.to('ms').magnitude:.2f}", qtwonv=f"{t_nv:.2f}", qtwoib=f"{t_ib:.1f}",
             qtwotp=c(2 * tpl.tp_size), qtwoover=f"{t_ib / tr.compute_time.to('ms').magnitude:.1f}")

    # ---- Q3: ASSUMED 2,048 workers, each late on 0.1 percent of steps (chapter's StragglerFreeStep1K logic)
    q3n, q3p = 2048, 0.001
    clean = (1 - q3p) ** q3n
    n.update(qthreen=c(q3n), qthreep=f"{q3p * 100:.1f}", qthreeontime=f"{1 - q3p:.3f}", qthreeclean=f"{clean * 100:.0f}",
             qthreelate=f"{(1 - clean) * 100:.0f}", qthreeexp=f"{q3n * q3p:.1f}")
    s1k = Scenarios.StragglerFreeStep1K
    n.update(chapn=c(s1k.n_components), chapp=f"{(1 - s1k.availability) * 100:.0f}", chapclean=f"{s1k.all_up * 100:.3f}")

    # ---- Q4: ASSUMED pilot trace tr(Sigma) = 2.4e6, |grad|^2 = 200; local batch 16; 4,096 workers proposed
    tr_sig, gn2, b_loc, q4n = 2.4e6, 200.0, 16, 4096
    bcrit = tr_sig / gn2
    glob = q4n * b_loc
    n.update(qfourtrace=f"{tr_sig / 1e6:.1f}", qfournorm=f"{gn2:.0f}", qfourlocal=c(b_loc), qfourn=c(q4n),
             qfourcrit=c(bcrit), qfouruseful=c(bcrit / b_loc), qfourglobal=c(glob), qfourmult=f"{glob / bcrit:.1f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
