"""Numbers for the Fleet Efficiency study sheet (Volume II, Chapter 9).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to performance_engineering_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
"""
from pathlib import Path

from mlsysim.engine.scenarios import Scenarios


def numbers() -> dict:
    n = {}

    # ---- Scheduling goodput: the shared pool snapshot (cell SchedulingGoodputSnapshot).
    snap = Scenarios.SharedPoolSchedulingSnapshot
    sched = snap.scheduling_goodput
    assert abs(sched - (snap.assigned_share - snap.incomplete_gang_share)) < 1e-12
    n.update(assigned=f"{snap.assigned_share:.2f}", incomplete=f"{snap.incomplete_gang_share:.2f}",
             unassigned=f"{snap.unassigned_share:.2f}", sched=f"{sched:.2f}")

    # ---- Runtime goodput: five owners' terms (cell RuntimeGoodputComposition).
    rs = Scenarios.Llama2_70BRuntimeGoodput128
    r = rs.result
    terms = dict(startup=r.startup_share, input=r.input_wait_share, checkpoint=r.checkpoint_share,
                 recovery=r.recovery_share, preempt=r.preemption_share)
    loss = sum(terms.values())
    assert abs(loss - r.runtime_loss) < 1e-12
    for k, v in terms.items():
        n[f"term{k}"] = f"{v * 100:.2f}"
    # The printed total (one decimal) must equal the sum of the printed terms to that precision.
    printed_sum = sum(float(n[f"term{k}"]) for k in terms)
    assert f"{printed_sum:.1f}" == f"{loss * 100:.1f}"
    runtime = r.runtime_goodput
    n.update(runtimeloss=f"{loss * 100:.1f}", runtime=f"{runtime:.3f}", runtimetwo=f"{runtime:.2f}",
             tstartup=f"{r.startup_time.to('minute').magnitude:.1f}",
             trecovery=f"{r.recovery_time.to('minute').magnitude:.1f}",
             mtbfhours=f"{r.mtbf.to('hour').magnitude:.0f}",
             mtbfdays=f"{r.mtbf.to('day').magnitude:.0f}",
             tpreempt=f"{r.preemption_cost.to('minute').magnitude:.1f}",
             preemptday=f"{rs.preemption_rate.to('1/day').magnitude:.0f}",
             windowdays=f"{rs.window.to('day').magnitude:.0f}",
             ckptinterval=f"{r.interval.to('minute').magnitude:.0f}")

    # ---- Program goodput: fleet MFU of the retained steps (cell FleetEfficiencyCalc).
    mf = Scenarios.Llama2_70BFleetMfu128
    useful_pflop = mf.fleet_useful_work.to("flop").magnitude / 1e15
    peak_tflops = mf.hardware.compute.peak_flops.to("TFLOPs/second").magnitude
    step_ms = mf.fleet_step_time.to("millisecond").magnitude
    budget_pflop = mf.fleet_devices * peak_tflops * step_ms / 1e3 / 1e3   # TFLOP/s x ms -> PFLOP
    program = mf.fleet_mfu
    assert abs(useful_pflop / budget_pflop - program) < 1e-9
    assert abs(float(f"{useful_pflop:.1f}") / float(f"{budget_pflop:.1f}") - float(f"{program:.2f}")) < 0.005
    n.update(gpus=f"{mf.fleet_devices}", usefulpflop=f"{useful_pflop:.1f}", peaktflops=f"{peak_tflops:.0f}",
             stepms=f"{step_ms:.0f}", budgetpflop=f"{budget_pflop:.1f}", program=f"{program:.2f}",
             nodemfu=f"{mf.node_mfu:.2f}", nodestepms=f"{mf.node_step_time.to('millisecond').magnitude:.0f}")

    # ---- The product (cell GoodputDecomposition, Scenarios.Llama2_70BSharedPoolGoodput128).
    pool = Scenarios.Llama2_70BSharedPoolGoodput128
    g = pool.result
    assert pool.scheduling == sched and pool.runtime == runtime and pool.program == program
    mpg = g.mpg
    # Printed MPG must reproduce from the printed factors.
    assert f"{float(n['sched']) * float(n['runtime']) * float(n['program']):.3f}" == f"{mpg:.3f}"
    price = pool.device_hour_rate.to("USD/hour").magnitude
    useful = g.useful_rate.to("USD/hour").magnitude
    assert abs(useful - price / mpg) < 1e-9
    n.update(mpg=f"{mpg:.3f}", mpgpct=f"{mpg * 100:.0f}", jobproduct=f"{g.job_product:.2f}",
             shares=f"{g.scheduling_share:.2f}", sharer=f"{g.runtime_share:.3f}", sharep=f"{g.program_share:.3f}",
             sharespct=f"{g.scheduling_share * 100:.0f}", sharerpct=f"{g.runtime_share * 100:.0f}",
             sharepppct=f"{g.program_share * 100:.0f}",
             price=f"{price:.0f}", useful=f"{useful:.2f}", costmult=f"{float(f'{useful:.2f}') / price:.1f}")
    assert abs(g.scheduling_share + g.runtime_share + g.program_share + mpg - 1) < 1e-12

    # Front picture: one stacked bar of offered capacity, 230 pt = 100 percent.
    width = 230.0
    edges = [0.0]
    for part in (g.scheduling_share, g.runtime_share, g.program_share, mpg):
        edges.append(edges[-1] + width * part)
    for name, x in zip(("xa", "xb", "xc", "xd", "xe"), edges):
        n[name] = f"{x:.1f}"

    # ---- Your turn: MPG at each owner's ceiling (pool.ceilings; g.mpg_at_ceiling).
    ce = pool.ceilings
    n.update(ceilsched=f"{ce['S']:.2f}", ceilruntime=f"{ce['R']:.3f}", ceilprogram=f"{ce['P']:.2f}")
    for k, key in (("S", "sched"), ("R", "runtime"), ("P", "program")):
        n[f"mpgceil{key}"] = f"{g.mpg_at_ceiling[k]:.3f}"
        n[f"ratioceil{key}"] = f"{g.ceiling_ratio[k]:.3f}"
        n[f"gainceil{key}"] = f"{(g.ceiling_ratio[k] - 1) * 100:.0f}"
    # Printed results reproduce from printed inputs.
    f = lambda k: float(n[k])
    assert f"{f('ceilsched') * f('runtime') * f('program'):.3f}" == n["mpgceilsched"]
    assert f"{f('sched') * f('ceilruntime') * f('program'):.3f}" == n["mpgceilruntime"]
    assert f"{f('sched') * f('runtime') * f('ceilprogram'):.3f}" == n["mpgceilprogram"]
    n["fullsched"] = f"{g.full_scheduling_gain * 100:.0f}"           # a perfect scheduler, in points of capacity
    n["mpgperfectsched"] = f"{g.job_product:.3f}"

    # ---- Mistake: high MFU. Llama 3 405B's program goodput (MFU) on its dedicated cluster.
    llama = Scenarios.Llama3_405BGoodput16K
    n["llamaprogram"] = f"{llama.program:.2f}"

    # ---- Q1: ASSUMED four forced preemptions a day at the chapter's per-event cost.
    q_rate = 4
    # Computed from the printed per-event cost and printed totals, so the answer reproduces from the sheet.
    q_term = q_rate * float(n["tpreempt"]) / (24 * 60)
    q_loss = float(n["runtimeloss"]) / 100 - float(n["termpreempt"]) / 100 + q_term
    assert abs(q_loss - (loss - r.preemption_share + q_rate * r.preemption_cost.to("day").magnitude)) < 0.001
    n.update(qrate=f"{q_rate}", qpreempt=f"{q_term * 100:.2f}", qloss=f"{q_loss * 100:.1f}",
             qruntime=f"{1 - q_loss:.3f}")
    assert q_term < r.input_wait_share                              # input waits still the largest term

    # ---- Q2: ASSUMED autotuned build: P from 0.48 to 0.52, R from 0.904 to 0.85 (longer startups).
    p_new, r_new = 0.52, 0.85
    rp = p_new / float(n["program"])
    rr = r_new / float(n["runtime"])
    n.update(qpnew=f"{p_new:.2f}", qrnew=f"{r_new:.2f}", qratiop=f"{rp:.4f}", qratior=f"{rr:.4f}")
    # Printed product comes from the printed ratios; it stays within a rounding step of the exact one.
    shown = float(n["qratiop"]) * float(n["qratior"])
    assert f"{shown:.3f}" == f"{rp * rr:.3f}"
    n.update(qratio=f"{shown:.3f}", qgain=f"{(shown - 1) * 100:.1f}")

    # ---- Q4: batch-one decode on the 8-GPU replica (cell RooflineDiagnostic).
    d = Scenarios.Llama2_70BDecodeFloorTp8.result
    n.update(decodems=f"{d.memory_time.to('millisecond').magnitude:.1f}",
             decodeutil=f"{d.peak_fraction * 100:.1f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
