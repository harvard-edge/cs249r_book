"""Numbers for the Fleet Orchestration study sheet (Volume II, Chapter 8).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to fleet_orchestration_studysheet_numbers.tex next to
this file. Inputs the sheet chooses for its own questions are marked ASSUMED (built as variants
of the chapter's scenarios). A printed result is computed from the rounded values the sheet prints
beside it, and checked against the unrounded model value.

Units: GB/s are decimal bytes per second, per direction, as in the chapter; prices are USD per
GPU-hour at the chapter's scenario rates.
"""
import math
from pathlib import Path

from mlsysim import Hardware, ReferenceStats, Systems
from mlsysim import Scenarios as SC


def fx(x, nd):
    return f"{x:.{nd}f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def mag(q, unit):
    return q.to(unit).magnitude


def numbers() -> dict:
    n = {}

    # ---- Queue card (chapter: lego-queuing-theory; M/G/1 at 80 percent, c_s = 1 and 3)
    qu, qm = SC.Mg1ExponentialJobQueueAt80, SC.Mg1MixedJobQueueAt80
    assert qu.utilization == qm.utilization
    near(qu.load_factor * qu.variability_factor, qu.wait_ratio, 1e-9)
    near(qm.load_factor * qm.variability_factor, qm.wait_ratio, 1e-9)
    n.update(loadref=fx(qm.utilization, 1), loadfactor=f"{qm.load_factor:.0f}",
             cvuniform=f"{qu.service_cv:.0f}", cvmixed=f"{qm.service_cv:.0f}",
             varuniform=f"{qu.variability_factor:.0f}", varmixed=f"{qm.variability_factor:.0f}",
             waituniform=f"{qu.wait_ratio:.0f}", waitmixed=f"{qm.wait_ratio:.0f}")

    # Slowdown, the chapter's own example (objectives section): a one-hour job that waits nine hours
    slow_run, slow_wait = 1, 9                           # chapter inputs (hours), typed in its prose
    n.update(slowrun=f"{slow_run}", slowwait=f"{slow_wait}", slowdown=f"{(slow_wait + slow_run) / slow_run:.0f}")

    # Check yourself Q1, ASSUMED: the mixed queue pushed to 90 percent offered load
    q1 = qm.variant(name="ASSUMED: mixed queue at 90 percent", description="study sheet question",
                    utilization=0.9)
    near(round(q1.load_factor) * round(qm.variability_factor), q1.wait_ratio, 1e-6)
    n.update(qload=fx(q1.utilization, 1), qloadfactor=f"{q1.load_factor:.0f}", qwait=f"{q1.wait_ratio:.0f}",
             qslow=f"{q1.wait_ratio + 1:.0f}")            # slowdown of a job of mean length: (W + T) / T, T = mean

    # ---- DRF card (chapter: lego-drf-allocation, DrfTwoTenantExample)
    d = SC.DrfTwoTenantExample
    dr = d.result
    cap = d.capacity
    ta, tb = dr["tenants"]["A"], dr["tenants"]["B"]
    n.update(drfabundles=f"{dr['bundle_counts']['A']}", drfbbundles=f"{dr['bundle_counts']['B']}",
             drfshare=fx(ta["dominant_share"], 2), drfbcpu=fx(tb["shares"]["cpu_cores"], 3))
    assert ta["dominant_share"] == tb["dominant_share"]

    # Check yourself Q2, ASSUMED: tenant B replaced by a preprocessing-heavy tenant C on the same cluster
    cb = {"accelerators": 1.0, "cpu_cores": 8.0, "memory_gb": 16.0, "network_gbps": 10.0}
    q2 = d.variant(name="ASSUMED: tenant A and a CPU-heavy tenant C", description="study sheet question",
                   bundles={"A": d.bundles["A"], "C": cb})
    r2 = q2.result
    a_b, c_b = d.bundles["A"], cb
    a_d = max(a_b[k] / cap[k] for k in cap)
    c_d = max(c_b[k] / cap[k] for k in cap)
    assert max(cap, key=lambda k: c_b[k] / cap[k]) == "cpu_cores"
    ca, cc = r2["bundle_counts"]["A"], r2["bundle_counts"]["C"]
    near(r2["tenants"]["A"]["dominant_share"], ca * a_d, 1e-9)
    near(r2["tenants"]["C"]["dominant_share"], cc * c_d, 1e-9)
    assert r2["used"]["cpu_cores"] == cap["cpu_cores"]  # cores saturate first
    idle_gpus = cap["accelerators"] - r2["used"]["accelerators"]
    n.update(capgpus=f"{cap['accelerators']:.0f}", capcores=f"{cap['cpu_cores']:.0f}",
             capmem=f"{cap['memory_gb']:.0f}", capnet=f"{cap['network_gbps']:.0f}",
             agpu=f"{a_b['accelerators']:.0f}", acores=f"{a_b['cpu_cores']:.0f}", amem=f"{a_b['memory_gb']:.0f}",
             anet=f"{a_b['network_gbps']:.0f}",
             cgpu=f"{c_b['accelerators']:.0f}", ccores=f"{c_b['cpu_cores']:.0f}", cmem=f"{c_b['memory_gb']:.0f}",
             cnet=f"{c_b['network_gbps']:.0f}",
             ashare=fx(a_d, 3), cshare=fx(c_d, 2), abundles=f"{ca}", cbundles=f"{cc}",
             qdrfshare=fx(r2["tenants"]["A"]["dominant_share"], 2), qgpusused=f"{r2['used']['accelerators']:.0f}",
             qgpusidle=f"{idle_gpus:.0f}", acorestotal=f"{ca * a_b['cpu_cores']:.0f}",
             ccorestotal=f"{cc * c_b['cpu_cores']:.0f}")
    near(r2["tenants"]["A"]["dominant_share"], r2["tenants"]["C"]["dominant_share"], 1e-9)

    # ---- Placement table + Q4 (chapter: FleetTopologyInterconnect, TensorParallelExchange1GB)
    nv = mag(Hardware.Cloud.H100.nvlink.bandwidth_per_direction, "GB/s")
    ndr = mag(Systems.Fabrics.InfiniBand_NDR.bandwidth, "GB/s")
    o2, o4 = Systems.Fabrics.InfiniBand_NDR_Oversub2to1, Systems.Fabrics.InfiniBand_NDR_Oversub4to1
    spine_hi = mag(o2.bandwidth, "GB/s") / o2.oversubscription_ratio
    spine_lo = mag(o4.bandwidth, "GB/s") / o4.oversubscription_ratio
    t = SC.TensorParallelExchange1GB
    tgb = mag(t.payload, "GB")
    tnv, tib = mag(t.result.times["NVLink"], "ms"), mag(t.result.times["InfiniBand NDR"], "ms")
    near(tgb / nv * 1000, tnv, 1e-9)
    near(tgb / ndr * 1000, tib, 1e-9)
    n.update(nvlink=f"{nv:.0f}", ndr=f"{ndr:.0f}", spinelo=fx(spine_lo, 1), spinehi=f"{spine_hi:.0f}",
             tpgb=f"{tgb:.0f}", tpnv=fx(tnv, 1), tpib=f"{tib:.0f}", tpslow=f"{tib / tnv:.0f}")

    # ---- Utilization tiers (chapter: UtilizationTiersExample, ReferenceStats.UtilizationTierProfile)
    u = ReferenceStats.UtilizationTierProfile
    al, ac, pr = int(u.AllocatedGpus), int(u.ActiveEquivalents), int(u.ProductiveEquivalents)
    n.update(tieralloc=f"{al}", tieractive=f"{ac}", tierprod=f"{pr}",
             activepct=f"{100 * ac / al:.0f}", prodpct=f"{100 * pr / al:.0f}")

    # ---- Spot card, worked example (chapter: SpotTrainingEconomics, Llama2_70BSpotRun512)
    s = SC.Llama2_70BSpotRun512
    od, sp = mag(s.on_demand_rate, "USD/hour"), mag(s.spot_rate, "USD/hour")
    rh, ck, ints = s.run_hours, s.checkpoint_extra_hours, s.interruptions
    rst = mag(s.restart_per_interruption, "hour")
    billed = s.spot_total_hours
    near(rh + ck + ints * rst, billed, 1e-9)
    ratio = billed / rh
    eff = sp * ratio
    near(round(sp * round(ratio, 2), 2), round(eff, 2), 1e-9)   # printed 0.70 x printed ratio -> printed price
    near(1 - eff / od, s.savings_share, 1e-9)                    # effective price reproduces the bill's saving
    near(s.gpus * billed * sp, s.spot_total_cost, 1e-6)
    near(s.gpus * rh * od, s.on_demand_cost, 1e-6)
    n.update(ondemand=f"{od:.0f}", spotrate=fx(sp, 2), discountpct=f"{100 * (1 - sp / od):.0f}",
             wmodel="Llama-2-70B", wgpus=f"{s.gpus}", wdays=f"{s.run_days:.0f}", whours=f"{rh:.0f}",
             ckptpct=f"{100 * s.checkpoint_overhead_fraction:.0f}", ckpth=fx(ck, 1),
             perday=f"{s.interruptions_per_day}", wint=f"{ints}", restarth=fx(rst, 1), restarttotal=f"{ints * rst:.0f}",
             wbilled=fx(billed, 1), wratio=fx(ratio, 2), weff=fx(eff, 2),
             wodcost=f"{s.on_demand_cost:,.0f}", wspotcost=f"{s.spot_total_cost:,.0f}",
             wsavepct=f"{100 * s.savings_share:.0f}", wlongerpct=f"{100 * s.longer_share:.0f}",
             # thousands of dollars, for the to-scale bars
             wodk=f"{s.on_demand_cost / 1000:.0f}", wspotk=f"{s.spot_total_cost / 1000:.0f}",
             drawod=fx(s.on_demand_cost / 1000, 3), drawbase=fx(s.spot_base_cost / 1000, 3),
             drawover=fx((s.spot_total_cost - s.spot_base_cost) / 1000, 3))

    # ---- Problem to finish (chapter: SpotDiscountFallacy, SpotInterruptionEffectiveCost)
    f = SC.SpotInterruptionEffectiveCost
    assert mag(f.spot_rate, "USD/hour") == sp and mag(f.on_demand_rate, "USD/hour") == od
    yint = mag(f.retained_interval, "hour")
    ylost = mag(f.lost_per_interruption, "hour")
    ybilled = yint + ylost
    near(ybilled / yint, f.billed_over_retained, 1e-12)
    near(sp * f.billed_over_retained, f.effective_rate, 1e-12)
    # break-even retained interval t: spot * (t + lost) / t = on-demand
    be = ylost * sp / (od - sp)
    near(sp * (be + ylost) / be, od, 1e-9)
    near(round(ylost, 2) * round(sp, 2) / round(od - sp, 2), be, 1e-9)
    n.update(yckpt=f"{mag(f.checkpoint_overhead, 'minute'):.0f}", yrestart=f"{mag(f.restart_overhead, 'minute'):.0f}",
             yint=f"{yint:.0f}", ylosth=fx(ylost, 2), ybilled=fx(ybilled, 2), yratio=fx(ybilled / yint, 3),
             yeff=fx(f.effective_rate, 2), ysavepct=f"{100 * f.effective_savings:.0f}",
             gap=fx(od - sp, 2), beh=fx(be, 2), bemin=f"{be * 60:.0f}")

    # ---- Mistake: half a gang (chapter: gang-scheduling definition and ClusterEconomicsContext)
    gang = SC.DenseLLMPretrainingReference.system.total_accelerators
    idle = SC.Training10KIdleHour
    near(idle.gpus * mag(idle.device_hour_rate, "USD/hour"), idle.cost, 1e-9)
    n.update(gang=f"{gang:,}", gangheld=f"{gang // 2}", cluster=f"{idle.gpus:,}", clusterhour=f"{idle.cost:,.0f}")

    # ---- Mistake: one utilization number (chapter: UtilizationSnapshotPitfall, FleetUtilizationSnapshot)
    sn = ReferenceStats.FleetUtilizationSnapshot
    asg, inc, inw = float(sn.AssignedShare), float(sn.IncompleteGangShare), float(sn.InputWaitShare)
    n.update(assigned=f"{100 * asg:.0f}", incomplete=f"{100 * inc:.0f}", inputwait=f"{100 * inw:.0f}",
             complete=f"{100 * (asg - inc):.0f}", executing=f"{100 * (asg - inc - inw):.0f}")

    # ---- Mistake: quotas sized to peaks (chapter: fallacy "Setting static quotas based on peak demand";
    # the peaks and the shared cluster size are the chapter's own prose inputs)
    peak_a, peak_b, shared = 500, 300, 600
    assert shared >= max(peak_a, peak_b)
    n.update(peaka=f"{peak_a}", peakb=f"{peak_b}", peaksum=f"{peak_a + peak_b}", shared=f"{shared}",
             extra=f"{peak_a + peak_b - shared}", extrapct=f"{100 * (peak_a + peak_b - shared) / (peak_a + peak_b):.0f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
