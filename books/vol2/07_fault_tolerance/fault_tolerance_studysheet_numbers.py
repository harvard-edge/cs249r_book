"""Numbers for the Fault Tolerance study sheet (Volume II, Chapter 7).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to fault_tolerance_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Storage sizes are decimal (TB = 10^12 bytes, GB = 10^9 bytes), as in the chapter.
"""
import math
from pathlib import Path

from mlsysim import Systems, hour
from mlsysim.engine.scenarios import Scenarios


def c(x):
    return f"{x:,.0f}"


def numbers() -> dict:
    n = {}

    # ---- Fleet failure card and worked example: the chapter's four-subsystem cascade
    # (Scenarios.FullSystemFailures10K, cell FailureCascadeExample).
    cas = Scenarios.FullSystemFailures10K
    r = cas.result
    gpus = cas.n_devices
    per_node = cas.devices_per_node
    mttf = {k: v.to("hour").magnitude for k, v in cas.component_mttf.items()}
    assert r.component_rates["power"] == 0          # 4+2 redundant supplies: no series term
    rate_gpu = r.component_rates["gpu"] * 1e6       # failures per million GPU-hours
    rate_pcie = r.component_rates["pcie"] * 1e6
    rate_net = r.component_rates["network"] * 1e6
    rate_total = r.device_rate * 1e6
    sys_rate = round(r.system_rate, 4)              # the chapter displays 0.2583 and inverts that
    mtbf_h = 1 / sys_rate
    per_day = 24 * sys_rate
    gpu_only_mtbf = mttf["gpu"] / gpus
    n.update(fleetgpus=c(gpus), gpusnode=c(per_node), mttfgpu=c(mttf["gpu"]), mttfpcie=c(mttf["pcie"]),
             mttfnic=c(mttf["network"]), rategpu=f"{rate_gpu:.0f}", ratepcie=f"{rate_pcie:.0f}",
             ratenet=f"{rate_net:.2f}", ratetotal=f"{rate_total:.2f}", sysrate=f"{sys_rate:.4f}",
             mtbfcascade=f"{mtbf_h:.2f}", perday=f"{per_day:.1f}", gpuonlymtbf=f"{gpu_only_mtbf:.0f}")

    # ---- Young-Daly card and worked example: 175B checkpoint on the 10,000-GPU cluster
    # (Scenarios.Gpt3CheckpointInterval10K, cells FaultToleranceCheckpointRecap and YoungDalyOptimal).
    yd = Scenarios.Gpt3CheckpointInterval10K
    yr = yd.result
    ckpt_tb = yd.checkpoint_bytes.to("TB").magnitude
    bw = float(Systems.Reliability.Recovery.checkpoint_write_bw_gbs)
    t_write = yr.write_time.to("second").magnitude
    assert abs(t_write - ckpt_tb * 1e3 / bw) < 1e-6
    m_s = yr.mtbf.to("second").magnitude
    assert abs(yr.mtbf.to("hour").magnitude - float(f"{mtbf_h:.2f}")) < 1e-9   # same MTBF as the cascade
    tau = math.sqrt(2 * t_write * m_s)
    assert abs(tau - yr.interval.to("second").magnitude) < 1e-6
    waste = math.sqrt(2 * t_write / m_s)
    hour_s = 3600.0
    w_write_hourly = t_write / hour_s
    w_rework_hourly = hour_s / (2 * m_s)
    assert abs((w_write_hourly + w_rework_hourly) - yd.waste_at(1 * hour)) < 1e-9
    n.update(ckpttb=f"{ckpt_tb:.2f}", bpp=f"{yd.bytes_per_param.magnitude:.0f}",
             writebw=f"{bw:.0f}", twrite=f"{t_write:.1f}", mtbfsec=c(m_s), radicand=c(2 * t_write * m_s),
             tauopt=f"{tau:.0f}", tauoptmin=f"{tau / 60:.1f}", wasteopt=f"{waste * 100:.1f}",
             wastehalf=f"{waste * 50:.1f}", hourwrite=f"{w_write_hourly * 100:.2f}",
             hourrework=f"{w_rework_hourly * 100:.1f}", hourwaste=f"{(w_write_hourly + w_rework_hourly) * 100:.1f}")

    # Front picture: waste W(T) = T_write/T + T/(2 MTBF), T from 2 to 70 min.
    # x = 24 + 3 * minutes (pt); y = 3.2 * percent (pt), clipped at 25 percent.
    def px(mins):
        return 24 + 3 * mins

    def py(frac):
        return 3.2 * frac * 100

    def w(mins):
        t = mins * 60
        return t_write / t + t / (2 * m_s)

    pts = []
    for i in range(0, 69 * 4 + 1):
        mins = 2 + i / 4
        if w(mins) <= 0.25:
            pts.append(f"({px(mins):.1f},{py(w(mins)):.1f})")
    n["wastecurve"] = " ".join(pts)
    n.update(xopt=f"{px(tau / 60):.1f}", yopt=f"{py(waste):.1f}", xhour=f"{px(60):.1f}",
             yhour=f"{py(w(60)):.1f}")

    # Your turn: cluster doubles, so (independent failures) the system MTBF halves; write time fixed.
    m2 = m_s / 2
    tau2 = math.sqrt(2 * t_write * m2)
    waste2 = math.sqrt(2 * t_write / m2)
    n.update(mtbfdouble=c(m2), gpusdouble=c(2 * gpus), taudouble=f"{tau2:.0f}", taudoublemin=f"{tau2 / 60:.1f}",
             wastedouble=f"{waste2 * 100:.1f}", factortau=f"{tau / tau2:.2f}", factorwaste=f"{waste2 / waste:.2f}")

    # ---- Recovery-budget card: the chapter's cold-restart terms (Scenarios.Gpt3RecoveryBudget1K, cell RecoveryBudget)
    rb = Scenarios.Gpt3RecoveryBudget1K.result
    det, rst, load, warm = (x.to("second").magnitude for x in (rb.detect, rb.restart, rb.reload, rb.warmup))
    total = det + rst + load + warm
    assert abs(total - rb.total.to("second").magnitude) < 1e-9
    n.update(tdetect=f"{det:.0f}", trestart=f"{rst / 60:.0f}", tload=f"{load:.1f}", twarmup=f"{warm / 60:.0f}",
             trecovery=f"{total / 60:.1f}", trecoverysec=f"{total:.1f}")

    # ---- Q1: ASSUMED hot spare cuts the restart term to one minute; cascade cadence of failures.
    rst_hot = 60.0
    total_hot = det + rst_hot + load + warm
    day_cold = total / 60 * float(f"{per_day:.1f}")
    day_hot = total_hot / 60 * float(f"{per_day:.1f}")
    saved_min = day_cold - day_hot
    saved_gpu_h = saved_min / 60 * gpus
    spare_gpu_h = per_node * 24
    n.update(trestarthot=f"{rst_hot / 60:.0f}", trecoveryhot=f"{total_hot / 60:.1f}", trecoveryhotsec=f"{total_hot:.1f}",
             daycold=f"{day_cold:.1f}", dayhot=f"{day_hot:.1f}", savedmin=f"{saved_min:.1f}",
             savedgpuh=c(saved_gpu_h), sparegpuh=c(spare_gpu_h), sparetimes=f"{saved_gpu_h / spare_gpu_h:.0f}")

    # ---- Replica card: chapter's 99 percent replica (Scenarios.ServingReplica99, x2, x3; cell ReplicaAvailability)
    s1, s2, s3 = Scenarios.ServingReplica99, Scenarios.ServingReplicas99x2, Scenarios.ServingReplicas99x3
    n.update(replicaone=f"{s1.availability * 100:.0f}",
             downone=f"{s1.replicated_downtime_per_year.to('hour').magnitude / 24:.2f}",
             downtwo=f"{s2.replicated_downtime_per_year.to('minute').magnitude:.1f}",
             downthree=f"{s3.replicated_downtime_per_year.to('second').magnitude:.1f}")
    # Q2: ASSUMED 99.5 percent replicas, target 99.99 percent.
    a, target = 0.995, 0.9999
    reps = next(k for k in range(1, 10) if 1 - (1 - a) ** k >= target)
    n.update(qavail=f"{a * 100:.1f}", qtarget=f"{target * 100:.2f}", qmiss=f"{(1 - a):.3f}", qreps=f"{reps}",
             qrepsavail=f"{(1 - (1 - a) ** reps) * 100:.4f}", qoneavail=f"{a * 100:.1f}")

    # ---- Q3: ASSUMED one GPU of the chapter's 1,024-GPU straggler job throttles to 80 percent speed
    # (Scenarios.HalfSpeedStraggler1K supplies the job, step time and GPU-hour price).
    st = Scenarios.HalfSpeedStraggler1K
    step = st.step_time.to("second").magnitude
    speed = 0.80
    slow = step / speed
    keep = step / slow
    price = st.device_hour_rate.to("USD/hour").magnitude
    idle = st.n_devices * (1 - keep) * price
    n.update(qgpus=c(st.n_devices), qstep=f"{step:.0f}", qspeed=f"{speed * 100:.0f}", qslow=f"{slow:.2f}",
             qkeep=f"{keep * 100:.0f}", qlost=f"{(1 - keep) * 100:.0f}", qprice=f"{price:.0f}", qidle=c(idle))

    # ---- Mistake: hourly checkpoints. On the 1,000-GPU row (GPU-only MTBF 50 h) with a 5-minute write
    # (Scenarios.FiveMinuteCheckpoint1K, cell YoungDalyIntuitionFallacy) the optimum is far above an hour.
    f5 = Scenarios.FiveMinuteCheckpoint1K
    n.update(onekgpus=c(round(mttf["gpu"] / f5.mtbf.to("hour").magnitude)),
             onekmtbf=f"{f5.mtbf.to('hour').magnitude:.0f}", onekwrite=f"{f5.write_time.to('minute').magnitude:.0f}",
             oneklong=f"{f5.result.interval.to('minute').magnitude:.0f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
