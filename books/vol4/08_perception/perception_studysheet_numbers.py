"""Numbers for the Sensor Perception study sheet (Volume IV, Chapter 8).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to perception_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios


def f(x, nd):
    return f"{x:,.{nd}f}"


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    cam = rm.nav_camera

    # ---- Worked example: the navigation camera's latency waterfall (chapter: SensorInformationAge)
    nav = Scenarios.WarehouseNavCameraAge
    age = nav.result
    stages = [s[1].to("ms").magnitude for s in age.stages]
    expohalf, readout, transport, dma, isp, backbone, ipc = stages
    v = nav.speed.to("m/s").magnitude
    t_age = age.age.to("ms").magnitude
    assert abs(sum(round(s, 1) for s in stages) - t_age) < 1e-9
    assert abs(t_age - aisle.perception_age.to("ms").magnitude) < 1e-9
    travel = v * t_age  # mm, since m/s x ms = mm
    assert abs(travel - age.age_travel.to("mm").magnitude) < 1e-9
    sync = aisle.clock_conversion_bound.to("ms").magnitude
    threshold = t_age + sync
    assert abs(threshold - aisle.dispatch_age_limit.to("ms").magnitude) < 1e-9
    stop = Scenarios.WarehouseBaseStopFullDelay.result
    before = Scenarios.WarehouseChunkLease.result.running_total.to("mm").magnitude
    after = stop.required_distance.to("mm").magnitude
    clear = aisle.d_clear.to("mm").magnitude
    remain = stop.spare.to("mm").magnitude
    assert abs(round(before, 1) + round(travel, 1) - after) < 1e-6
    assert abs(clear - after - remain) < 1e-6
    n.update(expohalf=f(expohalf, 1), exposure=f(cam.exposure_time.to("ms").magnitude, 0), readout=f(readout, 1),
             transport=f(transport, 1), dma=f(dma, 1), isp=f(isp, 1), backbone=f(backbone, 1), ipc=f(ipc, 1),
             speed=f(v, 1), age=f(t_age, 1), travel=f(travel, 1), sync=f(sync, 1), threshold=f(threshold, 1),
             before=f(before, 1), after=f(after, 1), clearmm=f(clear, 0), remain=f(remain, 1))
    # picture: to-scale stage strip, 1 ms = 4 pt
    edges = [0.0]
    for s in stages:
        edges.append(edges[-1] + s)
    for name, i in (("pzero", 0), ("pexp", 1), ("pread", 2), ("pisp", 5), ("pback", 6), ("pdisp", 7)):
        n[name] = f(edges[i] * 4, 2)

    # ---- Front: arrival stamp and the timing slips at the drive limit (chapter: ArrivalStampOffset,
    # RollingShutterSkew, CaptureTimingScale; margin_clearance_slip_ladder)
    arrival = age.arrival_offset.to("ms").magnitude
    assert abs(arrival - (expohalf + readout + transport + dma)) < 1e-9
    arrivaltravel = age.arrival_travel.to("mm").magnitude
    shear = Scenarios.WarehouseNavCameraRollingShutter.result
    skew = shear.skew.to("mm").magnitude
    trigger = age.trigger_travel.to("mm").magnitude
    loc = aisle.delta_loc.to("mm").magnitude
    side = aisle.side_clearance.to("mm").magnitude
    n.update(arrival=f(arrival, 1), arrivaltravel=f(arrivaltravel, 1), skew=f(skew, 1), trigger=f(trigger, 1),
             trigtime=f(age.trigger_offset.to("ms").magnitude, 0), loc=f(loc, 0), side=f(side, 0),
             imu=f(rm.imu.sample_rate.to("Hz").magnitude, 0), frameperiod=f(age.frame_period.to("ms").magnitude, 1),
             fps=f(cam.frame_rate.to("Hz").magnitude, 0))
    # tile: lateral miss from a mount error at range (chapter: ExtrinsicErrorBudget)
    mount = Scenarios.WarehouseLongAisleMountError
    mountmm = mount.result.angle_lateral.to("mm").magnitude
    assert round(mountmm) == 262 and mountmm > side
    n.update(mountmm=f(mountmm, 0), mountdeg=f(mount.angle_error.to("degree").magnitude, 0),
             mountrange=f(mount.distance.to("m").magnitude, 0))
    # picture: slips against the localization allowance, 1 mm = 1.9 pt
    for k, val in (("btravel", travel), ("barrival", arrivaltravel), ("bskew", skew), ("btrigger", trigger), ("barallow", loc)):
        n[k] = f(val * 1.9, 2)

    # ---- Your turn: the same pipeline batching four frames (chapter: TemporalBatching)
    wait = age.batch_waits[0].to("ms").magnitude
    batched = age.batched_age.to("ms").magnitude
    assert abs(round(t_age, 1) + round(wait, 0) - batched) < 1e-6
    btravel = v * batched
    extra = btravel - round(travel, 1)
    n.update(batch=f(nav.batch_frames, 0), wait=f(wait, 0), batched=f(batched, 1), batchtravel=f(btravel, 1),
             extra=f(extra, 1), extrashare=f(100 * extra / round(remain, 1), 0))

    # ---- Q1: ASSUMED slower base; same readout (chapter: RollingShutterSkew, eq rolling-shutter skew)
    vslow = 0.8
    skewslow = vslow * round(readout, 1)
    n.update(slow=f(vslow, 1), skewslow=f(skewslow, 1), skewshare=f(100 * skewslow / loc, 0))

    # ---- Q2: illustrative stereo rig, ASSUMED repetitive siding doubles the disparity spread
    rig = Scenarios.WarehouseStereoRigDepthError
    b = rig.baseline.to("m").magnitude
    fpx = rig.focal_length_px
    sd = rig.disparity_sigma_px
    sd2 = 2 * sd
    zrack = aisle.d_clear.to("m").magnitude
    fb = fpx * b
    sigrack = zrack ** 2 * sd / fb * 1e3
    assert abs(sigrack - rig.result.sigma_target.to("mm").magnitude) < 1e-9
    sigrack2 = zrack ** 2 * sd2 / fb * 1e3
    zloc2 = math.sqrt((loc / 1e3) * fb / sd2)
    zloc = math.sqrt((loc / 1e3) * fb / sd)
    assert abs(zloc - rig.result.range_at_loc_allowance.to("m").magnitude) < 1e-9
    n.update(locm=f(loc / 1e3, 2), baseline=f(b, 1), focal=f(fpx, 0), disp=f(sd, 1), disptwo=f(sd2, 1), rack=f(zrack, 2),
             fb=f(fb, 0), sigrack=f(sigrack, 1), sigracktwo=f(sigrack2, 1), sigmeter=f(sigrack2 / 1e3, 4), zloc=f(zloc, 1), zloctwo=f(zloc2, 2))

    # ---- Q3: ASSUMED speed; the gap a frame stamped on arrival leaves, and the offset it implies
    vq = 1.2
    gap = vq * round(arrival, 1)
    n.update(qspeed=f(vq, 1), gap=f(gap, 1), gaptime=f(round(gap, 1) / vq, 1))

    # ---- Q4: ASSUMED dispatch-to-admission interval
    adm = 8.0
    current = round(t_age, 1) + adm + round(sync, 1)
    ctravel = v * current
    n.update(admit=f(adm, 0), current=f(current, 1), currenttravel=f(ctravel, 1),
             admextra=f(v * adm, 1))  # clock-bound travel is absorbed into delta_loc (chapter)
    n.update(batchedclk=f(batched + round(sync, 1), 1))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
