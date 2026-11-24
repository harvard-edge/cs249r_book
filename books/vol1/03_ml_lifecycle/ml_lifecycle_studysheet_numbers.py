"""Numbers for the ML Lifecycle study sheet (Volume I, Chapter 3).

Every value the sheet prints is computed here from mlsysim's scenarios, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to ml_lifecycle_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
"""
from pathlib import Path

from mlsysim import load_series
from mlsysim.engine.scenarios import Scenarios as SC


def g(x):
    return f"{x:g}"


def pct(x, nd=0):
    return f"{x * 100:.{nd}f}"


def numbers() -> dict:
    n = {}

    # Worked example: the day-151 memory limit (constraint-propagation cell)
    lc = SC.RetinopathyScreenerLateMemoryLimit
    need_mb, budget_mb = lc.model_memory.to("MB").magnitude, lc.memory_budget.to("MB").magnitude
    tasks = {k: v.to("day").magnitude for k, v in lc.repair_tasks.items()}
    prep, train, val, pack = (tasks[k] for k in ("Revise preparation", "Train new weights",
                                                 "Repeat validation", "Package the release"))
    repair = lc.repair_elapsed.to("day").magnitude
    assert repair == prep + train + val + pack
    built_days = lc.handoff_day - lc.request_day + 1                  # days 1..150 built without the limit
    n.update(needgb=g(lc.model_memory.to("GB").magnitude), budget=g(budget_mb), over=g(need_mb / budget_mb),
             wprep=g(prep), wtrain=g(train), wval=g(val), wpack=g(pack), wrepair=g(repair),
             dayfound=g(lc.discovery_day), dayhand=g(lc.handoff_day), builtdays=g(built_days),
             sensday=pct(lc.validation_sensitivity[120]), wasteratio=f"{built_days / repair:.0f}")

    # Release floors and the measured slices (RetinopathyScreenerRequirements)
    rq = SC.RetinopathyScreenerRequirements
    n.update(sensfloor=pct(rq.sensitivity_floor), specfloor=pct(rq.specificity_floor),
             sensall=pct(rq.slice_sensitivity["all clinic photos"]),
             senscat=pct(rq.slice_sensitivity["cataract patients"]))

    # Your turn: ASSUMED durations for targeted collection and grading; train/validate/package from the chapter
    ycollect, ygrade = 10, 5                                           # ASSUMED days
    n.update(ycollect=g(ycollect), ygrade=g(ygrade), yrepair=g(ycollect + ygrade + prep + train + val + pack),
             ycatgap=g(round((rq.sensitivity_floor - rq.slice_sensitivity["cataract patients"]) * 100)))

    # Formula card: the busy clinic's daily upload (bandwidth-vs-compute-calc)
    up = SC.RuralClinicRetinalUpload
    rate_mbs = up.uplink.to("MB/s").magnitude
    photo_mb = up.file_size.to("MB").magnitude
    n.update(cpat=g(up.cases_per_day), cphoto=g(up.files_per_case), cmb=g(photo_mb),
             cdaily=f"{up.daily_volume.to('MB').magnitude:,.0f}", cmbit=g(up.uplink.to("megabit/s").magnitude),
             crate=g(rate_mbs), csec=f"{up.upload_time.to('s').magnitude:,.0f}",
             chours=f"{up.upload_time.to('hour').magnitude:.1f}", cshift=g(up.window.to("hour").magnitude),
             cshare=pct(up.window_share), cshiftsec=f"{up.window.to('s').magnitude:,.0f}", cone=g(photo_mb / rate_mbs), csumkb=g(up.summary_size.to("KB").magnitude),
             cred=f"{up.reduction:,.0f}")

    # Q1: ASSUMED clinic on a slower link, same photo size and summary size as the chapter
    qpat, qmbit = 60, 1.0                                              # ASSUMED patients/day, Mb/s
    qrate = qmbit / 8                                                  # MB/s
    qdaily = qpat * up.files_per_case * photo_mb
    qsec = qdaily / qrate
    qshare = qsec / up.window.to("s").magnitude
    qsum_kb = qpat * up.summary_size.to("KB").magnitude
    n.update(qpat=g(qpat), qmbit=g(qmbit), qrate=g(qrate), qdaily=f"{qdaily:,.0f}", qsec=f"{qsec:,.0f}",
             qhours=f"{qsec / 3600:.1f}", qshare=f"{qshare * 100:.0f}", qone=g(photo_mb / qrate),
             qsumkb=f"{qsum_kb:,.0f}", qred=f"{qdaily * 1000 / qsum_kb:,.0f}")

    # Q2: ASSUMED share of screened patients with referable disease
    prev = 0.08
    n.update(prev=pct(prev), alwaysacc=pct(1 - prev), alwaysspec=pct(1.0))

    # Mistake 1: Thai clinics (thai-ungradable-share cell; Beede et al. 2020 counts, as the chapter loads them)
    rejected, submitted = 393, 1838
    n.update(thairej=f"{rejected:,}", thaisub=f"{submitted:,}", thaipct=f"{rejected / submitted * 100:.0f}")

    # Mistake 2: the iteration tax (iteration-tax-calc)
    it = SC.SlowVersusFastIteration
    n.update(slowstart=g(it.slow_start), faststart=g(it.fast_start), slowruns=g(it.slow_runs),
             fastruns=g(it.fast_effective_runs), slowfinal=g(it.slow_final), fastfinal=g(it.fast_final),
             ideas=f"{it.ideas_ratio:.1f}", startgap=g(it.slow_start - it.fast_start),
             slowgain=g(it.slow_gain), fastgain=g(it.fast_gain), ceiling=g(it.ceiling), weeks=g(it.window.to("week").magnitude))
    # Where the hours go (data-scientist-time-allocation cell; CrowdFlower 2016 survey CSV)
    survey = {r["activity"]: int(r["pct_respondents"])
              for r in load_series("vol1/03_ml_lifecycle/data/crowdflower_2016_time_survey.csv").entries}
    n["datawork"] = g(survey["Cleaning and organizing data"] + survey["Collecting datasets"])
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
