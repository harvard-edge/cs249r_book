"""Numbers for the Data Selection study sheet (Volume I, Chapter 9).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to data_selection_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Derived results are computed from the rounded values the sheet prints beside them, so a
student who redoes the arithmetic from the page gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Hardware
from mlsysim.engine.scenarios import Scenarios


def c(x):
    return f"{x:,.0f}"


def numbers() -> dict:
    n = {}
    h100 = Hardware.Cloud.H100

    # ---- Data wall: GPT-2 XL against WebText (chapter: GPT2DataWall)
    wall = Scenarios.Gpt2XlWebTextCorpus
    tpp = wall.tokens_per_param                                     # fitted balance, about 20
    n.update(balance=c(tpp), flopsper=c(wall.flops_per_param_token),
             gptparams=f"{wall.parameters / 1e9:.1f}", gpttokens=f"{wall.unique_tokens.magnitude / 1e9:.0f}",
             gpttpp=f"{wall.unique_tokens_per_param:.1f}", gptpasses=c(round(wall.passes)))

    # ---- ICR: half-size coreset, one epoch (chapter: IcrCoresetComparison)
    icr = Scenarios.ResNet50HalfImageNetIcr
    n.update(icrratio=f"{icr.icr_ratio:.1f}", gainfull=f"{icr.gain_full:.0f}", gainsub=f"{icr.gain_subset:.1f}",
             icrkeep=f"{icr.keep_fraction * 100:.0f}")

    # ---- Front picture: one ImageNet epoch of ResNet-50 on one H100, by stage (chapter: ResNetEpochOnH100)
    ep = Scenarios.ResNet50HalfCoresetEpochH100
    t_epoch = round(ep.epoch_time.to("s").magnitude)                  # 126 s
    t_pruned = round(ep.pruned_epoch_time.to("s").magnitude)
    t_store = round(ep.storage_time.to("s").magnitude)
    t_link = round(ep.link_time.to("s").magnitude)
    t_host = round(ep.host_epoch_time.to("s").magnitude)
    n.update(pepoch=c(t_epoch), ppruned=c(t_pruned), pstore=c(t_store), plink=c(t_link), phost=c(t_host),
             archive=c(ep.archive_bytes.to("GB").magnitude), materialized=c(ep.materialized_bytes.to("GB").magnitude),
             nvmebw=c(h100.storage.bandwidth.to("GB/s").magnitude),
             hostrate=c(ep.host_rate.to("count/s").magnitude))

    # ---- Worked example: 10% coreset, 100 epochs, one H100 (chapter: SelectionInequalityCalc, SelectionBottleneckAnchor)
    cs = Scenarios.ResNet50TenPercentCoresetH100
    E = cs.epochs
    keep = cs.keep_fraction
    t_ep = round(cs.epoch_time.to("s").magnitude)                     # printed epoch time, s
    t_sub_ep = round(t_ep * keep, 1)                                  # 12.6 s
    t_score = t_ep / 3                                                # forward only: a third of fwd + bwd
    assert abs(cs.score_time().to("s").magnitude / cs.epoch_time.to("s").magnitude - 1 / 3) < 1e-9
    full_min = E * t_ep / 60
    sub_min = E * t_sub_ep / 60
    static_min = sub_min + t_score / 60
    resc_min = E * t_score / 60
    dyn_min = resc_min + sub_min
    n.update(wepochs=c(E), wkeep=f"{keep * 100:.0f}", wepoch=c(t_ep), wsubep=f"{t_sub_ep:.1f}", wscore=c(t_score),
             wfull=c(full_min), wsub=c(sub_min), wstatic=f"{static_min:.1f}",
             wstaticsave=f"{(1 - static_min / full_min) * 100:.0f}", wresc=c(resc_min), wdyn=c(dyn_min),
             wdynsave=f"{(1 - dyn_min / full_min) * 100:.0f}", wratio=f"{resc_min / sub_min:.1f}")
    # agree with the chapter's scenario (it prints 210 min, 57 percent)
    assert abs(cs.full_time.to("min").magnitude - full_min) < 1
    assert abs(cs.rescoring_saving_fraction - (1 - dyn_min / full_min)) < 0.01

    # ---- Your turn: same 100-epoch run, only the fraction kept changes to half (chapter: ResNetEpochOnH100 keep)
    E2, keep2 = E, ep.keep_fraction
    t_sub2 = t_ep * keep2
    full2 = E2 * t_ep
    static2 = E2 * t_sub2 + t_score
    dyn2 = E2 * (t_sub2 + t_score)
    n.update(yepochs=c(E2), ykeep=f"{keep2 * 100:.0f}", ysubep=c(t_sub2),
             yfull=c(full2), ystatic=c(static2), ystaticsave=f"{(1 - static2 / full2) * 100:.1f}",
             ydyn=c(dyn2), ydynsave=f"{(1 - dyn2 / full2) * 100:.1f}",
             yrescper=f"{t_score / t_sub2:.2f}")

    # ---- Mistake 3: synthetic-only training, accuracy over five recursive generations
    # (chapter: FpFallacyCalc.synthetic_gen1_acc_pct / synthetic_gen5_acc_pct; illustrative chapter scenario values,
    # not in mlsysim, copied from the chapter's LEGO cell)
    gen1, gen5, gens = 95.0, 78.0, 5
    n.update(m3first=c(gen1), m3last=c(gen5), m3gens=c(gens), m3drop=c(gen1 - gen5))

    # ---- Q1: ASSUMED 3B-parameter model, 30B unique tokens, 1.8e21 FLOPs budget
    p, d_unique, budget = 3e9, 30e9, 1.8e21
    have = d_unique / p
    processed = budget / (wall.flops_per_param_token * p)
    n.update(q1p=f"{p / 1e9:.0f}", q1d=f"{d_unique / 1e9:.0f}", q1c="1.8",  # budget mantissa, ASSUMED
             q1cexp=f"{int(math.log10(budget))}", q1budget=f"{budget / 10 ** int(math.log10(budget)):.1f}\\times 10^{{{int(math.log10(budget))}}}", q1have=f"{have:.0f}",
             q1need=f"{tpp * p / 1e9:.0f}", q1proc=f"{processed / 1e9:.0f}", q1passes=f"{processed / d_unique:.1f}")
    assert abs(float(n["q1c"]) * 10 ** int(n["q1cexp"]) - budget) < 1e6

    # ---- Q2: ASSUMED host with 8 cores at the chapter's loader-worker rate, against the H100's demand
    echo = Scenarios.ResNet50EchoingH100
    demand = echo.accelerator_rate.to("count/s").magnitude            # 10,100 images/s
    worker = Scenarios.ResNet50A100InputFeed.worker_rate.to("count/s").magnitude  # 375 images/s per core
    cores = 8                                                          # ASSUMED
    supply = cores * worker
    r = round(demand / supply, 2)
    n.update(q2demand=c(demand), q2worker=c(worker), q2cores=c(cores), q2supply=c(supply),
             q2r=f"{r:.2f}", q2e=c(math.ceil(r)), q2util=f"{supply / demand * 100:.0f}",
             chdemand=c(demand), chsupply=c(echo.host_rate.to("count/s").magnitude), chr=f"{echo.ratio:.2f}",
             chcores=c(echo.host_rate.to("count/s").magnitude / worker))

    # ---- Q3: ASSUMED pool of 500K images, 0.2% rare, 5% coreset, rare floor 300
    pool, rare_share, keep3, floor = 500_000, 0.002, 0.05, 300
    rare = pool * rare_share
    budget3 = pool * keep3
    expected = rare * keep3
    n.update(q3pool=c(pool), q3share=f"{rare_share * 100:.1f}", q3keep=c(keep3 * 100), q3floor=c(floor),
             q3rare=c(rare), q3budget=c(budget3), q3exp=c(expected), q3rest=c(budget3 - floor))

    # ---- Q4: ASSUMED dedup pipeline: $40,000 build + $8,000 compute, $6,000 saved per run
    build, once, save = 40_000, 8_000, 6_000
    inv = build + once
    n.update(q4build=c(build), q4once=c(once), q4inv=c(inv), q4save=c(save),
             q4even=c(inv / save), q4roione=f"{(save - inv) / inv * 100:.1f}",
             q4n=c(20), q4roin=c((20 * save - inv) / inv * 100))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
