"""Numbers for the Sustainable AI study sheet (Volume II, Chapter 15).

Every value the sheet prints is computed here from mlsysim and the chapter's data, as in the
chapter's LEGO cells, and written as \\setnum{key}{value} to sustainable_ai_studysheet_numbers.tex
next to this file. Inputs the sheet chooses for its own questions are marked ASSUMED (built as
variants of the chapter's scenarios). A printed result is computed from the rounded values the sheet
prints beside it, and checked against the unrounded model value.

Units: energy in kWh (decimal), carbon in kg or t (metric tonnes) of CO2e, grid factors in g/kWh.
No accelerator memory budget appears on this sheet, so the GiB rule does not arise.
"""
import csv
from pathlib import Path

from mlsysim import Hardware, Infrastructure, Systems
from mlsysim import Scenarios as SC

HERE = Path(__file__).resolve().parent
DAYS_PER_YEAR = 365
HOURS_PER_DAY = 24


def fx(x, digits):
    return f"{x:,.{digits}f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def mag(q, unit):
    return q.to(unit).magnitude


def prompt_rows():
    with open(HERE / "data" / "gemini_prompt_energy.csv") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def numbers() -> dict:
    n = {}
    n.update(daysperyear=f"{DAYS_PER_YEAR}", hoursperday=f"{HOURS_PER_DAY}")

    us, qc = Infrastructure.Grids.US_Avg, Infrastructure.Grids.Quebec
    us_ci, qc_ci = us.carbon_intensity_g_kwh, qc.carbon_intensity_g_kwh
    n.update(usci=f"{us_ci:.0f}", qcci=f"{qc_ci:.0f}", uspue=fx(us.pue, 2),
             usciskg=fx(us_ci / 1000, 3), qccikg=fx(qc_ci / 1000, 3))

    # ---- Rack: GPU TDP subtotal vs IT load (chapter: RackPowerBudget, RackItSubtotal)
    rack = Systems.Racks.DGX_H100_4Node
    gpu_kw, it_kw = mag(rack.accelerator_power, "kW"), mag(rack.it_power, "kW")
    it_ratio = it_kw / gpu_kw
    n.update(rackgpu=fx(gpu_kw, 1), rackitload=fx(it_kw, 1), itratio=fx(it_ratio, 3),
             itabovepct=fx(100 * (it_ratio - 1), 1))

    # ---- Embodied anchors (chapter: AcceleratorEmbodiedFootprints)
    a100_kg, h100_kg = Hardware.Cloud.A100.embodied_carbon_kg, Hardware.Cloud.H100.embodied_carbon_kg
    life_ext = Infrastructure.Pricing.Capital.AcceleratorExtendedServiceLifeYears.rate
    life_std = Infrastructure.Pricing.Capital.AcceleratorServiceLifeYears.rate
    n.update(aembodied=f"{a100_kg:.0f}", hembodied=f"{h100_kg:.0f}", lifeext=f"{life_ext:.0f}",
             lifestd=f"{life_std:.0f}")

    # ---- Worked example (chapter: TrainingEmissions, Scenarios.Lab64A100TrainingFootprint)
    s = SC.Lab64A100TrainingFootprint
    r = s.result
    gpus = Systems.Clusters.Lab_64_A100.total_accelerators
    tdp_w = mag(Hardware.Cloud.A100.tdp, "W")
    hours = mag(r.duration, "hour")
    days = hours / HOURS_PER_DAY
    acc = mag(r.accelerator_energy, "kWh")
    it = mag(r.it_energy, "kWh")
    fac = mag(r.facility_energy, "kWh")
    near(gpus * tdp_w * hours / 1000, acc, 1e-9)
    near(acc * it_ratio, it, 1e-6)
    near(it * us.pue, fac, 1e-6)
    acc_p, it_p, fac_p = round(acc, 1), round(round(acc, 1) * round(it_ratio, 3)), None
    near(acc_p * round(it_ratio, 3), it, 0.5)                       # printed 8,601.6 x 1.375 -> 11,827
    fac_p = round(it_p * round(us.pue, 2))
    assert fac_p == round(fac), (fac_p, fac)                           # printed 11,827 x 1.12 -> 13,246
    op_us = mag(r.carbon["US average"], "kg")
    op_qc = mag(r.carbon["Quebec"], "kg")
    near(fac * us_ci / 1000, op_us, 1e-6)
    assert round(fac_p * us_ci / 1000) == round(op_us)                 # 13,246 x 0.429 -> 5,683 kg
    emb = mag(r.embodied_allocated, "kg")
    per_day = a100_kg / (life_ext * DAYS_PER_YEAR)
    near(gpus * days * per_day, emb, 1e-9)
    assert round(gpus * days * round(per_day, 3)) == round(emb)        # 64 x 14 x 0.103 -> 92 kg
    share_us = emb / (op_us + emb)
    near(share_us, r.embodied_share["US average"], 1e-12)
    assert round(100 * round(emb) / (round(op_us) + round(emb)), 1) == round(100 * share_us, 1)
    n.update(wgpus=f"{gpus}", wtdp=f"{tdp_w:.0f}", whours=f"{hours:.0f}", wdays=f"{days:.0f}",
             wacc=fx(acc, 1), wituse=f"{it_p:,}", wfac=f"{fac_p:,}", wopus=f"{op_us:,.0f}",
             woptons=fx(op_us / 1000, 1), wperday=fx(per_day, 3), wdevdays=f"{gpus * days:.0f}",
             wemb=f"{emb:.0f}", wsharepct=fx(100 * share_us, 1), wmfgtons=fx(gpus * a100_kg / 1000, 1))

    # ---- Finish: same facility energy on Quebec's grid
    assert round(fac_p * qc_ci / 1000) == round(op_qc)                 # 13,246 x 0.020 -> 265 kg
    share_qc = emb / (op_qc + emb)
    near(share_qc, r.embodied_share["Quebec"], 1e-12)
    assert round(100 * round(emb) / (round(op_qc) + round(emb))) == round(100 * share_qc)
    n.update(fopqc=f"{op_qc:.0f}", ftotal=f"{round(op_qc) + round(emb):.0f}",
             fsharepct=f"{100 * share_qc:.0f}", fratio=fx(round(op_us) / round(op_qc), 1))

    # ---- Q1, ASSUMED: two million prompts a day at the chapter's measured per-prompt energy
    # (PromptEnergyBoundary: 0.24 Wh full stack vs 0.10 Wh accelerator only), against the run above.
    rows = prompt_rows()
    full = sum(float(x["comprehensive_wh"]) for x in rows if x["comprehensive_counted"] == "1")
    narrow = sum(float(x["existing_wh"]) for x in rows if x["existing_counted"] == "1")
    near(full, 0.24, 1e-9)
    near(narrow, 0.10, 1e-9)
    prompts_m = 2                                                       # ASSUMED, millions of prompts per day
    q = SC.TenMillionQueriesAgainstLab64Training.variant(
        name="ASSUMED: two million full-stack prompts a day", description="study sheet question",
        requests_per_day=prompts_m * 1e6, energy_per_request=SC.TenMillionQueriesAgainstLab64Training
        .energy_per_request * (full / 1.0))
    day_full = prompts_m * 1e6 * full / 1000                            # kWh per day
    day_narrow = prompts_m * 1e6 * narrow / 1000
    near(mag(q.result.daily_energy, "kWh"), day_full, 1e-6)
    near(q.result.crossover_days, fac / day_full, 1e-9)
    n.update(qprompts=f"{prompts_m}", qfull=fx(full, 2), qnarrow=fx(narrow, 2),
             qdayfull=f"{day_full:.0f}", qdaynarrow=f"{day_narrow:.0f}",
             qdaysfull=fx(fac_p / round(day_full), 1), qdaysnarrow=fx(fac_p / round(day_narrow), 1),
             qbound=fx(full / narrow, 1))

    # ---- Crossover card + Q2 (chapter: MatMulEnergyAnalysis, DecodeEnergySplit)
    d = SC.BatchOneDecodeWeightEnergy
    dr = d.result
    e_op, e_byte = mag(d.energy_per_flop, "pJ/flop"), mag(d.energy_per_byte, "pJ/byte")
    near(e_byte / e_op, dr.crossover_intensity, 1e-12)
    bpp, fpp = mag(d.bytes_per_param, "byte"), d.flops_per_param
    mv, cp = mag(dr.movement_energy, "pJ"), mag(dr.compute_energy, "pJ")
    near(bpp * e_byte, mv, 1e-9)
    near(fpp * e_op, cp, 1e-9)
    batch = 4                                                           # ASSUMED batch for Q2
    mv_b = mv / batch                                                   # one weight load shared by B tokens
    share_b = mv_b / (mv_b + cp)
    n.update(eop=f"{e_op:.0f}", ebyte=f"{e_byte:.0f}", crossover=f"{dr.crossover_intensity:.0f}",
             fpp=f"{fpp:.0f}", bpp=f"{bpp:.0f}", dint=f"{dr.intensity:.0f}", dmove=f"{mv:.0f}",
             dcomp=f"{cp:.0f}", dshare=f"{100 * dr.movement_share:.0f}", qbatch=f"{batch}",
             qint=f"{batch * fpp / bpp:.0f}", qmove=f"{mv_b:.0f}", qshare=f"{100 * share_b:.0f}",
             balance=f"{dr.balance_batch:.0f}")

    # ---- Q3, ASSUMED: the replacement saves 35 percent (chapter: H100ReplacementPayback at 50 percent)
    p = SC.H100ReplacementPayback
    pv = p.variant(name="ASSUMED: replacement saves 35 percent", description="study sheet question",
                   saved_fraction=0.35)
    h_kw = mag(Hardware.Cloud.H100.tdp, "kW")
    saved_kw = h_kw * pv.saved_fraction
    kwh_day = saved_kw * HOURS_PER_DAY
    kg_us, kg_qc = kwh_day * us_ci / 1000, kwh_day * qc_ci / 1000
    days_us, days_qc = h100_kg / kg_us, h100_kg / kg_qc
    near(days_us, mag(pv.result.payback["US average"], "day"), 1e-9)
    near(days_qc, mag(pv.result.payback["Quebec"], "day"), 1e-9)
    assert days_qc / DAYS_PER_YEAR > life_std                           # never repays inside a standard life
    n.update(rsave=f"{100 * pv.saved_fraction:.0f}", rtdp=f"{1000 * h_kw:.0f}", rsavedkw=fx(saved_kw, 3),
             rkwh=fx(kwh_day, 2), rkgus=fx(kg_us, 2), rkgqc=fx(kg_qc, 4),
             rdaysus=f"{h100_kg / round(kg_us, 2):.0f}", rweeksus=fx(h100_kg / round(kg_us, 2) / 7, 1),
             rdaysqc=f"{h100_kg / round(kg_qc, 4):,.0f}",
             ryearsqc=fx(h100_kg / round(kg_qc, 4) / DAYS_PER_YEAR, 1))

    # ---- Mistake: size by GPU TDP. ASSUMED 10 MW grid feed, US-average facility PUE
    feed_mw = 10
    it_cap_kw = feed_mw * 1000 / us.pue
    racks_tdp, racks_it = int(it_cap_kw // gpu_kw), int(it_cap_kw // it_kw)
    n.update(feed=f"{feed_mw}", itcap=fx(it_cap_kw / 1000, 2), rackstdp=f"{racks_tdp}",
             racksit=f"{racks_it}", racksover=f"{racks_tdp - racks_it}")

    # ---- Mistake: low PUE is green. ASSUMED pairing of registry grids and facility cooling classes
    iowa, france = Infrastructure.Grids.Iowa, Infrastructure.Grids.France
    pue_liq, pue_leg = Infrastructure.FacilityCooling.LiquidCooled.pue, Infrastructure.FacilityCooling.Legacy.pue
    g_a = pue_liq * iowa.carbon_intensity_g_kwh
    g_b = pue_leg * france.carbon_intensity_g_kwh
    n.update(pueliq=fx(pue_liq, 2), pueleg=fx(pue_leg, 2), iowaci=f"{iowa.carbon_intensity_g_kwh:.0f}",
             franceci=f"{france.carbon_intensity_g_kwh:.0f}", pga=f"{g_a:.0f}", pgb=f"{g_b:.0f}",
             pgratio=fx(round(g_a) / round(g_b), 1))

    # ---- Mistake: rebound (chapter: ReboundScenario, Scenarios.TokenPriceRebound)
    rb = SC.TokenPriceRebound
    unit_gain = rb.unit_price_before / rb.unit_price_after
    near(rb.volume_factor / unit_gain, rb.result.net_factor, 1e-9)
    n.update(pold=fx(rb.unit_price_before, 2), pnew=fx(rb.unit_price_after, 3), unitgain=f"{unit_gain:.0f}",
             volgrow=f"{rb.volume_factor:.0f}", netgrow=fx(rb.result.net_factor, 1))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
