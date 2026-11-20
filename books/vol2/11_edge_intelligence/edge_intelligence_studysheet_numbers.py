"""Numbers for the Edge Intelligence study sheet (Volume II, Chapter 11).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to edge_intelligence_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
MB is decimal (10^6 bytes). Phone RAM is binary in the registry (GiB);
the sheet prints both forms for RAM and decimal MB for wire payloads.
"""
import math
from pathlib import Path

from mlsysim import Hardware, Platforms, ReferenceStats
from mlsysim.core.units import GB, MB, millisecond, second
from mlsysim.engine.scenarios import Scenarios


def mb(q):
    return q.to(MB).magnitude


def numbers() -> dict:
    n = {}

    # ---- Memory-budget card, bar and worked example: the chapter's 10M-parameter FP32 Adam update
    # (Scenarios.PhoneFullUpdateWorkingSet, cell EdgeTrainingMemoryStack).
    s = Scenarios.PhoneFullUpdateWorkingSet
    w, g, o, a = mb(s.weights), mb(s.gradients), mb(s.optimizer_state), mb(s.activations)
    total = w + g + o + a
    assert abs(total - mb(s.total)) < 1e-9
    assert abs(total / w - s.amplification) < 1e-9
    n.update(paramsmillion=f"{s.parameters / 1e6:.0f}", bytesparam=f"{s.bytes_per_param.to('byte').magnitude:.0f}",
             moments=f"{s.optimizer_moments}", actfactor=f"{s.activation_factor:.0f}",
             weightsmb=f"{w:.0f}", gradsmb=f"{g:.0f}", adammb=f"{o:.0f}", actmb=f"{a:.0f}", totalmb=f"{total:.0f}",
             amplification=f"{s.amplification:.0f}", floormult=f"{s.floor:.0f}", ceilingmult=f"{s.ceiling:.0f}")

    # Keyboard background allowance (ReferenceStats.EdgeAdaptationTierProfile, the chapter's operating point).
    prof = ReferenceStats.EdgeAdaptationTierProfile
    allowance = mb(prof.KeyboardBackgroundBudgetExample)
    ram = Platforms.Mobile.ram
    n.update(allowancemb=f"{allowance:.0f}", overmb=f"{total - allowance:.0f}",
             phoneramgib=f"{ram.to('GiB').magnitude:.0f}", phoneramgb=f"{ram.to(GB).magnitude:.1f}")

    # Front bar: 0.55 pt per MB (360 MB -> 198 pt); allowance as a dashed limit.
    scale = 0.55
    n.update(barw=f"{w * scale:.1f}", barg=f"{(w + g) * scale:.1f}", baro=f"{(w + g + o) * scale:.1f}",
             bartotal=f"{total * scale:.1f}", barallow=f"{allowance * scale:.1f}")

    # Update traffic per step, one DRAM crossing per tensor use (cell EdgeUpdateTraffic):
    # forward reads W, writes A; backward reads A and W, writes G; Adam reads W, G, two moments, writes W and moments.
    fwd = w + a
    bwd = a + w + g
    adam = (w + g + o) + (w + o)
    step = fwd + bwd + adam
    assert abs(step - mb(s.step_traffic)) < 1e-9
    bw = Hardware.Mobile.iPhone15Pro.memory.bandwidth.to(GB / second).magnitude
    assert abs(bw - s.bandwidth.to(GB / second).magnitude) < 1e-9
    floor_ms = step / 1e3 / bw * 1e3
    assert abs(floor_ms - s.step_floor.to(millisecond).magnitude) < 1e-9
    n.update(fwdmb=f"{fwd:.0f}", bwdmb=f"{bwd:.0f}", adamtrafficmb=f"{adam:.0f}", stepmb=f"{step:.0f}",
             trafficratio=f"{step / w:.0f}", phonebw=f"{bw:.1f}", stepms=f"{floor_ms:.1f}")

    # ---- Your turn: plain SGD keeps parameters and gradients only (chapter footnote on SGD): no moments.
    sgd = w + g + a
    n.update(sgdmb=f"{sgd:.0f}", sgdmult=f"{sgd / w:.0f}", sgdspare=f"{allowance - sgd:.0f}",
             sgdactshare=f"{a / sgd * 100:.0f}")

    # ---- Q1: upload of the chapter's reference adapter update over the chapter's 1 and 10 Mb/s uplinks
    # (Scenarios.FederatedAdapterUploadMobileUplink, cell EdgeRoundUploadTime). Deadline ASSUMED.
    up = Scenarios.FederatedAdapterUploadMobileUplink
    pay_mb = mb(up.payload)
    bits_per_byte = 8
    pay_mbit = pay_mb * bits_per_byte
    lo = up.links["low"].to("megabit/second").magnitude
    hi = up.links["high"].to("megabit/second").magnitude
    t_lo, t_hi = pay_mbit / lo, pay_mbit / hi
    assert abs(t_lo - up.transfer_time("low").to(second).magnitude) < 1e-6
    deadline_min = 5  # ASSUMED round upload deadline
    n.update(payloadmb=f"{pay_mb:.0f}",
             payloadmbit=f"{pay_mbit:.0f}", bitsperbyte=f"{bits_per_byte}", linkslow=f"{lo:.0f}", linkfast=f"{hi:.0f}",
             uploadslows=f"{t_lo:.0f}", uploadslowmin=f"{t_lo / 60:.1f}", uploadfasts=f"{t_hi:.0f}",
             deadlinemin=f"{deadline_min}")
    # Q1 clause: over-selection for a secure-aggregation round (chapter: about 130% of target selected).
    target_clients, miss_share = 100, 0.30  # ASSUMED target cohort and share of selected devices that miss
    select_raw = target_clients / (1 - miss_share)
    n.update(targetclients=f"{target_clients}", misspct=f"{miss_share * 100:.0f}",
             completefrac=f"{1 - miss_share:.2f}", selectraw=f"{select_raw:.1f}",
             selectclients=f"{math.ceil(select_raw)}", selectmult=f"{math.ceil(select_raw) / target_clients:.2f}")

    # ---- Q2: coverage ratio. Fleet shares from the chapter's tier profile; completed-update shares ASSUMED.
    budget_fleet, flagship_fleet = float(prof.BudgetFleetShare), float(prof.FlagshipFleetShare)
    budget_done, flagship_done = 0.05, 0.40  # ASSUMED shares of completed updates
    n.update(budgetfleet=f"{budget_fleet * 100:.0f}", flagshipfleet=f"{flagship_fleet * 100:.0f}",
             budgetdone=f"{budget_done * 100:.0f}", flagshipdone=f"{flagship_done * 100:.0f}",
             budgetcover=f"{budget_done / budget_fleet:.2f}", flagshipcover=f"{flagship_done / flagship_fleet:.1f}")

    # ---- Q4: admission. Thermal limit = the chapter's sustained mobile ceiling (ReferenceStats.MobilePower);
    # device draw, update increment and latencies ASSUMED.
    thermal = ReferenceStats.MobilePower.MobileMlSustainedHigh.to("watt").magnitude
    p_dev, p_upd = 2.5, 1.0                 # ASSUMED, watts
    lat, dlat, deadline_ms = 30, 15, 50     # ASSUMED, milliseconds: latency passes, power fails
    assert lat + dlat <= deadline_ms and p_dev + p_upd > thermal
    n.update(thermalw=f"{thermal:.0f}", devicew=f"{p_dev:.1f}", updatew=f"{p_upd:.1f}",
             sumw=f"{p_dev + p_upd:.1f}", latms=f"{lat}", dlatms=f"{dlat}", deadlinems=f"{deadline_ms}",
             summs=f"{lat + dlat}")

    # ---- Mistake: a week of photos against 4-bit updates (Scenarios.PhoneImagesVersusQuantizedUpdates,
    # cell FederatedSavings), at the chapter's daily and hourly cadences.
    fs = Scenarios.PhoneImagesVersusQuantizedUpdates
    raw, upd = mb(fs.raw_volume), mb(fs.update_size)
    breakeven = raw / upd
    assert abs(breakeven - fs.break_even_rounds) < 1e-9
    daily, hourly = fs.rounds_per_week["daily"], fs.rounds_per_week["hourly"]
    n.update(rawmb=f"{raw:.0f}", updatemb=f"{upd:.1f}", breakeven=f"{breakeven:.0f}",
             dailyrounds=f"{daily}", hourlyrounds=f"{hourly}",
             dailyupmb=f"{daily * upd:.1f}", hourlyupmb=f"{hourly * upd:.0f}")
    # ---- Mistake: latency forces local inference, not local learning (chapter, Simpler alternatives first:
    # a 33 ms camera deadline against 50-200 ms network round trips; prose values).
    n.update(cameradeadlinems="33", rttlowms="50", rtthighms="200")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
