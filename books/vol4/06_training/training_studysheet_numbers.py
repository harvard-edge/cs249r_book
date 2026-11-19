"""Numbers for the Policy Training study sheet (Volume IV, Chapter 6).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to training_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios


def f(x, nd, commas=True):
    return f"{x:,.{nd}f}" if commas else f"{x:.{nd}f}"


def sci(x, nd=1):
    """a x 10^b in LaTeX, as the chapter's fmt_sci prints it."""
    e = int(math.floor(math.log10(abs(x))))
    return f"{x / 10 ** e:.{nd}f}$\\times$10$^{{{e}}}$"


def numbers() -> dict:
    n = {}
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator

    # ---- Decoder budget (chapter: DecoderBudget, notebook "The renewal slack")
    b = Scenarios.LatchDecoderBudget
    ms = lambda q: q.to("ms").magnitude
    step, renew, act = ms(b.setpoint_step), ms(b.renewal_period), ms(b.one_pass_latency)
    enc, ddim, jit = ms(b.encode), ms(b.denoising_latency), ms(b.renewal_jitter)
    kd = b.denoising_steps
    per_eval = (ddim - enc) / kd
    assert abs(per_eval - ms(b.per_evaluation)) < 1e-12
    n.update(step=f(step, 0), renew=f(renew, 0), act=f(act, 0), enc=f(enc, 0), ddim=f(ddim, 0),
             jitter=f(jit, 0), evals=f(kd, 0), pereval=f(per_eval, 1),
             slackact=f(renew - act, 0), slackddim=f(renew - ddim, 0),
             overrun=f(ddim + jit - renew, 0), actend=f(act + jit, 0),
             actparams=f(rm.chunk_model.parameters.magnitude / 1e6, 0))
    assert abs((renew - act) - ms(b.one_pass_slack)) < 1e-12
    assert abs((ddim + jit - renew) - ms(b.denoising_overrun_under_jitter)) < 1e-12
    # picture: 1 ms = 3.6 pt
    n.update(penc=f(enc * 3.6, 1, False), pddim=f(ddim * 3.6, 1, False), pact=f(act * 3.6, 1, False),
             pstep=f(step * 3.6, 1, False), prenew=f(renew * 3.6, 1, False),
             pjit=f((ddim + jit) * 3.6, 1, False))

    # ---- Your turn: ASSUMED step distillation cuts the diffusion head to 8 evaluations, same costs
    k_turn = 8
    t_turn = enc + k_turn * round(per_eval, 1)
    n.update(turnevals=f(k_turn, 0), turnlat=f(t_turn, 0), turnslack=f(renew - t_turn, 0),
             turnend=f(t_turn + jit, 0))
    assert t_turn > step and renew - t_turn > jit

    # ---- Trial campaign (chapter: TrialWallClock, notebook "The reset wall")
    c = Scenarios.LatchTrialCampaign
    roll = c.rollout.to("s").magnitude
    auto = c.automated_reset.to("s").magnitude
    manual = c.manual_reset.to("s").magnitude
    day = 86400.0
    n.update(trials=sci(c.trials), rollout=f(roll, 0), resetauto=f(auto, 0), resetmanual=f(manual, 0),
             rigs=f(c.farm_rigs, 0), revper=f(c.reversals_per_trial, 0),
             daysmanual=f(c.manual_time.to("day").magnitude, 0),
             daysfarm=f(c.farm_manual_time.magnitude, 1),
             reversals=sci(c.reversals), simminutes=f(c.simulated_time.to("minute").magnitude, 0),
             daysauto=f(c.automated_time.to("day").magnitude, 0),
             seeded=f(c.seeded_trials, 0), seededhours=f(c.seeded_budget.to("hour").magnitude, 1))
    # Q1: ASSUMED a faster fixture resets in 2 s; the campaign runs on the farm's rig count
    reset_q = 2.0
    days_q = c.trials * (roll + reset_q) / c.farm_rigs / day
    sec_q = c.trials * (roll + reset_q) / c.farm_rigs
    assert round(round(sec_q, -3) / day, 1) == round(days_q, 1)  # the printed seconds reproduce the days
    n.update(qreset=f(reset_q, 0), qdays=f(days_q, 1), qsec=sci(c.trials * (roll + reset_q) / c.farm_rigs, 2))

    # ---- Impact peak (chapter: LatchContactExploit, notebook "The soft-spring exploit")
    s = Scenarios.LatchSoftSpringExploit
    m = s.moving_mass.to("kg").magnitude
    kp = s.site.k_latch.to("N/m").magnitude
    ksim = s.penalty_stiffness.to("N/m").magnitude
    vf = s.site.v_latch_high.to("m/s").magnitude
    vg = s.site.v_latch_approach.to("m/s").magnitude
    flim = s.site.f_latch_limit.to("N").magnitude
    fsim = vf * math.sqrt(ksim * m)
    frig = vf * math.sqrt(kp * m)
    assert abs(fsim - s.simulated_peak.to("N").magnitude) < 1e-9
    assert abs(frig - s.rigid_peak.to("N").magnitude) < 1e-9
    n.update(mass=f(m, 0), kplate=sci(kp, 1), ksim=f(ksim, 0), vfast=f(vf, 2), vguard=f(vg, 2),
             flim=f(flim, 0), fsim=f(fsim, 1), frigid=f(frig, 1), softer=f(kp / ksim, 1))
    # picture: paired bars, 1 N = 1 pt
    n.update(bsim=f(fsim, 1, False), brigid=f(frig, 1, False), blim=f(flim, 1, False))
    # Q2: ASSUMED approach slowed to 0.06 m/s, coasting unbraked
    vq = 0.06
    n.update(qv=f(vq, 2), qroot=f(math.sqrt(kp * m), 0), qrootsim=f(math.sqrt(ksim * m), 0),
             qfrigid=f(vq * math.sqrt(kp * m), 1), qfsim=f(vq * math.sqrt(ksim * m), 1),
             qover=f(vq * math.sqrt(kp * m) - flim, 1))
    assert vq * math.sqrt(kp * m) > flim
    # Q2 answer: the guarded approach the declared ODD admits, from the printed rounded root
    qfguard = vg * round(math.sqrt(kp * m))
    n.update(qfguard=f(qfguard, 1))
    assert qfguard < flim

    # ---- Declared ODD (chapter: PretrainSupport over the latch scenario table)
    t = Scenarios.LatchScenarioTable
    n.update(cells=f(t.cells, 0), supported=f(t.supported, 0), absent=f(t.cells - t.supported, 0),
             basename=rm.intent_model.name, baseparams=f(rm.intent_model.parameters.magnitude / 1e9, 0),
             offset=f(t.offsets[1].to("mm").magnitude, 0))
    assert t.supported == 1
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
