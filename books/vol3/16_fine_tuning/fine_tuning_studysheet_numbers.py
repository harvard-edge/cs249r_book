"""Numbers for the Trajectory Fine-Tuning study sheet (Volume III, Chapter 16).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(PerTurnExpansion, SelectedTargetAccounting, PackingOccupancy, DistributionBounds,
PreferenceMargin, AdapterStateInventory), and written as \\setnum{key}{value} to
fine_tuning_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value. Memory is decimal GB and MB, the chapter's declared convention;
the sheet prints no accelerator capacity.
"""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios as SC


def fx(x, nd):
    """Round half up, as a reader would, and format with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP))   # round(.., 9) drops float noise


def r(x, nd):
    return float(fx(x, nd))


def pct(x, nd=0):
    return fx(x * 100, nd)


def ci(x):
    """Integer with thousands separators."""
    return f"{round(x):,}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def numbers() -> dict:
    n = {}

    # ---- Worked example: where the loss lands (chapter: SelectedTargetAccounting, RecoveryRecordTargetMask)
    rec = SC.RecoveryRecordTargetMask
    res = rec.result
    stored = rec.prompt + rec.task + rec.rationale + rec.response + rec.observation
    assert stored == res.stored
    predicted = stored - 1                                           # the first boundary has no prediction
    assert predicted == res.predicted
    k = rec.response - rec.unselected_response                       # rationale kept as context only
    assert k == res.targets_without_rationale == res.selected_response
    share = k / stored
    obs_share = rec.observation / stored
    near(obs_share, res.observation_share, 1e-12)
    dil = predicted / k
    near(dil, res.dilution_without_rationale, 1e-12)
    n.update(wprompt=ci(rec.prompt), wtask=ci(rec.task), wrat=ci(rec.rationale), wresp=ci(rec.response),
             wrefused=ci(rec.unselected_response), wobs=ci(rec.observation), wstored=ci(stored),
             wpred=ci(predicted), wk=ci(k), wshare=pct(share, 1), wobsshare=pct(obs_share, 1),
             wdil=fx(dil, 1),
             # picture: segment widths on a 200 pt bar
             barobs=fx(200 * obs_share, 2), bark=fx(200 * share, 2),
             barctx=fx(200 * (1 - obs_share - share), 2))

    # ---- Your turn: observations doubled (chapter: RecoveryRecordTargetMaskDoubledObservations)
    dbl = SC.RecoveryRecordTargetMaskDoubledObservations
    dres = dbl.result
    assert dbl.observation == 2 * rec.observation and dbl.response == rec.response
    stored2 = stored + rec.observation
    assert stored2 == dres.stored
    pred2 = stored2 - 1
    dil2 = pred2 / k
    near(dil2, dres.dilution_without_rationale, 1e-12)
    n.update(yobs=ci(dbl.observation), ystored=ci(stored2), ypred=ci(pred2), yshare=pct(k / stored2, 1),
             ydil=fx(dil2, 1), yobsshare=pct(dbl.observation / stored2, 1))

    # ---- Front: per-invocation expansion (chapter: PerTurnExpansion)
    pt = SC.RepairTrajectoryPerCallExamples
    near(pt.result.expansion, (pt.turns + 1) / 2, 1e-12)
    n.update(ptturns=f"{pt.turns}", ptexp=fx(pt.result.expansion, 1))

    # ---- Front card and picture: imitation regret (chapter: DistributionBounds)
    b = SC.RepairPolicyImitationBounds
    h, eps, u = b.horizon_steps, b.per_step_error, b.recoverability
    ref_bound = min(h, h * h * eps)
    lrn_bound = min(h, u * h * eps)
    near(ref_bound, b.cloning_capped, 1e-9)
    near(lrn_bound, b.aggregation_capped, 1e-9)
    n.update(rh=f"{h}", reps=pct(eps), repsfrac=fx(eps, 2), ru=f"{u:.0f}", rref=fx(ref_bound, 0),
             rlrn=fx(lrn_bound, 1), rgap=fx(ref_bound / lrn_bound, 0))

    # ---- Mistake: packing (chapter: PackingOccupancy, EightExamplePackingBatch)
    pk = SC.EightExamplePackingBatch.result
    near(pk.padded_occupancy, pk.mean_length / pk.longest, 1e-12)
    n.update(pkpad=pct(pk.padded_occupancy, 1), pkpack=pct(pk.packed_occupancy, 1),
             pkratio=fx(pk.position_ratio, 0))

    # ---- Mistake: the preference score (chapter: PreferenceMargin, RepairPreferencePair)
    pp, pp2 = SC.RepairPreferencePair, SC.RepairPreferencePairDoubledBeta
    dw = pp.policy_preferred - pp.reference_preferred
    dl = pp.policy_dispreferred - pp.reference_dispreferred
    near(dw, pp.result.score_preferred, 1e-12)
    near(dl, pp.result.score_dispreferred, 1e-12)
    import math
    z = pp.beta * (dw - dl)
    z2 = pp2.beta * (dw - dl)
    sig = 1 / (1 + math.exp(-z))
    sig2 = 1 / (1 + math.exp(-z2))
    near(sig, pp.result.preference_score, 1e-12)
    near(sig2, pp2.result.preference_score, 1e-12)
    n.update(dpbeta=fx(pp.beta, 1), dpbetatwo=fx(pp2.beta, 1), dpdw=fx(dw, 0), dpdl=fx(dl, 0),
             dpz=fx(z, 1), dpztwo=fx(z2, 1), dpsig=fx(sig, 3), dpsigtwo=fx(sig2, 3),
             dpsigpct=pct(sig))

    # ---- Mistake: adapter state (chapter: AdapterStateInventory, Rank16AdaptersOnHypothetical8B)
    run = SC.Rank16AdaptersOnHypothetical8B
    base = run.workload.parameters.to("count").magnitude
    per_matrix = run.adapter_rank * (run.adapter_in_dim + run.adapter_out_dim)
    adapters = run.workload.layers * run.adapted_matrices * per_matrix
    assert adapters == run.adapter_count
    frozen_b = run.bytes_per_value.to("byte").magnitude
    trained_b = run.bytes_per_trained_parameter.to("byte").magnitude
    assert trained_b == 16 and frozen_b == 2
    base_gb = base * frozen_b / 1e9
    adapter_gb = adapters * trained_b / 1e9
    lora_gb = r(base_gb, 0) + r(adapter_gb, 2)                       # as the sheet adds its printed terms
    full_gb = base * trained_b / 1e9
    near(lora_gb, run.total.to("GB").magnitude, 0.005)
    near(full_gb, run.full_state.to("GB").magnitude, 1e-9)
    kept = lora_gb / full_gb
    near(kept, run.adapter_kept_share, 0.001)
    n.update(adbase=fx(base / 1e9, 0), adrank=f"{run.adapter_rank}", adfrac=pct(adapters / base, 2),
             adfrozenb=fx(frozen_b, 0), adtrainb=fx(trained_b, 0), adbasegb=fx(base_gb, 0),
             adadaptgb=fx(adapter_gb, 2), adlora=fx(lora_gb, 2), adfull=fx(full_gb, 0),
             adkept=pct(kept, 1), adcut=fx(full_gb / lora_gb, 1), adbound=fx(trained_b / frozen_b, 0))

    # ---- Check yourself
    # Regret bounds. ASSUMED: horizon 40, disagreement 1 percent on both distributions, u = 2
    qh, qe, qu = 40, 0.01, 2
    q_ref = min(qh, qh * qh * qe)
    q_lrn = min(qh, qu * qh * qe)
    q_match = q_lrn / (qh * qh)
    q_sat = 1 / qe
    assert q_ref < qh
    n.update(qh=f"{qh}", qe=pct(qe), qefrac=fx(qe, 2), qu=f"{qu}", qhsq=ci(qh * qh), qref=fx(q_ref, 0),
             qlrn=fx(q_lrn, 1), qgap=fx(q_ref / q_lrn, 0), qmatch=pct(q_match, 2), qsat=fx(q_sat, 0))

    # Batch weights. ASSUMED: one trace with 4,800 targets, 12 lookups with 200 each;
    # then the trace split into 16 per-call examples of 300 targets each
    lt, ns, st, pieces = 4800, 12, 200, 16
    assert lt % pieces == 0
    tok = lt / (lt + ns * st)
    ex = 1 / (ns + 1)
    ex_split = pieces / (ns + pieces)
    n.update(batchlong=ci(lt), batchcount=f"{ns}", batchshort=ci(st), btotal=ci(lt + ns * st), bpieces=f"{pieces}",
             bpiece=ci(lt // pieces), btok=pct(tok, 1), bex=pct(ex, 1), bexn=f"{ns + 1}",
             bexsplit=pct(ex_split, 1), bsplitn=f"{ns + pieces}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
