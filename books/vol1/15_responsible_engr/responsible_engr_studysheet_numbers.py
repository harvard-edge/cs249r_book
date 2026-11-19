"""Numbers for the Responsible Engineering study sheet (Volume I, Chapter 15).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to responsible_engr_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
"""
import math
from pathlib import Path

from mlsysim import Hardware, Infrastructure
from mlsysim.engine.scenarios import Scenarios


def g(x, nd=3):
    return f"{x:.{nd}g}"


def c(x):
    return f"{x:,.0f}"


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


def numbers() -> dict:
    n = {}

    # ---- Worked example: the credit service's two confusion matrices (fairness-metrics-calc)
    credit = Scenarios.CreditServiceTwoGroups
    gate = Scenarios.CreditServiceBiasFeedback.gate                     # release gate on the TPR gap, points
    a, b = credit.groups["A"], credit.groups["B"]
    for grp, m in (("A", a), ("B", b)):
        k = grp.lower()
        n.update({f"tp{k}": c(m["tp"]), f"fn{k}": c(m["fn"]), f"fp{k}": c(m["fp"]), f"tn{k}": c(m["tn"]),
                  f"total{k}": c(sum(m.values())), f"qualified{k}": c(m["tp"] + m["fn"]),
                  f"unqualified{k}": c(m["fp"] + m["tn"]), f"approved{k}": c(m["tp"] + m["fp"]),
                  f"correct{k}": c(m["tp"] + m["tn"]),
                  f"approval{k}": pct(credit.positive_rate(grp)), f"tpr{k}": pct(credit.tpr(grp)),
                  f"fpr{k}": pct(credit.fpr(grp)), f"fnr{k}": pct(credit.fnr(grp)),
                  f"accuracy{k}": pct(credit.accuracy(grp))})
    correct_all = a["tp"] + a["tn"] + b["tp"] + b["tn"]
    total_all = sum(a.values()) + sum(b.values())
    tpr_gap = 100 * (credit.tpr("A") - credit.tpr("B"))
    n.update(correctall=c(correct_all), totalall=c(total_all), accuracyall=pct(credit.overall_accuracy, 1),
             paritygap=f"{100 * (credit.positive_rate('A') - credit.positive_rate('B')):.0f}",
             tprgap=f"{tpr_gap:.0f}", fprgap=f"{100 * (credit.fpr('A') - credit.fpr('B')):.0f}",
             gate=f"{gate:.0f}", gatemultiple=f"{tpr_gap / gate:.0f}",
             fnrratio=f"{credit.fnr('B') / credit.fnr('A'):.0f}")

    # ---- Your turn: ASSUMED Group B matrix after a threshold change (same 1,000 qualified, 1,000 not)
    mtp, mfn, mfp, mtn = 870, 130, 280, 720
    m_tpr, m_fpr, m_appr = mtp / (mtp + mfn), mfp / (mfp + mtn), (mtp + mfp) / (mtp + mfn + mfp + mtn)
    n.update(newtp=c(mtp), newfn=c(mfn), newfp=c(mfp), newtn=c(mtn), newapproved=c(mtp + mfp),
             newtpr=pct(m_tpr), newfpr=pct(m_fpr), newapproval=pct(m_appr, 1),
             newtprgap=f"{100 * credit.tpr('A') - 100 * m_tpr:.0f}",
             newfprgap=f"{100 * m_fpr - 100 * credit.fpr('A'):.0f}",
             newparitygap=f"{100 * m_appr - 100 * credit.positive_rate('A'):.1f}")

    # ---- Bias feedback model: the chapter's notebook (bias-feedback-loop-calc), front curve and mistake
    loop = Scenarios.CreditServiceBiasFeedback
    n.update(launchgap=f"{loop.launch_gap:.1f}", auditgap=f"{loop.next_audit_gap:.1f}",
             feedbackfactor=f"{loop.factor:.2f}", sampledfactor=f"{loop.sampled_factor:.2f}",
             breachquarter=c(loop.breach_cycle), breachgap=f"{loop.gap_at_breach:.1f}",
             sampledgap=f"{loop.sampled_gap_at_breach:.1f}", breachexact=f"{loop.gap_at_breach:.4f}")
    # curve points (unrounded), for the TikZ device
    for kq in range(0, 9):
        n[f"riseq{kq}"] = f"{loop.gap_at(kq):.4f}"
        n[f"fallq{kq}"] = f"{loop.launch_gap * loop.sampled_factor ** kq:.4f}"

    # ---- Slice sample size (representation-stats-calc)
    budget = Scenarios.OnePercentGroupEvaluationBudget
    n.update(zscore=f"{budget.z:.2f}", worstvariance=f"{budget.variance:.2f}",
             confidence=pct(budget.confidence), halfwidth=f"{100 * budget.half_width:.0f}",
             pergroup=c(budget.target_per_group), groupshare=pct(budget.group_share),
             randomtotal=c(budget.random_collection), randommultiple=f"{budget.random_multiple:.0f}")

    # ---- Lifecycle cost and carbon of the DLRM ranking service (inference-cost-calc, tco-calc, carbon-factor-calc)
    tco = Scenarios.RankingServiceLifecycleCost
    tdp_kw = Hardware.Cloud.A100.tdp.to("kW").magnitude
    per_gpu_hour = Infrastructure.Pricing.Fleet.CarbonPerGpuHr.rate.to("kg").magnitude
    intensity = per_gpu_hour / tdp_kw                                      # kg CO2e per kWh
    shares = tco.shares
    n.update(gpupower=f"{tdp_kw:.1f}", intensity=f"{intensity:.1f}", kgpergpuhour=f"{per_gpu_hour:.2f}",
             tcototal=f"{tco.total_cost.to('USD').magnitude / 1e6:.2f}",
             sharetraining=pct(shares["training"], 1), shareinference=pct(shares["inference"], 1),
             shareoperations=pct(shares["operations"], 1),
             costratio=f"{tco.cost_ratio:.0f}", carbonratio=f"{tco.carbon_ratio:.0f}",
             servicems=f"{tco.service_time.to('ms').magnitude:.0f}",
             inferencerate=f"{tco.inference_rate.to('USD/hour').magnitude:.2f}",
             cyclecost=c(tco.cycle_cost.to("USD").magnitude), cyclehours=c(tco.cycle_gpu_hours.to("hour").magnitude),
             cycles=c(tco.cycles), years=c(tco.years), days=c(tco.service_days))

    # ---- Check yourself Q1 (ASSUMED audit readings within the chapter's credit-service loop)
    q1_launch, q1_audit, q1_sampled = 2.5, 3.0, loop.sampled_factor
    q1_factor = q1_audit / q1_launch
    q1_k = math.ceil(math.log(gate / q1_launch) / math.log(q1_factor))
    n.update(qonelaunch=f"{q1_launch:.1f}", qoneaudit=f"{q1_audit:.1f}", qonefactor=f"{q1_factor:.1f}",
             qonelog=f"{math.log(gate / q1_launch) / math.log(q1_factor):.1f}", qonequarters=c(q1_k),
             qonemonths=c(3 * q1_k), qonebreach=f"{q1_launch * q1_factor ** q1_k:.1f}",
             qonebefore=f"{q1_launch * q1_factor ** (q1_k - 1):.1f}",
             qonesampled=f"{q1_launch * q1_sampled ** q1_k:.1f}")

    # ---- Q2 (ASSUMED 36 million requests a day; the chapter's service time, prices, cycles and carbon factor)
    requests = 36e6
    hours_day = requests * tco.service_time.to("hour").magnitude
    inf_hours = hours_day * tco.service_days
    inf_cost = inf_hours * tco.inference_rate.to("USD/hour").magnitude
    train_cost = tco.cycles * tco.cycle_cost.to("USD").magnitude
    train_hours = tco.cycles * tco.cycle_gpu_hours.to("hour").magnitude
    n.update(qtworequests=f"{requests / 1e6:.0f}", qtwohoursday=c(hours_day), qtwohours=c(inf_hours),
             qtwoinfcost=c(inf_cost), qtwotraincost=c(train_cost), qtworatio=f"{inf_cost / train_cost:.1f}",
             qtwoservingsave=c(tco.serving_cut * inf_cost), qtwotrainsave=c(tco.training_cut * train_cost),
             qtwoservingcut=pct(tco.serving_cut),
             qtwoinfcarbon=f"{inf_hours * per_gpu_hour / 1000:.1f}", qtwotrainhours=c(train_hours),
             qtwotraincarbon=f"{train_hours * per_gpu_hour / 1000:.1f}")

    # ---- Q3 (ASSUMED relative changes on a common normalized baseline, as in alignment-gap-calc)
    clicks, satisfaction = 0.15, 0.08
    n.update(qthreeclicks=pct(clicks), qthreesat=pct(satisfaction), qthreeclicklevel=f"{1 + clicks:.2f}",
             qthreesatlevel=f"{1 - satisfaction:.2f}", qthreegap=f"{clicks + satisfaction:.2f}")

    # ---- Q4 (ASSUMED 2 percent group, the chapter's rounded per-group target; a 400-example slice)
    share, slice_n = 0.02, 400
    margin = budget.z * math.sqrt(budget.variance / slice_n)
    n.update(qfourshare=pct(share), qfourtotal=c(budget.target_per_group / share),
             qfourmultiple=f"{1 / share:.0f}", qfourslice=c(slice_n), qfourmargin=f"{100 * margin:.1f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
