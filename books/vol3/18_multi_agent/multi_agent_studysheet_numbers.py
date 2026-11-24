"""Numbers for the Multi-Agent Coordination study sheet (Volume III, Chapter 18).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(DelegationScreen, UslScreen, WriteCollision, ReviewCorrelation, ReviewFloor), and written as
\\setnum{key}{value} to multi_agent_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
import math
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


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def elapsed(f, gamma, n):
    """Normalized elapsed time of a fixed partition: (1-f) + f/N + gamma(N-1)."""
    return (1 - f) + f / n + gamma * (n - 1)


def numbers() -> dict:
    n = {}

    # ---- Worked example: the chapter's screen (DelegationScreen, Scenarios.RepairPartitionScreen)
    sc = SC.RepairPartitionScreen
    f, g, top = sc.parallel_fraction, sc.overhead_per_extra_worker, sc.max_workers
    times = {m: elapsed(f, g, m) for m in range(1, top + 1)}
    for m in times:
        near(times[m], sc.elapsed[m], 1e-12)
    best = min(times, key=times.get)
    assert best == sc.best_workers == 6
    near(1 / times[best], sc.best_speedup, 1e-12)
    near(1 / (1 - f), sc.speedup_cap, 1e-9)
    t4, t6, t7, t8 = r(times[4], 3), r(times[best], 3), r(times[7], 3), r(times[8], 3)
    assert t6 < t4 and t6 < t7 < t8
    near(t7, r(1 - f + r(f / 7, 3) + g * 6, 3), 1e-12)                 # printed parts sum to the printed total
    n.update(fpar=fx(f, 2), serial=fx(1 - f, 2), gamma=fx(g, 2), maxn=f"{top}",
             ofour=fx(f / 4, 2), osix=fx(f / 6, 3), oeight=fx(f / 8, 2),
             gfour=fx(g * 3, 2), gsix=fx(g * 5, 2), geight=fx(g * 7, 2),
             tfour=fx(t4, 3), tsix=fx(t6, 3), oseven=fx(f / 7, 3), tseven=fx(t7, 3), nextcount=f"{best + 1}", teight=fx(t8, 3), best=f"{best}",
             bestspeed=fx(1 / t6, 1), sqrtbest=fx(math.sqrt(f / g), 1), cap=fx(1 / (1 - f), 0))
    # the chapter's own sensitivity: halve the overhead
    hv = SC.RepairPartitionScreenHalvedOverhead
    n.update(ghalf=fx(hv.overhead_per_extra_worker, 2), besthalf=f"{hv.best_workers}",
             speedhalf=fx(hv.best_speedup, 1))
    # curve on the front: speedup 1/elapsed for N = 1..16 (plotted from the same model)
    n.update(curvef=fx(f, 2), curveg=fx(g, 2))

    # ---- Finish it. ASSUMED: serial work halved, f = 0.90, same gamma
    f2 = 0.90
    times2 = {m: elapsed(f2, g, m) for m in range(1, top + 1)}
    best2 = min(times2, key=times2.get)
    assert best2 == 7
    t2_6, t2_7, t2_8 = r(times2[6], 3), r(times2[7], 3), r(times2[8], 3)
    assert t2_7 < t2_6 and t2_7 < t2_8
    sp2 = 1 / t2_7
    assert sp2 > hv.best_speedup and best2 < hv.best_workers   # more speedup with fewer workers
    n.update(fnew=fx(f2, 2), serialnew=fx(1 - f2, 2), onsix=fx(f2 / 6, 2), tnsix=fx(t2_6, 3),
             onseven=fx(f2 / 7, 3), gseven=fx(g * 6, 2), tnseven=fx(t2_7, 3), tneight=fx(t2_8, 3),
             bestnew=f"{best2}", speednew=fx(sp2, 1), sqrtnew=fx(math.sqrt(f2 / g), 1),
             capnew=fx(1 / (1 - f2), 0))

    # ---- Mistake: more workers (USL with the screen's coefficients, Scenarios.RepairPartitionScalability)
    usl = SC.RepairPartitionScalability
    near(usl.contention, 1 - f, 1e-12)
    near(usl.coherency, g, 1e-12)
    n.update(slower=f"{usl.first_slower}")

    # ---- Mistake: clean merge (WriteCollision, Scenarios.ThousandFileSharedWrites)
    ws = SC.ThousandFileSharedWrites
    n.update(files=f"{ws.files:,}", writes=f"{ws.writes_per_worker}",
             evenodds=f"{ws.even_odds_fleet}")
    # the sheet prints the front card's approximation exp(-N(N-1)w^2/(2n)) at N = 2, so a student can check it
    disjoint_two = math.exp(-2 * 1 * ws.writes_per_worker ** 2 / (2 * ws.files))
    near(disjoint_two, 1 - ws.collision[2], 0.002)
    n.update(disjointtwo=pct(disjoint_two, 1))

    # ---- Mistake: reviewer agreement (ReviewCorrelation, ReviewFloor; Scenarios.FiveCorrelatedReviewers)
    pn = SC.FiveCorrelatedReviewers
    ind5 = pn.independent_majority(pn.reviewers)
    mix5 = pn.correlated_majority(pn.reviewers)
    mixl = pn.correlated_majority(pn.large_ensemble)
    assert ind5 < pn.floor < mixl < mix5 < pn.false_approval
    n.update(revn=f"{pn.reviewers}", revp=pct(pn.false_approval), revrho=fx(pn.correlation, 2),
             indfive=pct(ind5, 1), mixfive=pct(mix5, 1), revlarge=f"{pn.large_ensemble}",
             mixlarge=pct(mixl, 1), revfloor=pct(pn.floor, 1))

    # ---- Q1 (fan-out). ASSUMED: ten interchangeable workers, each on time with 0.95; any nine suffice
    q_p, q_n, q_k = 0.95, 10, 9
    q_all = q_p ** q_n
    q_one_late = q_n * q_p ** (q_n - 1) * (1 - q_p)
    q_enough = sum(math.comb(q_n, j) * q_p ** j * (1 - q_p) ** (q_n - j) for j in range(q_k, q_n + 1))
    near(q_enough, q_all + q_one_late, 1e-12)
    n.update(qp=pct(q_p), qpfrac=fx(q_p, 2), qn=f"{q_n}", qk=f"{q_k}", qlatefrac=fx(1 - q_p, 2),
             qninefrac=fx(q_p ** (q_n - 1), 3),
             qallfrac=fx(q_all, 3), qall=pct(q_all, 1), qonefrac=fx(q_one_late, 3),
             qenough=pct(r(q_all, 3) + r(q_one_late, 3), 1))
    near(r(q_all, 3) + r(q_one_late, 3), q_enough, 0.001)

    # ---- Q2 (shared writes). ASSUMED: 2,000 files, 20 writes per worker, uniform and independent
    # (w^2/n = 0.2, unlike the front card's 1,000 files and 10 writes, so the fleet answer must be computed)
    w_f, w_w = 2000, 20
    assert w_w ** 2 / w_f != ws.writes_per_worker ** 2 / ws.files
    expo = lambda m: m * (m - 1) * w_w ** 2 / (2 * w_f)
    q_three = 1 - math.exp(-expo(3))
    even = next(m for m in range(2, 50) if 1 - math.exp(-expo(m)) > 0.5)
    assert even == 4
    n.update(wfiles=f"{w_f:,}", wwrites=f"{w_w}", wexpthree=fx(expo(3), 1),
             wthree=pct(q_three, 0), weven=f"{even}", wexpeven=fx(expo(even), 1),
             wexpbelow=fx(expo(even - 1), 1), weventpct=pct(1 - math.exp(-expo(even)), 0),
             wbelow=f"{even - 1}", lntwo=fx(math.log(2), 2))

    # ---- Q1(d) (makespan bound). ASSUMED: 8 tasks of 10 min, longest dependency chain 3 tasks, 4 then 5 slots
    mk_tasks, mk_len, mk_chain, mk_slots = 8, 10, 3, 4
    work, chain = mk_tasks * mk_len, mk_chain * mk_len
    bound4 = max(work / mk_slots, chain)
    bound5 = max(work / (mk_slots + 1), chain)
    assert bound4 == bound5 == chain                                  # the chain binds: a fifth slot leaves the bound unchanged
    n.update(mktasks=f"{mk_tasks}", mklen=f"{mk_len}", mkchain=f"{mk_chain}", mkslots=f"{mk_slots}",
             mkslotsfive=f"{mk_slots + 1}", mkwork=f"{work}", mkpath=f"{chain}",
             mkfour=fx(work / mk_slots, 0), mkfive=fx(work / (mk_slots + 1), 0), mkbound=fx(bound4, 0))

    # ---- Q3 (token multiple). ASSUMED: delegation uses 2.5x tokens per attempt; baselines 50 and 30 percent
    m_r, base_hi, base_lo = 2.5, 0.50, 0.30
    need_hi, need_lo, ceiling = m_r * base_hi, m_r * base_lo, 1 / m_r
    assert need_hi > 1 and need_lo < 1 and base_lo < ceiling < base_hi
    near(ceiling, 1 / m_r, 1e-12)
    n.update(tr=fx(m_r, 1), basehigh=pct(base_hi), basehighfrac=fx(base_hi, 2), baselow=pct(base_lo),
             baselowfrac=fx(base_lo, 2), needhigh=pct(need_hi), needlow=pct(need_lo), baseceiling=pct(ceiling))

    # ---- Q4 (delegation tree). ASSUMED: $40 remaining, four proposed children at $12 each
    rem, kids, each = 40, 4, 12
    copied = kids * each
    admitted = rem // each
    left = rem - admitted * each
    equal = rem / kids
    assert copied > rem and admitted == 3 and left < each
    n.update(rrem=f"{rem}", rkids=f"{kids}", reach=f"{each}", rcopied=f"{copied}",
             radmit=f"{admitted}", rheld=f"{admitted * each}", rleft=f"{left}", requal=f"{equal:.0f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
