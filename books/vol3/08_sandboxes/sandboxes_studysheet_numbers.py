"""Numbers for the Agent Sandboxes study sheet (Volume III, Chapter 8).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(EnvelopeAmortization, SandboxPoolReserve), and written as \\setnum{key}{value} to
sandboxes_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value. The guest image is read from the registry in MiB and printed
with its decimal GB value.
"""
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios as SC
from mlsysim.physics.statistics import calc_poisson_reserve, calc_poisson_tail


def fx(x, nd):
    """Round half up, as a reader would, and format with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP))


def r(x, nd):
    return float(fx(x, nd))


def ints(x):
    return f"{x:,.0f}"


def pct(x, nd=0):
    return fx(x * 100, nd)


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def tail(mu, s):
    """Pr(Poisson(mu) >= s), written out as the sheet's card writes it."""
    return 1 - math.exp(-mu) * sum(mu ** j / math.factorial(j) for j in range(s))


def numbers() -> dict:
    n = {}

    # ---- Worked example: sizing a pool (chapter: SandboxPoolReserve, nbk-08-reserve-from-arrival-rate)
    sc = SC.CodingSandboxPoolTwoPerSecond
    res = sc.result
    lam = sc.arrival_rate.to("1/s").magnitude
    hold = sc.hold.to("s").magnitude
    restore_ms = sc.restore.to("ms").magnitude
    guest_mib = sc.guest_memory.to("MiB").magnitude
    guest_gb = sc.guest_memory.to("GB").magnitude
    dirty_mb = sc.dirty.to("MB").magnitude
    standby_mb = sc.standby.to("MB").magnitude
    target = sc.target_miss

    active = lam * hold
    near(active, res.active, 1e-9)
    mu = lam * restore_ms / 1000
    near(mu, res.restore_arrivals, 1e-12)
    p1, p2 = tail(mu, 1), tail(mu, 2)
    near(p1, res.initial_miss, 1e-12)
    near(p2, res.reserve_miss, 1e-12)
    assert p1 > target > p2 and res.reserve == 2
    reserve = res.reserve
    active_gb = active * dirty_mb / 1000
    standby_gb = reserve * standby_mb / 1000
    pool_gb = r(guest_gb, 3) + active_gb + standby_gb                # from the printed terms
    near(pool_gb, res.pool_memory.to("GB").magnitude, 0.001)
    private_gb = active * guest_gb                                   # 600 x 512 MiB, exact
    near(private_gb, res.private_memory.to("GB").magnitude, 1e-9)
    ratio = r(private_gb, 1) / r(pool_gb, 1)
    assert fx(ratio, 1) == fx(res.memory_ratio, 1), (ratio, res.memory_ratio)
    n.update(lam=fx(lam, 0), hold=fx(hold, 0), restore=fx(restore_ms, 0), restoresec=fx(restore_ms / 1000, 3),
             guestmib=fx(guest_mib, 0), guestgb=fx(guest_gb, 3), dirty=fx(dirty_mb, 0), standby=fx(standby_mb, 0),
             target=pct(target, 1), active=fx(active, 0), mu=fx(mu, 2), pone=pct(p1, 2), ptwo=pct(p2, 3),
             reserve=f"{reserve}", activegb=fx(active_gb, 1), standbygb=fx(standby_gb, 3),
             poolgb=fx(pool_gb, 1), privategb=fx(private_gb, 1), memratio=fx(ratio, 1),
             privatemib=ints(active * guest_mib))
    assert fx(active * guest_mib * 2**20 / 1e9, 1) == fx(private_gb, 1)   # 307,200 MiB is the printed 322.1 GB

    # ---- Your turn. ASSUMED: restores contend under burst concurrency and take 120 ms (8x), all else equal
    slow_ms = 120
    mu_s = lam * slow_ms / 1000
    tails = [tail(mu_s, s) for s in range(1, 6)]
    reserve_s = calc_poisson_reserve(mu_s, target)
    assert reserve_s == 4 and tails[2] > target > tails[3]
    near(tails[3], calc_poisson_tail(mu_s, 4), 1e-12)
    pool_s = r(guest_gb, 3) + active_gb + reserve_s * standby_mb / 1000
    n.update(slow=fx(slow_ms, 0), slowsec=fx(slow_ms / 1000, 2), muslow=fx(mu_s, 2),
             muslowsq=fx(mu_s ** 2 / 2, 4), muslowcube=fx(mu_s ** 3 / 6, 5),
             sone=pct(tails[0], 1), stwo=pct(tails[1], 2), sthree=pct(tails[2], 2), sfour=pct(tails[3], 3),
             reserveslow=f"{reserve_s}", addedmb=fx((reserve_s - reserve) * standby_mb, 0),
             poolslowgb=fx(pool_s, 1))

    # ---- Check yourself Q1. ASSUMED: 200 parser calls of 5 ms, the chapter's 15 ms setup, a 20 percent cap
    ea = SC.ParserCallsEnvelopeSetup
    setup = ea.setup.to("ms").magnitude
    near(setup, restore_ms, 1e-9)                                    # same illustrative restore profile
    calls, work, cap = 200, 5, 0.20
    fresh = calls * (setup + work)
    kept = setup + calls * work
    share = setup / (setup + work)
    w_cap = setup * (1 - cap) / cap
    assert w_cap > work
    n.update(qcalls=f"{calls}", qwork=fx(work, 0), qcap=pct(cap), qcapfrac=fx(cap, 1), qcapleft=fx(1 - cap, 1),
             qfresh=fx(fresh / 1000, 1), qkept=fx(kept / 1000, 3), qspeed=fx(r(fresh / 1000, 1) / r(kept / 1000, 3), 1),
             qstep=fx(setup + work, 0), qusefulsec=fx(calls * work / 1000, 0),
             qshare=pct(share), qwcap=fx(w_cap, 0), chapcap=pct(ea.overhead_cap))

    # ---- Mistake: first-write granularity (chapter: WorkspaceCost, nbk-08-workspace-copy-vs-overlay)
    wc = SC.CopyOnWriteRepairWorkspaces
    wr = wc.result
    ws = wc.workspaces
    fixture_gb = wc.fixture.to("GB").magnitude
    block_kb = wc.block.to("KB").magnitude
    file_gb = ws * fixture_gb
    block_total_kb = ws * block_kb
    near(file_gb, wr.file_copy.to("GB").magnitude, 1e-9)
    near(block_total_kb, wr.block_copy.to("KB").magnitude, 1e-9)
    n.update(ws=f"{ws}", fixture=fx(fixture_gb, 1), block=fx(block_kb, 0), filecopy=fx(file_gb, 1),
             blockcopy=f"{block_total_kb:,.0f}")

    # ---- Check yourself Q2. ASSUMED grant lifetimes: parent 300 s, children 120 s and 400 s
    n.update(gparent="300", gshort="120", glong="400")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
