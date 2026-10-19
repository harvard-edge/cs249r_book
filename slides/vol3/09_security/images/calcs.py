#!/usr/bin/env python3
"""Every derived number in Lecture 9 (Agent Security), one block per slide.

Inputs are typed with their source. Sources:
  AS   = the rendered chapter, Agent Security (Volume III), dev text 2026-10-18.
  NM   = a Napkin Math notebook of that chapter (scenario values the chapter
         itself labels hypothetical; slides say "illustrative").
  ADJ  = AgentDojo published results table (Debenedetti et al., NeurIPS 2024),
         read through MLSysIM Literature.InjectionDefenseResults, which parses
         docs/results-table.html at github.com/ethz-spylab/agentdojo @ 089ed468.
  GU   = Gu et al., "Auditing Prompt Caching in Language Model APIs",
         ICML 2025 (arXiv 2502.07776v2), Table 1 and Section 4.2.
  NVD  = NIST National Vulnerability Database records, retrieved 2026-10-19.

Run from this folder: python3 calcs.py   (prints "label: value" per slide)
make_figs.py imports the functions below so figures and slides agree.
"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MLSYSIM = HERE.parents[3] / "mlsysim"      # MLSysBook-slides/mlsysim


def show(label, value):
    print(f"  {label}: {value}")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


# ------------------------------------------------- Exposure (equation 1)
# AS Sec. 9.2.1, equation 1 and Napkin Math 9.1: payload fraction 1 percent,
# repair reads 20 documents, research 100, screening miss rate 5 percent.
PAYLOAD = 0.01
READS = {"repair": 20, "research": 100, "support": 6, "remediation": 10}  # AS Lighthouse 9.1
MISS = 0.05


def exposure(p, n):
    return 1 - (1 - p) ** n


def nm91():
    out = {k: exposure(PAYLOAD, n) for k, n in READS.items()}
    out["repair_linear"] = PAYLOAD * READS["repair"]
    out["repair_screened"] = exposure(PAYLOAD * MISS, READS["repair"])
    out["research_screened"] = exposure(PAYLOAD * MISS, READS["research"])
    out["research_per_1000_screened"] = 1000 * out["research_screened"]
    out["research_over_support"] = out["research"] / out["support"]
    return out


# ------------------------------------------- Multi-turn form (equation 5)
# AS Sec. 9.8, equation 5. Illustrative: the repair agent's 30-step budget
# (Lighthouse 1.1) with 2 untrusted reads per turn at the 1 percent rate.
TURNS = 30
READS_PER_TURN = 2


def turns():
    p_turn = exposure(PAYLOAD, READS_PER_TURN)
    return dict(p_turn=p_turn, trajectory=exposure(p_turn, TURNS),
                same_as_docs=exposure(PAYLOAD, READS_PER_TURN * TURNS))


# ------------------------------------------------ Effective permission
# AS Sec. 9.7.1: an engineer with read access to 50 repositories asks for a
# fix in one (connection-pool).
REPOS_HELD = 50
REPOS_TASK = 1


def effperm():
    return dict(blocked=REPOS_HELD - REPOS_TASK, share=REPOS_TASK / REPOS_HELD)


# ------------------------------------------------ NM 9.2, policy A vs B
# AS Sec. 9.9.2, Napkin Math 9.2 table 4: 100 attacked and 100 benign
# episodes per policy (hypothetical counts).
N_ATT = 100
N_BEN = 100
A = dict(confirmed=20, unresolved=5, legit_under_attack=65, benign_accepted=90, false_block=5)
B = dict(confirmed=5, unresolved=10, legit_under_attack=55, benign_accepted=80, false_block=15)
ATTEMPT = 2.10   # illustrative attempt cost, the introduction's NM 1.2 value


def nm92():
    a_lo, b_lo = A["confirmed"] / N_ATT, B["confirmed"] / N_ATT
    a_hi = (A["confirmed"] + A["unresolved"]) / N_ATT
    b_hi = (B["confirmed"] + B["unresolved"]) / N_ATT
    pa, pb = A["benign_accepted"] / N_BEN, B["benign_accepted"] / N_BEN
    return dict(a_lo=a_lo, b_lo=b_lo, a_hi=a_hi, b_hi=b_hi,
                rel_observed=1 - b_lo / a_lo, rel_both_upper=1 - b_hi / a_hi,
                rel_least_favorable=1 - b_hi / a_lo,
                p_a=pa, p_b=pb, cost_rise=pa / pb - 1,
                ceff_a=ATTEMPT / pa, ceff_b=ATTEMPT / pb,
                extra_false_blocks=B["false_block"] - A["false_block"])


# ------------------------------------------------- Zero-success bound
# AS Sec. 9.9.3 and footnote 11: 95 percent upper bound with zero successes
# in n independent attacked trials, 1 - 0.05**(1/n); Checkpoint 9.3 uses 50.
def bound(n, conf=0.95):
    return 1 - (1 - conf) ** (1 / n)


def trials_needed(rate, conf=0.95):
    return math.ceil(math.log(1 - conf) / math.log(1 - rate))


def zero():
    return dict(b10=bound(10), b50=bound(50), b100=bound(100), b1000=bound(1000),
                rule3_50=3 / 50, need_1pct=trials_needed(0.01), rule3_1pct=3 / 0.01)


# ------------------------------------------------- AgentDojo (real data)
def agentdojo_rows():
    sys.path.insert(0, str(MLSYSIM))
    from mlsysim import Literature  # noqa: E402
    return Literature.InjectionDefenseResults.entries


ABLATED = "gpt-4o-2024-05-13"
EARLY, LATE = "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022"
DEFENSES = ["spotlighting_with_delimiting", "transformers_pi_detector",
            "tool_filter", "repeat_user_prompt"]


def agentdojo():
    rows = agentdojo_rows()

    def row(model, defense="none", attack="important_instructions"):
        return next(r for r in rows if r["model"] == model and r["defense"] == defense
                    and r["attack"] == attack)
    plain = [r for r in rows if r["defense"] == "none" and r["attack"] == "important_instructions"]
    out = dict(n_models=len(plain),
               asr_lo=min(r["targeted_asr_pct"] for r in plain),
               asr_hi=max(r["targeted_asr_pct"] for r in plain),
               early=row(EARLY), late=row(LATE), base=row(ABLATED))
    out["defenses"] = {d: row(ABLATED, d) for d in DEFENSES}
    out["plain"] = plain
    out["early_over_late"] = out["early"]["targeted_asr_pct"] / out["late"]["targeted_asr_pct"]
    return out


# ------------------------------------------- Prefix-cache timing channel
# AS Sec. 9.8.4 (modeled on the chapter's reference node): a 2,048-token
# probe waits about 83 ms of prefill on a miss and about 5 ms on a hit;
# 169 ms at 4,096 tokens. GU Sec. 4.2: caching detected at 8 of 17
# providers, global sharing at 7.
MISS_2K_MS, HIT_MS, MISS_4K_MS = 83, 5, 169
GU_PROVIDERS, GU_CACHING, GU_GLOBAL = 17, 8, 7


def cache():
    return dict(gap_2k=MISS_2K_MS - HIT_MS, gap_4k=MISS_4K_MS - HIT_MS,
                ratio_2k=MISS_2K_MS / HIT_MS)


def main():
    print("Lecture 9 derived values")
    e = nm91()
    print("[Estimate: exposure, NM 9.1]")
    show("repair, 20 reads", pct(e["repair"]))
    show("research, 100 reads", pct(e["research"]))
    show("repair linear estimate p*n", pct(e["repair_linear"], 0))
    show("repair behind 5% miss screen", pct(e["repair_screened"]))
    show("research behind 5% miss screen", pct(e["research_screened"]))
    show("research screened per 1,000 tasks", f"{e['research_per_1000_screened']:.0f}")
    print("[Lighthouse 9.1 table]")
    show("support, 6 reads", pct(e["support"]))
    show("remediation, 10 reads", pct(e["remediation"]))
    show("research / support exposure", f"{e['research_over_support']:.1f}x")
    t = turns()
    print("[Multi-turn, equation 5 (illustrative)]")
    show("per-turn exposure, 2 reads", pct(t["p_turn"]))
    show("30-turn trajectory", pct(t["trajectory"]))
    show("same as 60 documents", pct(t["same_as_docs"]))
    p = effperm()
    print("[Effective permission]")
    show("repositories the token reaches but the task does not", p["blocked"])
    n = nm92()
    print("[Estimate: NM 9.2]")
    show("A confirmed", pct(n["a_lo"], 0))
    show("B confirmed", pct(n["b_lo"], 0))
    show("relative reduction, observed", pct(n["rel_observed"], 0))
    show("A upper (unresolved = disclosure)", pct(n["a_hi"], 0))
    show("B upper", pct(n["b_hi"], 0))
    show("relative reduction, both upper", pct(n["rel_both_upper"], 0))
    show("least favorable (B upper vs A confirmed)", pct(n["rel_least_favorable"], 0))
    show("benign acceptance A, B", f"{n['p_a']:.2f}, {n['p_b']:.2f}")
    show("cost per accepted benign task rise", pct(n["cost_rise"]))
    show("C_eff A at $2.10 per attempt", f"${n['ceff_a']:.2f}")
    show("C_eff B at $2.10 per attempt", f"${n['ceff_b']:.2f}")
    show("extra false blocks per 100", n["extra_false_blocks"])
    z = zero()
    print("[Zero-success bound]")
    for k in ("b10", "b50", "b100", "b1000"):
        show(f"95% upper bound, n={k[1:]}", pct(z[k]))
    show("rule of three at n=50", pct(z["rule3_50"]))
    show("clean episodes needed for 1%", z["need_1pct"])
    show("rule-of-three estimate for 1%", f"{z['rule3_1pct']:.0f}")
    a = agentdojo()
    print("[AgentDojo, important_instructions attack]")
    show("undefended models", a["n_models"])
    show("ASR range", f"{a['asr_lo']:.0f}% to {a['asr_hi']:.0f}%")
    show("Claude 3.5 Sonnet (Jun / Oct 2024) utility", f"{a['early']['utility_pct']:.0f}%, {a['late']['utility_pct']:.0f}%")
    show("Claude 3.5 Sonnet (Jun / Oct 2024) ASR", f"{a['early']['targeted_asr_pct']:.0f}%, {a['late']['targeted_asr_pct']:.0f}%")
    show("ratio early/late ASR", f"{a['early_over_late']:.0f}x")
    show("GPT-4o undefended utility / ASR", f"{a['base']['utility_pct']:.0f}% / {a['base']['targeted_asr_pct']:.0f}%")
    for d, r in a["defenses"].items():
        show(f"GPT-4o + {d} utility / ASR", f"{r['utility_pct']:.0f}% / {r['targeted_asr_pct']:.0f}%")
    c = cache()
    print("[Prefix-cache timing]")
    show("2,048-token miss minus hit", f"{c['gap_2k']} ms")
    show("4,096-token miss minus hit", f"{c['gap_4k']} ms")
    show("2,048-token miss / hit", f"{c['ratio_2k']:.0f}x")


if __name__ == "__main__":
    main()
