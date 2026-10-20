#!/usr/bin/env python3
"""Every derived number in Lecture 6 (Long-Term Memory), one block per slide.

Inputs carry their source as a comment. "LTM" is the rendered chapter,
Volume III, Chapter 6, Long-Term Memory (dev text read 2026-10-18).
Run from this folder: python3 calcs.py. make_figs.py imports these values.
"""

import csv
import math
from pathlib import Path

HERE = Path(__file__).parent

# ------------------------------------------------------------------ hook
# Moffatt v. Air Canada, 2024 BCCRT 149, paras 18, 40, 44 (decision text).
PAID = 1630.36            # para 40: total paid for the two flights, CA$
SHOULD_HAVE_PAID = 979.48  # para 40: bereavement fare total, CA$
DAMAGES = 650.88          # para 44a
INTEREST = 36.14          # para 44b, pre-judgment interest
CRT_FEES = 125.00         # para 44c
damages_check = PAID - SHOULD_HAVE_PAID
award_total = DAMAGES + INTEREST + CRT_FEES

# ------------------------------------------------- NM 6.1 records that fit
WINDOW, OUTPUT_RESERVE, BASE_INPUT = 32_768, 2_048, 24_576  # LTM NM 6.1
ENVELOPE, BODY, META = 128, 512, 64                          # LTM NM 6.1
allowance = WINDOW - OUTPUT_RESERVE - BASE_INPUT
per_record = BODY + META
fit = (allowance - ENVELOPE) // per_record
used = ENVELOPE + fit * per_record
remainder = allowance - used
body_only = allowance // BODY
body_only_tokens = ENVELOPE + body_only * per_record
overshoot = body_only_tokens - allowance
meta_doubled_fit = (allowance - ENVELOPE) // (BODY + 2 * META)  # Checkpoint 6.1

# ------------------------------------- SWE-bench BM25 context (Jimenez 2024)
# Jimenez et al., ICLR 2024, Table 2 (Claude 2 % resolved) and Table 3
# (BM25 recall of oracle files, "Avg." row), by maximum context.
SWEB_CONTEXT = [13, 27, 50]          # thousand tokens
SWEB_RECALL = [29.58, 44.41, 51.06]  # percent
SWEB_RESOLVED = [1.96, 1.87, 1.22]   # percent

# ----------------------------------------- SWE-agent vs RAG (Yang 2024)
# Yang et al., NeurIPS 2024, Table 1, SWE-bench Lite, same model (GPT-4 Turbo).
LITE_RAG, LITE_SHELL, LITE_AGENT = 2.67, 11.00, 18.00
agent_over_rag = LITE_AGENT / LITE_RAG

# ---------------------------------------- CPython Condition docstrings
# CPython v3.14.4, Lib/threading.py lines 338, 409, 439 (first docstring lines).
DOC = {
    "wait": "Wait until notified or until a timeout occurs.",
    "notify": "Wake up one or more threads waiting on this condition, if any.",
    "notify_all": "Wake up all threads waiting on this condition.",
}


def words(s):
    return {w.strip(".,").lower() for w in s.split()}


shared_notify_all = len(words(DOC["notify"]) & words(DOC["notify_all"]))
union_notify_all = len(words(DOC["notify"]) | words(DOC["notify_all"]))
jaccard_notify_all = shared_notify_all / union_notify_all

# ----------------------------------------- NM 6.2 exact-scan floor
N_VEC, DIM, B_FP16 = 10_000_000, 768, 2   # LTM NM 6.2
PEAK_FLOPS, BW = 1e12, 50e9                # LTM NM 6.2 reference host
BUDGET_S = 10e-3                            # LTM NM 6.2
SEARCHES = 20                               # LTM NM 6.2
scan_bytes = N_VEC * DIM * B_FP16
scan_flops = 2 * N_VEC * DIM
intensity = scan_flops / scan_bytes
ridge = PEAK_FLOPS / BW
t_stream = scan_bytes / BW
t_math = scan_flops / PEAK_FLOPS
t_traj = SEARCHES * t_stream
t_10k = 10_000 * DIM * B_FP16 / BW
ceiling = BUDGET_S * BW / (DIM * B_FP16)

# ----------------------------- measured scan on one laptop (scan_bench.py)
rows = []
with open(HERE / "scan_bench.csv") as f:
    r = csv.reader(f)
    meta = next(r)
    next(r)
    for n, b, t in r:
        rows.append((int(n), int(b), float(t)))
MACHINE = meta[1]
achieved = [b / t / 1e9 for n, b, t in rows]
gflops = [2 * n * DIM / t / 1e9 for n, b, t in rows]
# least-squares fit of t = bytes / BW through the origin
bw_fit = sum(b * b for _, b, _ in rows) / sum(b * t for _, b, t in rows)
B_FP32 = 4
laptop_ceiling = BUDGET_S * bw_fit / (DIM * B_FP32)

# DDR5-4800 RDIMM in the photo (Commons description): 4,800 MT/s x 8 bytes
dimm_bw = 4800e6 * 8

# --------------------------------------- ANN memory (LTM section 6.6.2, Table 5)
LINKS, LINK_B, PQ_M = 32, 8, 64   # LTM: 32 eight-byte links; 64 subvectors
hnsw_links = N_VEC * LINKS * LINK_B
hnsw_total = scan_bytes + hnsw_links
pq_codes = N_VEC * PQ_M
pq_ratio = scan_bytes / pq_codes
lut_bytes = PQ_M * 256 * 4
bytes_per_vec_exact = DIM * B_FP16

# ----------------------------------------------- NM 6.3 depth budget
CHILDREN, NODE_TOK, EDGE_ENV = 3, 180, 100  # LTM NM 6.3
ALLOWANCE = 6_144


def nodes(depth, c=CHILDREN):
    return (c ** (depth + 1) - 1) // (c - 1)


depth_nodes = [nodes(d) for d in range(4)]
depth_tokens = [n * NODE_TOK + EDGE_ENV for n in depth_nodes]
max_nodes = (ALLOWANCE - EDGE_ENV) // NODE_TOK

# ---------------------------------------- 10:00 vs 11:00 (LTM section 6.2.2)
PREF_A = "Schedule no meetings before 10:00 AM"
PREF_B = "Schedule no meetings before 11:00 AM"
pref_shared = len(words(PREF_A) & words(PREF_B))
pref_total = len(words(PREF_A))

# ----------------------------------------------- PoisonedRAG (Zou et al.)
# Zou et al., USENIX Security 2025 (arXiv 2402.07867v3), Table 1, black-box,
# PaLM 2 as the generator; 5 malicious texts per target question.
POISON = [("NQ", 2_681_468, 0.97), ("HotpotQA", 5_233_329, 0.99),
          ("MS-MARCO", 8_841_823, 0.91)]
POISON_N = 5
poison_per_million = [(name, POISON_N / size * 1e6) for name, size, _ in POISON]

# ------------------------------------------ NM 6.4 who pays for freshness
FILES, CHUNKS = 8, 16                                    # LTM NM 6.4
PARSE, POSTING, LOCAL_EMB, REMOTE_EMB = 3.5e-3, 2e-3, 28e-3, 150e-3
EXCL_W, EXCL_R, CANDS = 50e-9, 30e-9, 50
sync_local = FILES * PARSE + CHUNKS * POSTING + LOCAL_EMB
sync_remote_batch = FILES * PARSE + CHUNKS * POSTING + REMOTE_EMB
sync_remote_serial = FILES * PARSE + CHUNKS * POSTING + FILES * REMOTE_EMB
excl_write = FILES * EXCL_W
excl_query = CANDS * EXCL_R
excl_ratio = sync_local / excl_write

# ------------------------------------------ NM 6.5 two stores (typed values)
STORE_A_AUX, STORE_B_AUX = 0.36, 8.52      # LTM NM 6.5, auxiliary / source
STORE_A_REIDX, STORE_B_REIDX = 34.1e-3, 612e-3
store_byte_ratio = STORE_B_AUX / STORE_A_AUX
store_time_ratio = STORE_B_REIDX / STORE_A_REIDX

# --------------------------- Checkpoint 6.4: cost per accepted task, B vs A
P_A, P_B, MAINT = 0.50, 0.60, 0.10   # LTM Checkpoint 6.4 (attempt cost of A = 1)
ceff_a = 1.0 / P_A
ceff_b = (1.0 + MAINT) / P_B
ceff_change = ceff_b / ceff_a - 1
breakeven_p = (1.0 + MAINT) / ceff_a

# ------------------------------------------ Liu et al. 2024, Table 6 and 1
# GPT-3.5-Turbo, 20 retrieved documents (~4K tokens), answer at index 0..19.
LIU_POS = [1, 5, 10, 15, 20]
LIU_ACC = [75.8, 57.2, 53.8, 55.4, 63.2]
LIU_CLOSED, LIU_ORACLE = 56.1, 88.3
liu_drop = LIU_ACC[0] - min(LIU_ACC)

# ------------------------------------------- BM25 Example 6.1 (notes check)
N_DOCS, AVGDL, DF, K1, BB = 100_000, 400, 5, 1.2, 0.75
idf = math.log(1 + (N_DOCS - DF + 0.5) / (DF + 0.5))


def bm25(tf, dl):
    return idf * tf * (K1 + 1) / (tf + K1 * (1 - BB + BB * dl / AVGDL))


bm25_short, bm25_long = bm25(1, 80), bm25(6, 4000)


if __name__ == "__main__":
    p = print
    p("HOOK  damages check (paid - should have paid):", f"{damages_check:.2f}")
    p("HOOK  award total CA$:", f"{award_total:.2f}")
    p("NM6.1 allowance:", allowance, " per record:", per_record, " fit:", fit,
      " used:", used, " remainder:", remainder)
    p("NM6.1 body-only count:", body_only, " serialized:", body_only_tokens,
      " overshoot:", overshoot, " metadata doubled fit:", meta_doubled_fit)
    p("SWE-agent / RAG on Lite:", f"{agent_over_rag:.1f}x")
    p("CPython notify vs notify_all shared words:", shared_notify_all, "of",
      union_notify_all, f"Jaccard {jaccard_notify_all:.2f}")
    p("NM6.2 bytes GB:", scan_bytes / 1e9, " GFLOP:", scan_flops / 1e9,
      " intensity:", intensity, " ridge:", ridge)
    p("NM6.2 stream ms:", f"{t_stream * 1e3:.1f}", " math ms:", f"{t_math * 1e3:.2f}",
      " 20 searches s:", f"{t_traj:.1f}", " 10k ms:", f"{t_10k * 1e3:.2f}",
      " ceiling:", f"{ceiling:,.0f}")
    p("MEASURED", MACHINE, " achieved GB/s:", [f"{a:.0f}" for a in achieved],
      " GFLOP/s:", [f"{g:.0f}" for g in gflops])
    p("MEASURED fit BW GB/s:", f"{bw_fit / 1e9:.0f}", " 10 ms ceiling FP32:",
      f"{laptop_ceiling:,.0f}")
    p("DIMM DDR5-4800 peak GB/s:", f"{dimm_bw / 1e9:.1f}")
    p("ANN HNSW links GB:", hnsw_links / 1e9, " total GB:", hnsw_total / 1e9,
      " PQ codes GB:", pq_codes / 1e9, " ratio:", pq_ratio, " LUT KB:",
      f"{lut_bytes / 1e3:.1f}", " bytes/vector exact:", bytes_per_vec_exact)
    p("NM6.3 nodes:", depth_nodes, " tokens:", depth_tokens, " max nodes:", max_nodes)
    p("PREF shared words:", pref_shared, "of", pref_total)
    p("POISON per million:", [(n, f"{v:.2f}") for n, v in poison_per_million])
    p("NM6.4 sync local ms:", f"{sync_local * 1e3:.0f}", " remote batch ms:",
      f"{sync_remote_batch * 1e3:.0f}", " remote serial s:", f"{sync_remote_serial:.2f}")
    p("NM6.4 exclusion write us:", f"{excl_write * 1e6:.1f}", " query us:",
      f"{excl_query * 1e6:.1f}", " ratio:", f"{excl_ratio:,.0f}")
    p("NM6.5 byte ratio:", f"{store_byte_ratio:.0f}", " time ratio:", f"{store_time_ratio:.0f}")
    p("CP6.4 Ceff A:", f"{ceff_a:.2f}", " Ceff B:", f"{ceff_b:.2f}",
      " change:", f"{ceff_change * 100:.1f}%", " break-even p:", f"{breakeven_p:.2f}")
    p("LIU drop first to worst (points):", f"{liu_drop:.1f}")
    p("BM25 idf:", f"{idf:.3f}", " short:", f"{bm25_short:.2f}", " long:", f"{bm25_long:.2f}")
