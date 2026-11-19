"""Numbers for the Long-Term Memory study sheet (Volume III, Chapter 6).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(RetrievalPayloadBudget, VectorScanLatencyFloor, GraphNeighborhoodExpansion),
and written as \\setnum{key}{value} to long_term_memory_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value. Host memory bandwidth is a decimal rate (GB/s), so scan bytes
are printed in decimal GB; no binary capacity enters a budget on this sheet.
"""
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios as SC


def fx(x, nd):
    """Round half up, as a reader would, and format with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP))


def r(x, nd):
    return float(fx(x, nd))


def pct(x, nd=0):
    return fx(x * 100, nd)


def ints(x):
    return f"{x:,.0f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def tree_nodes(children, depth):
    """Seed plus every level of an exact tree: (children^(depth+1) - 1) / (children - 1)."""
    return (children ** (depth + 1) - 1) // (children - 1)


def numbers() -> dict:
    n = {}

    # ---- Front + mistake: the retrieval allowance and whole records (chapter: RetrievalPayloadBudget)
    pb = SC.RetrievalRecordsInAllowance
    pr = pb.result
    allowance = pb.window - pb.output_reserve - pb.other_input
    assert allowance == pr.allowance
    record = pb.body + pb.metadata
    records = (allowance - pb.envelope) // record
    assert record == pr.record and records == pr.records
    body_only = allowance // pb.body                                  # a body-only count, as the chapter sizes it
    body_only_used = pb.envelope + body_only * record                 # what those records really serialize to
    assert body_only == pr.body_only and body_only_used == pr.body_only_used
    n.update(window=ints(pb.window), outreserve=ints(pb.output_reserve), otherin=ints(pb.other_input),
             allowance=ints(allowance), allowshare=pct(allowance / pb.window),
             envelope=ints(pb.envelope), body=ints(pb.body), meta=ints(pb.metadata), record=ints(record),
             records=ints(records), bodyonly=ints(body_only), bodyonlyused=ints(body_only_used),
             overshoot=ints(body_only_used - allowance),
             # plain values for drawing, in thousands of tokens (pgfmath reads no separators and stops near 16,384)
             windowk=f"{pb.window / 1000:g}", otherink=f"{pb.other_input / 1000:g}", allowancek=f"{allowance / 1000:g}",
             allowanceraw=f"{allowance}")

    # ---- Worked example: a typed neighborhood against the allowance (chapter: GraphNeighborhoodExpansion)
    gs = SC.TypedTraversalDepthThree
    gr = gs.result
    assert gs.window == pb.window and gr.allowance == allowance
    nodes = tree_nodes(gs.children, gs.depth)
    need = nodes * gs.node_tokens + gs.envelope_tokens
    fit = (allowance - gs.envelope_tokens) // gs.node_tokens
    shallow = tree_nodes(gs.children, gs.depth - 1)
    shallow_need = shallow * gs.node_tokens + gs.envelope_tokens
    assert (nodes, need, fit, shallow, shallow_need) == (gr.nodes, gr.required, gr.fit, gr.shallow_nodes,
                                                          gr.shallow_required)
    assert need > allowance >= shallow_need
    n.update(children=f"{gs.children}", depth=f"{gs.depth}", shdepth=f"{gs.depth - 1}",
             nodetok=ints(gs.node_tokens), edgetok=ints(gs.envelope_tokens),
             levelone=f"{gs.children}", leveltwo=f"{gs.children ** 2}", levelthree=f"{gs.children ** 3}",
             nodes=ints(nodes), need=ints(need), nodetokall=ints(nodes * gs.node_tokens),
             fit=ints(fit), fitroom=ints(allowance - gs.envelope_tokens),
             shnodes=ints(shallow), shneed=ints(shallow_need), shnodetok=ints(shallow * gs.node_tokens),
             # tokens needed at each depth, as printed beside the bars, and plain values for drawing
             **{f"depthneed{w}": ints(tree_nodes(gs.children, k) * gs.node_tokens + gs.envelope_tokens)
                for k, w in enumerate(("zero", "one", "two", "three"))},
             **{f"depthraw{w}": f"{tree_nodes(gs.children, k) * gs.node_tokens + gs.envelope_tokens}"
                for k, w in enumerate(("zero", "one", "two", "three"))})
    assert [tree_nodes(gs.children, k) * gs.node_tokens + gs.envelope_tokens for k in range(4)] == gr.tokens_by_depth

    # ---- Your turn. ASSUMED: a wider walk, 4 children per node; node records and edges reserve unchanged
    y_c, y_tok = 4, gs.node_tokens
    y_two = tree_nodes(y_c, 2)
    y_three = tree_nodes(y_c, 3)
    y_two_need = y_two * y_tok + gs.envelope_tokens
    y_three_need = y_three * y_tok + gs.envelope_tokens
    y_fit = (allowance - gs.envelope_tokens) // y_tok
    assert y_two_need <= allowance < y_three_need and y_two < y_fit < y_three
    n.update(ychildren=f"{y_c}", ytok=ints(y_tok), ytwo=ints(y_two), ythree=ints(y_three),
             ytwoneed=ints(y_two_need), ythreeneed=ints(y_three_need), yfit=ints(y_fit),
             ylevelone=f"{y_c}", yleveltwo=f"{y_c ** 2}", ylevelthree=f"{y_c ** 3}")

    # ---- Front card: the exact-scan floor (chapter: VectorScanLatencyFloor, Scenarios.ExactScanTenMillionVectors)
    es = SC.ExactScanTenMillionVectors
    er = es.result
    bw = es.host.memory.bandwidth.to("GB/s").magnitude
    peak = es.host.compute.peak_flops.to("TFLOP/s").magnitude
    b = es.bytes_per_value.to("byte").magnitude
    per_vec = es.dimension * b
    scan_gb = es.vectors * per_vec / 1e9
    floor_ms = scan_gb / bw * 1e3
    near(floor_ms, er.stream_time.to("ms").magnitude, 1e-9)
    filt_ms = es.filtered_vectors * per_vec / (bw * 1e9) * 1e3
    near(filt_ms, er.filtered_time.to("ms").magnitude, 1e-12)
    budget_ms = es.budget.to("ms").magnitude
    ceiling = budget_ms * 1e-3 * bw * 1e9 / per_vec
    near(ceiling, er.budget_vectors, 1e-6)
    ridge = peak * 1e12 / (bw * 1e9)
    near(ridge, er.ridge.magnitude, 1e-9)
    near(2 / b, er.intensity.magnitude, 1e-12)
    n.update(scanvec=ints(es.vectors), scandim=ints(es.dimension), scanb=f"{b:g}", hostbw=f"{bw:g}",
             vecbytes=ints(per_vec), scangb=fx(scan_gb, 2), scanfloor=fx(floor_ms, 1),
             filtvec=ints(es.filtered_vectors), filtfloor=fx(filt_ms, 2), scanbudget=f"{budget_ms:g}",
             ceiling=ints(round(ceiling, -3)), intensity=f"{2 / b:g}", ridge=f"{ridge:.0f}",
             # plotted curve: log10 of floor in ms against log10 of eligible vectors
             logk=f"{math.log10(per_vec / (bw * 1e9) * 1e3):.6f}",
             logbudget=f"{math.log10(budget_ms):.6f}", logceil=f"{math.log10(ceiling):.6f}")

    # ---- Check yourself
    # Q1 ASSUMED: 2,000,000 vectors of 1,024 FP16 components on the same host, a 20 ms budget
    q_n, q_d, q_budget = 2_000_000, 1_024, 20
    q_vec = q_d * b
    q_gb = q_n * q_vec / 1e9
    q_floor = q_gb / bw * 1e3
    q_ceil = q_budget * 1e-3 * bw * 1e9 / q_vec
    assert q_floor > q_budget and q_ceil < q_n
    q_b32 = 4                                                          # FP32 bytes per component
    n.update(qvec=ints(q_n), qdim=ints(q_d), qbudget=f"{q_budget}", qvecbytes=ints(q_vec),
             qgb=fx(q_gb, 3), qfloor=fx(q_floor, 2), qceil=ints(math.floor(q_ceil)),
             qbfour=f"{q_b32}", qintensityfour=f"{2 / q_b32:g}", qfloorfour=fx(q_floor * q_b32 / b, 2))

    # Q4 ASSUMED: policy A spends $0.80 per attempt at 40 percent acceptance; policy B adds $0.20 of
    # background maintenance per attempt and reaches 55 percent
    a_cost, a_acc, b_add, b_acc = 0.80, 0.40, 0.20, 0.55
    b_cost = r(a_cost + b_add, 2)
    a_eff = r(a_cost / a_acc, 2)
    b_eff = r(b_cost / b_acc, 2)
    b_even = b_cost / a_eff
    assert b_eff < a_eff and b_acc > b_even
    n.update(pacost=fx(a_cost, 2), paacc=pct(a_acc), paaccfrac=fx(a_acc, 2), pbadd=fx(b_add, 2),
             pbacc=pct(b_acc), pbaccfrac=fx(b_acc, 2), pbcost=fx(b_cost, 2), paeff=fx(a_eff, 2),
             pbeff=fx(b_eff, 2), pbeven=pct(b_even))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
