"""Numbers for the Serving at Scale study sheet (Volume II, Chapter 10).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to inference_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Memory: accelerator HBM is read from the registry in GiB and printed as GiB with its decimal
GB value beside it (80 GiB = 85.9 GB); all other sizes are decimal (GB = 10^9 bytes).
"""
import math
from pathlib import Path

from mlsysim import Hardware
from mlsysim.engine.scenarios import Scenarios


def c(x):
    return f"{x:,.0f}"


def numbers() -> dict:
    n = {}

    # ---- KV budget card and worked example: the chapter's long-context admission scenario
    # (Scenarios.Llama3_70BLongContextAdmissionDgxH100, cell KVCacheCapacityEstimator).
    sc = Scenarios.Llama3_70BLongContextAdmissionDgxH100
    r = sc.result
    m = sc.workload
    gpus = sc.accelerators
    hbm_gib = Hardware.Cloud.H100.memory.capacity.to("GiB").magnitude
    hbm_gb = Hardware.Cloud.H100.memory.capacity.to("GB").magnitude
    node_gb = r.hbm_total.to("GB").magnitude
    params_b = m.parameters.magnitude / 1e9
    bytes_fp16 = sc.kv_bytes_per_elem.to("byte").magnitude
    weights_gb = r.weights.to("GB").magnitude
    assert abs(weights_gb - params_b * bytes_fp16) < 1e-6
    reserve_gb = sc.reserve.to("GB").magnitude
    per_token = r.kv_per_token.to("byte").magnitude
    assert per_token == 2 * m.layers * m.kv_heads * r.head_dim * bytes_fp16
    context = sc.context_tokens
    per_req_gb = r.kv_per_request.to("GB").magnitude
    budget_gb = r.kv_budget.to("GB").magnitude
    assert abs(budget_gb - (node_gb - weights_gb - reserve_gb)) < 1e-6
    batch = math.floor(budget_gb / per_req_gb)
    assert batch == r.max_requests
    spare_gb = budget_gb - batch * per_req_gb
    n.update(gpus=c(gpus), hbmgib=f"{hbm_gib:.0f}", hbmgb=f"{hbm_gb:.1f}", nodegib=f"{gpus * hbm_gib:.0f}",
             nodegb=f"{node_gb:.1f}", params=f"{params_b:.1f}", layers=c(m.layers), kvheads=c(m.kv_heads),
             headdim=c(r.head_dim), fpbytes=f"{bytes_fp16:.0f}", weightsgb=f"{weights_gb:.1f}",
             reservegb=f"{reserve_gb:.0f}", pertoken=c(per_token), pertokenmb=f"{per_token / 1e6:.2f}",
             context=c(context), contextk=f"{context / 1024:.0f}", perrequest=f"{per_req_gb:.1f}",
             budget=f"{budget_gb:.1f}", maxbatch=c(batch), sparegb=f"{spare_gb:.1f}",
             kvtotalgb=f"{batch * per_req_gb:.1f}")
    # Stacked-bar device: node HBM to scale, 230 pt for the whole node.
    sc_pt = 230 / node_gb
    n.update(barweights=f"{weights_gb * sc_pt:.1f}", barreserve=f"{reserve_gb * sc_pt:.1f}",
             barkv=f"{batch * per_req_gb * sc_pt:.1f}", barone=f"{per_req_gb * sc_pt:.2f}")

    # ---- Your turn: INT8 KV cache (1 byte) vs INT4 weights (0.5 byte per parameter), same node and reserve.
    int8 = 1.0
    int4 = 0.5
    per_req8 = context * per_token * int8 / bytes_fp16 / 1e9
    batch8 = math.floor(budget_gb / per_req8)
    w4 = params_b * int4
    budget4 = node_gb - w4 - reserve_gb
    batch4 = math.floor(budget4 / per_req_gb)
    assert math.floor(float(f"{budget_gb:.1f}") / float(f"{per_req8:.1f}")) == batch8
    assert math.floor(float(f"{budget4:.1f}") / float(f"{per_req_gb:.1f}")) == batch4
    n.update(pertoken8=c(per_token * int8 / bytes_fp16), perrequest8=f"{per_req8:.1f}", batchkveight=c(batch8),
             weightsfour=f"{w4:.1f}", budgetfour=f"{budget4:.1f}", batchweightsfour=c(batch4))

    # ---- Mistake: INT4 70B on one H100 (registry 80 GiB), 128K-token requests, no reserve.
    # From the printed (rounded) values, so the printed sums reproduce.
    one = round(w4, 1) + round(per_req_gb, 1)
    two = round(w4, 1) + 2 * round(per_req_gb, 1)
    assert one < hbm_gb < two
    n.update(onefit=f"{one:.1f}", twofit=f"{two:.1f}")

    # ---- Batching card and curve: the chapter's decode knee (Scenarios.LlmDecodeBatchingKnee).
    k = Scenarios.LlmDecodeBatchingKnee
    t0 = k.fixed_time.to("ms").magnitude
    tr = k.per_request_time.to("ms").magnitude
    knee = t0 / tr
    assert knee == k.knee
    ceil_rate = 1000 / tr
    n.update(tfixed=f"{t0:.0f}", treq=f"{tr:.1f}", knee=c(knee), ceiling=c(ceil_rate))
    # Curve: x = 20 + 0.5 * B pt for B in 0..400; y = 70 * mu / ceiling pt.
    pts = []
    for b in range(0, 401, 5):
        mu = 1000 * b / (t0 + b * tr)
        pts.append(f"({20 + 0.5 * b:.1f},{70 * mu / ceil_rate:.2f})")
    n["curve"] = " ".join(pts)
    n.update(xknee=f"{20 + 0.5 * knee:.1f}", yknee=f"{70 * 0.5:.1f}", halfceiling=c(ceil_rate / 2))

    # ---- Q1: ASSUMED decode step 30 ms fixed + 0.6 ms per sequence, TPOT target 50 ms, batch 200.
    q0, qr, target, qb = 30.0, 0.6, 50.0, 200
    qknee = q0 / qr
    qceil = 1000 / qr
    qstep = q0 + qb * qr
    qmu = 1000 * qb / qstep
    qmax = math.floor((target - q0) / qr)
    n.update(qfixed=f"{q0:.0f}", qreq=f"{qr:.1f}", qtarget=f"{target:.0f}", qbatch=c(qb), qknee=c(qknee),
             qceiling=c(qceil), qstep=f"{qstep:.0f}", qrate=c(qmu), qshare=f"{100 * qmu / qceil:.0f}",
             qextra=f"{qb * qr:.0f}", qmax=c(qmax), qmaxstep=f"{q0 + qmax * qr:.1f}",
             qhead=f"{target - q0:.0f}")

    # ---- Fan-out card: the chapter's 100 leaves at p99 (Scenarios.ServingFanout100Leaves).
    f = Scenarios.ServingFanout100Leaves
    slow = 1 - f.all_up
    leaf_budget = 1 - f.required_availability(f.availability)
    n.update(leaves=c(f.n_components), leafpct=f"{100 * f.availability:.0f}", slowpct=f"{100 * slow:.0f}",
             leafbudget=f"{100 * leaf_budget:.2f}")
    # Q2: ASSUMED 40 leaves, each at its own p99.
    ql = 40
    qslow = 1 - f.availability ** ql
    qleaf = 1 - f.availability ** (1 / ql)
    assert (1 - round(qleaf, 5)) ** ql >= f.availability     # the printed leaf percentile holds the target
    n.update(qleaves=c(ql), qslowpct=f"{100 * qslow:.1f}", qleafbudget=f"{100 * qleaf:.3f}",
             qleafpct=f"{100 * (1 - qleaf):.3f}", qfast=f"{f.availability:.2f}")

    # ---- Lifetime cost card: the chapter's first-year LLM scenario (Scenarios.LlmServingFirstYear).
    s = Scenarios.LlmServingFirstYear
    n.update(trainm=f"{s.training_cost / 1e6:.0f}", users=f"{s.daily_users / 1e6:.0f}",
             perday=c(s.requests_per_user_per_day), perrequestcost=f"{s.cost_per_request:.3f}",
             servingm=f"{s.annual_serving_cost / 1e6:.2f}", ratio=f"{s.serving_to_training:.1f}",
             crossweeks=f"{s.crossover_weeks:.0f}")
    # ---- Little's law card and Q4: the chapter's vision service (Scenarios.VisionServingLittlesLaw,
    # cell LittlesLawExample: 1,000 req/s, 100 ms mean, batches of 8 in 80 ms, 80% target -> 100 in flight, 13 replicas).
    ll = Scenarios.VisionServingLittlesLaw
    assert round(ll.in_flight) == 100 and ll.replicas_needed == 13
    lb = ll.batch_size
    lsvc = ll.batch_service_time.to("ms").magnitude
    lu = ll.utilization_target
    # Q4: ASSUMED traffic grows to 1,500 req/s at 120 ms mean time in system; same replicas and target.
    q4rate, q4lat = 1500, 120.0
    q4inflight = q4rate * q4lat / 1000
    q4rep = lb / (lsvc / 1000)
    q4floor = math.ceil(q4rate / q4rep)
    q4need_raw = q4rate / (lu * q4rep)
    q4need = math.ceil(q4need_raw)
    n.update(lbatch=c(lb), lsvc=f"{lsvc:.0f}", lutil=f"{100 * lu:.0f}", lutilfrac=f"{lu:.1f}",
             qlrate=c(q4rate), qllat=f"{q4lat:.0f}", qllats=f"{q4lat / 1000:.2f}", qlinflight=c(q4inflight),
             qlrep=c(q4rep), qlfloor=c(q4floor), qlneedraw=f"{q4need_raw:.2f}", qlneed=c(q4need),
             qlsvcs=f"{lsvc / 1000:.2f}")

    # ---- Mistake: continuous batching (Scenarios.FourRequestDecodeBatch).
    fb = Scenarios.FourRequestDecodeBatch
    n.update(slots=c(fb.slot_iterations), useful=c(fb.useful_iterations), idlepct=f"{100 * fb.idle_share:.1f}",
             advanced=c(fb.requests_advanced), iterations=c(fb.continuous_iterations))

    # ---- Q3: chunked prefill (Scenarios.LongPromptChunkedPrefill).
    cp = Scenarios.LongPromptChunkedPrefill
    assert cp.iterations == math.ceil(cp.prompt_tokens / cp.chunk_tokens)
    n.update(prompt=c(cp.prompt_tokens), chunk=c(cp.chunk_tokens), chunkratio=f"{cp.prompt_tokens / cp.chunk_tokens:.1f}",
             chunkiters=c(cp.iterations))

    # ---- Mistake: average-throughput sizing (chapter pitfall: +25% traffic at 80% utilization),
    # cold start from Scenarios.Llama70BReplicaColdStart.
    util, grow = 0.80, 0.25
    cs = Scenarios.Llama70BReplicaColdStart
    n.update(util=f"{100 * util:.0f}", growth=f"{100 * grow:.0f}", growfactor=f"{1 + grow:.2f}",
             newutil=f"{100 * util * (1 + grow):.0f}", coldmin=f"{cs.cold_start_time.to('minute').magnitude:.1f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
