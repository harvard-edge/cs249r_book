import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import html
    import sys
    from pathlib import Path

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    import mlsysim
    from mlsysim import Hardware, Models, Systems
    try:
        from mlsysim import DistributedModel, Fleet
    except ImportError:
        from mlsysim.engine.solvers import DistributedModel
        from mlsysim.engine import Fleet
    from mlsysim.labs.components import MathPeek
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        RationaleChallenge,
        RationaleEvaluation,
        evaluate_rationale,
        render_distributed_memory_breakdown,
        render_distributed_step_breakdown,
        build_lab_report,
        get_lab_metadata,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        DistributedModel,
        Fleet,
        Models,
        RationaleChallenge,
        Systems,
        build_lab_report,
        get_lab_metadata,
        mo,
        render_distributed_memory_breakdown,
        render_distributed_step_breakdown,
    )


@app.cell
def _(Models, Systems):
    # Volume II: The Scale Level - Distributed Cluster & Frontier Model
    dgx_node = Systems.Nodes.DGX_H100
    cluster_fabric = Systems.Fabrics.InfiniBand_NDR
    llama3_70b = Models.Language.Llama3_70B

    # Cluster Hardware Dimensions
    num_nodes = 8
    gpus_per_node = int(getattr(dgx_node, "accelerators_per_node", 8))
    total_gpus = num_nodes * gpus_per_node  # 64 H100 GPUs
    vram_per_gpu_gb = 80.0
    intra_node_bw_gbs = 900.0   # NVLink 4.0 bidirectional
    inter_node_bw_gbs = 50.0    # InfiniBand NDR (400 Gbps = 50 GB/s)

    # Workload Parameters
    param_count = float(llama3_70b.parameters.magnitude)
    hidden_dim = llama3_70b.hidden_dim
    num_layers = llama3_70b.layers
    num_heads = llama3_70b.heads
    return (
        cluster_fabric,
        dgx_node,
        gpus_per_node,
        inter_node_bw_gbs,
        intra_node_bw_gbs,
        llama3_70b,
        num_layers,
        param_count,
        total_gpus,
        vram_per_gpu_gb,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    cluster_fabric,
    dgx_node,
    inter_node_bw_gbs,
    intra_node_bw_gbs,
    llama3_70b,
    mo,
    num_layers,
    param_count,
    total_gpus,
    vram_per_gpu_gb,
):
    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell">
      <div class="mlsysbook-lab-header" style="border-left: 6px solid #A51C30; background: #FFFFFF; padding: 24px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
          ML Systems Textbook &middot; Volume II &middot; Chapter 05 &middot; Lab 05
        </div>
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 0 0 10px 0; line-height: 1.2;">
          3D Parallelism &amp; Distributed Training Scaling
        </h1>
        <p style="font-size: 1.05rem; color: #334155; line-height: 1.6; margin: 0 0 16px 0;">
          Decompose frontier LLM training across a 64-GPU supercomputer using Tensor Parallelism (TP), Pipeline Parallelism (PP), and Data Parallelism with ZeRO memory sharding.
        </p>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Cluster: 8x {dgx_node.name} Nodes ({total_gpus}x H100 {vram_per_gpu_gb:.0f}GB)
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Intra-Node: NVLink 4.0 ({intra_node_bw_gbs:.0f} GB/s)
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Inter-Node: {cluster_fabric.name} ({inter_node_bw_gbs:.0f} GB/s)
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Workload: {llama3_70b.name} ({param_count/1e9:.1f}B Params, {num_layers} Layers)
          </span>
          <span style="background: #FEF2F2; color: #A51C30; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; border: 1px solid #FECACA;">
            Baseline Memory: 1,145 GB/GPU (OOM Wall)
          </span>
        </div>
      </div>

      <div class="mlsysbook-panel" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #0F172A; font-size: 1.15rem; font-weight: 700;">
          System Scenario: Training a 70B Frontier LLM on 64 GPUs
        </h3>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 12px;">
          You are the lead distributed training architect responsible for training <strong>Llama-3-70B</strong> on an 8-node cluster of DGX H100 servers.
          Each node hosts 8 GPUs connected via NVLink (900 GB/s), while nodes communicate across 400 Gbps InfiniBand (50 GB/s).
        </p>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 12px;">
          <strong>The Engineering Crisis:</strong> Standard Data Parallelism (DP=64) requires 1,145 GB of VRAM per GPU (141 GB weights + 141 GB gradients + 847 GB Adam FP32 optimizer states + activations), causing an immediate Out-Of-Memory (OOM) crash against the 80 GB physical HBM3 limit.
          To fit the model, you must partition parameters and activations using 3D Parallelism (TP &times; PP &times; DP = 64) and ZeRO memory sharding.
        </p>
        <div style="background: #F8FAFC; border-left: 4px solid #006395; padding: 12px 16px; border-radius: 4px; font-size: 0.9rem; color: #1E293B; line-height: 1.6;">
          <strong>The Fundamental Laws of 3D Parallelism:</strong><br/>
          &bull; <strong>Cluster Topology Constraint:</strong> <code>TP &times; PP &times; DP = 64 GPUs</code><br/>
          &bull; <strong>Megatron TP Communication:</strong> 2 AllReduce operations per transformer layer on activations across the TP group.<br/>
          &bull; <strong>1F1B Pipeline Bubble Tax:</strong> <code>F_bubble = (PP - 1) / (PP - 1 + M)</code>, where <code>M</code> is the microbatch count.<br/>
          &bull; <strong>ZeRO Memory Partitioning:</strong> ZeRO-1 shards optimizer states across DP; ZeRO-2 shards gradients; ZeRO-3 shards weights.
        </div>
      </div>
    </div>
    """)
    mo.vstack([ACADEMIC_LAB_CSS, header_html])
    return


@app.cell(hide_code=True)
def _(mo):
    # ZONE B: Prediction Widget (Gated Hypothesis Lock)
    pred_scaling_radio = mo.ui.radio(
        options={
            "A) Pipeline Bubble Tax: The 1F1B schedule bubble expands beyond 50%, starving pipeline stages.": "bubble",
            "B) Cross-Node Network Cliff: TP AllReduce crosses the InfiniBand boundary (50 GB/s vs 900 GB/s NVLink), causing ~38x communication latency explosion.": "tp_cliff",
            "C) PCIe Queue Saturation: Host-to-device driver queues overflow as CPU prefetching stalls.": "pcie",
            "D) FP16 Numerical Underflow: High worker count causes gradient accumulation loss precision collapse.": "numerical",
        },
        label="Hypothesis Lock: If you set Tensor Parallelism TP=16 to fit weights across 2 nodes (8 GPUs/node), which failure mode occurs?",
    )
    pred_scaling_card = mo.vstack([
        mo.Html("""
        <div class="mlsysbook-panel" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #A51C30; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
          <div style="font-size: 0.75rem; font-weight: 700; color: #A51C30; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
            Required Engineering Gate
          </div>
          <h3 style="margin: 0 0 8px 0; color: #0F172A; font-size: 1.15rem; font-weight: 700;">
            1. Formulate Your Scaling Hypothesis
          </h3>
          <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">
            In large-scale distributed ML, topological placement determines efficiency.
            Before touching the 3D parallelism sliders, predict the primary systems bottleneck if Tensor Parallelism crosses the physical node boundary:
          </p>
        </div>
        """),
        pred_scaling_radio,
    ])
    pred_scaling_card
    return (pred_scaling_radio,)


@app.cell
def _(mo):
    # ZONE B: Interactive 3D Parallelism Simulation Controls
    tp_radio = mo.ui.radio(
        options={"1": 1, "2": 2, "4": 4, "8": 8, "16": 16},
        value="8",
        label="Tensor Parallelism (TP) — Model slicing across GPUs",
    )
    pp_radio = mo.ui.radio(
        options={"1": 1, "2": 2, "4": 4, "8": 8},
        value="1",
        label="Pipeline Parallelism (PP) — Layer slicing across stages",
    )
    microbatch_slider = mo.ui.slider(
        start=4,
        stop=64,
        step=4,
        value=16,
        label="Microbatches per Global Batch (M)",
    )
    zero_dropdown = mo.ui.dropdown(
        options={
            "0": "ZeRO-0 (No Sharding — Full Replicated Parameters & States)",
            "1": "ZeRO-1 (Shard Adam Optimizer States across DP)",
            "2": "ZeRO-2 (Shard Optimizer States + Gradients across DP)",
            "3": "ZeRO-3 (Shard Optimizer + Gradients + Parameters across DP)",
        },
        value="1",
        label="ZeRO Memory Optimization Stage",
    )
    recompute_radio = mo.ui.radio(
        options={
            "selective": "Selective Recomputation (Discard attention activation projections)",
            "full": "Full Recomputation (Recompute all intermediate activations)",
            "none": "No Recomputation (Stash all forward activations in VRAM)",
        },
        value="selective",
        label="Activation Memory Management",
    )
    return (
        microbatch_slider,
        pp_radio,
        recompute_radio,
        tp_radio,
        zero_dropdown,
    )


@app.cell
def _(
    DistributedModel,
    Fleet,
    cluster_fabric,
    dgx_node,
    gpus_per_node,
    llama3_70b,
    microbatch_slider,
    mo,
    param_count,
    pp_radio,
    pred_scaling_radio,
    recompute_radio,
    total_gpus,
    tp_radio,
    vram_per_gpu_gb,
    zero_dropdown,
):
    # ZONE C: Gate execution behind the hypothesis prediction lock
    mo.stop(
        pred_scaling_radio.value is None,
        mo.Html("""
        <div style="background: #F8FAFC; border: 1px dashed #94A3B8; border-radius: 8px; padding: 24px; text-align: center; margin: 24px 0;">
          <div style="font-size: 1.4rem; margin-bottom: 8px;">🔒</div>
          <div style="font-weight: 700; color: #1E293B; font-size: 1.05rem;">Cluster Simulator Locked</div>
          <p style="color: #64748B; font-size: 0.9rem; max-width: 540px; margin: 6px auto 0 auto;">
            Engineering decisions require prior hypothesis formation.
            Commit to a scaling prediction above to activate the live mlsysim distributed physics engine.
          </p>
        </div>
        """),
    )

    curr_tp = int(tp_radio.value)
    curr_pp = int(pp_radio.value)
    curr_m = int(microbatch_slider.value)
    zero_val_str = str(zero_dropdown.value).lower()
    if "3" in zero_val_str:
        curr_zero = 3
    elif "2" in zero_val_str:
        curr_zero = 2
    elif "1" in zero_val_str:
        curr_zero = 1
    else:
        curr_zero = 0
    curr_recomp = str(recompute_radio.value)

    # Topology validation: TP * PP * DP = 64
    infeasible_topology = (curr_tp * curr_pp) > total_gpus
    curr_dp = max(1, total_gpus // (curr_tp * curr_pp)) if not infeasible_topology else 0

    # Physical Memory Breakdown per GPU (Gigabytes)
    # Weights and Gradients: 2 bytes per param in FP16
    bpp = 2.0
    weights_total_gb = (param_count * bpp) / 1e9
    grads_total_gb = (param_count * bpp) / 1e9
    # Adam FP32 optimizer: 12 bytes per param (master weights + momentum + variance)
    opt_total_gb = (param_count * 12.0) / 1e9

    # Sharding across TP and PP
    model_shards = max(1, curr_tp * curr_pp)

    # Apply ZeRO sharding across DP dimension
    dp_shards = max(1, curr_dp)
    mem_weights = (weights_total_gb / model_shards) / (dp_shards if curr_zero == 3 else 1.0)
    mem_grads = (grads_total_gb / model_shards) / (dp_shards if curr_zero >= 2 else 1.0)
    mem_opt = (opt_total_gb / model_shards) / (dp_shards if curr_zero >= 1 else 1.0)

    # Activation memory per GPU
    if curr_recomp == "full":
        mem_act = 4.0 / model_shards
    elif curr_recomp == "none":
        mem_act = 60.0 / model_shards
    else:  # selective
        mem_act = 16.0 / model_shards

    vram_allocated_gb = mem_weights + mem_grads + mem_opt + mem_act
    is_oom = vram_allocated_gb > vram_per_gpu_gb
    tp_spans_nodes = curr_tp > gpus_per_node

    # Simulate distributed physics with mlsysim.DistributedModel
    fleet = Fleet(
        name="DGX-H100-Cluster",
        node=dgx_node,
        fabric=cluster_fabric,
        count=total_gpus // gpus_per_node,
    )
    dist_solver = DistributedModel()

    if not infeasible_topology:
        dist_res = dist_solver.solve(
            model=llama3_70b,
            fleet=fleet,
            tp_size=curr_tp,
            pp_size=curr_pp,
            batch_size=total_gpus,
            microbatch_count=curr_m,
            zero_stage=curr_zero,
            activation_recomputation=(curr_recomp != "none"),
        )
        step_ms = float(dist_res.step_latency_total.m_as("ms"))
        tp_comm_ms = float(dist_res.tp_communication_latency.m_as("ms"))
        dp_comm_ms = float(dist_res.dp_communication_latency.m_as("ms"))
        bubble_ms = float(dist_res.pipeline_bubble_latency.m_as("ms"))
        bubble_frac = float(dist_res.bubble_fraction)
        scaling_eff = float(dist_res.scaling_efficiency)
        compute_ms = max(0.0, step_ms - (tp_comm_ms + dp_comm_ms + bubble_ms))
        mfu_val = float(dist_res.node_profile.mfu)
        tokens_per_sec = (total_gpus * 4096) / (step_ms / 1000.0) if step_ms > 0 else 0.0
    else:
        dist_res = None
        step_ms = 99999.0
        tp_comm_ms = 0.0
        dp_comm_ms = 0.0
        bubble_ms = 0.0
        bubble_frac = 1.0
        scaling_eff = 0.0
        compute_ms = 0.0
        mfu_val = 0.0
        tokens_per_sec = 0.0
    return (
        bubble_frac,
        bubble_ms,
        compute_ms,
        curr_tp,
        curr_zero,
        dp_comm_ms,
        infeasible_topology,
        is_oom,
        mem_act,
        mem_grads,
        mem_opt,
        mem_weights,
        mfu_val,
        scaling_eff,
        step_ms,
        tokens_per_sec,
        tp_comm_ms,
        tp_spans_nodes,
        vram_allocated_gb,
    )


@app.cell
def _(
    RationaleChallenge,
    curr_tp,
    infeasible_topology,
    is_oom,
    pred_scaling_radio,
    step_ms,
    tp_comm_ms,
    tp_spans_nodes,
    vram_allocated_gb,
):
    # ZONE D: Senior Architect Pedagogical Audit
    scaling_challenge = RationaleChallenge(
        question="Which physical failure mode occurs when scaling TP to 16 on an 8-GPU-per-node cluster?",
        metric_label="TP Communication Overhead",
        options={
            "bubble": "Pipeline Bubble Tax",
            "tp_cliff": "Cross-Node Network Cliff (InfiniBand TP AllReduce)",
            "pcie": "PCIe Queue Saturation",
            "numerical": "FP16 Numerical Underflow",
        },
        mechanisms={
            "tp_nvlink_cliff": "TP AllReduce traverses 50 GB/s InfiniBand instead of 900 GB/s NVLink, causing ~38x communication latency explosion.",
            "pp_bubble_idle": "Pipeline stages idle while waiting for microbatch activations to traverse the ring.",
            "zero_prefetch_stall": "Parameter all-gather operations exhaust inter-node bisection bandwidth.",
            "driver_queue_overflow": "Kernel dispatch bottlenecks on single host thread.",
        },
        correct_option="tp_cliff",
        correct_mechanism="tp_nvlink_cliff",
        concept_title="Topological Interconnect Hierarchy & 3D Parallelism Placement",
        chapter_reference="ML Systems Textbook Vol. II, §5.3",
        literature_source="Megatron-LM (Shoeybi et al., 2019) & ZeRO (Rajbhandari et al., 2020)",
        fallacy_explanation="Engineers often treat a GPU cluster as a flat compute pool. In reality, the 18x bandwidth drop from NVLink (900 GB/s) to InfiniBand (50 GB/s) means fine-grained per-layer AllReduces (TP) must NEVER cross the physical node boundary.",
    )

    student_pred = str(pred_scaling_radio.value)
    pred_is_correct = student_pred == "tp_cliff"

    audit_items = []
    if infeasible_topology:
        audit_status = "🔴 INFEASIBLE TOPOLOGY"
        audit_items.append("❌ **Topology Over-allocation:** <code>TP &times; PP &gt; 64</code>. Total allocated model ranks exceed the physical 64-GPU cluster.")
    elif is_oom:
        audit_status = "🔴 MEMORY WALL COLLAPSE (OOM)"
        audit_items.append(f"❌ **Out of Memory:** VRAM allocation ({vram_allocated_gb:.1f} GB) exceeds 80 GB H100 limit. Enable ZeRO-1 or increase TP/PP to shard parameters and optimizer states.")
    elif tp_spans_nodes:
        audit_status = "⚠️ INTER-NODE TP BANDWIDTH CLIFF"
        audit_items.append(f"⚠️ **Severe Network Degrade:** TP={curr_tp} crosses node boundaries. TP AllReduce over 400 Gbps InfiniBand adds {tp_comm_ms:.0f} ms to every step (~38x penalty vs NVLink). Keep TP &le; 8!")
    else:
        audit_status = "🟢 PRODUCTION-GRADE 3D TOPOLOGY"
        audit_items.append("✅ **Topology Aligned:** TP stays strictly inside the 8-GPU NVLink boundary (900 GB/s).")
        audit_items.append(f"✅ **Memory Safe:** Model fits in VRAM ({vram_allocated_gb:.1f} GB / 80 GB).")
        audit_items.append(f"✅ **Step Latency:** {step_ms:,.0f} ms per global batch across 64 GPUs.")

    audit_card = f"""
    <div class="mlsysbook-panel" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid {'#A51C30' if is_oom or infeasible_topology or tp_spans_nodes else '#16A34A'}; border-radius: 8px; padding: 20px; margin-top: 16px;">
      <div style="font-size: 0.75rem; font-weight: 700; color: {'#A51C30' if is_oom or infeasible_topology or tp_spans_nodes else '#16A34A'}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
        Architectural Evaluation &middot; {audit_status}
      </div>
      <h3 style="margin: 0 0 10px 0; color: #0F172A; font-size: 1.15rem; font-weight: 700;">
        Senior Systems Architect Audit
      </h3>
      <p style="color: #334155; line-height: 1.5; font-size: 0.95rem; margin-bottom: 12px;">
        {'<strong>Hypothesis Confirmed:</strong> You correctly anticipated the inter-node network cliff!' if pred_is_correct else '<strong>Hypothesis Miss:</strong> Review the topological interconnect hierarchy below.'}
      </p>
      <ul style="margin: 0 0 12px 20px; padding: 0; color: #475569; font-size: 0.9rem; line-height: 1.6;">
        {''.join(f'<li>{item}</li>' for item in audit_items)}
      </ul>
      <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px 16px; border-radius: 6px; font-size: 0.85rem; color: #64748B;">
        <strong>Literature Principle:</strong> {scaling_challenge.literature_source} &mdash; {scaling_challenge.fallacy_explanation}
      </div>
    </div>
    """
    return audit_card, pred_is_correct


@app.cell
def _(
    bubble_frac,
    bubble_ms,
    compute_ms,
    curr_tp,
    dp_comm_ms,
    infeasible_topology,
    is_oom,
    mem_act,
    mem_grads,
    mem_opt,
    mem_weights,
    mfu_val,
    mo,
    render_distributed_memory_breakdown,
    render_distributed_step_breakdown,
    scaling_eff,
    step_ms,
    tokens_per_sec,
    tp_comm_ms,
    tp_spans_nodes,
    vram_allocated_gb,
    vram_per_gpu_gb,
):
    # ZONE E: Systems Visualizer & Interactive Instrumentation
    mem_chart = render_distributed_memory_breakdown(
        weights_gb=mem_weights,
        grads_gb=mem_grads,
        opt_gb=mem_opt,
        act_gb=mem_act,
        vram_capacity_gb=vram_per_gpu_gb,
    )

    step_chart = render_distributed_step_breakdown(
        compute_ms=compute_ms,
        tp_comm_ms=tp_comm_ms,
        dp_comm_ms=dp_comm_ms,
        bubble_ms=bubble_ms,
        tp_spans_nodes=tp_spans_nodes,
    )

    # State badges
    if infeasible_topology:
        status_banner = mo.callout(mo.md("🛑 **Infeasible Topology:** $TP \\times PP > 64$. Reduce TP or PP."), kind="danger")
    elif is_oom:
        status_banner = mo.callout(mo.md(f"🛑 **Out-of-Memory (OOM):** Per-GPU VRAM requirement (**{vram_allocated_gb:.1f} GB**) exceeds the **80 GB** H100 memory wall! Enable ZeRO-1/2/3 or increase TP/PP sharding."), kind="danger")
    elif tp_spans_nodes:
        status_banner = mo.callout(mo.md(f"⚠️ **Inter-Node TP Cliff:** TP={curr_tp} spans across nodes over InfiniBand (50 GB/s vs 900 GB/s NVLink). Communication latency surged to **{tp_comm_ms:.0f} ms**!"), kind="warn")
    elif bubble_frac > 0.20:
        status_banner = mo.callout(mo.md(f"⚠️ **Pipeline Bubble Waste:** Pipeline idle time fraction is **{bubble_frac*100:.1f}%**! Increase microbatch count M to amortize the 1F1B bubble."), kind="warn")
    else:
        status_banner = mo.callout(mo.md(f"✅ **Optimal 3D Operating Regime:** VRAM fits (**{vram_allocated_gb:.1f} GB** &le; 80 GB), TP confined to NVLink node, MFU = **{mfu_val*100:.1f}%**."), kind="success")

    metrics_html = mo.Html(f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0;">
      <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px; border-top: 3px solid {'#A51C30' if is_oom else '#16A34A'};">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Per-GPU VRAM</div>
        <div style="font-size: 1.4rem; font-weight: 800; color: {'#A51C30' if is_oom else '#0F172A'};">{vram_allocated_gb:.1f} GB</div>
        <div style="font-size: 0.8rem; color: #64748B;">Capacity: {vram_per_gpu_gb:.0f} GB ({'OOM 💥' if is_oom else 'Fits ✅'})</div>
      </div>
      <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px; border-top: 3px solid #006395;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Step Latency</div>
        <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A;">{step_ms:,.0f} ms</div>
        <div style="font-size: 0.8rem; color: #64748B;">TP Comm: {tp_comm_ms:.0f} ms</div>
      </div>
      <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px; border-top: 3px solid #D97706;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Pipeline Bubble</div>
        <div style="font-size: 1.4rem; font-weight: 800; color: {'#A51C30' if bubble_frac > 0.25 else '#0F172A'};">{bubble_frac*100:.1f}%</div>
        <div style="font-size: 0.8rem; color: #64748B;">Idle time: {bubble_ms:.0f} ms</div>
      </div>
      <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px; border-top: 3px solid #4A777A;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Cluster Throughput</div>
        <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A;">{tokens_per_sec:,.0f} tok/s</div>
        <div style="font-size: 0.8rem; color: #64748B;">Scaling Eff: {scaling_eff*100:.1f}%</div>
      </div>
    </div>
    """)

    dashboard_view = mo.vstack([
        status_banner,
        metrics_html,
        mo.ui.plotly(mem_chart),
        mo.ui.plotly(step_chart),
    ])
    return (dashboard_view,)


@app.cell
def _(
    audit_card,
    build_lab_report,
    curr_zero,
    dashboard_view,
    get_lab_metadata,
    is_oom,
    mem_act,
    mem_grads,
    mem_opt,
    mem_weights,
    microbatch_slider,
    mo,
    pp_radio,
    pred_is_correct,
    pred_scaling_radio,
    recompute_radio,
    step_ms,
    tokens_per_sec,
    tp_radio,
    tp_spans_nodes,
    vram_allocated_gb,
    zero_dropdown,
):
    # ZONE F: Design Ledger & Curriculum Tab Layout
    metadata = get_lab_metadata("vol2/lab_05_dist_train.py")
    report = build_lab_report(
        metadata,
        track="cloud_fleet",
        predictions={"hypothesis": str(pred_scaling_radio.value)},
        knob_settings={
            "tp_size": int(tp_radio.value),
            "pp_size": int(pp_radio.value),
            "microbatches": int(microbatch_slider.value),
            "zero_stage": curr_zero,
            "recomputation": str(recompute_radio.value),
        },
        decisions={
            "vram_gb": vram_allocated_gb,
            "step_latency_ms": step_ms,
            "throughput_tokens_sec": tokens_per_sec,
        },
    )

    part_a_view = mo.vstack([
        mo.Html("""
        <div style="margin-bottom: 14px;">
          <h3 style="margin: 0 0 4px 0; color: #0F172A; font-size: 1.2rem;">Part A: 3D Cluster Topology &amp; Scaling Explorer</h3>
          <p style="color: #64748B; font-size: 0.92rem; margin: 0;">
            Adjust Tensor Parallelism, Pipeline Parallelism, and ZeRO stages to observe real-time trade-offs between VRAM footprint, communication overhead, and bubble idle time.
          </p>
        </div>
        """),
        tp_radio,
        pp_radio,
        microbatch_slider,
        zero_dropdown,
        recompute_radio,
        dashboard_view,
    ])

    part_b_view = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px;">
          <h3 style="margin-top: 0; color: #0F172A; font-size: 1.15rem;">Part B: Memory Hierarchy &amp; ZeRO Sharding</h3>
          <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
            A 70B parameter model in mixed-precision (FP16 weights + FP32 Adam optimizer) requires over <strong>1.1 Terabytes</strong> of state without sharding.
            Observe how your selected 3D configuration shards each component:
          </p>
          <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 16px 0; color: #1E293B;">
            <thead>
              <tr style="background: #F8FAFC; border-bottom: 2px solid #CBD5E1; text-align: left;">
                <th style="padding: 10px;">Component</th>
                <th style="padding: 10px;">Unsharded Size</th>
                <th style="padding: 10px;">Current Per-GPU Allocation</th>
                <th style="padding: 10px;">Active Sharding Mechanism</th>
              </tr>
            </thead>
            <tbody>
              <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 10px; font-weight: 600;">Model Weights</td>
                <td style="padding: 8px 10px;">141.2 GB</td>
                <td style="padding: 8px 10px; font-weight: 700; color: #006395;">{mem_weights:.1f} GB</td>
                <td style="padding: 8px 10px;">Sharded across TP={int(tp_radio.value)} &times; PP={int(pp_radio.value)}{' &times; DP' if curr_zero == 3 else ''}</td>
              </tr>
              <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 10px; font-weight: 600;">Gradients</td>
                <td style="padding: 8px 10px;">141.2 GB</td>
                <td style="padding: 8px 10px; font-weight: 700; color: #4A777A;">{mem_grads:.1f} GB</td>
                <td style="padding: 8px 10px;">Sharded across TP={int(tp_radio.value)} &times; PP={int(pp_radio.value)}{' &times; DP' if curr_zero >= 2 else ''}</td>
              </tr>
              <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 10px; font-weight: 600;">Adam Optimizer States</td>
                <td style="padding: 8px 10px;">847.2 GB</td>
                <td style="padding: 8px 10px; font-weight: 700; color: {'#A51C30' if is_oom else '#E06D53'};">{mem_opt:.1f} GB</td>
                <td style="padding: 8px 10px;">Sharded across TP={int(tp_radio.value)} &times; PP={int(pp_radio.value)}{' &times; DP' if curr_zero >= 1 else ''}</td>
              </tr>
              <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 10px; font-weight: 600;">Forward Activations</td>
                <td style="padding: 8px 10px;">~60.0 GB (unrecomputed)</td>
                <td style="padding: 8px 10px; font-weight: 700; color: #D97706;">{mem_act:.1f} GB</td>
                <td style="padding: 8px 10px;">{str(recompute_radio.value).capitalize()}</td>
              </tr>
              <tr style="background: #F8FAFC; font-weight: 700; border-top: 2px solid #CBD5E1;">
                <td style="padding: 10px;">Total Per-GPU VRAM</td>
                <td style="padding: 10px;">1,189.6 GB</td>
                <td style="padding: 10px; color: {'#A51C30' if is_oom else '#16A34A'};">{vram_allocated_gb:.1f} GB</td>
                <td style="padding: 10px;">{'OOM 💥 (Exceeds 80 GB)' if is_oom else 'FITS IN VRAM ✅'}</td>
              </tr>
            </tbody>
          </table>
        </div>
        """),
    ])

    part_c_view = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px;">
          <h3 style="margin-top: 0; color: #0F172A; font-size: 1.15rem;">Part C: Interconnect Physics &amp; The Cross-Node Cliff</h3>
          <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
            A critical systems pitfall in distributed ML is confusing intra-node bandwidth with inter-node fabric bandwidth:
          </p>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;">
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; padding: 14px; border-radius: 6px;">
              <h4 style="margin: 0 0 6px 0; color: #166534; font-size: 0.95rem;">Intra-Node NVLink 4.0</h4>
              <div style="font-size: 1.3rem; font-weight: 800; color: #15803D;">900 GB/s</div>
              <p style="font-size: 0.84rem; color: #166534; margin: 6px 0 0 0;">
                Sub-microsecond latency and massive crossbar switch throughput. Ideal for high-frequency TP AllReduce operations (2 per layer).
              </p>
            </div>
            <div style="background: {'#FEF2F2' if tp_spans_nodes else '#F8FAFC'}; border: 1px solid {'#FECACA' if tp_spans_nodes else '#E2E8F0'}; padding: 14px; border-radius: 6px;">
              <h4 style="margin: 0 0 6px 0; color: {'#991B1B' if tp_spans_nodes else '#334155'}; font-size: 0.95rem;">Inter-Node InfiniBand NDR</h4>
              <div style="font-size: 1.3rem; font-weight: 800; color: {'#B91C1C' if tp_spans_nodes else '#0F172A'};">50 GB/s (400 Gbps)</div>
              <p style="font-size: 0.84rem; color: {'#991B1B' if tp_spans_nodes else '#64748B'}; margin: 6px 0 0 0;">
                18&times; lower bandwidth than NVLink. When TP spans nodes (TP &gt; 8), AllReduce latency explodes from 84 ms to over 3,200 ms!
              </p>
            </div>
          </div>
        </div>
        """),
    ])

    tabs = mo.ui.tabs({
        "Part A: 3D Topology Explorer": part_a_view,
        "Part B: Memory Hierarchy & ZeRO": part_b_view,
        "Part C: Interconnect Physics": part_c_view,
        "Senior Architect Audit": mo.Html(audit_card),
        "Design Ledger": mo.vstack([
            mo.md("### Verification & Ledger Status"),
            mo.md(f"**Hypothesis Assessment:** {'VERIFIED FIRST-PRINCIPLES RATIO ✅' if pred_is_correct else 'HYPOTHESIS MISS ⚠️'}"),
            mo.md(f"**VRAM Requirement:** `{vram_allocated_gb:.1f} GB` (H100 80 GB limit)"),
            mo.md(f"**Step Latency:** `{step_ms:,.0f} ms` | **Cluster Throughput:** `{tokens_per_sec:,.0f} tokens/s`"),
        ]),
    })
    tabs
    return


if __name__ == "__main__":
    app.run()
