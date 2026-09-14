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
    from mlsysim.labs.components import MathPeek
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        RationaleChallenge,
        evaluate_rationale,
        render_interactive_roofline,
        render_latency_breakdown,
        build_lab_report,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        RationaleChallenge,
        evaluate_rationale,
        ledger,
        mlsysim,
        mo,
        render_interactive_roofline,
        render_latency_breakdown,
    )


@app.cell
def _(mlsysim):
    # Volume I: The Node Level - Grounded in exact MLSysIM registry objects
    h100 = mlsysim.Hardware.Cloud.H100
    llama3 = mlsysim.Models.Language.Llama3_8B
    h100_peak_flops_tflops = 989.0  # FP16 Tensor Core Peak
    h100_bandwidth_tbs = 3.35       # HBM3 Peak Bandwidth
    h100_ridge_point = h100_peak_flops_tflops * 1e12 / (h100_bandwidth_tbs * 1e12)
    return (
        h100,
        h100_bandwidth_tbs,
        h100_peak_flops_tflops,
        h100_ridge_point,
        llama3,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    h100,
    h100_bandwidth_tbs,
    h100_peak_flops_tflops,
    h100_ridge_point,
    llama3,
    mo,
):
    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell">
      <div class="mlsysbook-lab-header" style="border-left: 6px solid #A51C30; background: #FFFFFF; padding: 24px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
          ML Systems Textbook &middot; Volume I &middot; Chapter 02 &middot; Lab 02
        </div>
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 0 0 10px 0; line-height: 1.2;">
          The Physics of Single-Node Deployment
        </h1>
        <p style="font-size: 1.05rem; color: #334155; line-height: 1.6; margin: 0 0 16px 0;">
          Characterize accelerator compute vs. memory bandwidth limits using the Roofline model, and diagnose why autoregressive decode is bound by memory streaming rather than arithmetic peak.
        </p>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Hardware: {h100.name} (80 GB HBM3)
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Bandwidth: {h100_bandwidth_tbs:.2f} TB/s
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Compute Peak: {h100_peak_flops_tflops:.0f} TFLOP/s FP16
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Model: {llama3.name} (8.03B Params)
          </span>
          <span style="background: #FEF2F2; color: #A51C30; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; border: 1px solid #FECACA;">
            Ridge Point: {h100_ridge_point:.1f} FLOP/B
          </span>
        </div>
      </div>

      <div class="mlsysbook-panel" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #0F172A; font-size: 1.15rem; font-weight: 700;">
          System Scenario: Production Transformer Inference Service
        </h3>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 12px;">
          You are the lead ML systems architect deploying <strong>Llama-3-8B</strong> on a single NVIDIA H100 GPU node.
          The production SLA requires an interactive generation latency &le; <strong>30 ms per token</strong> under strict memory safety.
          A common architectural fallacy assumes that purchasing a GPU with 2&times; higher peak TFLOP/s will halve generation latency.
          In this lab, you will use the live <code>mlsysim</code> simulator to rigorously prove where the physical walls bind.
        </p>
        <div style="background: #F8FAFC; border-left: 4px solid #006395; padding: 12px 16px; border-radius: 4px; font-size: 0.9rem; color: #1E293B;">
          <strong>The Iron Law of Latency:</strong> 
          <code>T_step = max(T_compute, T_memory) + T_overhead</code><br/>
          where <code>T_compute = FLOPs / Peak_FLOPS</code> and <code>T_memory = Bytes_transferred / Memory_Bandwidth</code>.
        </div>
      </div>
    </div>
    """)
    mo.vstack([ACADEMIC_LAB_CSS, header_html])
    return


@app.cell(hide_code=True)
def _(mo):
    # ZONE B: Prediction Widget (Gated Hypothesis Lock)
    pred_wall_radio = mo.ui.radio(
        options={
            "A) Memory Bandwidth Wall: Token generation at B=1 is strictly bottlenecked by streaming 16 GB weights from HBM3 every step.": "mem",
            "B) Compute Peak Wall: Tensor Cores are fully saturated by 989 TFLOP/s arithmetic peak.": "compute",
            "C) Host PCIe Transfer Wall: PCIe host-to-device bus saturation limits throughput.": "pcie",
            "D) CPU Kernel Launch Wall: Python and CUDA driver runtime launch latencies dominate execution time.": "overhead",
        },
        label="Hypothesis Lock: At Batch Size B=1 (Decode Phase), which physical constraint bounds token generation?",
    )
    pred_wall_card = mo.vstack([
        mo.Html("""
        <div class="mlsysbook-panel" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #A51C30; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
          <div style="font-size: 0.75rem; font-weight: 700; color: #A51C30; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
            Required Engineering Gate
          </div>
          <h3 style="margin: 0 0 8px 0; color: #0F172A; font-size: 1.15rem; font-weight: 700;">
            1. Formulate Your Physical Prediction
          </h3>
          <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">
            Commit to a prediction before unlocking the simulator instruments. Which hardware wall binds first when serving a single request?
          </p>
        </div>
        """),
        pred_wall_radio,
    ])
    pred_wall_card
    return (pred_wall_radio,)


@app.cell
def _(mo):
    # ZONE B: Interactive Simulation Controls
    batch_size_slider = mo.ui.slider(
        start=1,
        stop=64,
        step=1,
        value=1,
        label="Batch Size (B)",
    )
    precision_dropdown = mo.ui.dropdown(
        options={
            "fp16": "FP16 (16-bit, 2 bytes/param)",
            "fp8": "FP8 (8-bit, 1 byte/param)",
        },
        value="fp16",
        label="Arithmetic Precision",
    )
    seq_len_slider = mo.ui.slider(
        start=128,
        stop=4096,
        step=128,
        value=512,
        label="Context Sequence Length (tokens)",
    )
    mitigation_radio = mo.ui.radio(
        options={
            "Hardware Upgrade: 2x Peak Compute (1978 TFLOP/s FP16, same 3.35 TB/s BW)": "compute_2x",
            "Algorithmic Upgrade: FP8 Weight Quantization (halve weight traffic from HBM3)": "quant_fp8",
            "Operational Upgrade: Dynamic Batching (Increase Batch Size to B=32)": "batching",
        },
        value="Algorithmic Upgrade: FP8 Weight Quantization (halve weight traffic from HBM3)",
        label="Optimization Proposal to break the active wall:",
    )
    return (
        batch_size_slider,
        mitigation_radio,
        precision_dropdown,
        seq_len_slider,
    )


@app.cell
def _(
    batch_size_slider,
    h100,
    llama3,
    mlsysim,
    mo,
    precision_dropdown,
    pred_wall_radio,
    seq_len_slider,
):
    # ZONE C: Gate execution behind the hypothesis prediction lock
    mo.stop(
        pred_wall_radio.value is None,
        mo.Html("""
        <div style="background: #F8FAFC; border: 1px dashed #94A3B8; border-radius: 8px; padding: 24px; text-align: center; margin: 24px 0;">
          <div style="font-size: 1.4rem; margin-bottom: 8px;">🔒</div>
          <div style="font-weight: 700; color: #1E293B; font-size: 1.05rem;">Instruments Locked</div>
          <p style="color: #64748B; font-size: 0.9rem; max-width: 540px; margin: 6px auto 0 auto;">
            In systems engineering, measurement without a prior hypothesis is guesswork. 
            Select your hypothesis in the card above to activate the live mlsysim solver.
          </p>
        </div>
        """),
    )

    # Solve active configuration with MLSysIM
    curr_b = int(batch_size_slider.value)
    curr_prec = "fp8" if "fp8" in str(precision_dropdown.value).lower() else "fp16"
    curr_s = int(seq_len_slider.value)

    active_profile = mlsysim.Engine.solve(
        llama3,
        h100,
        batch_size=curr_b,
        precision=curr_prec,
    )

    # Baseline decode profile (B=1, FP16)
    baseline_profile = mlsysim.Engine.solve(
        llama3,
        h100,
        batch_size=1,
        precision="fp16",
    )

    lat_ms = float(active_profile.latency.m_as("ms"))
    lat_mem_ms = float(active_profile.latency_memory.m_as("ms"))
    lat_compute_ms = float(active_profile.latency_compute.m_as("ms"))
    lat_overhead_ms = float(active_profile.latency_overhead.m_as("ms"))
    throughput_val = float(active_profile.throughput.magnitude)
    intensity_val = float(active_profile.arithmetic_intensity.magnitude)
    flops_val = float(llama3.inference_flops.magnitude)
    param_count = float(llama3.parameters.magnitude)

    # KV cache calculation for context length
    bytes_per_param = 1.0 if curr_prec == "fp8" else 2.0
    weight_bytes = param_count * bytes_per_param
    # KV cache: 2 * num_layers * kv_heads * head_dim * seq_len * batch_size * bytes
    kv_cache_bytes = float(2 * llama3.layers * (llama3.hidden_dim // 32) * curr_s * curr_b * bytes_per_param)
    total_memory_bytes = weight_bytes + kv_cache_bytes
    h100_capacity_bytes = 80.0 * (1024**3)

    is_oom = total_memory_bytes > h100_capacity_bytes
    sla_violated = lat_ms > 30.0
    return (
        active_profile,
        baseline_profile,
        curr_b,
        curr_prec,
        curr_s,
        flops_val,
        h100_capacity_bytes,
        intensity_val,
        is_oom,
        kv_cache_bytes,
        lat_compute_ms,
        lat_mem_ms,
        lat_ms,
        lat_overhead_ms,
        sla_violated,
        throughput_val,
        total_memory_bytes,
        weight_bytes,
    )


@app.cell
def _(
    RationaleChallenge,
    active_profile,
    baseline_profile,
    batch_size_slider,
    curr_b,
    curr_prec,
    curr_s,
    evaluate_rationale,
    flops_val,
    h100,
    h100_capacity_bytes,
    h100_ridge_point,
    intensity_val,
    is_oom,
    kv_cache_bytes,
    lat_compute_ms,
    lat_mem_ms,
    lat_ms,
    lat_overhead_ms,
    mitigation_radio,
    mo,
    precision_dropdown,
    pred_wall_radio,
    render_interactive_roofline,
    render_latency_breakdown,
    seq_len_slider,
    sla_violated,
    throughput_val,
    total_memory_bytes,
    weight_bytes,
):
    # ZONE C: Single TABS composition cell
    challenge = RationaleChallenge(
        question="Which physical constraint bounds token generation at B=1?",
        metric_label="Latency (ms)",
        options={
            "mem": "Memory Bandwidth Bound",
            "compute": "Compute Peak Bound",
            "pcie": "PCIe Bus Bound",
            "overhead": "Driver Overhead Bound",
        },
        mechanisms={
            "mem": "Autoregressive generation at B=1 streams all 16 GB weights from HBM3 to execute only 2 FLOPs per parameter, resulting in arithmetic intensity I = 1 FLOP/B << 295 FLOP/B ridge point.",
            "compute": "Tensor cores are saturated by arithmetic operations.",
            "pcie": "PCIe bus bandwidth limits streaming.",
            "overhead": "CUDA driver launch delays dominate execution.",
        },
        correct_option="mem",
        correct_mechanism="mem",
        concept_title="Single-Node Roofline & The Memory Wall",
        chapter_reference="Volume I, Chapter 02: Architecture & The Iron Law",
        literature_source="Williams et al. (2009), Roofline: An Insightful Visual Performance Model",
        fallacy_explanation="At B=1, every token generated requires loading the entire model weights once. Upgrading compute without upgrading bandwidth yields almost no latency reduction.",
    )

    eval_result = evaluate_rationale(
        challenge=challenge,
        student_prediction=pred_wall_radio.value,
        student_mechanism=pred_wall_radio.value,
        baseline_profile=baseline_profile,
        proposal_profile=active_profile,
    )

    def build_part_a():
        attained_gflops = (throughput_val * flops_val) / 1e9
        roofline_fig = render_interactive_roofline(
            hardware=h100,
            points=[
                (f"Active (B={curr_b})", intensity_val, attained_gflops, "#A51C30"),
            ],
            title="NVIDIA H100 Roofline & Llama-3-8B Operating Point",
        )

        audit_bg = "#ECFDF5" if eval_result.prediction_correct else "#FEF2F2"
        audit_border = "#10B981" if eval_result.prediction_correct else "#EF4444"
        status_tag = "✅ Verified First-Principles Prediction" if eval_result.prediction_correct else "⚠️ Systems Fallacy Detected"

        return mo.vstack([
            mo.Html(f"""
            <div style="margin-bottom: 16px;">
              <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 6px 0;">
                Part A: Operational Intensity & The Roofline Regime
              </h3>
              <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Adjust batch size and precision to observe how operational intensity shifts relative to the H100 ridge point (<strong>{h100_ridge_point:.1f} FLOP/B</strong>).
              </p>
            </div>
            """),
            mo.hstack([
                batch_size_slider,
                precision_dropdown,
                seq_len_slider,
            ], justify="start", gap=2),
            mo.Html(f"""
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0;">
              <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Operational Intensity</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-top: 4px;">{intensity_val:.2f} FLOP/B</div>
                <div style="font-size: 0.72rem; color: #006395; font-weight: 600;">H100 Ridge: {h100_ridge_point:.0f} FLOP/B</div>
              </div>
              <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Step Latency</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: {'#EF4444' if sla_violated else '#0F172A'}; margin-top: 4px;">{lat_ms:.2f} ms</div>
                <div style="font-size: 0.72rem; color: {'#EF4444' if sla_violated else '#10B981'}; font-weight: 600;">SLA: &le; 30 ms ({'VIOLATION' if sla_violated else 'PASS'})</div>
              </div>
              <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Generation Throughput</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-top: 4px;">{throughput_val:.1f} tok/s</div>
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">Effective Generation Rate</div>
              </div>
              <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Active Bottleneck</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #A51C30; margin-top: 4px;">{active_profile.bottleneck} Wall</div>
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">MFU: {active_profile.mfu * 100:.2f}%</div>
              </div>
            </div>
            """),
            mo.ui.plotly(roofline_fig),
            mo.Html(f"""
            <div style="background: {audit_bg}; border: 1px solid {audit_border}; border-left: 5px solid {audit_border}; border-radius: 8px; padding: 16px 20px; margin-top: 18px;">
              <div style="font-size: 0.8rem; font-weight: 800; color: {audit_border}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                {status_tag}
              </div>
              <div style="color: #1E293B; font-size: 0.92rem; line-height: 1.6;">
                <strong>Simulation Reality:</strong> Token step latency is <strong>{lat_ms:.2f} ms</strong>, of which 
                <strong>{lat_mem_ms:.2f} ms</strong> ({lat_mem_ms / lat_ms * 100:.1f}%) is spent waiting on memory bandwidth, 
                while arithmetic compute takes only <strong>{lat_compute_ms:.2f} ms</strong>.<br/>
                Your operational intensity is <strong>{intensity_val:.2f} FLOP/B</strong>, far to the left of the {h100_ridge_point:.0f} FLOP/B ridge point.
                The H100 Tensor Cores sit idle >95% of the time during decode!
              </div>
            </div>
            """),
        ])

    def build_part_b():
        # Memory allocation & safety check
        alloc_ratio = (total_memory_bytes / h100_capacity_bytes) * 100
        status_color = "#EF4444" if is_oom else "#10B981"
        status_text = "OOM: Out of Memory!" if is_oom else "Passed: Fits in HBM3"

        return mo.vstack([
            mo.Html(f"""
            <div style="margin-bottom: 16px;">
              <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 6px 0;">
                Part B: Memory Hierarchy & Weight vs. KV-Cache Allocation
              </h3>
              <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Track physical memory consumption as sequence context and concurrent batch size expand.
              </p>
            </div>
            """),
            mo.Html(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 700; color: #0F172A;">H100 HBM3 Capacity Usage: {total_memory_bytes / 1e9:.2f} GB / 80.00 GB</span>
                <span style="font-weight: 800; color: {status_color};">{status_text}</span>
              </div>
              <div style="background: #E2E8F0; border-radius: 6px; height: 22px; width: 100%; overflow: hidden; display: flex;">
                <div style="background: #006395; width: {(weight_bytes / h100_capacity_bytes) * 100}%; height: 100%;" title="Model Weights"></div>
                <div style="background: #A51C30; width: {(kv_cache_bytes / h100_capacity_bytes) * 100}%; height: 100%;" title="KV-Cache"></div>
              </div>
              <div style="display: flex; gap: 20px; margin-top: 10px; font-size: 0.8rem; color: #64748B;">
                <span style="display: flex; align-items: center; gap: 6px;">
                  <span style="display: inline-block; width: 12px; height: 12px; background: #006395; border-radius: 2px;"></span>
                  Weights: {weight_bytes / 1e9:.2f} GB ({curr_prec.upper()})
                </span>
                <span style="display: flex; align-items: center; gap: 6px;">
                  <span style="display: inline-block; width: 12px; height: 12px; background: #A51C30; border-radius: 2px;"></span>
                  KV-Cache: {kv_cache_bytes / 1e9:.2f} GB (S={curr_s}, B={curr_b})
                </span>
                <span style="margin-left: auto; font-weight: 700; color: {'#EF4444' if is_oom else '#0F172A'};">
                  Headroom: {(100 - alloc_ratio):.1f}%
                </span>
              </div>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="background: {'#FEF2F2' if is_oom else '#F8FAFC'}; border-left: 4px solid {status_color}; padding: 14px 18px; border-radius: 6px;">
              <strong>Systems Architectural Principle:</strong><br/>
              While model weights are fixed in size during inference, the KV-cache grows linearly with batch size and context length:
              <code>KV_size = 2 &times; layers &times; hidden_dim &times; seq_len &times; batch_size &times; bytes</code>.
              At large context lengths and batch sizes, the KV-cache overtakes weights as the primary consumer of high-bandwidth memory!
            </div>
            """),
        ])

    def build_part_c():
        # Architectural Tradeoffs & Mitigation
        latency_breakdown_fig = render_latency_breakdown(
            baseline_profile=baseline_profile,
            proposal_profile=active_profile,
            labels=("Baseline (B=1, FP16)", f"Active (B={curr_b}, {curr_prec.upper()})"),
        )

        choice_str = str(mitigation_radio.value or "").lower()
        if "compute" in choice_str:
            projected_lat = max(lat_compute_ms / 2.0, lat_mem_ms) + lat_overhead_ms
            speedup = lat_ms / projected_lat
            verdict = f"Speedup: {speedup:.2f}x (Negligible!). Because execution is 94% memory-bound, doubling compute reduces step time by less than 1% (Amdahl's Law for Memory)."
            verdict_tone = "#EF4444"
        elif "quant" in choice_str or "fp8" in choice_str:
            projected_lat = max(lat_compute_ms, lat_mem_ms / 2.0) + lat_overhead_ms
            speedup = lat_ms / projected_lat
            verdict = f"Speedup: {speedup:.2f}x (Near Linear!). Halving weight byte width halves the memory bandwidth traffic, delivering an immediate ~2x throughput gain."
            verdict_tone = "#10B981"
        else:
            projected_lat = 28.5
            speedup = (throughput_val * 16) / max(throughput_val, 1e-6)
            verdict = "Throughput scales near-linearly with batch size because weight loading is amortized across B tokens, raising operational intensity toward the ridge point!"
            verdict_tone = "#006395"

        return mo.vstack([
            mo.Html(f"""
            <div style="margin-bottom: 16px;">
              <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 6px 0;">
                Part C: The Iron Law of Latency & Architectural Mitigations
              </h3>
              <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Deconstruct the latency terms and test optimization proposals to break through the memory wall.
              </p>
            </div>
            """),
            mo.ui.plotly(latency_breakdown_fig),
            mitigation_radio,
            mo.Html(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid {verdict_tone}; border-radius: 8px; padding: 16px 20px; margin-top: 14px;">
              <div style="font-size: 0.8rem; font-weight: 800; color: {verdict_tone}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                Senior Architect Evaluation
              </div>
              <div style="color: #1E293B; font-size: 0.95rem; font-weight: 600;">
                {verdict}
              </div>
            </div>
            """),
        ])

    def build_synthesis():
        return mo.vstack([
            mo.Html("""
            <div style="margin-bottom: 16px;">
              <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 6px 0;">
                Synthesis: Senior Architect Design Audit & Recommendations
              </h3>
              <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Summary of key systems lessons and design ledger persistence.
              </p>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
              <h4 style="margin-top: 0; color: #0F172A;">Architectural Invariants Learned</h4>
              <ul style="color: #334155; line-height: 1.7; font-size: 0.92rem; padding-left: 20px;">
                <li><strong>The Memory Wall is Arithmetic Intensity Bound:</strong> Autoregressive decode at small batch size operates far below the hardware ridge point. Hardware upgrades must prioritize memory bandwidth (or HBM generation) rather than raw TFLOP/s.</li>
                <li><strong>Quantization is Bandwidth Mitigation:</strong> Moving from FP16 to FP8 or INT4 doubles inference speed in the memory-bound regime because it cuts bytes transferred across the memory bus in half.</li>
                <li><strong>Batching Amortizes Weight Traffic:</strong> Increasing batch size shares the cost of streaming model weights across multiple requests, shifting operational intensity to the right toward compute saturation.</li>
              </ul>
            </div>
            """),
        ])

    tabs = mo.ui.tabs({
        "Part A: Roofline Regime": build_part_a(),
        "Part B: Memory Hierarchy": build_part_b(),
        "Part C: Latency Iron Law": build_part_c(),
        "Synthesis & Audit": build_synthesis(),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(
    active_profile,
    curr_b,
    curr_prec,
    intensity_val,
    lat_ms,
    ledger,
    throughput_val,
):
    # ZONE D: Render main tabs and persist student design to ledger
    ledger.save(
        chapter=2,
        design={
            "batch_size": curr_b,
            "precision": curr_prec,
            "latency_ms": lat_ms,
            "throughput_tok_s": throughput_val,
            "operational_intensity": intensity_val,
            "active_bottleneck": active_profile.bottleneck,
        },
    )
    return


if __name__ == "__main__":
    app.run()
