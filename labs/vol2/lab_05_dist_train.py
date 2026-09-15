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
        gated_hypothesis_card,
        instrumentation_console,
        report_export_panel,
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
        gated_hypothesis_card,
        get_lab_metadata,
        instrumentation_console,
        mo,
        render_distributed_memory_breakdown,
        render_distributed_step_breakdown,
        report_export_panel,
    )


@app.cell
def _(mo):
    # Top-Level Universal Track Selector
    track_dropdown = mo.ui.dropdown(
        options={
            "☁️ Cloud Supercomputing Track (64x H100 Cluster & 3D Parallelism)": "cloud",
            "🤖 Edge & Embodied Track (Multi-Robot Fleet Swarm & Wi-Fi Mesh)": "embodied",
            "📱 Mobile Track (On-Device Federated Learning Across Apple Silicon)": "mobile",
            "⚡ TinyML Track (Microcontroller BLE Mesh & Sensor Swarm)": "tinyml",
        },
        value="☁️ Cloud Supercomputing Track (64x H100 Cluster & 3D Parallelism)",
        label="Select Course / Industry Track",
    )
    return (track_dropdown,)


@app.cell
def _(Models, Systems, track_dropdown):
    # Declarative Track Matrix - Maps academic track to authentic hardware & workload
    track_id = track_dropdown.value or "cloud"
    if "embodied" in str(track_id).lower() or "robot" in str(track_id).lower():
        track_key = "embodied"
        cluster_name = "16x Autonomous Mobile Robots (Jetson AGX Orin)"
        fabric_name = "5.0 GHz Ad-hoc Wi-Fi Mesh (100 MB/s)"
        workload_name = "ViT-Base Robot Perception (86M Params, 12 Layers)"
        num_nodes = 16
        gpus_per_node = 1
        total_gpus = 16
        vram_per_gpu_gb = 32.0
        intra_node_bw_gbs = 204.8  # Orin memory bus
        inter_node_bw_gbs = 0.1    # 100 MB/s Wi-Fi
        param_count = 86_000_000.0
        hidden_dim = 768
        num_layers = 12
        baseline_mem_str = "1.38 GB / Robot"
        scenario_title = "System Scenario: Synchronizing 16 Autonomous Robots over Wireless Mesh"
        scenario_text = (
            "You are the perception and coordination lead for a swarm of 16 autonomous mobile robots inspecting an industrial facility. "
            "Each robot runs a vision transformer perception model locally, and coordinates state updates over an ad-hoc Wi-Fi mesh. "
            "The hard safety deadline requires all robots to exchange perception gradients within a 20 ms control loop."
        )
        law_1 = "Mesh Ingress Constraint: Total mesh throughput capped at 100 MB/s aggregate bandwidth across 16 robots."
        law_2 = "Perception Deadline Law: T_comm + T_compute must be <= 20 ms to prevent obstacle detection stale-state collisions."
        law_3 = "Memory Footprint: On-device model states share 32 GB LPDDR5 with real-time camera DMA ring buffers."
        law_4 = "Decentralized Ring AllReduce: Minimizes RF contention compared to central coordinator star topology."
        hyp_prompt = "Hypothesis Lock: If all 16 robots attempt uncompressed AllReduce synchronization over the 100 MB/s mesh, which failure occurs?"
        hyp_options = {
            "A) Real-Time Loop Deadline Violation: Gradient exchange takes >80 ms, violating the 20 ms control cycle and triggering safety e-stops.": "tp_cliff",
            "B) Robotic Motor Brownout: High RF antenna power draw causes 24V battery bus voltage drop.": "wrong_1",
            "C) Camera DMA Collision: CSI camera frame ingestion stalls due to unified memory lockup.": "wrong_2",
            "D) Tensor Core Throttling: Orin SoC exceeds thermal threshold within 2 seconds.": "wrong_3",
        }
    elif "mobile" in str(track_id).lower() or "phone" in str(track_id).lower():
        track_key = "mobile"
        cluster_name = "100x Apple Silicon Devices (iPhone & Mac)"
        fabric_name = "Wi-Fi 6 Dynamic Uplink (50 MB/s)"
        workload_name = "Llama-3.2-1B On-Device Assistant (1.23B Params, 16 Layers)"
        num_nodes = 100
        gpus_per_node = 1
        total_gpus = 100
        vram_per_gpu_gb = 8.0
        intra_node_bw_gbs = 100.0  # Unified memory
        inter_node_bw_gbs = 0.05   # 50 MB/s Wi-Fi
        param_count = 1_230_000_000.0
        hidden_dim = 2048
        num_layers = 16
        baseline_mem_str = "19.6 GB / Device (OOM Crash)"
        scenario_title = "System Scenario: Federated Adaptation Across 100 Personal Devices"
        scenario_text = (
            "You are the on-device AI platform lead deploying privacy-preserving federated fine-tuning across 100 personal devices. "
            "Full FP16 model updates require transmitting 2.46 GB of weights and gradients over residential Wi-Fi, which triggers operating system kills and battery drain warnings."
        )
        law_1 = "Client Battery Guardrail: Model adaptation must consume < 1% battery per training round."
        law_2 = "Asynchronous Straggler Law: Slowest 5% of devices (thermal throttling on battery) dictate round duration."
        law_3 = "Parameter-Efficient Fine-Tuning: LoRA rank r=16 reduces trainable weights from 1.23B down to 3.2M params."
        law_4 = "Differential Privacy Budget: Noise injection bounds privacy leak per federated epoch."
        hyp_prompt = "Hypothesis Lock: When training 1.23B parameters directly on 8 GB mobile devices without LoRA or sharding, what causes failure?"
        hyp_options = {
            "A) Unified Memory OOM Eviction: iOS / Android OS memory manager terminates the background process when allocation exceeds 4 GB.": "tp_cliff",
            "B) Flash Storage Wear-Out: Flash NAND cells degrade after 100 training iterations.": "wrong_1",
            "C) Neural Engine Quantization Crash: CoreML rejects dynamic backpropagation graphs.": "wrong_2",
            "D) Bluetooth RF Collision: Personal peripheral disconnect occurs during weight broadcast.": "wrong_3",
        }
    elif "tiny" in str(track_id).lower() or "mcu" in str(track_id).lower():
        track_key = "tinyml"
        cluster_name = "64x ESP32-S3 / Cortex-M55 Microcontrollers"
        fabric_name = "BLE 5.0 Mesh Radio (1 Mbps = 125 KB/s)"
        workload_name = "TinyConv Anomaly Detector (250K Params, 6 Layers)"
        num_nodes = 64
        gpus_per_node = 1
        total_gpus = 64
        vram_per_gpu_gb = 0.000512  # 512 KB SRAM
        intra_node_bw_gbs = 0.0016  # Internal bus
        inter_node_bw_gbs = 0.000125 # 125 KB/s BLE
        param_count = 250_000.0
        hidden_dim = 64
        num_layers = 6
        baseline_mem_str = "4.0 MB / MCU (OOM: Exceeds 512 KB SRAM)"
        scenario_title = "System Scenario: Distributed Wake-Word Detection on a 64-Sensor BLE Mesh"
        scenario_text = (
            "You are the firmware lead for a low-power acoustic sensor mesh deployed in an industrial plant. "
            "64 battery-powered microcontrollers must collaboratively adapt to background acoustic noise using distributed learning over a 1 Mbps Bluetooth Low Energy (BLE) mesh."
        )
        law_1 = "Severe Comm-to-Compute Asymmetry: BLE radio bandwidth (125 KB/s) is 1,000x slower than MCU arithmetic throughput."
        law_2 = "Static Tensor Arena Limit: Dynamic heap allocation is prohibited; all states must fit in 256 KB static SRAM."
        law_3 = "Energy Harvesting Envelope: Active radio transmission consumes 30 mW vs 2 mW in deep sleep duty cycle."
        law_4 = "Integer-Only Arithmetic: MCU lacks hardware FP32 FPU; updates require fixed-point INT8 quantization."
        hyp_prompt = "Hypothesis Lock: In a 64-node microcontroller swarm over 1 Mbps BLE, where is 98% of the wall-clock step time spent?"
        hyp_options = {
            "A) Radio Serialization Latency: Transmitting 500 KB weights across the 125 KB/s BLE mesh completely dominates the 15 ms MCU compute time.": "tp_cliff",
            "B) MCU ALU Saturation: Integer multiplication overflows the 32-bit register accumulator.": "wrong_1",
            "C) Flash Memory Read Stalls: Execute-in-place (XIP) bus contention blocks CPU instructions.": "wrong_2",
            "D) Battery Voltage Sag: Coin-cell internal resistance drops below 1.8V reset threshold.": "wrong_3",
        }
    else:  # Cloud
        track_key = "cloud"
        dgx_node = Systems.Nodes.DGX_H100
        cluster_fabric = Systems.Fabrics.InfiniBand_NDR
        llama3_70b = Models.Language.Llama3_70B
        cluster_name = f"8x {dgx_node.name} Nodes (64x H100 80GB)"
        fabric_name = f"{cluster_fabric.name} (50 GB/s) + NVLink 4.0 (900 GB/s)"
        workload_name = f"{llama3_70b.name} (70.6B Params, 80 Layers)"
        num_nodes = 8
        gpus_per_node = int(getattr(dgx_node, "accelerators_per_node", 8))
        total_gpus = 64
        vram_per_gpu_gb = 80.0
        intra_node_bw_gbs = 900.0
        inter_node_bw_gbs = 50.0
        param_count = float(llama3_70b.parameters.magnitude)
        hidden_dim = llama3_70b.hidden_dim
        num_layers = llama3_70b.layers
        baseline_mem_str = "1,145 GB / GPU (OOM Wall)"
        scenario_title = "System Scenario: Training a 70B Frontier LLM on 64 GPUs"
        scenario_text = (
            "You are the lead distributed training architect responsible for training Llama-3-70B on an 8-node cluster of DGX H100 servers. "
            "Each node hosts 8 GPUs connected via NVLink (900 GB/s), while nodes communicate across 400 Gbps InfiniBand (50 GB/s). "
            "Standard Data Parallelism (DP=64) requires 1,145 GB VRAM per GPU, causing an immediate OOM crash against the 80 GB limit. "
            "You must partition parameters and activations using 3D Parallelism (TP x PP x DP = 64) and ZeRO memory sharding."
        )
        law_1 = "Cluster Topology Constraint: TP x PP x DP = 64 GPUs"
        law_2 = "Megatron TP Communication: 2 AllReduce operations per transformer layer on activations across the TP group."
        law_3 = "1F1B Pipeline Bubble Tax: F_bubble = (PP - 1) / (PP - 1 + M), where M is the microbatch count."
        law_4 = "ZeRO Memory Partitioning: ZeRO-1 shards optimizer states across DP; ZeRO-2 shards gradients; ZeRO-3 shards weights."
        hyp_prompt = "Hypothesis Lock: If you set Tensor Parallelism TP=16 to fit weights across 2 nodes (8 GPUs/node), which failure mode occurs?"
        hyp_options = {
            "B) Cross-Node Network Cliff: TP AllReduce crosses the InfiniBand boundary (50 GB/s vs 900 GB/s NVLink), causing ~38x communication latency explosion.": "tp_cliff",
            "A) Pipeline Bubble Tax: The 1F1B schedule bubble expands beyond 50%, starving pipeline stages.": "bubble",
            "C) PCIe Queue Saturation: Host-to-device driver queues overflow as CPU prefetching stalls.": "pcie",
            "D) FP16 Numerical Underflow: High worker count causes gradient accumulation loss precision collapse.": "numerical",
        }

    # Safe fallbacks for simulation objects if not set above
    try:
        dgx_node
    except NameError:
        dgx_node = Systems.Nodes.DGX_H100
    try:
        cluster_fabric
    except NameError:
        cluster_fabric = Systems.Fabrics.InfiniBand_NDR
    try:
        llama3_70b
    except NameError:
        llama3_70b = Models.Language.Llama3_70B
    return (
        baseline_mem_str,
        cluster_fabric,
        cluster_name,
        dgx_node,
        fabric_name,
        gpus_per_node,
        hyp_options,
        hyp_prompt,
        law_1,
        law_2,
        law_3,
        law_4,
        llama3_70b,
        param_count,
        scenario_text,
        scenario_title,
        total_gpus,
        vram_per_gpu_gb,
        workload_name,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    baseline_mem_str,
    cluster_name,
    fabric_name,
    law_1,
    law_2,
    law_3,
    law_4,
    mo,
    scenario_text,
    scenario_title,
    track_dropdown,
    workload_name,
):
    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell">
      <div style="margin-bottom: 16px;">
        {track_dropdown}
      </div>
      <div class="mlsysbook-lab-header" style="border-left: 6px solid #A51C30; background: #FFFFFF; padding: 24px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
          ML Systems Textbook &middot; Volume II &middot; Chapter 05 &middot; Lab 05
        </div>
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 0 0 10px 0; line-height: 1.2;">
          3D Parallelism &amp; Distributed Scaling
        </h1>
        <p style="font-size: 1.05rem; color: #334155; line-height: 1.6; margin: 0 0 16px 0;">
          Decompose frontier deep learning models across distributed nodes using Tensor Parallelism (TP), Pipeline Parallelism (PP), and Data Parallelism with ZeRO memory sharding.
        </p>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Cluster: {cluster_name}
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Fabric: {fabric_name}
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Workload: {workload_name}
          </span>
          <span style="background: #FEF2F2; color: #A51C30; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; border: 1px solid #FECACA;">
            Baseline: {baseline_mem_str}
          </span>
        </div>
      </div>

      <div class="mlsysbook-panel" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #0F172A; font-size: 1.15rem; font-weight: 700;">
          {scenario_title}
        </h3>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 12px;">
          {scenario_text}
        </p>
        <div style="background: #F8FAFC; border-left: 4px solid #006395; padding: 12px 16px; border-radius: 4px; font-size: 0.9rem; color: #1E293B; line-height: 1.6;">
          <strong>The Fundamental Laws of This Architecture:</strong>
          <ul class="mlsysbook-list" style="margin: 8px 0 0 0; padding-left: 1.25rem;">
            <li>{law_1}</li>
            <li>{law_2}</li>
            <li>{law_3}</li>
            <li>{law_4}</li>
          </ul>
        </div>
      </div>
    </div>
    """)
    mo.vstack([ACADEMIC_LAB_CSS, header_html])
    return


@app.cell
def _(gated_hypothesis_card, hyp_options, hyp_prompt, mo):
    # ZONE B: Prediction Widget (Gated Hypothesis Lock)
    pred_scaling_radio = mo.ui.radio(
        options=hyp_options,
        value=None,
    )
    hypothesis_card = gated_hypothesis_card(
        pred_scaling_radio,
        title="1. Formulate Your Scaling Hypothesis",
        subtitle=hyp_prompt,
        gate_label="Required Engineering Gate",
        accent="#A51C30",
    )
    hypothesis_card
    return (pred_scaling_radio,)


@app.cell
def _(mo):
    # ZONE B: Interactive 3D Parallelism Simulation Controls
    tp_radio = mo.ui.radio(
        options={"1": 1, "2": 2, "4": 4, "8": 8, "16": 16},
        value="8",
        label="Tensor Parallelism (TP) — Model slicing across devices",
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


@app.cell(hide_code=True)
def _(
    instrumentation_console,
    microbatch_slider,
    mo,
    pp_radio,
    recompute_radio,
    tp_radio,
    zero_dropdown,
):
    controls_layout = mo.hstack([
        mo.vstack([tp_radio, pp_radio]),
        mo.vstack([zero_dropdown, microbatch_slider, recompute_radio]),
    ], widths="equal", gap=1.5)
    console = instrumentation_console(
        controls_layout,
        title="2. Interactive 3D Parallelism & Memory Controls",
        subtitle="Configure Tensor (TP), Pipeline (PP), and ZeRO memory sharding across the cluster.",
    )
    console
    return


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

    # Activation memory per GPU (using mlsysim.physics.memory)
    try:
        from mlsysim.physics.memory import calc_activation_memory
        act_bytes = calc_activation_memory(
            n_layers=int(num_layers),
            seq_len=4096 if track_key == "cloud" else 512,
            batch_size=max(1, curr_m),
            hidden_dim=int(hidden_dim),
            n_heads=64 if track_key == "cloud" else 12,
            precision_bytes=bpp,
            strategy=curr_recomp,
        )
        mem_act = (act_bytes.m_as("byte") / 1e9) / max(1, curr_tp)
    except Exception:
        if curr_recomp == "full":
            mem_act = (4.0 * curr_m) / model_shards
        elif curr_recomp == "none":
            mem_act = (60.0 * curr_m) / model_shards
        else:  # selective
            mem_act = (16.0 * curr_m) / model_shards

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
        if curr_zero == 3:
            # ZeRO-3 / FSDP requires forward AllGather, backward AllGather, backward ReduceScatter (1.5x volume)
            dp_comm_ms *= 1.5
            step_ms += dp_comm_ms * 0.333
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
        You predicted: <code>{pred_scaling_radio.value}</code> &mdash; Actual system regime: <strong>{audit_status}</strong>.<br/>
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
    cluster_name,
    curr_zero,
    dashboard_view,
    fabric_name,
    get_lab_metadata,
    infeasible_topology,
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
    report_export_panel,
    scenario_text,
    step_ms,
    tokens_per_sec,
    tp_radio,
    tp_spans_nodes,
    track_dropdown,
    vram_allocated_gb,
    vram_per_gpu_gb,
    workload_name,
):
    metadata = get_lab_metadata("vol2/lab_05_dist_train.py")
    report = build_lab_report(
        metadata,
        track=str(track_dropdown.value),
        scenario=scenario_text,
        learning_objectives=(
            "Analyze 3D parallelism trade-offs across tensor, pipeline, and data parallel dimensions.",
            "Characterize memory footprint reductions from ZeRO stage 1, 2, and 3 partitioning.",
            "Evaluate network fabric bottlenecks and the cross-node tensor parallelism latency cliff.",
            "Synthesize a balanced distributed scaling topology satisfying VRAM and throughput constraints.",
        ),
        predictions={"hypothesis": str(pred_scaling_radio.value)},
        knob_settings={
            "tp_size": int(tp_radio.value),
            "pp_size": int(pp_radio.value),
            "microbatches": int(microbatch_slider.value),
            "zero_stage": curr_zero,
            "recomputation": str(recompute_radio.value),
        },
        evidence_summary={
            "vram_allocated_gb": round(vram_allocated_gb, 2),
            "vram_per_gpu_gb": vram_per_gpu_gb,
            "is_oom": is_oom,
            "step_latency_ms": round(step_ms, 1),
            "cluster_throughput_tok_s": round(tokens_per_sec, 1),
            "tp_spans_nodes": tp_spans_nodes,
            "infeasible_topology": infeasible_topology,
        },
        final_decision={
            "tp_size": int(tp_radio.value),
            "pp_size": int(pp_radio.value),
            "microbatches": int(microbatch_slider.value),
            "zero_stage": curr_zero,
            "recomputation": str(recompute_radio.value),
            "vram_gb": vram_allocated_gb,
            "step_latency_ms": step_ms,
            "throughput_tokens_sec": tokens_per_sec,
            "topology_status": "APPROVED" if (not is_oom and not infeasible_topology) else "INVIOLATE_LIMIT_BREACH",
        },
        big_takeaways=(
            "Tensor Parallelism requires high-bandwidth intra-node interconnects (NVLink); spanning nodes across lower-bandwidth fabrics triggers severe latency penalties.",
            "Pipeline Parallelism mitigates communication overhead across nodes but introduces a bubble tax inversely proportional to microbatch depth.",
            "ZeRO memory partitioning dramatically reduces per-device optimizer and parameter footprints without changing computational graph semantics.",
            "Balanced 3D parallelism co-designs TP, PP, DP, and ZeRO against memory limits and physical network topology.",
        ),
        reflections={
            "diagnosis": "Verified 3D parallelism trade-offs against physical hardware limits.",
            "tradeoff": f"Configured TP={int(tp_radio.value)}, PP={int(pp_radio.value)}, ZeRO-{curr_zero} for {workload_name}.",
            "residual_risk": "Real-world scaling may experience stragglers, NCCL communication jitter, or thermal throttling.",
        },
        residual_risk=(
            "Simulated step latencies assume ideal network fabric utilization without congestion or stragglers. "
            "Validate cluster execution traces with production profiling tools (e.g. PyTorch Profiler, Nsight Systems)."
        ),
        source_trace={
            "workload": workload_name,
            "cluster": cluster_name,
            "fabric": fabric_name,
            "chapter_anchors": (
                "#sec-dist-train-3d-parallelism",
                "#sec-dist-train-zero-memory",
                "#sec-dist-train-interconnect-cliffs",
            ),
        },
        result_snapshot={
            "tp_size": int(tp_radio.value),
            "pp_size": int(pp_radio.value),
            "microbatches": int(microbatch_slider.value),
            "zero_stage": curr_zero,
            "recomputation": str(recompute_radio.value),
            "vram_allocated_gb": vram_allocated_gb,
            "step_ms": step_ms,
            "tokens_per_sec": tokens_per_sec,
            "is_oom": is_oom,
            "infeasible_topology": infeasible_topology,
            "pred_is_correct": pred_is_correct,
        },
    )

    def build_part_a():
        return mo.vstack([
            mo.Html("""
            <div style="margin-bottom: 14px;">
              <h3 style="margin: 0 0 4px 0; color: #0F172A; font-size: 1.2rem;">Part A: 3D Cluster Topology &amp; Scaling Explorer</h3>
              <p style="color: #64748B; font-size: 0.92rem; margin: 0;">
                Real-time trade-offs between VRAM footprint, communication overhead, and bubble idle time based on your configuration in the Simulation Knobs above.
              </p>
            </div>
            """),
            dashboard_view,
        ])

    def build_part_b():
        return mo.vstack([
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

    def build_part_c():
        return mo.vstack([
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

    def build_synthesis():
        return mo.vstack([
            mo.Html(audit_card),
            mo.Html("""
            <div class="mlsysbook-panel" style="border-left: 4px solid #A51C30; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">FINAL VERIFICATION & SIGN-OFF</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Lead Architect Authorization</h4>
                <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #475569;">
                    Confirm your 3D parallelism topology, verify all physical invariants, and export your engineering audit record.
                </p>
            </div>
            """),
            report_export_panel(report),
        ])

    tabs = mo.ui.tabs({
        "Part A: 3D Topology Explorer": build_part_a(),
        "Part B: Memory Hierarchy & ZeRO": build_part_b(),
        "Part C: Interconnect Physics": build_part_c(),
        "Synthesis": build_synthesis(),
    })

    hud = mo.Html(f"""
    <div class="lab-hud">
        <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab 05</span></div>
        <div><span class="hud-label">TRACK</span> <span class="hud-value">{cluster_name}</span></div>
        <div><span class="hud-label">VRAM</span> <span class="hud-value">{vram_allocated_gb:.1f} GB</span></div>
        <div><span class="hud-label">STEP</span> <span class="hud-value">{step_ms:,.0f} ms</span></div>
        <div><span class="hud-label">THROUGHPUT</span> <span class="hud-value">{tokens_per_sec:,.0f} tok/s</span></div>
        <div><span class="hud-label">STATUS</span> <span style="color:{'#10B981' if pred_is_correct and not is_oom and not infeasible_topology else '#EF4444'}; font-family:var(--font-mono); font-weight:700;">{'VERIFIED' if pred_is_correct else 'ACTIVE'}</span></div>
    </div>
    <div class="mlsysbook-panel">
      <h2>Design Ledger &amp; Verification</h2>
      <div class="mlsysbook-grid">
        <div class="mlsysbook-field"><strong>Hypothesis Assessment</strong>{'VERIFIED FIRST-PRINCIPLES RATIO ✅' if pred_is_correct else 'HYPOTHESIS MISS ⚠️'}</div>
        <div class="mlsysbook-field"><strong>VRAM Allocation</strong>{vram_allocated_gb:.1f} GB / {vram_per_gpu_gb:.0f} GB</div>
        <div class="mlsysbook-field"><strong>Step Latency</strong>{step_ms:,.0f} ms</div>
        <div class="mlsysbook-field"><strong>Cluster Throughput</strong>{tokens_per_sec:,.0f} tokens/s</div>
        <div class="mlsysbook-field"><strong>Topology Ranks</strong>TP={int(tp_radio.value)} &times; PP={int(pp_radio.value)} &times; DP={64 // max(1, int(tp_radio.value) * int(pp_radio.value))}</div>
        <div class="mlsysbook-field"><strong>ZeRO Stage</strong>ZeRO-{curr_zero}</div>
      </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
