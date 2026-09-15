import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 02: Physics of Deployment · MLSysBook")


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
        gated_hypothesis_card,
        instrumentation_console,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        RationaleChallenge,
        evaluate_rationale,
        gated_hypothesis_card,
        instrumentation_console,
        ledger,
        mlsysim,
        mo,
        render_interactive_roofline,
        render_latency_breakdown,
    )


@app.cell
def _(mo):
    # Top-Level Universal Track Selector
    track_dropdown = mo.ui.dropdown(
        options={
            "☁️ Cloud Supercomputing Track (NVIDIA H100 & Llama-3-8B)": "cloud",
            "🤖 Edge & Embodied Track (NVIDIA Jetson Orin & YOLOv8 Perception)": "embodied",
            "📱 Mobile Track (Apple Silicon A17 Pro & MobileNetV2)": "mobile",
            "⚡ TinyML Track (ESP32-S3 Microcontroller & Anomaly Detector)": "tinyml",
        },
        value="☁️ Cloud Supercomputing Track (NVIDIA H100 & Llama-3-8B)",
        label="Select Course / Industry Track",
    )
    return (track_dropdown,)


@app.cell
def _(mlsysim, track_dropdown):
    track_id = track_dropdown.value or "cloud"
    if "embodied" in str(track_id).lower() or "orin" in str(track_id).lower():
        track_key = "embodied"
        active_hardware = mlsysim.Hardware.Edge.JetsonOrinNX
        active_model = mlsysim.Models.Vision.YOLOv8_Nano
        hw_name = "NVIDIA Jetson Orin NX (16 GB LPDDR5)"
        model_name = "YOLOv8-Nano Robot Perception (3.2M Params)"
        peak_flops_tflops = float(active_hardware.compute.peak_flops.m_as("TFLOPs / second"))
        bandwidth_gbs = float(active_hardware.memory.bandwidth.m_as("GB / second"))
        ridge_point = (peak_flops_tflops * 1e12) / (bandwidth_gbs * 1e9)
        vram_capacity_bytes = 16.0 * (1024**3)
        sla_target_ms = 20.0
        sla_label = "Real-Time Control Loop (50 Hz / 20 ms SLA)"
        scenario_title = "System Scenario: Autonomous Mobile Robot Perception"
        scenario_text = (
            "You are the autonomous navigation lead deploying real-time vision perception on an NVIDIA Jetson Orin NX. "
            "To prevent high-speed collisions, the robot's perception pipeline has a strict hard real-time safety deadline of 20 ms per frame. "
            "Engineers frequently assume that because YOLOv8-Nano has only 3.2M parameters, latency will easily beat 5 ms. "
            "However, streaming high-resolution sensor frames and large activation maps across the unified LPDDR5 bus can saturate memory bandwidth."
        )
        laws = [
            "Real-Time Safety Deadline: Frame latency must strictly satisfy T_frame <= 20 ms to prevent obstacle detection stale-state collisions.",
            "Unified Memory Contention: GPU inference shares 102 GB/s LPDDR5 bandwidth with camera DMA ingestion and OS processes.",
            "Tensor Core Quantization: INT8 Tensor Cores deliver 100 TOP/s (4x FP16 throughput) and halve activation footprint.",
            "Batch Amortization Limit: Embodied robotics operates at Batch Size B=1; latency cannot be amortized across concurrent requests."
        ]
        hyp_prompt = "Hypothesis Lock: At Batch Size B=1 on Jetson Orin, which physical subsystem bounds frame processing time?"
        hyp_options = {
            "A) Memory Bandwidth Streaming: Transferring intermediate layer feature maps across the 102 GB/s LPDDR5 bus dominates frame latency.": "mem",
            "B) Arithmetic Core Peak: The 25 TFLOP/s FP16 cores are 100% saturated with matrix multiplications.": "compute",
            "C) Camera DMA Interface: PCIe / MIPI CSI bus driver serialization stalls GPU thread dispatch.": "overhead",
            "D) SoC Thermal Throttle: Thermal management forces immediate 50% CPU underclocking.": "pcie",
        }
        precision_options = {
            "fp16": "FP16 (16-bit float, 2 bytes/param)",
            "int8": "INT8 (8-bit quantized, 1 byte/param)",
        }
        mitigation_options = {
            "Algorithmic Upgrade: TensorRT INT8 Quantization (4x compute peak, halve weight/activation traffic)": "quant_int8",
            "Hardware Upgrade: 2x Peak Compute (50 TFLOP/s, same 102 GB/s LPDDR5 bandwidth)": "compute_2x",
            "Architectural Upgrade: Feature Map Tiling in L2 Cache (Reduce DRAM roundtrips)": "tiling",
        }
    elif "mobile" in str(track_id).lower() or "a17" in str(track_id).lower():
        track_key = "mobile"
        active_hardware = mlsysim.Hardware.Mobile.iPhone15Pro
        active_model = mlsysim.Models.Vision.MobileNetV2
        hw_name = "Apple Silicon A17 Pro (8 GB Unified Memory)"
        model_name = "MobileNetV2 Visual Classifier (3.5M Params)"
        peak_flops_tflops = float(active_hardware.compute.peak_flops.m_as("TFLOPs / second"))
        bandwidth_gbs = float(active_hardware.memory.bandwidth.m_as("GB / second"))
        ridge_point = (peak_flops_tflops * 1e12) / (bandwidth_gbs * 1e9)
        vram_capacity_bytes = 8.0 * (1024**3)
        sla_target_ms = 16.67
        sla_label = "Interactive 60 FPS UI Deadline (16.7 ms SLA)"
        scenario_title = "System Scenario: On-Device Continuous Vision Assistant"
        scenario_text = (
            "You are the on-device ML architect optimizing a real-time vision classifier running in the background of a smartphone. "
            "To deliver smooth 60 FPS interaction without triggering OS background process termination or thermal warnings, "
            "inference must execute under 16.7 ms while drawing less than 3W of average SoC power. "
            "Every gigabyte transferred over mobile LPDDR5 consumes roughly 100 pJ/bit, making memory transfers the primary driver of battery drain."
        )
        laws = [
            "DRAM Energy Penalty: Moving data to/from off-chip DRAM consumes ~100x more energy per bit than on-chip arithmetic.",
            "Thermal Dissipation Limit: Continuous execution must stay within a strict passive thermal budget (<= 5W total SoC TDP).",
            "Depthwise Separable Efficiency: Decouples spatial filtering from channel projection to slash total FLOPs.",
            "Unified Memory Contention: Mobile CPU, GPU, and Neural Engine compete for shared 100 GB/s memory bandwidth."
        ]
        hyp_prompt = "Hypothesis Lock: In continuous on-device mobile inference, why does memory bandwidth dominate battery life and speed?"
        hyp_options = {
            "A) DRAM Streaming Tax: Streaming weights and activation tensors from off-chip DRAM accounts for the vast majority of latency and thermal dissipation.": "mem",
            "B) FP16 ALU Saturation: Mobile GPU ALUs run out of pipeline register stages.": "compute",
            "C) Flash Storage Read Bottleneck: Reading model checkpoints from NAND Flash blocks execution.": "pcie",
            "D) OS Context Switch Overhead: Mobile OS thread scheduler introduces 10 ms jitter per inference.": "overhead",
        }
        precision_options = {
            "fp16": "FP16 (16-bit, 2 bytes/param)",
            "int8": "INT8 (8-bit, 1 byte/param)",
        }
        mitigation_options = {
            "Algorithmic Upgrade: INT8 Weight & Activation Quantization (Cuts memory traffic & energy by 50%)": "quant_int8",
            "Hardware Upgrade: 2x Peak Compute (70 TFLOP/s, same 100 GB/s bandwidth)": "compute_2x",
            "Kernel Fusion: Fused Depthwise + Pointwise Layers (Keep activations in on-chip SRAM)": "fusion",
        }
    elif "tiny" in str(track_id).lower() or "esp32" in str(track_id).lower():
        track_key = "tinyml"
        active_hardware = mlsysim.Hardware.Tiny.ESP32_S3
        active_model = mlsysim.Models.Tiny.AnomalyDetector
        hw_name = "ESP32-S3 AI Microcontroller (512 KB SRAM, 8 MB Flash)"
        model_name = "Tiny AnomalyDetector (270K Params)"
        peak_flops_tflops = float(active_hardware.compute.peak_flops.m_as("TFLOPs / second"))
        bandwidth_gbs = float(active_hardware.memory.bandwidth.m_as("GB / second"))
        ridge_point = (peak_flops_tflops * 1e12) / (bandwidth_gbs * 1e9)
        vram_capacity_bytes = 512.0 * 1024
        sla_target_ms = 10.0
        sla_label = "Coin-Cell Energy Budget (10 ms SLA / 0.4W TDP)"
        scenario_title = "System Scenario: Industrial Acoustic Sensor on Coin-Cell Power"
        scenario_text = (
            "You are the embedded firmware lead deploying predictive maintenance anomaly detection on a factory floor sensor. "
            "The microcontroller runs on a CR2032 coin cell that must last 2 years. "
            "The MCU operates with 512 KB of fast internal SRAM and 8 MB of external SPI Flash. "
            "Reading weights from Flash via Execute-In-Place (XIP) has a throughput of only 80 MB/s—12x slower than SRAM. "
            "If execution stalls waiting on Flash memory bandwidth, the MCU stays in active high-power state longer, rapidly killing the battery."
        )
        laws = [
            "Flash XIP Bandwidth Cliff: External SPI Flash read rate (80 MB/s) is 12x slower than internal SRAM (960 MB/s).",
            "Static SRAM Arena Limit: Total activations and model state must fit within 512 KB static SRAM (no dynamic heap allocations).",
            "Duty Cycle Energy Conservation: Active MCU draw is 400 mW vs 10 uW in deep sleep; latency directly determines battery life.",
            "Fixed-Point Precision: Integer-only pipeline avoids expensive software emulation of floating-point arithmetic."
        ]
        hyp_prompt = "Hypothesis Lock: When running inference directly from Flash XIP on the ESP32-S3, which hardware constraint binds first?"
        hyp_options = {
            "A) Flash XIP Memory Wall: Streaming weights over the 80 MB/s SPI bus throttles the MCU, keeping it in high-power state.": "mem",
            "B) Integer ALU Saturation: The dual Xtensa LX7 cores run out of integer multiplication cycles.": "compute",
            "C) Hardware Watchdog Timeout: Internal timer triggers because execution takes > 1 second.": "overhead",
            "D) SRAM Cell Voltage Decay: High clock rate causes memory bit flips in internal registers.": "pcie",
        }
        precision_options = {
            "int8": "INT8 (8-bit integer, 1 byte/param)",
            "fp16": "FP16 / FP32 (Emulated float, 2 bytes/param)",
        }
        mitigation_options = {
            "Algorithmic Upgrade: INT8 Quantized SRAM Layer Tiling (Keep active weights in SRAM)": "quant_int8",
            "Hardware Upgrade: 2x Peak Core Clock (Same 80 MB/s SPI Flash bus)": "compute_2x",
            "Duty-Cycled Burst Wakeup: Wake MCU only upon acoustic threshold trigger": "duty_cycle",
        }
    else:  # Cloud (Default)
        track_key = "cloud"
        active_hardware = mlsysim.Hardware.Cloud.H100
        active_model = mlsysim.Models.Language.Llama3_8B
        hw_name = "NVIDIA H100 SXM5 (80 GB HBM3)"
        model_name = "Llama-3-8B Autoregressive LLM (8.03B Params)"
        peak_flops_tflops = float(active_hardware.compute.peak_flops.m_as("TFLOPs / second"))
        bandwidth_tbs = float(active_hardware.memory.bandwidth.m_as("TB / second"))
        bandwidth_gbs = bandwidth_tbs * 1000.0
        ridge_point = (peak_flops_tflops * 1e12) / (bandwidth_tbs * 1e12)
        vram_capacity_bytes = 80.0 * (1024**3)
        sla_target_ms = 30.0
        sla_label = "Interactive Generation SLA (<= 30 ms / token)"
        scenario_title = "System Scenario: Production Transformer Inference Service"
        scenario_text = (
            "You are the lead ML systems architect deploying Llama-3-8B on a single NVIDIA H100 GPU node. "
            "The production SLA requires an interactive generation latency <= 30 ms per token under strict memory safety. "
            "A common architectural fallacy assumes that purchasing a GPU with 2x higher peak TFLOP/s will halve generation latency. "
            "In this lab, you will use the live mlsysim simulator to rigorously prove where the physical walls bind."
        )
        laws = [
            "The Iron Law of Latency: T_step = max(T_compute, T_memory) + T_overhead.",
            "Weight Streaming at B=1: In autoregressive decode, every token step must stream all 16 GB weights from HBM3.",
            "Autoregressive Arithmetic Intensity: Computing 2 FLOPs per 2-byte weight gives I = 1.0 FLOP/B << 295 FLOP/B ridge point.",
            "Batch Amortization: Increasing batch size B shares weight memory traffic across B queries, raising operational intensity toward compute saturation."
        ]
        hyp_prompt = "Hypothesis Lock: At Batch Size B=1 (Decode Phase), which physical constraint bounds token generation?"
        hyp_options = {
            "A) Memory Bandwidth Wall: Token generation at B=1 is strictly bottlenecked by streaming 16 GB weights from HBM3 every step.": "mem",
            "B) Compute Peak Wall: Tensor Cores are fully saturated by 989 TFLOP/s arithmetic peak.": "compute",
            "C) Host PCIe Transfer Wall: PCIe host-to-device bus saturation limits throughput.": "pcie",
            "D) CPU Kernel Launch Wall: Python and CUDA driver runtime launch latencies dominate execution time.": "overhead",
        }
        precision_options = {
            "fp16": "FP16 (16-bit float, 2 bytes/param)",
            "fp8": "FP8 (8-bit float, 1 byte/param)",
        }
        mitigation_options = {
            "Algorithmic Upgrade: FP8 Weight Quantization (halve weight traffic from HBM3)": "quant_fp8",
            "Hardware Upgrade: 2x Peak Compute (1978 TFLOP/s FP16, same 3.35 TB/s BW)": "compute_2x",
            "Operational Upgrade: Dynamic Batching (Increase Batch Size to B=32)": "batching",
        }
    return (
        active_hardware,
        active_model,
        bandwidth_gbs,
        hw_name,
        hyp_options,
        hyp_prompt,
        laws,
        mitigation_options,
        model_name,
        peak_flops_tflops,
        precision_options,
        ridge_point,
        scenario_text,
        scenario_title,
        sla_target_ms,
        track_key,
        vram_capacity_bytes,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    bandwidth_gbs,
    hw_name,
    laws,
    mo,
    model_name,
    peak_flops_tflops,
    ridge_point,
    scenario_text,
    scenario_title,
    track_dropdown,
):
    laws_html = "".join([f"<li>{law}</li>" for law in laws])
    bw_str = f"{bandwidth_gbs / 1000.0:.2f} TB/s" if bandwidth_gbs >= 1000.0 else f"{bandwidth_gbs:.1f} GB/s"
    compute_str = f"{peak_flops_tflops:.0f} TFLOP/s" if peak_flops_tflops >= 1.0 else f"{peak_flops_tflops * 1000.0:.1f} GFLOP/s"

    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell">
      <div style="margin-bottom: 16px;">
        {track_dropdown}
      </div>
      <div class="mlsysbook-lab-header" style="border-left: 6px solid #A51C30; background: #FFFFFF; padding: 24px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
          ML Systems Textbook &middot; Volume I &middot; Chapter 02 &middot; Lab 02
        </div>
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 0 0 10px 0; line-height: 1.2;">
          The Physics of Single-Node Deployment
        </h1>
        <p style="font-size: 1.05rem; color: #334155; line-height: 1.6; margin: 0 0 16px 0;">
          Characterize accelerator compute vs. memory bandwidth limits using the Roofline model, and diagnose why execution is bound by memory streaming rather than arithmetic peak.
        </p>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Hardware: {hw_name}
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Bandwidth: {bw_str}
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Compute Peak: {compute_str}
          </span>
          <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
            Workload: {model_name}
          </span>
          <span style="background: #FEF2F2; color: #A51C30; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; border: 1px solid #FECACA;">
            Ridge Point: {ridge_point:.1f} FLOP/B
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
        <div style="background: #F8FAFC; border-left: 4px solid #006395; padding: 12px 16px; border-radius: 4px; font-size: 0.9rem; color: #1E293B;">
          <strong>The Architectural Principles of This Track:</strong>
          <ul class="mlsysbook-list" style="margin: 8px 0 4px 0;">
            {laws_html}
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
    pred_wall_radio = mo.ui.radio(
        options=hyp_options,
        value=None,
    )
    hypothesis_card = gated_hypothesis_card(
        pred_wall_radio,
        title="1. Formulate Your Physical Prediction",
        subtitle=hyp_prompt,
        gate_label="Required Engineering Gate",
        accent="#A51C30",
    )
    hypothesis_card
    return (pred_wall_radio,)


@app.cell
def _(mitigation_options, mo, precision_options, track_key):
    # ZONE B: Interactive Simulation Controls
    batch_size_slider = mo.ui.slider(
        start=1,
        stop=32 if track_key in ["cloud", "mobile", "embodied"] else 4,
        step=1,
        value=1,
        label="Batch Size (B)",
    )
    precision_dropdown = mo.ui.dropdown(
        options=precision_options,
        value=list(precision_options.keys())[0],
        label="Arithmetic Precision",
    )
    seq_len_slider = mo.ui.slider(
        start=128 if track_key == "cloud" else 1,
        stop=2048 if track_key == "cloud" else 16,
        step=128 if track_key == "cloud" else 1,
        value=512 if track_key == "cloud" else 1,
        label="Context Length (Tokens)" if track_key == "cloud" else "Sensor Frames / Patches",
    )
    mitigation_radio = mo.ui.radio(
        options=mitigation_options,
        value=list(mitigation_options.keys())[0],
        label="Optimization Proposal to Break the Active Wall:",
    )
    return (
        batch_size_slider,
        mitigation_radio,
        precision_dropdown,
        seq_len_slider,
    )


@app.cell(hide_code=True)
def _(
    batch_size_slider,
    instrumentation_console,
    mitigation_radio,
    mo,
    precision_dropdown,
    seq_len_slider,
):
    controls_layout = mo.hstack([
        mo.vstack([batch_size_slider, precision_dropdown]),
        mo.vstack([seq_len_slider, mitigation_radio]),
    ], widths="equal", gap=2)
    console = instrumentation_console(
        controls_layout,
        title="2. Interactive Hardware & Workload Controls",
        subtitle="Vary concurrency, numerical precision, and operational parameters to explore the live Roofline:",
    )
    console
    return


@app.cell
def _(
    active_hardware,
    active_model,
    batch_size_slider,
    mlsysim,
    mo,
    precision_dropdown,
    pred_wall_radio,
    seq_len_slider,
    sla_target_ms,
    track_key,
    vram_capacity_bytes,
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
    curr_prec_raw = str(precision_dropdown.value).lower()
    if "fp8" in curr_prec_raw:
        curr_prec = "fp8"
    elif "int8" in curr_prec_raw:
        curr_prec = "int8"
    else:
        curr_prec = "fp16"
    curr_s = int(seq_len_slider.value)

    active_profile = mlsysim.Engine.solve(
        active_model,
        active_hardware,
        batch_size=curr_b,
        precision=curr_prec,
    )

    # Baseline profile (B=1, FP16 or INT8)
    baseline_prec = "int8" if track_key == "tinyml" else "fp16"
    baseline_profile = mlsysim.Engine.solve(
        active_model,
        active_hardware,
        batch_size=1,
        precision=baseline_prec,
    )

    lat_ms = float(active_profile.latency.m_as("ms"))
    lat_mem_ms = float(active_profile.latency_memory.m_as("ms"))
    lat_compute_ms = float(active_profile.latency_compute.m_as("ms"))
    lat_overhead_ms = float(active_profile.latency_overhead.m_as("ms"))
    throughput_val = float(active_profile.throughput.magnitude)
    intensity_val = float(active_profile.arithmetic_intensity.magnitude)
    flops_val = float(active_model.inference_flops.magnitude) if hasattr(active_model, "inference_flops") else 1.0
    param_count = float(active_model.parameters.magnitude) if hasattr(active_model, "parameters") else 1.0

    bytes_per_param = 1.0 if curr_prec in ["fp8", "int8"] else 2.0
    weight_bytes = param_count * bytes_per_param

    if track_key == "cloud":
        # KV cache calculated via mlsysim model method accounting for GQA heads
        if hasattr(active_model, "get_kv_cache_size"):
            kv_cache_bytes = float(active_model.get_kv_cache_size(seq_len=curr_s, batch_size=curr_b, precision=bytes_per_param * ureg.byte).m_as("byte"))
        else:
            kv_cache_bytes = float(2 * active_model.layers * getattr(active_model, "kv_heads", 8) * (active_model.hidden_dim // getattr(active_model, "heads", 32)) * curr_s * curr_b * bytes_per_param)
        activation_label = "KV-Cache"
        activation_bytes = kv_cache_bytes
    elif track_key == "embodied":
        # Feature maps across layers
        activation_bytes = float(curr_b * 640 * 640 * 3 * 2 * bytes_per_param)
        activation_label = "Feature Maps"
    elif track_key == "mobile":
        activation_bytes = float(curr_b * 224 * 224 * 3 * 4 * bytes_per_param)
        activation_label = "Activations"
    else:  # tinyml
        activation_bytes = float(curr_b * 32 * 1024 * bytes_per_param)
        activation_label = "Tensor Arena"

    total_memory_bytes = weight_bytes + activation_bytes
    is_oom = total_memory_bytes > vram_capacity_bytes
    sla_violated = lat_ms > sla_target_ms
    return (
        activation_bytes,
        activation_label,
        active_profile,
        baseline_profile,
        curr_b,
        curr_prec,
        flops_val,
        intensity_val,
        is_oom,
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
    activation_bytes,
    activation_label,
    active_hardware,
    active_model,
    active_profile,
    baseline_profile,
    curr_b,
    curr_prec,
    evaluate_rationale,
    flops_val,
    hyp_options,
    hyp_prompt,
    intensity_val,
    is_oom,
    lat_compute_ms,
    lat_mem_ms,
    lat_ms,
    lat_overhead_ms,
    mitigation_radio,
    mo,
    pred_wall_radio,
    render_interactive_roofline,
    render_latency_breakdown,
    ridge_point,
    sla_target_ms,
    sla_violated,
    throughput_val,
    total_memory_bytes,
    track_key,
    vram_capacity_bytes,
    weight_bytes,
):
    # ZONE C: Single TABS composition cell
    challenge = RationaleChallenge(
        question=hyp_prompt,
        metric_label="Latency (ms)",
        options=hyp_options,
        mechanisms={k: k for k in hyp_options},
        correct_option=list(hyp_options.keys())[0],
        correct_mechanism=list(hyp_options.keys())[0],
        concept_title="Single-Node Roofline & The Memory Wall",
        chapter_reference="Volume I, Chapter 02: Architecture & The Iron Law",
        literature_source="Williams et al. (2009), Roofline: An Insightful Visual Performance Model",
        fallacy_explanation="When operational intensity lies below the ridge point, upgrading compute capacity without increasing memory bandwidth yields negligible latency improvement (Amdahl's Law for Memory).",
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
            hardware=active_hardware,
            points=[
                (f"Active (B={curr_b})", intensity_val, attained_gflops, "#A51C30"),
            ],
            title=f"{active_hardware.name} Roofline & {active_model.name} Operating Point",
        )

        audit_bg = "#ECFDF5" if eval_result.prediction_correct else "#FEF2F2"
        audit_border = "#10B981" if eval_result.prediction_correct else "#EF4444"
        status_tag = "✅ Verified First-Principles Prediction" if eval_result.prediction_correct else "⚠️ Systems Fallacy Detected"

        return mo.vstack([
            mo.Html(f"""
            <div style="margin-bottom: 16px;">
              <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 6px 0;">
                Part A: Operational Intensity &amp; The Roofline Regime
              </h3>
              <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Observe how operational intensity compares to the hardware ridge point (<strong>{ridge_point:.1f} FLOP/B</strong>) on {active_hardware.name}.
              </p>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0;">
              <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Operational Intensity</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-top: 4px;">{intensity_val:.2f} FLOP/B</div>
                <div style="font-size: 0.72rem; color: #006395; font-weight: 600;">Ridge: {ridge_point:.0f} FLOP/B</div>
              </div>
              <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Step Latency</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: {'#EF4444' if sla_violated else '#0F172A'}; margin-top: 4px;">{lat_ms:.2f} ms</div>
                <div style="font-size: 0.72rem; color: {'#EF4444' if sla_violated else '#10B981'}; font-weight: 600;">SLA: &le; {sla_target_ms:.1f} ms ({'VIOLATION' if sla_violated else 'PASS'})</div>
              </div>
              <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Throughput</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-top: 4px;">{throughput_val:.1f} inferences/s</div>
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">Effective Service Rate</div>
              </div>
              <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Active Bottleneck</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #A51C30; margin-top: 4px;">{active_profile.bottleneck} Wall</div>
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">MFU: {getattr(active_profile, 'mfu', 0.0) * 100:.2f}%</div>
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
                <p style="margin: 0 0 6px 0;">
                  You predicted: <code>{pred_wall_radio.value}</code> &mdash; Actual bottleneck: <strong>{active_profile.bottleneck} Wall</strong>.
                </p>
                <strong>Simulation Reality:</strong> Execution step latency is <strong>{lat_ms:.2f} ms</strong>, of which 
                <strong>{lat_mem_ms:.2f} ms</strong> ({lat_mem_ms / max(lat_ms, 1e-6) * 100:.1f}%) is memory transfer overhead, 
                while arithmetic compute takes <strong>{lat_compute_ms:.2f} ms</strong>.<br/>
                Your operational intensity is <strong>{intensity_val:.2f} FLOP/B</strong> compared to the {ridge_point:.0f} FLOP/B ridge point.
                The accelerator ALUs sit starved for data during single-item execution!
              </div>
            </div>
            """),
        ])

    def build_part_b():
        alloc_ratio = (total_memory_bytes / max(vram_capacity_bytes, 1.0)) * 100
        status_color = "#EF4444" if is_oom else "#10B981"
        status_text = "OOM: Out of Memory!" if is_oom else "Passed: Fits in Memory"
        cap_str = f"{vram_capacity_bytes / 1e9:.2f} GB" if vram_capacity_bytes >= 1e9 else f"{vram_capacity_bytes / 1024:.0f} KB"
        tot_str = f"{total_memory_bytes / 1e9:.2f} GB" if total_memory_bytes >= 1e9 else f"{total_memory_bytes / 1024:.1f} KB"
        wt_str = f"{weight_bytes / 1e9:.2f} GB" if weight_bytes >= 1e9 else f"{weight_bytes / 1024:.1f} KB"
        act_str = f"{activation_bytes / 1e9:.2f} GB" if activation_bytes >= 1e9 else f"{activation_bytes / 1024:.1f} KB"

        return mo.vstack([
            mo.Html(f"""
            <div style="margin-bottom: 16px;">
              <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 6px 0;">
                Part B: Memory Hierarchy &amp; Allocation Safety
              </h3>
              <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Track physical memory consumption as sequence context, resolution, and batch size expand.
              </p>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 700; color: #0F172A;">{active_hardware.name} Capacity Usage: {tot_str} / {cap_str}</span>
                <span style="font-weight: 800; color: {status_color};">{status_text}</span>
              </div>
              <div style="background: #E2E8F0; border-radius: 6px; height: 22px; width: 100%; overflow: hidden; display: flex;">
                <div style="background: #006395; width: {min(100.0, (weight_bytes / max(vram_capacity_bytes, 1.0)) * 100)}%; height: 100%;" title="Model Weights"></div>
                <div style="background: #A51C30; width: {min(100.0, (activation_bytes / max(vram_capacity_bytes, 1.0)) * 100)}%; height: 100%;" title="{activation_label}"></div>
              </div>
              <div style="display: flex; gap: 20px; margin-top: 10px; font-size: 0.8rem; color: #64748B;">
                <span style="display: flex; align-items: center; gap: 6px;">
                  <span style="display: inline-block; width: 12px; height: 12px; background: #006395; border-radius: 2px;"></span>
                  Weights: {wt_str} ({curr_prec.upper()})
                </span>
                <span style="display: flex; align-items: center; gap: 6px;">
                  <span style="display: inline-block; width: 12px; height: 12px; background: #A51C30; border-radius: 2px;"></span>
                  {activation_label}: {act_str} (B={curr_b})
                </span>
                <span style="margin-left: auto; font-weight: 700; color: {'#EF4444' if is_oom else '#0F172A'};">
                  Headroom: {max(0.0, 100.0 - alloc_ratio):.1f}%
                </span>
              </div>
            </div>
            <div class="mlsysbook-panel" style="background: {'#FEF2F2' if is_oom else '#F8FAFC'}; border-left: 4px solid {status_color}; padding: 14px 18px; border-radius: 6px;">
              <strong>Systems Architectural Principle:</strong><br/>
              While model weights remain fixed in size during inference, activations and KV-cache buffers grow dynamically with batch size and context length.
              In memory-constrained hardware (e.g., 512 KB MCU SRAM or shared mobile LPDDR5), activation footprint frequently dictates the maximum feasible batch size before an Out-Of-Memory crash occurs!
            </div>
            """),
        ])

    def build_part_c():
        baseline_prec_label = "INT8" if track_key == "tinyml" else "FP16"
        latency_breakdown_fig = render_latency_breakdown(
            baseline_profile=baseline_profile,
            proposal_profile=active_profile,
            labels=(f"Baseline (B=1, {baseline_prec_label})", f"Active (B={curr_b}, {curr_prec.upper()})"),
        )

        choice_str = str(mitigation_radio.value or "").lower()
        if "compute" in choice_str:
            projected_lat = max(lat_compute_ms / 2.0, lat_mem_ms) + lat_overhead_ms
            speedup = lat_ms / max(projected_lat, 1e-6)
            verdict = f"Speedup: {speedup:.2f}x (Negligible!). Because execution is memory-bandwidth bound (low operational intensity), doubling peak compute reduces step time by less than 2% (Amdahl's Law for Memory)."
            verdict_tone = "#EF4444"
        elif "quant" in choice_str or "fp8" in choice_str or "int8" in choice_str:
            projected_lat = max(lat_compute_ms, lat_mem_ms / 2.0) + lat_overhead_ms
            speedup = lat_ms / max(projected_lat, 1e-6)
            verdict = f"Speedup: {speedup:.2f}x (Near Linear!). Halving weight byte width cuts memory bus traffic in half, delivering immediate latency reduction and energy savings."
            verdict_tone = "#10B981"
        else:
            speedup = (throughput_val * 4) / max(throughput_val, 1e-6)
            verdict = "Throughput scales near-linearly because weight streaming overhead is amortized across items or kept on-chip in fast SRAM/cache."
            verdict_tone = "#006395"

        return mo.vstack([
            mo.Html(f"""
            <div style="margin-bottom: 16px;">
              <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 6px 0;">
                Part C: The Iron Law of Latency &amp; Architectural Mitigations
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
            mo.Html(f"""
            <div style="margin-bottom: 16px;">
              <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 6px 0;">
                Synthesis: Senior Architect Design Audit &amp; Recommendations
              </h3>
              <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Summary of key systems invariants learned on the <strong>{active_hardware.name}</strong> track.
              </p>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
              <h4 style="margin-top: 0; color: #0F172A;">Architectural Invariants Learned</h4>
              <ul class="mlsysbook-list" style="margin: 8px 0 4px 0;">
                <li><strong>The Memory Wall is Arithmetic Intensity Bound:</strong> Single-item inference operates far below the hardware ridge point. Hardware upgrades must prioritize memory bandwidth (or memory generation) rather than raw peak compute.</li>
                <li><strong>Quantization is Bandwidth Mitigation:</strong> Lower precision (INT8/FP8) doubles speed in the memory-bound regime because it cuts bytes transferred across the memory bus in half.</li>
                <li><strong>Batch Amortization:</strong> Increasing batch size shares the cost of streaming model weights across multiple queries, shifting operational intensity to the right toward compute saturation.</li>
                <li><strong>On-Device Energy Proportionality:</strong> Off-chip memory transfers consume ~100x more energy than on-chip arithmetic, making memory streaming the dominant factor in battery drain.</li>
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
    active_hardware,
    active_model,
    active_profile,
    curr_b,
    curr_prec,
    intensity_val,
    lat_ms,
    ledger,
    mo,
    throughput_val,
    track_key,
):
    # ZONE D: Persist student design to ledger
    ledger.save(
        chapter=2,
        design={
            "track": track_key,
            "hardware": active_hardware.name,
            "model": active_model.name,
            "batch_size": curr_b,
            "precision": curr_prec,
            "latency_ms": lat_ms,
            "throughput": throughput_val,
            "operational_intensity": intensity_val,
            "active_bottleneck": active_profile.bottleneck,
        },
    )
    return mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">02 &middot; ML Systems &amp; Roofline</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{track_key}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">HARDWARE</span>
        <span class="hud-value">{active_hardware.name}</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)


if __name__ == "__main__":
    app.run()
