#!/usr/bin/env python3
"""
Generator for TinyTorch Tier 3 (Optimizations) Mechanical SVG Diagrams.
Adheres strictly to tinytorch/quarto/tools/diagrams/STYLE.md:
- 680px viewBox width
- Height sized to content + padding
- ~85% greyscale, max 1 accent node in #fff1e8 fill / #ff8246 stroke
- Uniform 1pt stroke weights, rx="2" rounded corners
- Standard polygon arrowheads, TeX Gyre Heros / Helvetica font stack
- Synchronizes output to narrative_book/ and quarto/ asset directories
"""

from pathlib import Path

DEST_DIRS = [
    Path("tinytorch/narrative_book/assets/images/diagrams"),
    Path("tinytorch/quarto/assets/images/diagrams"),
]

HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="680" height="{height}" viewBox="0 0 680 {height}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="680" height="{height}" fill="#ffffff"/>
<defs>
  <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#1f2937"/>
  </marker>
  <marker id="arrow-gray" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#9ca3af"/>
  </marker>
  <marker id="arrow-orange" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#ff8246"/>
  </marker>
</defs>
"""

FOOTER = "</svg>\n"


def write_svg(filename: str, content: str):
    for d in DEST_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Generated: {filename}")


# -------------------------------------------------------------
# 1. Ch 14: 14_weight-streaming-reuse.svg
# -------------------------------------------------------------
def gen_14_profiling():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">THE ROOFLINE REGIME: INFERENCE VS TRAINING COMPUTE INTENSITY</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Why single-token generation starves hardware ALUs: arithmetic intensity R = FLOPs / Byte</text>

  <!-- Left Box: Single Token Inference (Memory-Bound) -->
  <rect x="35" y="65" width="295" height="225" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">1. Autoregressive Decode (Batch=1)</text>
  <text x="45" y="96" font-size="8.5" fill="#6b7280">Memory-Bound Regime: R = 1 FLOP / Byte</text>

  <!-- Flow in DRAM -->
  <rect x="45" y="112" width="275" height="42" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="53" y="126" font-size="8" font-family="monospace" fill="#1f2937">Stream 7B Weights from VRAM (14 GB / token)</text>
  <text x="53" y="142" font-size="7.5" font-family="monospace" fill="#6b7280">Memory bus at 100% capacity (1000 GB/s)</text>

  <line x1="182" y1="156" x2="182" y2="174" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- ALU Utilization -->
  <rect x="45" y="176" width="275" height="46" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="53" y="191" font-size="8" font-family="monospace" fill="#1f2937">Hardware Tensor Cores: 97% IDLE</text>
  <text x="53" y="207" font-size="7.5" font-family="monospace" fill="#6b7280">Each weight byte used exactly ONCE before eviction</text>

  <text x="45" y="244" font-size="8.5" font-weight="bold" fill="#1f2937">Bottleneck: Memory Bandwidth</text>
  <text x="45" y="258" font-size="8" fill="#6b7280">Faster GPUs with more TFLOPs do not speed</text>
  <text x="45" y="272" font-size="8" fill="#6b7280">up single-user generation without HBM upgrades.</text>

  <!-- Right Box: Batched Training (Compute-Bound, ACCENT) -->
  <rect x="345" y="65" width="300" height="225" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="357" y="83" font-size="10" font-weight="700" fill="#1f2937">2. Batched Training / Prefill (Batch=64)</text>
  <text x="357" y="96" font-size="8.5" fill="#6b7280">Compute-Bound Regime: R = 64 FLOPs / Byte</text>

  <!-- Stream in DRAM -->
  <rect x="357" y="112" width="276" height="42" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="365" y="126" font-size="8" font-family="monospace" fill="#1f2937">Stream Weights ONCE per Batch</text>
  <text x="365" y="142" font-size="7.5" font-family="monospace" fill="#6b7280">Loaded into SRAM / register cache once</text>

  <line x1="495" y1="156" x2="495" y2="174" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>

  <!-- ALU Utilization -->
  <rect x="357" y="176" width="276" height="46" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="365" y="191" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">Hardware Tensor Cores: 95% SATURATED</text>
  <text x="365" y="207" font-size="7.5" font-family="monospace" fill="#374151">Each weight multiplied across 64 token vectors</text>

  <text x="357" y="244" font-size="8.5" font-weight="bold" fill="#1f2937">Peak FLOP Efficiency</text>
  <text x="357" y="258" font-size="8" fill="#6b7280">High arithmetic intensity amortizes memory transfer,</text>
  <text x="357" y="272" font-size="8" fill="#6b7280">allowing execution to hit the roofline ceiling.</text>
{FOOTER}"""
    write_svg("14_weight-streaming-reuse.svg", body)


# -------------------------------------------------------------
# 2. Ch 15: 15_affine-quantization-grid.svg
# -------------------------------------------------------------
def gen_15_quantization():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">AFFINE UNIFORM QUANTIZATION: CONTINUOUS FLOAT32 TO DISCRETE INT8</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">q = clamp(round(x / S) + Z, -128, 127): exact zero representation eliminates padding error</text>

  <!-- Upper Axis: Float32 Continuous Space -->
  <rect x="35" y="65" width="610" height="95" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="82" font-size="10" font-weight="700" fill="#1f2937">Continuous Float32 Domain [alpha = -3.2, beta = +4.8]</text>
  <text x="450" y="82" font-size="8" font-family="monospace" fill="#6b7280">Scale S = (beta - alpha) / 255 = 0.03137</text>

  <!-- Number Line Float32 -->
  <line x1="60" y1="115" x2="620" y2="115" stroke="#1f2937" stroke-width="1.5"/>

  <!-- Ticks & Values -->
  <line x1="80" y1="110" x2="80" y2="120" stroke="#1f2937" stroke-width="1"/>
  <text x="80" y="132" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">-3.20 (min)</text>

  <line x1="220" y1="110" x2="220" y2="120" stroke="#1f2937" stroke-width="1"/>
  <text x="220" y="132" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">-1.50</text>

  <!-- Zero Point Float Highlight -->
  <line x1="320" y1="106" x2="320" y2="124" stroke="#ff8246" stroke-width="2"/>
  <circle cx="320" cy="115" r="3" fill="#ff8246"/>
  <text x="320" y="102" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#ff8246">0.00 (Real Zero)</text>
  <text x="320" y="132" text-anchor="middle" font-size="8" font-family="monospace" fill="#ff8246">x = 0.0</text>

  <line x1="450" y1="110" x2="450" y2="120" stroke="#1f2937" stroke-width="1"/>
  <text x="450" y="132" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">+2.10</text>

  <line x1="590" y1="110" x2="590" y2="120" stroke="#1f2937" stroke-width="1"/>
  <text x="590" y="132" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">+4.80 (max)</text>

  <!-- Lower Axis: Discrete INT8 Grid (ACCENT) -->
  <rect x="35" y="172" width="610" height="118" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="45" y="189" font-size="10" font-weight="700" fill="#1f2937">Quantized Discrete INT8 Domain [-128, +127]</text>
  <text x="430" y="189" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">Zero-point Z = round(-alpha / S) - 128 = -26</text>

  <!-- INT8 Number line -->
  <line x1="60" y1="225" x2="620" y2="225" stroke="#ff8246" stroke-width="1.5"/>

  <!-- Left Clamping Shelf -->
  <rect x="45" y="215" width="35" height="20" rx="1" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="62" y="228" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">Clamp</text>
  <text x="80" y="244" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">-128</text>

  <!-- Zero Point INT8 tick -->
  <line x1="320" y1="216" x2="320" y2="234" stroke="#ff8246" stroke-width="2"/>
  <circle cx="320" cy="225" r="3" fill="#ff8246"/>
  <text x="320" y="244" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">q = -26 (Z)</text>

  <!-- Right Clamping Shelf -->
  <line x1="590" y1="220" x2="590" y2="230" stroke="#1f2937" stroke-width="1"/>
  <text x="590" y="244" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">+127</text>
  <rect x="595" y="215" width="35" height="20" rx="1" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="612" y="228" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">Clamp</text>

  <!-- Mapping projection lines -->
  <line x1="80" y1="135" x2="80" y2="215" stroke="#9ca3af" stroke-dasharray="3,2"/>
  <line x1="320" y1="135" x2="320" y2="215" stroke="#ff8246" stroke-width="1"/>
  <line x1="590" y1="135" x2="590" y2="215" stroke="#9ca3af" stroke-dasharray="3,2"/>

  <text x="45" y="266" font-size="8.5" fill="#1f2937">Zero Distortion Invariant: Zero in float32 maps exactly to integer Z without rounding error.</text>
  <text x="45" y="280" font-size="8" fill="#6b7280">This ensures tensor padding (0.0) and ReLU activations do not accumulate numerical bias during inference.</text>
{FOOTER}"""
    write_svg("15_affine-quantization-grid.svg", body)


# -------------------------------------------------------------
# 3. Ch 16: 16_svd-rank-breakeven.svg
# -------------------------------------------------------------
def gen_16_compression():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">LOW-RANK COMPRESSION: SVD FACTORIZATION &amp; BREAK-EVEN RANK</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">W ~ A @ B: When does low-rank matrix factorization actually reduce model size and latency?</text>

  <!-- Left: Matrix Factorization Diagrams -->
  <rect x="35" y="65" width="310" height="225" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">1. Truncated SVD Factorization</text>
  <text x="45" y="96" font-size="8.5" fill="#6b7280">Dense W (M x N) factored into two tall-skinny matrices</text>

  <!-- Dense Matrix W -->
  <rect x="45" y="112" width="60" height="80" rx="1" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1"/>
  <text x="75" y="150" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#1f2937">W</text>
  <text x="75" y="165" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">(M x N)</text>

  <text x="116" y="152" font-size="12" font-family="monospace" font-weight="bold" fill="#1f2937">~</text>

  <!-- Matrix A (M x r) -->
  <rect x="132" y="112" width="28" height="80" rx="1" fill="#f8f9fa" stroke="#9ca3af" stroke-width="1"/>
  <text x="146" y="150" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#1f2937">A</text>
  <text x="146" y="165" text-anchor="middle" font-size="7" font-family="monospace" fill="#6b7280">(M x r)</text>

  <text x="168" y="152" font-size="10" font-family="monospace" fill="#1f2937">@</text>

  <!-- Matrix B (r x N) -->
  <rect x="184" y="138" width="60" height="28" rx="1" fill="#f8f9fa" stroke="#9ca3af" stroke-width="1"/>
  <text x="214" y="154" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#1f2937">B</text>
  <text x="214" y="178" text-anchor="middle" font-size="7" font-family="monospace" fill="#6b7280">(r x N)</text>

  <rect x="45" y="202" width="285" height="42" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="53" y="217" font-size="8" font-family="monospace" fill="#1f2937">Dense Params: M * N = 4096 * 4096 = 16.7M</text>
  <text x="53" y="233" font-size="8" font-family="monospace" fill="#1f2937">Low-Rank Params: r * (M + N)</text>

  <text x="45" y="260" font-size="8.5" fill="#6b7280">FLOPs scale linearly with rank r.</text>
  <text x="45" y="274" font-size="8.5" fill="#6b7280">For r &lt;&lt; min(M, N), compute drops by 4x-8x.</text>

  <!-- Right: Break-Even Rank Scree Plot (ACCENT) -->
  <rect x="360" y="65" width="285" height="225" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="372" y="83" font-size="10" font-weight="700" fill="#1f2937">2. Parameter Break-Even Curve</text>
  <text x="372" y="96" font-size="8.5" fill="#6b7280">Break-even condition: r_break = (M * N) / (M + N)</text>

  <!-- Plot Frame -->
  <line x1="385" y1="205" x2="625" y2="205" stroke="#1f2937" stroke-width="1.2"/>
  <line x1="385" y1="205" x2="385" y2="115" stroke="#1f2937" stroke-width="1.2"/>
  <text x="385" y="108" font-size="7.5" font-family="monospace" fill="#6b7280">Total Parameters</text>
  <text x="625" y="218" text-anchor="end" font-size="7.5" font-family="monospace" fill="#6b7280">Rank r -&gt;</text>

  <!-- Flat line: Original uncompressed size -->
  <line x1="385" y1="150" x2="625" y2="150" stroke="#9ca3af" stroke-dasharray="3,2" stroke-width="1"/>
  <text x="390" y="144" font-size="7.5" font-family="monospace" fill="#9ca3af">Original M*N (Dense)</text>

  <!-- Slanted line: Low rank param growth: r*(M+N) -->
  <line x1="385" y1="195" x2="615" y2="115" stroke="#ff8246" stroke-width="1.5"/>

  <!-- Intersection dot -->
  <circle cx="500" cy="155" r="4" fill="#ff8246"/>
  <text x="500" y="170" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">r_break</text>

  <!-- Zones -->
  <rect x="390" y="165" width="80" height="32" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="430" y="178" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">SAVINGS</text>
  <text x="430" y="190" text-anchor="middle" font-size="7" fill="#6b7280">Size &lt; Dense</text>

  <rect x="530" y="115" width="80" height="32" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="570" y="128" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">BLOAT</text>
  <text x="570" y="140" text-anchor="middle" font-size="7" fill="#6b7280">Size &gt; Dense</text>

  <text x="372" y="246" font-size="8.5" font-weight="bold" fill="#1f2937">The Square Matrix Theorem:</text>
  <text x="372" y="260" font-size="8" fill="#6b7280">When M = N, r_break = N / 2.</text>
  <text x="372" y="274" font-size="8" fill="#6b7280">Compressing above N/2 rank increases memory and FLOPs!</text>
{FOOTER}"""
    write_svg("16_svd-rank-breakeven.svg", body)


# -------------------------------------------------------------
# 4. Ch 17: 17_kernel-fusion-traffic.svg
# -------------------------------------------------------------
def gen_17_acceleration():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">KERNEL FUSION: DRAM TRAFFIC COLLAPSE</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Comparing unfused PyTorch dispatch vs fused single-kernel execution: eliminating DRAM round-trips</text>

  <!-- Left: Unfused Execution (8 DRAM Trips) -->
  <rect x="35" y="65" width="295" height="225" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">1. Unfused Eager (PyTorch Default)</text>
  <text x="45" y="96" font-size="8.5" fill="#6b7280">GELU: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))</text>

  <!-- DRAM round trips -->
  <rect x="45" y="112" width="275" height="110" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="53" y="126" font-size="8" font-family="monospace" fill="#1f2937">Kernel 1: Mul (x^3)   -&gt; DRAM Read, DRAM Write</text>
  <text x="53" y="142" font-size="8" font-family="monospace" fill="#1f2937">Kernel 2: Mul + Add   -&gt; DRAM Read, DRAM Write</text>
  <text x="53" y="158" font-size="8" font-family="monospace" fill="#1f2937">Kernel 3: Tanh        -&gt; DRAM Read, DRAM Write</text>
  <text x="53" y="174" font-size="8" font-family="monospace" fill="#1f2937">Kernel 4: Add (1+)    -&gt; DRAM Read, DRAM Write</text>
  <text x="53" y="190" font-size="8" font-family="monospace" fill="#1f2937">Kernel 5: Mul (0.5*x) -&gt; DRAM Read, DRAM Write</text>
  <text x="53" y="208" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">Total DRAM Traffic: 8 Round-trips (16 transfers)</text>

  <text x="45" y="244" font-size="8.5" font-weight="bold" fill="#1f2937">Bandwidth Choke:</text>
  <text x="45" y="258" font-size="8" fill="#6b7280">CPUs spend 80% of execution time waiting for</text>
  <text x="45" y="272" font-size="8" fill="#6b7280">intermediate tensors to travel across DDR bus.</text>

  <!-- Right: Fused Single Kernel (ACCENT) -->
  <rect x="345" y="65" width="300" height="225" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="357" y="83" font-size="10" font-weight="700" fill="#1f2937">2. Fused Custom Kernel (TinyTorch)</text>
  <text x="357" y="96" font-size="8.5" fill="#6b7280">Single loop over CPU / GPU registers</text>

  <!-- Fused Register pipeline -->
  <rect x="357" y="112" width="276" height="110" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="365" y="126" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">1. Load x into CPU Register r0 (ONCE)</text>
  <text x="365" y="142" font-size="8" font-family="monospace" fill="#ff8246">2. r1 = r0 * r0 * r0 * 0.044715</text>
  <text x="365" y="158" font-size="8" font-family="monospace" fill="#ff8246">3. r2 = tanhf(0.797884 * (r0 + r1))</text>
  <text x="365" y="174" font-size="8" font-family="monospace" fill="#ff8246">4. r3 = 0.5f * r0 * (1.0f + r2)</text>
  <text x="365" y="190" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">5. Store r3 to DRAM Out (ONCE)</text>
  <text x="365" y="208" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">Total DRAM Traffic: Exactly 1 Load, 1 Store</text>

  <text x="357" y="244" font-size="8.5" font-weight="bold" fill="#1f2937">8x Bandwidth Reduction:</text>
  <text x="357" y="258" font-size="8" fill="#6b7280">Intermediate values never leave CPU registers.</text>
  <text x="357" y="272" font-size="8" fill="#6b7280">Delivers 3x-6x wall-clock speedup for activation blocks.</text>
{FOOTER}"""
    write_svg("17_kernel-fusion-traffic.svg", body)


# -------------------------------------------------------------
# 5. Ch 18: 18_kv-cache-state-machine.svg
# -------------------------------------------------------------
def gen_18_memoization():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">KV CACHE STATE MACHINE: STATIC BUFFER POINTER ADVANCEMENT</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Eliminating O(N^2) dynamic allocations: preallocated ring buffer with advancing write cursor</text>

  <!-- Preallocated Buffer Bar (ACCENT) -->
  <rect x="35" y="65" width="610" height="120" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">Preallocated KV Tensor Buffer: Shape [Batch=1, Heads=4, MaxSeq=8, Dim=16]</text>
  <text x="430" y="83" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">Static Memory: Zero malloc per step</text>

  <!-- Token Slots in Buffer -->
  <!-- Active Slots 0..3 -->
  <rect x="45" y="100" width="65" height="42" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <text x="77" y="117" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">K[:, :, 0]</text>
  <text x="77" y="131" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">'The'</text>

  <rect x="115" y="100" width="65" height="42" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <text x="147" y="117" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">K[:, :, 1]</text>
  <text x="147" y="131" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">'quick'</text>

  <rect x="185" y="100" width="65" height="42" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <text x="217" y="117" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">K[:, :, 2]</text>
  <text x="217" y="131" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">'brown'</text>

  <!-- Current Write Slot -->
  <rect x="255" y="100" width="65" height="42" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="2"/>
  <text x="287" y="117" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">K[:, :, 3]</text>
  <text x="287" y="131" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">'fox' (NEW)</text>

  <!-- Write cursor arrow -->
  <line x1="287" y1="168" x2="287" y2="148" stroke="#ff8246" stroke-width="1.5" marker-end="url(#arrow-orange)"/>
  <text x="287" y="180" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">curr_pos = 3</text>

  <!-- Inactive Slots 4..7 (Preallocated capacity) -->
  <rect x="325" y="100" width="65" height="42" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="357" y="124" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">Slot 4</text>

  <rect x="395" y="100" width="65" height="42" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="427" y="124" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">Slot 5</text>

  <rect x="465" y="100" width="65" height="42" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="497" y="124" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">Slot 6</text>

  <rect x="535" y="100" width="65" height="42" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="567" y="124" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">Slot 7</text>

  <!-- Bottom Panel: Slicing Mechanics -->
  <rect x="35" y="196" width="610" height="94" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="213" font-size="9.5" font-weight="700" fill="#1f2937">Slicing Operation: Valid Attention Range = cache[:, :, :curr_pos+1]</text>
  <text x="45" y="230" font-size="8" font-family="monospace" fill="#1f2937">1. Write step: k_cache[:, :, curr_pos:curr_pos+1] = new_k (zero copy insert)</text>
  <text x="45" y="244" font-size="8" font-family="monospace" fill="#1f2937">2. Query step: Q (1 token) @ K[:, :, :4].T -&gt; Attention shape: [Batch, Heads, 1, 4]</text>
  <text x="45" y="260" font-size="8.5" fill="#6b7280">FLOP Complexity per token generation drops from O(N^2) to O(N).</text>
  <text x="45" y="274" font-size="8.5" fill="#6b7280">Eliminates Python garbage collection pauses and memory fragmentation completely.</text>
{FOOTER}"""
    write_svg("18_kv-cache-state-machine.svg", body)


# -------------------------------------------------------------
# 6. Ch 19: 19_latency-anatomy-distribution.svg
# -------------------------------------------------------------
def gen_19_benchmarking():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">BENCHMARKING METHODOLOGY: WARMUP &amp; TAIL LATENCY ANATOMY</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Why standard deviation is insufficient: discarded cold-start warmup and asymmetric P95/P99 latency tails</text>

  <!-- Left: Timeline with Warmup Discard -->
  <rect x="35" y="65" width="295" height="225" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">1. Execution Timeline (Runs 1..100)</text>
  <text x="45" y="96" font-size="8.5" fill="#6b7280">Warmup phase must be discarded before timing</text>

  <!-- Cold Start Spike Box -->
  <rect x="45" y="112" width="75" height="75" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="82" y="128" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">Cold Starts</text>
  <line x1="55" y1="175" x2="55" y2="135" stroke="#9ca3af" stroke-width="3"/>
  <line x1="75" y1="175" x2="75" y2="148" stroke="#9ca3af" stroke-width="3"/>
  <line x1="95" y1="175" x2="95" y2="155" stroke="#9ca3af" stroke-width="3"/>
  <text x="82" y="196" text-anchor="middle" font-size="7" font-family="monospace" fill="#9ca3af">DISCARDED</text>

  <!-- Steady State Runs Box -->
  <rect x="130" y="112" width="185" height="75" rx="1" fill="#f8f9fa" stroke="#9ca3af" stroke-width="1"/>
  <text x="222" y="128" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">Steady State (Runs 10..100)</text>
  <!-- Uniform low bars -->
  <line x1="145" y1="175" x2="145" y2="165" stroke="#1f2937" stroke-width="2"/>
  <line x1="160" y1="175" x2="160" y2="164" stroke="#1f2937" stroke-width="2"/>
  <line x1="175" y1="175" x2="175" y2="166" stroke="#1f2937" stroke-width="2"/>
  <line x1="190" y1="175" x2="190" y2="164" stroke="#1f2937" stroke-width="2"/>
  <line x1="205" y1="175" x2="205" y2="165" stroke="#1f2937" stroke-width="2"/>
  <line x1="220" y1="175" x2="220" y2="165" stroke="#1f2937" stroke-width="2"/>
  <line x1="235" y1="175" x2="235" y2="163" stroke="#1f2937" stroke-width="2"/>
  <line x1="250" y1="175" x2="250" y2="165" stroke="#1f2937" stroke-width="2"/>
  <line x1="265" y1="175" x2="265" y2="164" stroke="#1f2937" stroke-width="2"/>
  <line x1="280" y1="175" x2="280" y2="165" stroke="#1f2937" stroke-width="2"/>
  <line x1="295" y1="175" x2="295" y2="166" stroke="#1f2937" stroke-width="2"/>
  <text x="222" y="196" text-anchor="middle" font-size="7" font-family="monospace" fill="#1f2937">MEASURED WINDOW</text>

  <text x="45" y="244" font-size="8.5" font-weight="bold" fill="#1f2937">Warmup Sources Discarded:</text>
  <text x="45" y="258" font-size="8" fill="#6b7280">1. OS Page Faults (allocating physical frames)</text>
  <text x="45" y="272" font-size="8" fill="#6b7280">2. CPU Branch Predictor / Instruction Cache warming</text>

  <!-- Right: Asymmetric Latency PDF (ACCENT) -->
  <rect x="345" y="65" width="300" height="225" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="357" y="83" font-size="10" font-weight="700" fill="#1f2937">2. Latency Probability Distribution</text>
  <text x="357" y="96" font-size="8.5" fill="#6b7280">Heavy right tail caused by OS interrupts &amp; GC pauses</text>

  <!-- Axes -->
  <line x1="365" y1="190" x2="625" y2="190" stroke="#1f2937" stroke-width="1.2"/>
  <line x1="365" y1="190" x2="365" y2="115" stroke="#1f2937" stroke-width="1.2"/>
  <text x="625" y="202" text-anchor="end" font-size="7.5" font-family="monospace" fill="#6b7280">Latency (ms) -&gt;</text>

  <!-- Skewed curve staying strictly above y=190 baseline -->
  <path d="M 370 190 C 385 140, 405 125, 420 125 C 440 125, 460 165, 490 175 C 520 185, 550 178, 570 178 C 590 178, 605 186, 620 190" fill="none" stroke="#ff8246" stroke-width="2"/>

  <!-- P50 Median Line -->
  <line x1="420" y1="125" x2="420" y2="190" stroke="#ff8246" stroke-dasharray="2,2"/>
  <text x="420" y="118" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">P50: 1.2ms</text>

  <!-- P95 Line -->
  <line x1="490" y1="175" x2="490" y2="190" stroke="#1f2937" stroke-dasharray="2,2"/>
  <text x="490" y="166" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#1f2937">P95: 2.8ms</text>

  <!-- P99 Tail Line -->
  <line x1="570" y1="178" x2="570" y2="190" stroke="#1f2937" stroke-dasharray="2,2"/>
  <text x="570" y="168" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#1f2937">P99: 5.4ms</text>

  <text x="357" y="244" font-size="8.5" font-weight="bold" fill="#1f2937">The Mean Trap:</text>
  <text x="357" y="258" font-size="8" fill="#6b7280">A few 5ms spikes pull Mean to 1.8ms (misleading by 50%).</text>
  <text x="357" y="272" font-size="8" fill="#6b7280">Always report P50 (median) + P99 for production SLAs.</text>
{FOOTER}"""
    write_svg("19_latency-anatomy-distribution.svg", body)


# -------------------------------------------------------------
# 7. Ch 20: 20_stacking-waterfall-amdahl.svg
# -------------------------------------------------------------
def gen_20_capstone():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">OPTIMIZATION STACKING WATERFALL &amp; AMDAHL'S CEILING</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">How optimizations compound: each technique tackles a distinct bottleneck until reaching the Python runtime floor</text>

  <!-- Bar 1: Pure Python Baseline -->
  <rect x="45" y="75" width="90" height="150" rx="1" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1"/>
  <text x="90" y="95" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#1f2937">Baseline</text>
  <text x="90" y="110" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">100.0 ms</text>
  <text x="90" y="145" text-anchor="middle" font-size="7.5" fill="#6b7280">Naive loops</text>
  <text x="90" y="160" text-anchor="middle" font-size="7.5" fill="#6b7280">&amp; dynamic</text>
  <text x="90" y="175" text-anchor="middle" font-size="7.5" fill="#6b7280">allocations</text>

  <!-- Arrow -->
  <line x1="140" y1="150" x2="155" y2="150" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Bar 2: + Vectorization -->
  <rect x="160" y="125" width="90" height="100" rx="1" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1"/>
  <text x="205" y="145" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#1f2937">+ Vectorized</text>
  <text x="205" y="160" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">25.0 ms</text>
  <text x="205" y="185" text-anchor="middle" font-size="7.5" fill="#6b7280">BLAS GEMM</text>
  <text x="205" y="200" text-anchor="middle" font-size="7.5" font-weight="bold" fill="#1f2937">4.0x Speedup</text>

  <!-- Arrow -->
  <line x1="255" y1="150" x2="270" y2="150" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Bar 3: + Kernel Fusion -->
  <rect x="275" y="165" width="90" height="60" rx="1" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1"/>
  <text x="320" y="185" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#1f2937">+ Fused</text>
  <text x="320" y="200" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">10.0 ms</text>
  <text x="320" y="215" text-anchor="middle" font-size="7.5" font-weight="bold" fill="#1f2937">10.0x Speedup</text>

  <!-- Arrow -->
  <line x1="370" y1="180" x2="385" y2="180" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Bar 4: + INT8 Quantization -->
  <rect x="390" y="195" width="90" height="30" rx="1" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1"/>
  <text x="435" y="210" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#1f2937">+ INT8 Quant</text>
  <text x="435" y="222" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">4.0 ms (25x)</text>

  <!-- Arrow -->
  <line x1="485" y1="205" x2="500" y2="205" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>

  <!-- Bar 5: + KV Caching (ACCENT) -->
  <rect x="505" y="207" width="115" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="562" y="220" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#ff8246">+ KV Cache: 1.2 ms (83x)</text>

  <!-- Ground Line -->
  <line x1="40" y1="225" x2="630" y2="225" stroke="#1f2937" stroke-width="1.5"/>

  <!-- Bottom: Amdahl's Law Explanation -->
  <rect x="35" y="238" width="610" height="54" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="254" font-size="9.5" font-weight="700" fill="#1f2937">Amdahl's Law Floor: Speedup_max = 1 / ( (1 - P) + P / S )</text>
  <text x="45" y="269" font-size="8" fill="#6b7280">Once tensor math is 99% optimized, Python interpreter dispatch overhead (the unaccelerated 1%) becomes the primary bottleneck.</text>
  <text x="45" y="283" font-size="8" fill="#6b7280">Further speedups require escaping the Python GIL into ahead-of-time compiled native C++ / CUDA kernels.</text>
{FOOTER}"""
    write_svg("20_stacking-waterfall-amdahl.svg", body)


# -------------------------------------------------------------
# 8. Ch 21: 21_compiler-lowering-fusion.svg
# -------------------------------------------------------------
def gen_21_extensions():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">COMPILER LOWERING: EAGER AST TO FUSED KERNEL IR</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">From Python source trees to intermediate representation (IR) DAGs and hardware-specific SIMD micro-kernels</text>

  <!-- Stage 1: Eager Python AST -->
  <rect x="35" y="65" width="180" height="225" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">1. Python Source / AST</text>
  <text x="45" y="96" font-size="8.5" fill="#6b7280">Eager expression tree</text>

  <rect x="45" y="112" width="160" height="58" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="52" y="126" font-size="8" font-family="monospace" fill="#1f2937">def gelu(x):</text>
  <text x="52" y="140" font-size="7.5" font-family="monospace" fill="#1f2937">  t1 = x * 0.5</text>
  <text x="52" y="154" font-size="7.5" font-family="monospace" fill="#1f2937">  t2 = tanh(0.797 * x)</text>
  <text x="52" y="166" font-size="7.5" font-family="monospace" fill="#1f2937">  return t1 * (1.0 + t2)</text>

  <text x="45" y="195" font-size="8" fill="#6b7280">Interpreted statement-by-statement.</text>
  <text x="45" y="209" font-size="8" fill="#6b7280">Produces 4 separate C API calls,</text>
  <text x="45" y="223" font-size="8" fill="#6b7280">4 DRAM allocation events, and</text>
  <text x="45" y="237" font-size="8" fill="#6b7280">GIL acquisition overhead.</text>

  <!-- Arrow -->
  <line x1="215" y1="172" x2="235" y2="172" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Stage 2: Computational Graph IR (ACCENT) -->
  <rect x="245" y="65" width="205" height="225" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="257" y="83" font-size="10" font-weight="700" fill="#1f2937">2. Fused Computational IR</text>
  <text x="257" y="96" font-size="8.5" fill="#6b7280">Operator fusion cluster</text>

  <!-- Fused Cluster Box -->
  <rect x="255" y="112" width="185" height="85" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <text x="263" y="127" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">[FusedPointwiseCluster_0]</text>
  <text x="263" y="142" font-size="7.5" font-family="monospace" fill="#1f2937">Input: %0 = Tensor(float32, [N])</text>
  <text x="263" y="156" font-size="7.5" font-family="monospace" fill="#1f2937">Ops: [Mul, Tanh, Add, Mul]</text>
  <text x="263" y="170" font-size="7.5" font-family="monospace" fill="#1f2937">Output: %1 = Tensor(float32, [N])</text>
  <text x="263" y="186" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">Dead Stores Eliminated: t1, t2</text>

  <text x="257" y="222" font-size="8.5" fill="#6b7280">Passes performed by compiler:</text>
  <text x="257" y="236" font-size="8" fill="#6b7280">1. Common Subexpression Elimination</text>
  <text x="257" y="250" font-size="8" fill="#6b7280">2. Loop tiling and affine fusion</text>
  <text x="257" y="264" font-size="8" fill="#6b7280">3. Constant folding (0.79788)</text>

  <!-- Arrow -->
  <line x1="450" y1="172" x2="470" y2="172" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>

  <!-- Stage 3: Low-Level C / SIMD Kernel -->
  <rect x="480" y="65" width="165" height="225" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="490" y="83" font-size="10" font-weight="700" fill="#1f2937">3. Lowered C / SIMD</text>
  <text x="490" y="96" font-size="8.5" fill="#6b7280">Hardware vector instructions</text>

  <rect x="490" y="112" width="145" height="85" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="496" y="126" font-size="7.5" font-family="monospace" fill="#1f2937">#pragma omp simd</text>
  <text x="496" y="139" font-size="7.5" font-family="monospace" fill="#1f2937">for(int i=0; i&lt;N; i+=8) {{</text>
  <text x="496" y="152" font-size="7.5" font-family="monospace" fill="#1f2937">  __m256 vx = _mm256_load(x+i);</text>
  <text x="496" y="165" font-size="7.5" font-family="monospace" fill="#1f2937">  __m256 vy = fused_gelu(vx);</text>
  <text x="496" y="178" font-size="7.5" font-family="monospace" fill="#1f2937">  _mm256_store(out+i, vy);</text>
  <text x="496" y="191" font-size="7.5" font-family="monospace" fill="#1f2937">}}</text>

  <text x="490" y="222" font-size="8" fill="#6b7280">AVX2 / NEON 8-wide vectors.</text>
  <text x="490" y="236" font-size="8" fill="#6b7280">Zero Python interpreter</text>
  <text x="490" y="250" font-size="8" fill="#6b7280">involvement inside loop.</text>
  <text x="490" y="264" font-size="8" fill="#6b7280">Bypasses GIL entirely.</text>
{FOOTER}"""
    write_svg("21_compiler-lowering-fusion.svg", body)


# -------------------------------------------------------------
# 9. Milestone 03: milestone_03_pareto-frontier.svg
# -------------------------------------------------------------
def gen_milestone_03():
    h = 330
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="300" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">THE PRODUCTION PARETO FRONTIER: LATENCY VS COMPRESSION TRADE-OFF</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Navigating multi-objective deployment: Pareto-optimal configurations subject to the R >= 0.99 accuracy gate</text>

  <!-- Main Plot Canvas (ACCENT) -->
  <rect x="35" y="65" width="375" height="235" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">Throughput Speedup vs Memory Compression</text>

  <!-- Axes -->
  <line x1="65" y1="250" x2="390" y2="250" stroke="#1f2937" stroke-width="1.5"/>
  <line x1="65" y1="250" x2="65" y2="95" stroke="#1f2937" stroke-width="1.5"/>
  <text x="390" y="278" text-anchor="end" font-size="8" font-family="monospace" fill="#1f2937">Inference Speedup -&gt;</text>
  <text x="70" y="105" font-size="8" font-family="monospace" fill="#1f2937">Compression Ratio -&gt;</text>

  <!-- Ticks -->
  <text x="65" y="262" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">1x</text>
  <text x="130" y="262" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">5x</text>
  <text x="200" y="262" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">15x</text>
  <text x="270" y="262" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">35x</text>
  <text x="340" y="262" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">70x</text>

  <!-- Pareto Frontier Curve -->
  <path d="M 75 240 Q 145 195, 220 145 T 330 110" fill="none" stroke="#ff8246" stroke-width="2.5"/>

  <!-- Config 1: Baseline -->
  <circle cx="75" cy="240" r="4" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
  <text x="83" y="235" font-size="7" font-family="monospace" fill="#1f2937">FP32 (1x, 1x)</text>

  <!-- Config 2: Fused FP16 -->
  <circle cx="145" cy="195" r="4" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
  <text x="153" y="190" font-size="7" font-family="monospace" fill="#1f2937">Fused FP16 (6x, 2x)</text>

  <!-- Config 3: INT8 Quantized -->
  <circle cx="220" cy="145" r="4" fill="#ffffff" stroke="#ff8246" stroke-width="2"/>
  <text x="228" y="140" font-size="7" font-family="monospace" font-weight="bold" fill="#ff8246">INT8 (18x, 4x)</text>

  <!-- Config 4: INT8 + KV Cache (Pareto Peak) -->
  <circle cx="330" cy="110" r="5" fill="#ff8246" stroke="#ffffff" stroke-width="1.5"/>
  <text x="330" y="98" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">INT8 + KV Cache (65x, 4.2x)</text>

  <!-- Suboptimal points inside curve -->
  <circle cx="165" cy="225" r="3" fill="#9ca3af"/>
  <text x="173" y="228" font-size="6" font-family="monospace" fill="#9ca3af">Naive INT8 (no fusion)</text>

  <circle cx="240" cy="185" r="3" fill="#9ca3af"/>
  <text x="248" y="188" font-size="6" font-family="monospace" fill="#9ca3af">Over-pruned SVD (r=64)</text>

  <!-- Right: Legend and Rules -->
  <rect x="420" y="65" width="225" height="235" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="432" y="83" font-size="10" font-weight="700" fill="#1f2937">Pareto Selection Rules</text>
  <text x="432" y="96" font-size="8.5" fill="#6b7280">Multi-objective constraint bounds</text>

  <rect x="432" y="108" width="201" height="46" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="438" y="122" font-size="8" font-weight="bold" fill="#1f2937">Accuracy Floor (R &gt;= 0.99):</text>
  <text x="438" y="136" font-size="7.5" fill="#6b7280">Must preserve &gt;= 99% baseline accuracy.</text>
  <text x="438" y="148" font-size="7.5" fill="#6b7280">Strict gating requirement for deployment.</text>

  <rect x="432" y="160" width="201" height="46" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="438" y="174" font-size="8" font-weight="bold" fill="#1f2937">Dominated Solutions:</text>
  <text x="438" y="188" font-size="7.5" fill="#6b7280">Sub-frontier points are discarded:</text>
  <text x="438" y="200" font-size="7.5" fill="#6b7280">a frontier configuration offers higher speed.</text>

  <text x="432" y="226" font-size="8.5" font-weight="bold" fill="#1f2937">Production Verdict:</text>
  <text x="432" y="240" font-size="8" fill="#6b7280">INT8 Quantization + KV Caching sits</text>
  <text x="432" y="254" font-size="8" fill="#6b7280">at the Pareto peak, yielding 65x speedup</text>
  <text x="432" y="268" font-size="8" fill="#6b7280">with zero loss in model task accuracy.</text>
{FOOTER}"""
    write_svg("milestone_03_pareto-frontier.svg", body)


if __name__ == "__main__":
    gen_14_profiling()
    gen_15_quantization()
    gen_16_compression()
    gen_17_acceleration()
    gen_18_memoization()
    gen_19_benchmarking()
    gen_20_capstone()
    gen_21_extensions()
    gen_milestone_03()
    print("Tier 3 Mechanical SVGs generated successfully.")
