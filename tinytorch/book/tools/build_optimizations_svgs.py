#!/usr/bin/env python3
"""
Generator for TinyTorch Tier 3 (Optimizations) Mechanical SVG Diagrams.
Adheres strictly to Vol 3 Figure 1.10 standard & tinytorch/palette.md:
- 680px viewBox width
- Height sized to content + breathing room
- Canvas base is pure white (#ffffff), NO outer frame border stroke
- NO embedded canvas titles or subtitles (Quarto fig-cap owns the caption)
- Subsystem container cards with 24px header bands (rx="2")
- ~85% greyscale, max 1 accent node in #fff1e8 fill / #ff8246 stroke
- Uniform 1pt stroke weights, rx="2" rounded corners
- Standard polygon arrowheads, TeX Gyre Heros / Helvetica font stack
- Synchronizes output to book/ and quarto/ asset directories
"""

from pathlib import Path

DEST_DIRS = [
    Path("tinytorch/book/assets/images/diagrams"),
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
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Left Box: Single Token Inference (Memory-Bound) -->
  <rect x="25" y="20" width="300" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="300" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. AUTOREGRESSIVE DECODE (BATCH=1)</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">Memory-Bound Regime: R = 1 FLOP / Byte</text>

  <!-- Flow in DRAM -->
  <rect x="35" y="74" width="280" height="46" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="43" y="91" font-size="8" font-family="monospace" fill="#1f2937">Stream 7B Weights from VRAM (14 GB / token)</text>
  <text x="43" y="107" font-size="7.5" font-family="monospace" fill="#6b7280">Memory bus saturated at 100% capacity (1000 GB/s)</text>

  <line x1="175" y1="124" x2="175" y2="138" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- ALU Utilization -->
  <rect x="35" y="142" width="280" height="48" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="43" y="159" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">Hardware Tensor Cores: 97% IDLE</text>
  <text x="43" y="175" font-size="7.5" font-family="monospace" fill="#6b7280">Each weight byte used exactly ONCE before eviction</text>

  <text x="35" y="214" font-size="8.5" font-weight="bold" fill="#1f2937">Bottleneck: Memory Bandwidth</text>
  <text x="35" y="230" font-size="8.5" fill="#6b7280">Faster GPUs with higher TFLOPs do not accelerate</text>
  <text x="35" y="246" font-size="8.5" fill="#6b7280">single-user generation without HBM bus upgrades.</text>

  <!-- Right Box: Batched Training (Compute-Bound, ACCENT) -->
  <rect x="345" y="20" width="310" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="345" y="20" width="310" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="357" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. BATCHED PREFILL / TRAINING (BATCH=64)</text>
  <text x="357" y="58" font-size="8.5" fill="#6b7280">Compute-Bound Regime: R = 64 FLOPs / Byte</text>

  <!-- Stream in DRAM -->
  <rect x="357" y="74" width="286" height="46" rx="1" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
  <text x="365" y="91" font-size="8" font-family="monospace" fill="#1f2937">Stream Weights ONCE per Batch</text>
  <text x="365" y="107" font-size="7.5" font-family="monospace" fill="#6b7280">Loaded into SRAM / register cache tiles once</text>

  <line x1="500" y1="124" x2="500" y2="138" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

  <!-- ALU Utilization -->
  <rect x="357" y="142" width="286" height="48" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="365" y="159" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">Hardware Tensor Cores: 95% SATURATED</text>
  <text x="365" y="175" font-size="7.5" font-family="monospace" fill="#374151">Each weight multiplied across 64 token vectors</text>

  <text x="357" y="214" font-size="8.5" font-weight="bold" fill="#1f2937">Peak FLOP Efficiency</text>
  <text x="357" y="230" font-size="8.5" fill="#6b7280">High arithmetic intensity amortizes memory transfer,</text>
  <text x="357" y="246" font-size="8.5" fill="#6b7280">allowing execution to reach the hardware roofline ceiling.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Single-token autoregressive decoding is bound by memory bandwidth; batched prefill saturates ALU compute capacity.</text>
{FOOTER}"""
    write_svg("14_weight-streaming-reuse.svg", body)


# -------------------------------------------------------------
# 2. Ch 15: 15_affine-quantization-grid.svg
# -------------------------------------------------------------
def gen_15_quantization():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Upper Axis: Float32 Continuous Space -->
  <rect x="25" y="20" width="630" height="115" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="630" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. CONTINUOUS FLOAT32 DOMAIN [α = -3.2, β = +4.8]</text>
  <text x="440" y="36" font-size="8.5" font-family="monospace" fill="#6b7280">Scale S = (β - α) / 255 = 0.03137</text>

  <!-- Number Line Float32 -->
  <line x1="50" y1="82" x2="630" y2="82" stroke="#1f2937" stroke-width="1.5"/>

  <!-- Ticks & Values -->
  <line x1="70" y1="77" x2="70" y2="87" stroke="#1f2937" stroke-width="1"/>
  <text x="70" y="100" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">-3.20 (min)</text>

  <line x1="210" y1="77" x2="210" y2="87" stroke="#1f2937" stroke-width="1"/>
  <text x="210" y="100" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">-1.50</text>

  <!-- Zero Point Float Highlight -->
  <line x1="320" y1="73" x2="320" y2="91" stroke="#ff8246" stroke-width="2"/>
  <circle cx="320" cy="82" r="3" fill="#ff8246"/>
  <text x="320" y="68" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#ff8246">0.00 (Real Zero)</text>
  <text x="320" y="100" text-anchor="middle" font-size="8" font-family="monospace" fill="#ff8246">x = 0.0</text>

  <line x1="450" y1="77" x2="450" y2="87" stroke="#1f2937" stroke-width="1"/>
  <text x="450" y="100" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">+2.10</text>

  <line x1="590" y1="77" x2="590" y2="87" stroke="#1f2937" stroke-width="1"/>
  <text x="590" y="100" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">+4.80 (max)</text>

  <!-- Lower Axis: Discrete INT8 Grid (ACCENT) -->
  <rect x="25" y="150" width="630" height="135" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="25" y="150" width="630" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="35" y="166" font-size="9.5" font-weight="700" fill="#c85a17">2. DISCRETE INT8 QUANTIZED DOMAIN [-128, +127]</text>
  <text x="420" y="166" font-size="8.5" font-family="monospace" font-weight="bold" fill="#c85a17">Zero-point Z = round(-α / S) - 128 = -26</text>

  <!-- INT8 Number line -->
  <line x1="50" y1="208" x2="630" y2="208" stroke="#ff8246" stroke-width="1.5"/>

  <!-- Left Clamping Shelf -->
  <rect x="45" y="198" width="40" height="20" rx="1" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="65" y="211" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">Clamp</text>
  <text x="70" y="228" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">-128</text>

  <!-- Zero Point INT8 tick -->
  <line x1="320" y1="199" x2="320" y2="217" stroke="#ff8246" stroke-width="2"/>
  <circle cx="320" cy="208" r="3" fill="#ff8246"/>
  <text x="320" y="228" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#ff8246">q = -26 (Z)</text>

  <!-- Right Clamping Shelf -->
  <line x1="590" y1="203" x2="590" y2="213" stroke="#1f2937" stroke-width="1"/>
  <text x="590" y="228" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">+127</text>
  <rect x="595" y="198" width="40" height="20" rx="1" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="615" y="211" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">Clamp</text>

  <!-- Mapping projection lines -->
  <line x1="70" y1="105" x2="70" y2="198" stroke="#9ca3af" stroke-dasharray="3,2"/>
  <line x1="320" y1="105" x2="320" y2="198" stroke="#ff8246" stroke-width="1.2"/>
  <line x1="590" y1="105" x2="590" y2="198" stroke="#9ca3af" stroke-dasharray="3,2"/>

  <text x="35" y="254" font-size="8.5" font-weight="bold" fill="#1f2937">Zero Distortion Invariant: Zero in float32 maps exactly to integer Z without rounding error.</text>
  <text x="35" y="270" font-size="8.5" fill="#6b7280">This ensures tensor padding (0.0) and ReLU activations do not accumulate numerical bias during inference.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="306" text-anchor="middle" font-size="9" fill="#6b7280">Affine uniform quantization maps continuous FP32 values to discrete INT8 bins while strictly preserving exact zero representation.</text>
{FOOTER}"""
    write_svg("15_affine-quantization-grid.svg", body)


# -------------------------------------------------------------
# 3. Ch 16: 16_svd-rank-breakeven.svg
# -------------------------------------------------------------
def gen_16_compression():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Left: Matrix Factorization Diagrams -->
  <rect x="25" y="20" width="310" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="310" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. TRUNCATED SVD FACTORIZATION</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">Dense W (M x N) factored into A (M x r) @ B (r x N)</text>

  <!-- Dense Matrix W -->
  <rect x="40" y="74" width="60" height="75" rx="1" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
  <text x="70" y="110" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#1f2937">W</text>
  <text x="70" y="125" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">(M x N)</text>

  <text x="112" y="112" font-size="12" font-family="monospace" font-weight="bold" fill="#1f2937">~</text>

  <!-- Matrix A (M x r) -->
  <rect x="126" y="74" width="34" height="75" rx="1" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="143" y="108" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#1f2937">A</text>
  <text x="143" y="122" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">(M x r)</text>

  <text x="170" y="112" font-size="10" font-family="monospace" fill="#1f2937">@</text>

  <!-- Matrix B (r x N) -->
  <rect x="186" y="100" width="65" height="28" rx="1" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="218" y="116" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#1f2937">B</text>
  <text x="218" y="140" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">(r x N)</text>

  <rect x="35" y="162" width="290" height="44" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="43" y="178" font-size="8" font-family="monospace" fill="#1f2937">Dense Params: M * N = 4096 * 4096 = 16.7M</text>
  <text x="43" y="194" font-size="8" font-family="monospace" fill="#1f2937">Low-Rank Params: r * (M + N)</text>

  <text x="35" y="228" font-size="8.5" fill="#6b7280">FLOPs scale linearly with rank r.</text>
  <text x="35" y="244" font-size="8.5" fill="#6b7280">For r &lt;&lt; min(M, N), compute drops by 4x-8x.</text>

  <!-- Right: Break-Even Rank Scree Plot (ACCENT) -->
  <rect x="350" y="20" width="305" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="350" y="20" width="305" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="362" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. PARAMETER BREAK-EVEN CURVE</text>
  <text x="362" y="58" font-size="8.5" fill="#6b7280">Break-even condition: r_break = (M · N) / (M + N)</text>

  <!-- Plot Frame -->
  <line x1="380" y1="180" x2="630" y2="180" stroke="#1f2937" stroke-width="1.2"/>
  <line x1="380" y1="180" x2="380" y2="85" stroke="#1f2937" stroke-width="1.2"/>
  <text x="380" y="78" font-size="7.5" font-family="monospace" fill="#6b7280">Total Parameters</text>
  <text x="630" y="194" text-anchor="end" font-size="7.5" font-family="monospace" fill="#6b7280">Rank r -&gt;</text>

  <!-- Flat line: Original uncompressed size -->
  <line x1="380" y1="125" x2="630" y2="125" stroke="#9ca3af" stroke-dasharray="3,2" stroke-width="1"/>
  <text x="385" y="118" font-size="7.5" font-family="monospace" fill="#9ca3af">Original M*N (Dense)</text>

  <!-- Slanted line: Low rank param growth: r*(M+N) -->
  <line x1="380" y1="170" x2="620" y2="85" stroke="#ff8246" stroke-width="1.5"/>

  <!-- Intersection dot -->
  <circle cx="505" cy="125" r="4" fill="#ff8246"/>
  <text x="505" y="142" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">r_break</text>

  <!-- Zones -->
  <rect x="390" y="136" width="75" height="30" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="427" y="149" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">SAVINGS</text>
  <text x="427" y="160" text-anchor="middle" font-size="7" fill="#6b7280">Size &lt; Dense</text>

  <rect x="545" y="90" width="75" height="30" rx="1" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
  <text x="582" y="103" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">BLOAT</text>
  <text x="582" y="114" text-anchor="middle" font-size="7" fill="#6b7280">Size &gt; Dense</text>

  <text x="362" y="218" font-size="8.5" font-weight="bold" fill="#1f2937">The Square Matrix Theorem:</text>
  <text x="362" y="234" font-size="8.5" fill="#6b7280">When M = N, r_break = N / 2.</text>
  <text x="362" y="250" font-size="8.5" fill="#6b7280">Compressing above N/2 rank increases memory and FLOPs!</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Truncated SVD reduces parameter volume only when the chosen rank r is strictly below the break-even threshold r*.</text>
{FOOTER}"""
    write_svg("16_svd-rank-breakeven.svg", body)


# -------------------------------------------------------------
# 4. Ch 17: 17_kernel-fusion-traffic.svg
# -------------------------------------------------------------
def gen_17_acceleration():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Left: Unfused Execution (8 DRAM Trips) -->
  <rect x="25" y="20" width="300" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="300" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. UNFUSED EAGER (PYTORCH DEFAULT)</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">GELU: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))</text>

  <!-- DRAM round trips -->
  <rect x="35" y="74" width="280" height="110" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="43" y="90" font-size="8" font-family="monospace" fill="#1f2937">Kernel 1: Mul (x^3)   -&gt; DRAM Read, DRAM Write</text>
  <text x="43" y="106" font-size="8" font-family="monospace" fill="#1f2937">Kernel 2: Mul + Add   -&gt; DRAM Read, DRAM Write</text>
  <text x="43" y="122" font-size="8" font-family="monospace" fill="#1f2937">Kernel 3: Tanh        -&gt; DRAM Read, DRAM Write</text>
  <text x="43" y="138" font-size="8" font-family="monospace" fill="#1f2937">Kernel 4: Add (1+)    -&gt; DRAM Read, DRAM Write</text>
  <text x="43" y="154" font-size="8" font-family="monospace" fill="#1f2937">Kernel 5: Mul (0.5*x) -&gt; DRAM Read, DRAM Write</text>
  <text x="43" y="172" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">Total DRAM Traffic: 8 Round-trips (16 transfers)</text>

  <text x="35" y="206" font-size="8.5" font-weight="bold" fill="#1f2937">Bandwidth Choke:</text>
  <text x="35" y="222" font-size="8.5" fill="#6b7280">CPUs spend 80% of execution time waiting for</text>
  <text x="35" y="238" font-size="8.5" fill="#6b7280">intermediate tensors to travel across memory bus.</text>

  <!-- Right: Fused Single Kernel (ACCENT) -->
  <rect x="345" y="20" width="310" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="345" y="20" width="310" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="357" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. FUSED CUSTOM KERNEL (TINYTORCH)</text>
  <text x="357" y="58" font-size="8.5" fill="#6b7280">Single loop over CPU / GPU registers</text>

  <!-- Fused Register pipeline -->
  <rect x="357" y="74" width="286" height="110" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="365" y="90" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">1. Load x into CPU Register r0 (ONCE)</text>
  <text x="365" y="106" font-size="8" font-family="monospace" fill="#ff8246">2. r1 = r0 * r0 * r0 * 0.044715</text>
  <text x="365" y="122" font-size="8" font-family="monospace" fill="#ff8246">3. r2 = tanhf(0.797884 * (r0 + r1))</text>
  <text x="365" y="138" font-size="8" font-family="monospace" fill="#ff8246">4. r3 = 0.5f * r0 * (1.0f + r2)</text>
  <text x="365" y="154" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">5. Store r3 to DRAM Out (ONCE)</text>
  <text x="365" y="172" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">Total DRAM Traffic: Exactly 1 Load, 1 Store</text>

  <text x="357" y="206" font-size="8.5" font-weight="bold" fill="#1f2937">8x Bandwidth Reduction:</text>
  <text x="357" y="222" font-size="8.5" fill="#6b7280">Intermediate values never leave CPU registers.</text>
  <text x="357" y="238" font-size="8.5" fill="#6b7280">Delivers 3x-6x wall-clock speedup for activation blocks.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Operator fusion collapses multi-step DAG pipelines into a single kernel execution loop, eliminating high-latency memory round-trips.</text>
{FOOTER}"""
    write_svg("17_kernel-fusion-traffic.svg", body)


# -------------------------------------------------------------
# 5. Ch 18: 18_kv-cache-state-machine.svg
# -------------------------------------------------------------
def gen_18_memoization():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Preallocated Buffer Bar (ACCENT) -->
  <rect x="25" y="20" width="630" height="120" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="25" y="20" width="630" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#c85a17">PREALLOCATED STATIC KV BUFFER: [BATCH=1, HEADS=4, MAX_SEQ=8, DIM=16]</text>
  <text x="440" y="36" font-size="8.5" font-family="monospace" font-weight="bold" fill="#c85a17">Zero malloc per step</text>

  <!-- Token Slots in Buffer -->
  <!-- Active Slots 0..3 -->
  <rect x="35" y="55" width="68" height="44" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <text x="69" y="73" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">K[:, :, 0]</text>
  <text x="69" y="88" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">'The'</text>

  <rect x="108" y="55" width="68" height="44" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <text x="142" y="73" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">K[:, :, 1]</text>
  <text x="142" y="88" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">'quick'</text>

  <rect x="181" y="55" width="68" height="44" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <text x="215" y="73" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">K[:, :, 2]</text>
  <text x="215" y="88" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">'brown'</text>

  <!-- Current Write Slot -->
  <rect x="254" y="55" width="68" height="44" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="2"/>
  <text x="288" y="73" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">K[:, :, 3]</text>
  <text x="288" y="88" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">'fox' (NEW)</text>

  <!-- Write cursor arrow -->
  <line x1="288" y1="124" x2="288" y2="105" stroke="#ff8246" stroke-width="1.5" marker-end="url(#arrow-orange)"/>
  <text x="288" y="134" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">curr_pos = 3</text>

  <!-- Inactive Slots 4..7 (Preallocated capacity) -->
  <rect x="327" y="55" width="68" height="44" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="361" y="80" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">Slot 4</text>

  <rect x="400" y="55" width="68" height="44" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="434" y="80" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">Slot 5</text>

  <rect x="473" y="55" width="68" height="44" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="507" y="80" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">Slot 6</text>

  <rect x="546" y="55" width="68" height="44" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="580" y="80" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">Slot 7</text>

  <!-- Bottom Panel: Slicing Mechanics -->
  <rect x="25" y="152" width="630" height="115" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="152" width="630" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="168" font-size="9.5" font-weight="700" fill="#1f2937">ZERO-COPY SLICING &amp; ATTENTION QUERY</text>
  <text x="320" y="168" font-size="8.5" font-family="monospace" fill="#6b7280">Valid Attention Range = cache[:, :, :curr_pos+1]</text>

  <text x="35" y="196" font-size="8.5" font-family="monospace" fill="#1f2937">1. Write step: k_cache[:, :, curr_pos:curr_pos+1] = new_k (zero copy insert)</text>
  <text x="35" y="212" font-size="8.5" font-family="monospace" fill="#1f2937">2. Query step: Q (1 token) @ K[:, :, :4].T -&gt; Attention shape: [Batch, Heads, 1, 4]</text>
  <text x="35" y="234" font-size="8.5" font-weight="bold" fill="#ff8246">FLOP Complexity per token generation drops from O(N²) to O(N).</text>
  <text x="35" y="250" font-size="8.5" fill="#6b7280">Eliminates Python runtime allocations and memory fragmentation completely.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">KV caching replaces quadratic recomputation with static ring-buffer pointers, reducing incremental token generation to linear O(N) cost.</text>
{FOOTER}"""
    write_svg("18_kv-cache-state-machine.svg", body)


# -------------------------------------------------------------
# 6. Ch 19: 19_latency-anatomy-distribution.svg
# -------------------------------------------------------------
def gen_19_benchmarking():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Left: Timeline with Warmup Discard -->
  <rect x="25" y="20" width="300" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="300" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. EXECUTION TIMELINE (RUNS 1..100)</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">Warmup phase must be discarded before timing</text>

  <!-- Cold Start Spike Box -->
  <rect x="35" y="74" width="75" height="75" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="72" y="90" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">Cold Starts</text>
  <line x1="45" y1="135" x2="45" y2="98" stroke="#9ca3af" stroke-width="3"/>
  <line x1="65" y1="135" x2="65" y2="108" stroke="#9ca3af" stroke-width="3"/>
  <line x1="85" y1="135" x2="85" y2="116" stroke="#9ca3af" stroke-width="3"/>
  <text x="72" y="145" text-anchor="middle" font-size="7" font-family="monospace" fill="#9ca3af">DISCARDED</text>

  <!-- Steady State Runs Box -->
  <rect x="120" y="74" width="195" height="75" rx="1" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
  <text x="217" y="90" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">Steady State (Runs 10..100)</text>
  <!-- Uniform low bars -->
  <line x1="135" y1="135" x2="135" y2="125" stroke="#1f2937" stroke-width="2"/>
  <line x1="150" y1="135" x2="150" y2="124" stroke="#1f2937" stroke-width="2"/>
  <line x1="165" y1="135" x2="165" y2="126" stroke="#1f2937" stroke-width="2"/>
  <line x1="180" y1="135" x2="180" y2="124" stroke="#1f2937" stroke-width="2"/>
  <line x1="195" y1="135" x2="195" y2="125" stroke="#1f2937" stroke-width="2"/>
  <line x1="210" y1="135" x2="210" y2="125" stroke="#1f2937" stroke-width="2"/>
  <line x1="225" y1="135" x2="225" y2="123" stroke="#1f2937" stroke-width="2"/>
  <line x1="240" y1="135" x2="240" y2="125" stroke="#1f2937" stroke-width="2"/>
  <line x1="255" y1="135" x2="255" y2="124" stroke="#1f2937" stroke-width="2"/>
  <line x1="270" y1="135" x2="270" y2="125" stroke="#1f2937" stroke-width="2"/>
  <line x1="285" y1="135" x2="285" y2="126" stroke="#1f2937" stroke-width="2"/>
  <text x="217" y="145" text-anchor="middle" font-size="7" font-family="monospace" fill="#1f2937">MEASURED WINDOW</text>

  <text x="35" y="180" font-size="8.5" font-weight="bold" fill="#1f2937">Warmup Sources Discarded:</text>
  <text x="35" y="196" font-size="8.5" fill="#6b7280">1. OS Page Faults (allocating physical frames)</text>
  <text x="35" y="212" font-size="8.5" fill="#6b7280">2. CPU Branch Predictor / Instruction cache warming</text>
  <text x="35" y="228" font-size="8.5" fill="#6b7280">3. cuBLAS / Framework runtime workspace init</text>

  <!-- Right: Asymmetric Latency PDF (ACCENT) -->
  <rect x="345" y="20" width="310" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="345" y="20" width="310" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="357" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. LATENCY PROBABILITY DISTRIBUTION</text>
  <text x="357" y="58" font-size="8.5" fill="#6b7280">Heavy right tail caused by OS interrupts &amp; GC pauses</text>

  <!-- Axes -->
  <line x1="365" y1="165" x2="625" y2="165" stroke="#1f2937" stroke-width="1.2"/>
  <line x1="365" y1="165" x2="365" y2="85" stroke="#1f2937" stroke-width="1.2"/>
  <text x="625" y="178" text-anchor="end" font-size="7.5" font-family="monospace" fill="#6b7280">Latency (ms) -&gt;</text>

  <!-- Skewed curve -->
  <path d="M 370 165 C 385 110, 405 95, 420 95 C 440 95, 460 140, 490 150 C 520 160, 550 153, 570 153 C 590 153, 605 161, 620 165" fill="none" stroke="#ff8246" stroke-width="2"/>

  <!-- P50 Median Line -->
  <line x1="420" y1="95" x2="420" y2="165" stroke="#ff8246" stroke-dasharray="2,2"/>
  <text x="420" y="88" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">P50: 1.2ms</text>

  <!-- P95 Line -->
  <line x1="490" y1="150" x2="490" y2="165" stroke="#1f2937" stroke-dasharray="2,2"/>
  <text x="490" y="141" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#1f2937">P95: 2.8ms</text>

  <!-- P99 Tail Line -->
  <line x1="570" y1="153" x2="570" y2="165" stroke="#1f2937" stroke-dasharray="2,2"/>
  <text x="570" y="143" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#1f2937">P99: 5.4ms</text>

  <text x="357" y="200" font-size="8.5" font-weight="bold" fill="#1f2937">The Mean Trap:</text>
  <text x="357" y="216" font-size="8.5" fill="#6b7280">A few 5ms spikes pull Mean to 1.8ms (misleading by 50%).</text>
  <text x="357" y="232" font-size="8.5" fill="#6b7280">Always report P50 (median) + P99 for production SLAs.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Benchmarking requires discarding initial cold-start warmup runs and analyzing percentile distributions rather than misleading arithmetic means.</text>
{FOOTER}"""
    write_svg("19_latency-anatomy-distribution.svg", body)


# -------------------------------------------------------------
# 7. Ch 20: 20_stacking-waterfall-amdahl.svg
# -------------------------------------------------------------
def gen_20_capstone():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Waterfall Area -->
  <!-- Bar 1: Pure Python Baseline -->
  <rect x="45" y="45" width="90" height="145" rx="1" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
  <text x="90" y="65" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#1f2937">Baseline</text>
  <text x="90" y="80" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">100.0 ms</text>
  <text x="90" y="115" text-anchor="middle" font-size="7.5" fill="#6b7280">Naive loops</text>
  <text x="90" y="130" text-anchor="middle" font-size="7.5" fill="#6b7280">&amp; dynamic</text>
  <text x="90" y="145" text-anchor="middle" font-size="7.5" fill="#6b7280">allocations</text>

  <!-- Arrow -->
  <line x1="140" y1="120" x2="155" y2="120" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Bar 2: + Vectorization -->
  <rect x="160" y="95" width="90" height="95" rx="1" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
  <text x="205" y="115" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#1f2937">+ Vectorized</text>
  <text x="205" y="130" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">25.0 ms</text>
  <text x="205" y="155" text-anchor="middle" font-size="7.5" fill="#6b7280">BLAS GEMM</text>
  <text x="205" y="170" text-anchor="middle" font-size="7.5" font-weight="bold" fill="#1f2937">4.0x Speedup</text>

  <!-- Arrow -->
  <line x1="255" y1="130" x2="270" y2="130" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Bar 3: + Kernel Fusion -->
  <rect x="275" y="135" width="90" height="55" rx="1" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
  <text x="320" y="155" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#1f2937">+ Fused</text>
  <text x="320" y="170" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">10.0 ms</text>
  <text x="320" y="183" text-anchor="middle" font-size="7.5" font-weight="bold" fill="#1f2937">10.0x Speedup</text>

  <!-- Arrow -->
  <line x1="370" y1="155" x2="385" y2="155" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Bar 4: + INT8 Quantization -->
  <rect x="390" y="160" width="90" height="30" rx="1" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
  <text x="435" y="174" text-anchor="middle" font-size="8" font-weight="bold" fill="#1f2937">+ INT8 Quant</text>
  <text x="435" y="185" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">4.0 ms (25x)</text>

  <!-- Arrow -->
  <line x1="485" y1="175" x2="500" y2="175" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

  <!-- Bar 5: + KV Caching (ACCENT) -->
  <rect x="505" y="172" width="125" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="567" y="185" text-anchor="middle" font-size="8" font-weight="bold" fill="#ff8246">+ KV Cache: 1.2 ms (83x)</text>

  <!-- Ground Line -->
  <line x1="40" y1="190" x2="640" y2="190" stroke="#1f2937" stroke-width="1.5"/>

  <!-- Bottom: Amdahl's Law Explanation -->
  <rect x="25" y="205" width="630" height="68" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="205" width="630" height="22" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="220" font-size="9" font-weight="700" fill="#1f2937">AMDAHL'S LAW CEILING: SPEEDUP_MAX = 1 / ( (1 - P) + P / S )</text>
  <text x="35" y="244" font-size="8.5" fill="#6b7280">Once tensor math is 99% optimized, Python interpreter dispatch overhead (the unaccelerated 1%) becomes the dominant floor.</text>
  <text x="35" y="258" font-size="8.5" fill="#6b7280">Further speedups require escaping the Python GIL into ahead-of-time compiled native C++ / CUDA execution.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Compounding optimizations address orthogonal bottlenecks until Amdahl's Law shifts the constraint to language runtime overhead.</text>
{FOOTER}"""
    write_svg("20_stacking-waterfall-amdahl.svg", body)


# -------------------------------------------------------------
# 8. Ch 21: 21_compiler-lowering-fusion.svg
# -------------------------------------------------------------
def gen_21_extensions():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Stage 1: Eager Python AST -->
  <rect x="25" y="20" width="190" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="190" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. PYTHON SOURCE / AST</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">Eager expression syntax tree</text>

  <rect x="35" y="74" width="170" height="68" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="43" y="90" font-size="8" font-family="monospace" fill="#1f2937">def gelu(x):</text>
  <text x="43" y="104" font-size="7.5" font-family="monospace" fill="#1f2937">  t1 = x * 0.5</text>
  <text x="43" y="118" font-size="7.5" font-family="monospace" fill="#1f2937">  t2 = tanh(0.797 * x)</text>
  <text x="43" y="132" font-size="7.5" font-family="monospace" fill="#1f2937">  return t1 * (1.0 + t2)</text>

  <text x="35" y="166" font-size="8.5" fill="#6b7280">Interpreted statement-by-statement.</text>
  <text x="35" y="182" font-size="8.5" fill="#6b7280">Produces 4 separate C API calls,</text>
  <text x="35" y="198" font-size="8.5" fill="#6b7280">4 DRAM allocation events, and</text>
  <text x="35" y="214" font-size="8.5" fill="#6b7280">heavy Python GIL overhead.</text>

  <!-- Arrow -->
  <line x1="215" y1="147" x2="235" y2="147" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Stage 2: Computational Graph IR (ACCENT) -->
  <rect x="240" y="20" width="220" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="240" y="20" width="220" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="252" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. FUSED COMPUTATIONAL IR</text>
  <text x="252" y="58" font-size="8.5" fill="#6b7280">Operator fusion cluster</text>

  <!-- Fused Cluster Box -->
  <rect x="250" y="74" width="200" height="88" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <text x="258" y="90" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">[FusedPointwiseCluster_0]</text>
  <text x="258" y="105" font-size="7.5" font-family="monospace" fill="#1f2937">Input: %0 = Tensor(float32, [N])</text>
  <text x="258" y="119" font-size="7.5" font-family="monospace" fill="#1f2937">Ops: [Mul, Tanh, Add, Mul]</text>
  <text x="258" y="133" font-size="7.5" font-family="monospace" fill="#1f2937">Output: %1 = Tensor(float32, [N])</text>
  <text x="258" y="150" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">Dead Stores Eliminated: t1, t2</text>

  <text x="252" y="184" font-size="8.5" fill="#6b7280">Passes performed by compiler:</text>
  <text x="252" y="200" font-size="8.5" fill="#6b7280">1. Common Subexpression Elimination</text>
  <text x="252" y="216" font-size="8.5" fill="#6b7280">2. Loop tiling and affine fusion</text>
  <text x="252" y="232" font-size="8.5" fill="#6b7280">3. Constant folding (0.79788)</text>

  <!-- Arrow -->
  <line x1="460" y1="147" x2="480" y2="147" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

  <!-- Stage 3: Low-Level C / SIMD Kernel -->
  <rect x="485" y="20" width="170" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="485" y="20" width="170" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="495" y="36" font-size="9.5" font-weight="700" fill="#1f2937">3. LOWERED C / SIMD</text>
  <text x="495" y="58" font-size="8.5" fill="#6b7280">Hardware vector instructions</text>

  <rect x="495" y="74" width="150" height="88" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="501" y="89" font-size="7.5" font-family="monospace" fill="#1f2937">#pragma omp simd</text>
  <text x="501" y="102" font-size="7.5" font-family="monospace" fill="#1f2937">for(int i=0; i&lt;N; i+=8) {{</text>
  <text x="501" y="115" font-size="7.5" font-family="monospace" fill="#1f2937">  __m256 vx = _load(x+i);</text>
  <text x="501" y="128" font-size="7.5" font-family="monospace" fill="#1f2937">  __m256 vy = gelu(vx);</text>
  <text x="501" y="141" font-size="7.5" font-family="monospace" fill="#1f2937">  _store(out+i, vy);</text>
  <text x="501" y="154" font-size="7.5" font-family="monospace" fill="#1f2937">}}</text>

  <text x="495" y="184" font-size="8.5" fill="#6b7280">AVX2 / NEON 8-wide vectors.</text>
  <text x="495" y="200" font-size="8.5" fill="#6b7280">Zero Python interpreter</text>
  <text x="495" y="216" font-size="8.5" fill="#6b7280">overhead inside inner loop.</text>
  <text x="495" y="232" font-size="8.5" fill="#6b7280">Bypasses GIL entirely.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">The machine learning compiler pipeline lowers high-level eager expressions into intermediate graph representations before emitting fused SIMD assembly.</text>
{FOOTER}"""
    write_svg("21_compiler-lowering-fusion.svg", body)


# -------------------------------------------------------------
# 9. Milestone 03: milestone_03_pareto-frontier.svg
# -------------------------------------------------------------
def gen_milestone_03():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Main Plot Canvas (ACCENT) -->
  <rect x="25" y="20" width="385" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="25" y="20" width="385" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#c85a17">MULTI-OBJECTIVE PARETO FRONTIER</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">Inference Throughput Speedup vs Memory Compression Ratio</text>

  <!-- Axes -->
  <line x1="55" y1="230" x2="380" y2="230" stroke="#1f2937" stroke-width="1.5"/>
  <line x1="55" y1="230" x2="55" y2="75" stroke="#1f2937" stroke-width="1.5"/>
  <text x="380" y="258" text-anchor="end" font-size="8" font-family="monospace" fill="#1f2937">Inference Speedup -&gt;</text>
  <text x="60" y="85" font-size="8" font-family="monospace" fill="#1f2937">Compression Ratio -&gt;</text>

  <!-- Ticks -->
  <text x="55" y="242" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">1x</text>
  <text x="120" y="242" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">5x</text>
  <text x="190" y="242" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">15x</text>
  <text x="260" y="242" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">35x</text>
  <text x="330" y="242" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">70x</text>

  <!-- Pareto Frontier Curve -->
  <path d="M 65 220 Q 135 175, 210 125 T 320 90" fill="none" stroke="#ff8246" stroke-width="2.5"/>

  <!-- Config 1: Baseline -->
  <circle cx="65" cy="220" r="4" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
  <text x="73" y="215" font-size="7" font-family="monospace" fill="#1f2937">FP32 (1x, 1x)</text>

  <!-- Config 2: Fused FP16 -->
  <circle cx="135" cy="175" r="4" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
  <text x="143" y="170" font-size="7" font-family="monospace" fill="#1f2937">Fused FP16 (6x, 2x)</text>

  <!-- Config 3: INT8 Quantized -->
  <circle cx="210" cy="125" r="4" fill="#ffffff" stroke="#ff8246" stroke-width="2"/>
  <text x="218" y="120" font-size="7" font-family="monospace" font-weight="bold" fill="#ff8246">INT8 (18x, 4x)</text>

  <!-- Config 4: INT8 + KV Cache (Pareto Peak) -->
  <circle cx="320" cy="90" r="5" fill="#ff8246" stroke="#ffffff" stroke-width="1.5"/>
  <text x="320" y="80" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">INT8 + KV Cache (65x, 4.2x)</text>

  <!-- Suboptimal points inside curve -->
  <circle cx="155" cy="205" r="3" fill="#9ca3af"/>
  <text x="163" y="208" font-size="6" font-family="monospace" fill="#9ca3af">Naive INT8 (no fusion)</text>

  <circle cx="230" cy="165" r="3" fill="#9ca3af"/>
  <text x="238" y="168" font-size="6" font-family="monospace" fill="#9ca3af">Over-pruned SVD (r=64)</text>

  <!-- Right: Legend and Rules -->
  <rect x="425" y="20" width="230" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="425" y="20" width="230" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="435" y="36" font-size="9.5" font-weight="700" fill="#1f2937">PARETO SELECTION RULES</text>
  <text x="435" y="58" font-size="8.5" fill="#6b7280">Multi-objective constraint bounds</text>

  <rect x="435" y="74" width="210" height="50" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="443" y="90" font-size="8" font-weight="bold" fill="#1f2937">Accuracy Floor (R &gt;= 0.99):</text>
  <text x="443" y="104" font-size="7.5" fill="#6b7280">Must preserve &gt;= 99% baseline accuracy.</text>
  <text x="443" y="116" font-size="7.5" fill="#6b7280">Strict gating requirement for deployment.</text>

  <rect x="435" y="132" width="210" height="50" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="443" y="148" font-size="8" font-weight="bold" fill="#1f2937">Dominated Solutions:</text>
  <text x="443" y="162" font-size="7.5" fill="#6b7280">Sub-frontier points are discarded:</text>
  <text x="443" y="174" font-size="7.5" fill="#6b7280">a frontier configuration offers higher speed.</text>

  <text x="435" y="204" font-size="8.5" font-weight="bold" fill="#1f2937">Production Verdict:</text>
  <text x="435" y="220" font-size="8.5" fill="#6b7280">INT8 Quantization + KV Caching sits</text>
  <text x="435" y="236" font-size="8.5" fill="#6b7280">at the Pareto peak, yielding 65x speedup</text>
  <text x="435" y="252" font-size="8.5" fill="#6b7280">with zero loss in task accuracy.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">The production Pareto frontier identifies optimal deployment candidates that maximize speedup while satisfying accuracy and memory gates.</text>
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
