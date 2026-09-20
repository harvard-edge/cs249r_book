#!/usr/bin/env python3
"""
Generate High-Impact Pedagogical & Systems SVG Diagrams for TinyTorch.
Strict compliance with Vol 3 Figure 1.10 standard & tinytorch/palette.md:
- 680 width viewBox
- Pure white background (#ffffff), NO outer frame border stroke
- NO embedded canvas titles or subtitles (Quarto fig-cap owns the caption)
- Subsystem container cards with 24px header bands (rx="2")
- Palette: #ffffff, #f8fafc, #fff1e8, #f1f5f9, #9ca3af, #cbd5e1, #ff8246, #c85a17, #1f2937, #6b7280
- Max 1 accent node per diagram
- Fonts: TeX Gyre Heros, Helvetica Neue, Arial, sans-serif
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK_DIAGRAMS = HERE.parent / "assets" / "images" / "diagrams"

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

def save(filename: str, content: str):
    p = BOOK_DIAGRAMS / filename
    p.write_text(content, encoding="utf-8")
    print(f"Generated: {p}")

# -----------------------------------------------------------------------------
# 1. 01_slice-offset-stride.svg
# -----------------------------------------------------------------------------
def gen_01_slice_offset():
    h = 320
    content = f"""{HEADER.format(height=h)}
  <!-- Container Card: Shared 1D Physical DRAM Buffer -->
  <rect x="30" y="25" width="620" height="110" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="25" width="620" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="42" font-size="11" font-weight="700" fill="#1f2937">SHARED PHYSICAL DRAM SUBSTRATE (Flat 1D Byte Ribbon)</text>
  <text x="500" y="42" font-size="10" font-family="monospace" fill="#6b7280">6 x 4B = 24 Bytes</text>

  <!-- slot 0 -->
  <rect x="50" y="65" width="85" height="48" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="92" y="85" text-anchor="middle" font-size="10" font-family="monospace" fill="#6b7280">0x00: [0,0]</text>
  <text x="92" y="103" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937">1.0</text>

  <!-- slot 1 -->
  <rect x="145" y="65" width="85" height="48" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="187" y="85" text-anchor="middle" font-size="10" font-family="monospace" fill="#6b7280">0x04: [0,1]</text>
  <text x="187" y="103" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937">2.0</text>

  <!-- slot 2 -->
  <rect x="240" y="65" width="85" height="48" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="282" y="85" text-anchor="middle" font-size="10" font-family="monospace" fill="#6b7280">0x08: [0,2]</text>
  <text x="282" y="103" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937">3.0</text>

  <!-- slot 3 (Accent: Slice Start) -->
  <rect x="335" y="65" width="85" height="48" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.8"/>
  <text x="377" y="85" text-anchor="middle" font-size="10" font-family="monospace" fill="#c85a17">0x0C: [1,0]</text>
  <text x="377" y="103" text-anchor="middle" font-size="13" font-weight="700" fill="#c85a17">4.0</text>

  <!-- slot 4 (Accent: Slice Next) -->
  <rect x="430" y="65" width="85" height="48" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.8"/>
  <text x="472" y="85" text-anchor="middle" font-size="10" font-family="monospace" fill="#c85a17">0x10: [1,1]</text>
  <text x="472" y="103" text-anchor="middle" font-size="13" font-weight="700" fill="#c85a17">5.0</text>

  <!-- slot 5 -->
  <rect x="525" y="65" width="85" height="48" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="567" y="85" text-anchor="middle" font-size="10" font-family="monospace" fill="#6b7280">0x14: [1,2]</text>
  <text x="567" y="103" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937">6.0</text>

  <!-- Lower Panel: Two Lenses, Zero Copies -->
  <rect x="30" y="160" width="290" height="135" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="160" width="290" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="177" font-size="11" font-weight="700" fill="#1f2937">VIEW A: x (Original Tensor)</text>
  <text x="45" y="208" font-size="11" fill="#4b5563">Shape: <tspan font-family="monospace" font-weight="700" fill="#1f2937">(2, 3)</tspan></text>
  <text x="45" y="228" font-size="11" fill="#4b5563">Strides: <tspan font-family="monospace" font-weight="700" fill="#1f2937">(3, 1)</tspan> elements</text>
  <text x="45" y="248" font-size="11" fill="#4b5563">Base Offset: <tspan font-family="monospace" font-weight="700" fill="#1f2937">0x00</tspan> (index 0)</text>
  <text x="45" y="275" font-size="10" font-style="italic" fill="#6b7280">Reads all 6 slots in row-major order.</text>

  <!-- Right Card: Sliced View -->
  <rect x="360" y="160" width="290" height="135" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
  <rect x="360" y="160" width="290" height="26" rx="2" fill="#ff8246" stroke="#ff8246" stroke-width="1.5"/>
  <text x="375" y="177" font-size="11" font-weight="700" fill="#ffffff">VIEW B: x[1:, :2] (Zero-Copy Slice)</text>
  <text x="375" y="208" font-size="11" fill="#4b5563">Shape: <tspan font-family="monospace" font-weight="700" fill="#1f2937">(1, 2)</tspan></text>
  <text x="375" y="228" font-size="11" fill="#4b5563">Strides: <tspan font-family="monospace" font-weight="700" fill="#1f2937">(3, 1)</tspan> (unmodified)</text>
  <text x="375" y="248" font-size="11" fill="#4b5563">New Offset: <tspan font-family="monospace" font-weight="700" fill="#c85a17">0x0C</tspan> (pointer advances to slot 3)</text>
  <text x="375" y="275" font-size="10" font-weight="700" fill="#c85a17">0 bytes allocated; 0 values copied.</text>

  <!-- Arrow connecting Slice Card to Offset 0x0C -->
  <path d="M505,160 V130 H377 V118" stroke="#ff8246" stroke-width="1.8" fill="none" marker-end="url(#arrow-orange)"/>
{FOOTER}"""
    save("01_slice-offset-stride.svg", content)

# -----------------------------------------------------------------------------
# 2. 02_softmax-numerical-cliff.svg
# -----------------------------------------------------------------------------
def gen_02_softmax_cliff():
    h = 280
    content = f"""{HEADER.format(height=h)}
  <!-- Top Panel: Naive Softmax (Overflow Cliff) -->
  <rect x="30" y="20" width="620" height="110" rx="2" fill="#ffffff" stroke="#ef4444" stroke-width="1.2"/>
  <rect x="30" y="20" width="620" height="24" rx="2" fill="#fee2e2" stroke="#ef4444" stroke-width="1.2"/>
  <text x="45" y="37" font-size="11" font-weight="700" fill="#991b1b">NAIVE SOFTMAX: Catastrophic Float32 Overflow (The Numerical Cliff)</text>

  <text x="45" y="65" font-size="11" fill="#1f2937">Logits: <tspan font-family="monospace">[10.0, 50.0, 100.0]</tspan></text>
  <path d="M225,60 H255" stroke="#9ca3af" stroke-width="1.2" fill="none" marker-end="url(#arrow-gray)"/>
  <text x="268" y="65" font-size="10.5" fill="#4b5563">Exponentiate: <tspan font-family="monospace">e^100 ≈ 2.68 x 10^43 &gt; 3.4 x 10^38 (Float32 Max)</tspan></text>
  <path d="M45,85 H600" stroke="#fca5a5" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="45" y="105" font-size="11" font-weight="700" fill="#dc2626">Result: [2.2e4, 5.1e21, +Inf] / +Inf = [0.0, 0.0, NaN]  --> Model training collapses</text>

  <!-- Bottom Panel: Stable Softmax (Max Subtraction) -->
  <rect x="30" y="145" width="620" height="115" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
  <rect x="30" y="145" width="620" height="24" rx="2" fill="#ff8246" stroke="#ff8246" stroke-width="1.5"/>
  <text x="45" y="162" font-size="11" font-weight="700" fill="#ffffff">STABLE SOFTMAX: Safe LogSumExp with Running Maximum Subtraction</text>

  <text x="45" y="190" font-size="11" fill="#1f2937">Shift by m = max(z) = 100.0: <tspan font-family="monospace">z - 100 = [-90, -50, 0]</tspan></text>
  <path d="M375,185 H405" stroke="#ff8246" stroke-width="1.2" fill="none" marker-end="url(#arrow-orange)"/>
  <text x="415" y="190" font-size="11" fill="#1f2937">Exponentiate: <tspan font-family="monospace">e^0 = 1.0 (Exact)</tspan></text>
  <path d="M45,210 H600" stroke="#fdba74" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="45" y="235" font-size="11" font-weight="700" fill="#c85a17">Probabilities: [0.0000, 0.0000, 1.0000]  --> Perfectly stable, zero NaN risk.</text>
{FOOTER}"""
    save("02_softmax-numerical-cliff.svg", content)

# -----------------------------------------------------------------------------
# 3. 03_linear-dimension-flow.svg
# -----------------------------------------------------------------------------
def gen_03_linear_flow():
    h = 240
    content = f"""{HEADER.format(height=h)}
  <rect x="30" y="20" width="620" height="200" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="20" width="620" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="37" font-size="11" font-weight="700" fill="#1f2937">LINEAR LAYER DIMENSION FLOW: Why Weights Are Transposed (x @ W.T + b)</text>

  <rect x="50" y="70" width="110" height="70" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
  <text x="105" y="95" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">Input X</text>
  <text x="105" y="115" text-anchor="middle" font-size="11" font-family="monospace" fill="#6b7280">(B, D_in)</text>

  <text x="175" y="110" font-size="16" font-weight="700" fill="#1f2937">@</text>

  <rect x="205" y="60" width="130" height="90" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
  <text x="270" y="90" text-anchor="middle" font-size="12" font-weight="700" fill="#c85a17">Weight.T</text>
  <text x="270" y="110" text-anchor="middle" font-size="11" font-family="monospace" fill="#c85a17">(D_in, D_out)</text>
  <text x="270" y="130" text-anchor="middle" font-size="9" fill="#6b7280">W stored as (D_out, D_in)</text>

  <text x="350" y="110" font-size="16" font-weight="700" fill="#1f2937">+</text>

  <rect x="375" y="75" width="95" height="60" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
  <text x="422" y="100" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">Bias b</text>
  <text x="422" y="120" text-anchor="middle" font-size="11" font-family="monospace" fill="#6b7280">(1, D_out)*</text>

  <text x="485" y="110" font-size="16" font-weight="700" fill="#1f2937">=</text>

  <rect x="510" y="70" width="120" height="70" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
  <text x="570" y="95" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">Output Y</text>
  <text x="570" y="115" text-anchor="middle" font-size="11" font-family="monospace" fill="#1f2937">(B, D_out)</text>

  <text x="50" y="185" font-size="10" fill="#4b5563">* Inner dimension <tspan font-family="monospace" font-weight="700" fill="#c85a17">D_in</tspan> contracts during dot product. Row-major stride allows contiguous dot products.</text>
  <text x="50" y="202" font-size="10" fill="#4b5563">* Bias broadcasts across mini-batch <tspan font-family="monospace" font-weight="700">B</tspan> via zero-stride without memory allocation.</text>
{FOOTER}"""
    save("03_linear-dimension-flow.svg", content)

# -----------------------------------------------------------------------------
# 4. 06_gradient-fanin-accumulation.svg
# -----------------------------------------------------------------------------
def gen_06_gradient_fanin():
    h = 290
    content = f"""{HEADER.format(height=h)}
  <rect x="30" y="20" width="620" height="250" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="20" width="620" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="37" font-size="11" font-weight="700" fill="#1f2937">AUTOGRAD TAPE: The Multi-Branch Gradient Fan-in Accumulation Invariant</text>

  <!-- Upstream Node X -->
  <rect x="45" y="105" width="135" height="75" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.8"/>
  <text x="112" y="130" text-anchor="middle" font-size="12" font-weight="700" fill="#c85a17">Node X</text>
  <text x="112" y="148" text-anchor="middle" font-size="10" font-family="monospace" fill="#6b7280">x.grad += dL/dx</text>
  <text x="112" y="165" text-anchor="middle" font-size="9" font-weight="700" fill="#c85a17">Must accumulate!</text>

  <!-- Fork arrows forward -->
  <path d="M180,125 L275,75" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" fill="none" marker-end="url(#arrow-gray)"/>
  <text x="205" y="90" font-size="9" fill="#9ca3af">forward</text>

  <path d="M180,155 L275,205" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" fill="none" marker-end="url(#arrow-gray)"/>
  <text x="205" y="195" font-size="9" fill="#9ca3af">forward</text>

  <!-- Downstream Branch A -->
  <rect x="285" y="50" width="135" height="60" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
  <text x="352" y="75" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Op A (e.g. ReLU)</text>
  <text x="352" y="95" text-anchor="middle" font-size="10" font-family="monospace" fill="#6b7280">grad_A = dL/dA</text>

  <!-- Downstream Branch B -->
  <rect x="285" y="170" width="135" height="60" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
  <text x="352" y="195" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Op B (e.g. Skip)</text>
  <text x="352" y="215" text-anchor="middle" font-size="10" font-family="monospace" fill="#6b7280">grad_B = dL/dB</text>

  <!-- Backward Arrows back to Node X -->
  <path d="M285,85 C225,85 225,125 185,135" stroke="#ff8246" stroke-width="1.8" fill="none" marker-end="url(#arrow-orange)"/>
  <text x="220" y="125" font-size="10" font-weight="700" fill="#c85a17">+ vjp_A</text>

  <path d="M285,195 C225,195 225,155 185,145" stroke="#ff8246" stroke-width="1.8" fill="none" marker-end="url(#arrow-orange)"/>
  <text x="220" y="165" font-size="10" font-weight="700" fill="#c85a17">+ vjp_B</text>

  <!-- Right Explanatory Note -->
  <rect x="445" y="55" width="190" height="175" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="455" y="78" font-size="10.5" font-weight="700" fill="#1f2937">THE INVARIANT:</text>
  <text x="455" y="98" font-size="9" fill="#4b5563">Multivariable chain rule:</text>
  <text x="455" y="118" font-size="10" font-family="monospace" font-weight="700" fill="#c85a17">dL/dx = Σ dL/dy_i * dy_i/dx</text>
  <text x="455" y="145" font-size="9" fill="#4b5563">If <tspan font-family="monospace">backward()</tspan> assigned</text>
  <text x="455" y="160" font-size="9" fill="#4b5563">with <tspan font-family="monospace" font-weight="700" fill="#ef4444">=</tspan> instead of <tspan font-family="monospace" font-weight="700" fill="#c85a17">+=</tspan>,</text>
  <text x="455" y="178" font-size="9" fill="#4b5563">gradients from branch A</text>
  <text x="455" y="195" font-size="9" fill="#4b5563">would be silently overwritten!</text>
{FOOTER}"""
    save("06_gradient-fanin-accumulation.svg", content)

# -----------------------------------------------------------------------------
# 5. 09_im2col-unfolding.svg
# -----------------------------------------------------------------------------
def gen_09_im2col():
    h = 280
    content = f"""{HEADER.format(height=h)}
  <rect x="30" y="20" width="620" height="240" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="20" width="620" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="37" font-size="11" font-weight="700" fill="#1f2937">CONV2D ACCELERATION: Lowering Spatial Sliding Filters to GEMM via im2col</text>

  <rect x="50" y="65" width="130" height="110" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
  <text x="115" y="85" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Input Image X</text>
  <text x="115" y="102" text-anchor="middle" font-size="10" font-family="monospace" fill="#6b7280">(C_in, H, W)</text>
  <rect x="70" y="115" width="45" height="45" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
  <text x="92" y="142" text-anchor="middle" font-size="9" font-weight="700" fill="#c85a17">Patch</text>

  <path d="M190,135 H230" stroke="#ff8246" stroke-width="1.5" fill="none" marker-end="url(#arrow-orange)"/>
  <text x="193" y="125" font-size="9" font-weight="700" fill="#c85a17">im2col</text>

  <rect x="240" y="65" width="150" height="110" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
  <text x="315" y="85" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Unfolded Matrix X_col</text>
  <text x="315" y="102" text-anchor="middle" font-size="9" font-family="monospace" fill="#6b7280">(C_in*Kh*Kw, H_out*W_out)</text>
  <rect x="280" y="115" width="22" height="50" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="291" y="145" text-anchor="middle" font-size="8" font-weight="700" fill="#c85a17">col</text>

  <text x="400" y="125" font-size="14" font-weight="700" fill="#1f2937">@</text>

  <rect x="420" y="70" width="105" height="100" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
  <text x="472" y="90" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Filter Bank</text>
  <text x="472" y="110" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">W_row</text>
  <text x="472" y="130" text-anchor="middle" font-size="9" font-family="monospace" fill="#6b7280">(C_out, K_dim)</text>

  <text x="535" y="125" font-size="14" font-weight="700" fill="#1f2937">=</text>

  <rect x="550" y="75" width="95" height="90" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
  <text x="597" y="110" text-anchor="middle" font-size="11" font-weight="700" fill="#c85a17">Y_col</text>
  <text x="597" y="130" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">(C_out, H*W)</text>
  <text x="597" y="150" text-anchor="middle" font-size="8" fill="#c85a17">reshape to (C,H,W)</text>

  <text x="50" y="200" font-size="9.5" fill="#4b5563">• Overlapping spatial patches are unrolled into columns; spatial dot products become a single matrix multiply.</text>
  <text x="50" y="218" font-size="9.5" fill="#4b5563">• Trades memory footprint (K^2 buffer replication) for massive vendor GEMM library speedups.</text>
{FOOTER}"""
    save("09_im2col-unfolding.svg", content)

# -----------------------------------------------------------------------------
# 6. 10_bpe-merge-progression.svg
# -----------------------------------------------------------------------------
def gen_10_bpe_merge():
    h = 240
    content = f"""{HEADER.format(height=h)}
  <rect x="30" y="20" width="620" height="200" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="20" width="620" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="37" font-size="11" font-weight="700" fill="#1f2937">BYTE-PAIR ENCODING (BPE): Progressive Vocabulary Construction by Pair Frequency</text>

  <rect x="50" y="65" width="160" height="40" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="130" y="89" text-anchor="middle" font-size="11" font-family="monospace" fill="#1f2937">['l', 'o', 'w', '_']</text>
  <text x="50" y="120" font-size="9.5" fill="#6b7280">Step 0: Base byte tokens</text>

  <path d="M220,85 H260" stroke="#ff8246" stroke-width="1.5" fill="none" marker-end="url(#arrow-orange)"/>
  <text x="222" y="77" font-size="8.5" font-weight="700" fill="#c85a17">Rank #1</text>

  <rect x="270" y="65" width="160" height="40" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="350" y="89" text-anchor="middle" font-size="11" font-family="monospace" fill="#1f2937">['lo', 'w', '_']</text>
  <text x="270" y="120" font-size="9.5" fill="#6b7280">Step 1: Merge ('l', 'o') -> 'lo'</text>

  <path d="M440,85 H480" stroke="#ff8246" stroke-width="1.5" fill="none" marker-end="url(#arrow-orange)"/>
  <text x="442" y="77" font-size="8.5" font-weight="700" fill="#c85a17">Rank #2</text>

  <rect x="490" y="65" width="145" height="40" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
  <text x="562" y="89" text-anchor="middle" font-size="11" font-family="monospace" font-weight="700" fill="#c85a17">['low', '_']</text>
  <text x="490" y="120" font-size="9.5" font-weight="700" fill="#c85a17">Step 2: Merge ('lo', 'w') -> 'low'</text>

  <line x1="50" y1="145" x2="610" y2="145" stroke="#e2e8f0" stroke-width="1"/>
  <text x="50" y="168" font-size="9.5" fill="#4b5563">• BPE greedy compression resolves out-of-vocabulary (OOV) tokens by falling back to UTF-8 byte units.</text>
  <text x="50" y="186" font-size="9.5" fill="#4b5563">• Tokenizer dictionary stores exact merge rankings; inference repeats merges in identical priority order.</text>
{FOOTER}"""
    save("10_bpe-merge-progression.svg", content)

# -----------------------------------------------------------------------------
# 7. 11_rope-phase-clock.svg
# -----------------------------------------------------------------------------
def gen_11_rope_clock():
    h = 270
    content = f"""{HEADER.format(height=h)}
  <rect x="30" y="20" width="620" height="230" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="20" width="620" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="37" font-size="11" font-weight="700" fill="#1f2937">ROTARY POSITION EMBEDDING (RoPE): 2D Complex Subspace Phase Rotation</text>

  <circle cx="140" cy="135" r="50" fill="#f8fafc" stroke="#9ca3af" stroke-width="1.2"/>
  <line x1="85" y1="135" x2="195" y2="135" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="140" y1="80" x2="140" y2="190" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="140" y1="135" x2="175" y2="95" stroke="#ff8246" stroke-width="2" marker-end="url(#arrow-orange)"/>
  <text x="175" y="90" font-size="10" font-weight="700" fill="#c85a17">R_m * q</text>
  <text x="140" y="205" text-anchor="middle" font-size="9.5" fill="#1f2937">Query at pos m (angle m·θ)</text>

  <circle cx="330" cy="135" r="50" fill="#f8fafc" stroke="#9ca3af" stroke-width="1.2"/>
  <line x1="275" y1="135" x2="385" y2="135" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="330" y1="80" x2="330" y2="190" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="330" y1="135" x2="355" y2="85" stroke="#1f2937" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="360" y="85" font-size="10" font-weight="700" fill="#1f2937">R_n * k</text>
  <text x="330" y="205" text-anchor="middle" font-size="9.5" fill="#1f2937">Key at pos n (angle n·θ)</text>

  <rect x="435" y="65" width="200" height="145" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
  <text x="445" y="88" font-size="10.5" font-weight="700" fill="#c85a17">THE ROPE INVARIANT:</text>
  <text x="445" y="110" font-size="9" fill="#1f2937">Dot product of rotated vectors:</text>
  <text x="445" y="130" font-size="9.5" font-family="monospace" font-weight="700" fill="#c85a17">&lt;R_m q, R_n k&gt; =</text>
  <text x="445" y="148" font-size="9.5" font-family="monospace" font-weight="700" fill="#c85a17">q^T R_(n-m) k</text>
  <text x="445" y="173" font-size="9" fill="#4b5563">Depends strictly on relative</text>
  <text x="445" y="188" font-size="9" fill="#4b5563">distance (n - m), preserving shift invariance!</text>
{FOOTER}"""
    save("11_rope-phase-clock.svg", content)

# -----------------------------------------------------------------------------
# 8. 12_head-split-origami.svg
# -----------------------------------------------------------------------------
def gen_12_head_split():
    h = 260
    content = f"""{HEADER.format(height=h)}
  <rect x="30" y="20" width="620" height="220" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="20" width="620" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="37" font-size="11" font-weight="700" fill="#1f2937">MULTI-HEAD ATTENTION: Coordinate Origami (B, S, D) -> (B, H, S, d_k)</text>

  <rect x="50" y="70" width="135" height="90" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
  <text x="117" y="95" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Fused Projections</text>
  <text x="117" y="115" text-anchor="middle" font-size="10" font-family="monospace" fill="#6b7280">(B, Seq, D_model)</text>
  <text x="117" y="140" text-anchor="middle" font-size="9" fill="#6b7280">e.g. (1, 16, 64)</text>

  <path d="M195,115 H255" stroke="#ff8246" stroke-width="1.8" fill="none" marker-end="url(#arrow-orange)"/>
  <text x="200" y="105" font-size="8" font-weight="700" fill="#c85a17">reshape +</text>
  <text x="200" y="132" font-size="8" font-weight="700" fill="#c85a17">transpose(1,2)</text>

  <rect x="270" y="60" width="140" height="60" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
  <text x="340" y="85" text-anchor="middle" font-size="10" font-weight="700" fill="#c85a17">Head 0 Subspace</text>
  <text x="340" y="102" text-anchor="middle" font-size="9" font-family="monospace" fill="#c85a17">(B, S, d_k = 32)</text>

  <rect x="270" y="130" width="140" height="60" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
  <text x="340" y="155" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Head 1 Subspace</text>
  <text x="340" y="172" text-anchor="middle" font-size="9" font-family="monospace" fill="#6b7280">(B, S, d_k = 32)</text>

  <rect x="430" y="60" width="205" height="135" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
  <text x="440" y="82" font-size="10" font-weight="700" fill="#1f2937">WHY TRANSPOSE?</text>
  <text x="440" y="102" font-size="9" fill="#4b5563">By placing <tspan font-family="monospace" font-weight="700">H</tspan> before <tspan font-family="monospace" font-weight="700">S</tspan>:</text>
  <text x="440" y="122" font-size="10" font-family="monospace" font-weight="700" fill="#c85a17">Q @ K.T</text>
  <text x="440" y="145" font-size="9" fill="#4b5563">executes parallel 2D matrix multiplies</text>
  <text x="440" y="160" font-size="9" fill="#4b5563">across all heads simultaneously</text>
  <text x="440" y="175" font-size="9" fill="#4b5563">in a single hardware batch call!</text>
{FOOTER}"""
    save("12_head-split-origami.svg", content)

# -----------------------------------------------------------------------------
# 9. 15_zero-point-centering.svg
# -----------------------------------------------------------------------------
def gen_15_zero_point():
    h = 240
    content = f"""{HEADER.format(height=h)}
  <rect x="30" y="20" width="620" height="200" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="20" width="620" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="37" font-size="11" font-weight="700" fill="#1f2937">QUANTIZATION: Symmetric (Z_W = 0) vs Asymmetric Zero-Point Offset</text>

  <text x="50" y="75" font-size="10" font-weight="700" fill="#1f2937">ASYMMETRIC INT8 (Min-Max):</text>
  <line x1="50" y1="95" x2="380" y2="95" stroke="#9ca3af" stroke-width="2"/>
  <circle cx="50" cy="95" r="4" fill="#1f2937"/>
  <text x="50" y="115" text-anchor="middle" font-size="9" font-family="monospace">min (q=-128)</text>
  <circle cx="160" cy="95" r="5" fill="#ef4444"/>
  <text x="160" y="115" text-anchor="middle" font-size="9" font-family="monospace" fill="#ef4444">0.0 (q=Z)</text>
  <circle cx="380" cy="95" r="4" fill="#1f2937"/>
  <text x="380" y="115" text-anchor="middle" font-size="9" font-family="monospace">max (q=127)</text>

  <text x="50" y="150" font-size="10" font-weight="700" fill="#c85a17">SYMMETRIC INT8 (AbsMax):</text>
  <line x1="50" y1="170" x2="380" y2="170" stroke="#ff8246" stroke-width="2.5"/>
  <circle cx="50" cy="170" r="4" fill="#c85a17"/>
  <text x="50" y="190" text-anchor="middle" font-size="9" font-family="monospace">-M (q=-127)</text>
  <circle cx="215" cy="170" r="6" fill="#ff8246"/>
  <text x="215" y="190" text-anchor="middle" font-size="10" font-family="monospace" font-weight="700" fill="#c85a17">0.0 (q = Z = 0)</text>
  <circle cx="380" cy="170" r="4" fill="#c85a17"/>
  <text x="380" y="190" text-anchor="middle" font-size="9" font-family="monospace">+M (q=127)</text>

  <rect x="420" y="60" width="210" height="135" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="430" y="82" font-size="10" font-weight="700" fill="#c85a17">WHY SYMMETRIC WINS:</text>
  <text x="430" y="105" font-size="9" fill="#1f2937">Integer dot product:</text>
  <text x="430" y="125" font-size="9.5" font-family="monospace" fill="#c85a17">Σ (Q_x - Z_x)(Q_w - Z_w)</text>
  <text x="430" y="145" font-size="9" fill="#4b5563">When <tspan font-family="monospace" font-weight="700">Z_w = 0</tspan>, the cross-term</text>
  <text x="430" y="160" font-size="9.5" font-family="monospace" fill="#c85a17">- Z_x * Σ Q_w</text>
  <text x="430" y="178" font-size="8.5" fill="#4b5563">cancels completely on integer hardware!</text>
{FOOTER}"""
    save("15_zero-point-centering.svg", content)

# -----------------------------------------------------------------------------
# 10. 16_csr-three-arrays.svg
# -----------------------------------------------------------------------------
def gen_16_csr_arrays():
    h = 260
    content = f"""{HEADER.format(height=h)}
  <rect x="30" y="20" width="620" height="220" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="20" width="620" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="37" font-size="11" font-weight="700" fill="#1f2937">COMPRESSED SPARSE ROW (CSR): 3-Array Storage vs Dense 2D Grid</text>

  <rect x="50" y="65" width="130" height="120" rx="2" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
  <text x="115" y="85" text-anchor="middle" font-size="10" font-weight="700">Dense Grid (66% Zeros)</text>
  <text x="80" y="115" font-family="monospace" font-weight="700" fill="#c85a17">1.5</text> <text x="115" y="115" font-family="monospace" fill="#9ca3af">0.0</text> <text x="145" y="115" font-family="monospace" fill="#9ca3af">0.0</text>
  <text x="80" y="140" font-family="monospace" fill="#9ca3af">0.0</text> <text x="115" y="140" font-family="monospace" font-weight="700" fill="#c85a17">2.0</text> <text x="145" y="140" font-family="monospace" fill="#9ca3af">0.0</text>
  <text x="80" y="165" font-family="monospace" font-weight="700" fill="#c85a17">3.5</text> <text x="115" y="165" font-family="monospace" fill="#9ca3af">0.0</text> <text x="145" y="165" font-family="monospace" font-weight="700" fill="#c85a17">4.0</text>

  <path d="M195,125 H230" stroke="#ff8246" stroke-width="1.5" fill="none" marker-end="url(#arrow-orange)"/>

  <rect x="245" y="65" width="385" height="35" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="255" y="87" font-size="10" font-weight="700" fill="#c85a17">values:      [1.5, 2.0, 3.5, 4.0] (Nonzeros only)</text>

  <rect x="245" y="110" width="385" height="35" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1"/>
  <text x="255" y="132" font-size="10" font-family="monospace" fill="#1f2937">col_indices: [ 0,   1,   0,   2 ] (Columns)</text>

  <rect x="245" y="155" width="385" height="35" rx="2" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
  <text x="255" y="177" font-size="9.5" font-family="monospace" fill="#4b5563">row_ptr:     [ 0,   1,   2,   4 ] (Row slice bounds)</text>

  <text x="50" y="215" font-size="9.5" fill="#6b7280">• Each nonzero value requires an index; at low sparsity (&lt; 66%), index metadata exceeds saved zeros!</text>
{FOOTER}"""
    save("16_csr-three-arrays.svg", content)

# -----------------------------------------------------------------------------
# 11. 18_kv-cache-cursor.svg
# -----------------------------------------------------------------------------
def gen_18_kv_cursor():
    h = 240
    content = f"""{HEADER.format(height=h)}
  <rect x="30" y="20" width="620" height="200" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
  <rect x="30" y="20" width="620" height="26" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="45" y="37" font-size="11" font-weight="700" fill="#1f2937">KV CACHE STATE MACHINE: Preallocated Buffer with Advancing Cursor (seq_pos)</text>

  <text x="50" y="70" font-size="10" font-weight="700" fill="#1f2937">Persistent Preallocated Slots (max_seq_len = 5):</text>
  
  <rect x="50" y="85" width="100" height="50" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="100" y="105" text-anchor="middle" font-size="10" font-family="monospace">Slot 0 (t=0)</text>
  <text x="100" y="122" text-anchor="middle" font-size="9" fill="#16a34a">✓ Valid K,V</text>

  <rect x="160" y="85" width="100" height="50" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="210" y="105" text-anchor="middle" font-size="10" font-family="monospace">Slot 1 (t=1)</text>
  <text x="210" y="122" text-anchor="middle" font-size="9" fill="#16a34a">✓ Valid K,V</text>

  <rect x="270" y="85" width="100" height="50" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="2"/>
  <text x="320" y="105" text-anchor="middle" font-size="10" font-family="monospace" font-weight="700" fill="#c85a17">Slot 2 (t=2)</text>
  <text x="320" y="122" text-anchor="middle" font-size="9" font-weight="700" fill="#c85a17">WRITE HEAD</text>

  <rect x="380" y="85" width="100" height="50" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="430" y="115" text-anchor="middle" font-size="9" font-family="monospace" fill="#9ca3af">Reserved</text>

  <rect x="490" y="85" width="100" height="50" rx="2" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="540" y="115" text-anchor="middle" font-size="9" font-family="monospace" fill="#9ca3af">Reserved</text>

  <path d="M320,165 V140" stroke="#ff8246" stroke-width="2" fill="none" marker-end="url(#arrow-orange)"/>
  <text x="320" y="180" text-anchor="middle" font-size="10" font-family="monospace" font-weight="700" fill="#c85a17">cursor: seq_pos = 2</text>

  <text x="50" y="205" font-size="9.5" fill="#4b5563">• Prefill writes prompt slots; decode advances cursor by 1 each step. Single query attends to slots 0..seq_pos.</text>
{FOOTER}"""
    save("18_kv-cache-cursor.svg", content)

if __name__ == "__main__":
    BOOK_DIAGRAMS.mkdir(parents=True, exist_ok=True)
    gen_01_slice_offset()
    gen_02_softmax_cliff()
    gen_03_linear_flow()
    gen_06_gradient_fanin()
    gen_09_im2col()
    gen_10_bpe_merge()
    gen_11_rope_clock()
    gen_12_head_split()
    gen_15_zero_point()
    gen_16_csr_arrays()
    gen_18_kv_cursor()
    print("Regenerated all 11 pedagogical diagrams with refined spacing!")
