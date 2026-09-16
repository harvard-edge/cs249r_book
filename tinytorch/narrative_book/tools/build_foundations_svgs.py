#!/usr/bin/env python3
"""
Generate Tier 1 (Foundations) Mechanical SVG Diagrams for TinyTorch Narrative Book.
Strict compliance with STYLE.md:
- 680 width viewBox
- Palette: #ffffff, #f4f5f7, #fff1e8, #f8f9fa, #9ca3af, #ff8246, #1f2937, #6b7280
- Max 1 accent node per diagram
- Fonts: TeX Gyre Heros, Helvetica Neue, Arial, sans-serif
- Uniform stroke weights, crisp geometry, tactile systems intuition.
"""

import os
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


# -----------------------------------------------------------------------------
# 1. 00_physical-stack.svg
# -----------------------------------------------------------------------------
def gen_00_physical_stack():
    h = 390
    body = f"""{HEADER.format(height=h)}
  <!-- Layer 1: User Application Code -->
  <rect x="50" y="25" width="580" height="52" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <rect x="65" y="37" width="130" height="28" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="130" y="55" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">User Script</text>
  <text x="210" y="50" font-size="11" font-weight="700" fill="#1f2937">y = x @ W + b; loss = criterion(y, target); loss.backward()</text>
  <text x="210" y="65" font-size="9.5" fill="#6b7280">Python high-level operations declaring models, losses, and training loops</text>

  <path d="M340 77 V93" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

  <!-- Layer 2: Tensor Object Model -->
  <rect x="50" y="95" width="580" height="52" rx="3" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="65" y="107" width="130" height="28" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="130" y="125" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Tensor Object</text>
  <text x="210" y="120" font-size="11" font-weight="700" fill="#1f2937">Tensor(data, shape=(B, D), strides=(D, 1), requires_grad=True)</text>
  <text x="210" y="135" font-size="9.5" fill="#6b7280">Python wrapper holding metadata, gradient references, and creation operation</text>

  <path d="M340 147 V163" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

  <!-- Layer 3: Dynamic Autograd Tape (ACCENT) -->
  <rect x="50" y="165" width="580" height="56" rx="3" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="65" y="179" width="130" height="28" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <text x="130" y="197" text-anchor="middle" font-size="11" font-weight="700" fill="#ff8246">Autograd Tape</text>
  <text x="210" y="190" font-size="11" font-weight="700" fill="#1f2937">Directed Acyclic Graph (DAG) of Function closures</text>
  <text x="210" y="206" font-size="9.5" fill="#6b7280">Records backward closures, saved tensors, and topological execution order</text>

  <path d="M340 221 V237" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

  <!-- Layer 4: Strided Buffer Descriptor -->
  <rect x="50" y="239" width="580" height="52" rx="3" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="65" y="251" width="130" height="28" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="130" y="269" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Buffer View</text>
  <text x="210" y="264" font-size="11" font-weight="700" fill="#1f2937">Linear Indexing: offset = i * stride[0] + j * stride[1]</text>
  <text x="210" y="279" font-size="9.5" fill="#6b7280">Zero-copy slicing, transpositions, and broadcasting without data movement</text>

  <path d="M340 291 V307" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

  <!-- Layer 5: Physical Hardware RAM -->
  <rect x="50" y="309" width="580" height="52" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <rect x="65" y="321" width="130" height="28" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="130" y="339" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Physical DRAM</text>
  <text x="210" y="334" font-size="11" font-weight="700" fill="#1f2937">Contiguous Flat C-Array: 0x7fff0000 [ float32 | float32 | ... ]</text>
  <text x="210" y="349" font-size="9.5" fill="#6b7280">Contiguous 4-byte floating-point words aligned to 64-byte hardware cache lines</text>
{FOOTER}"""
    write_svg("00_physical-stack.svg", body)


# -----------------------------------------------------------------------------
# 2. 01_stride-zero-broadcast.svg
# -----------------------------------------------------------------------------
def gen_01_stride_zero_broadcast():
    h = 360
    body = f"""{HEADER.format(height=h)}
  <!-- Section 1: Physical Buffer in Memory -->
  <text x="40" y="30" font-size="12" font-weight="700" fill="#1f2937">1. Physical 1D Storage in Memory (shape [3], 12 bytes)</text>
  
  <!-- Buffer Cells -->
  <g transform="translate(40, 45)">
    <!-- Cell 0 -->
    <rect x="0" y="15" width="80" height="40" rx="0" fill="#f4f5f7" stroke="#1f2937" stroke-width="1.2"/>
    <text x="40" y="38" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937">10.0</text>
    <text x="40" y="10" text-anchor="middle" font-size="10" fill="#6b7280">offset 0 (0x00)</text>

    <!-- Cell 1 -->
    <rect x="80" y="15" width="80" height="40" rx="0" fill="#f4f5f7" stroke="#1f2937" stroke-width="1.2"/>
    <text x="120" y="38" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937">20.0</text>
    <text x="120" y="10" text-anchor="middle" font-size="10" fill="#6b7280">offset 1 (0x04)</text>

    <!-- Cell 2 -->
    <rect x="160" y="15" width="80" height="40" rx="0" fill="#f4f5f7" stroke="#1f2937" stroke-width="1.2"/>
    <text x="200" y="38" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937">30.0</text>
    <text x="200" y="10" text-anchor="middle" font-size="10" fill="#6b7280">offset 2 (0x08)</text>
    
    <text x="280" y="40" font-size="11" fill="#6b7280">← Stored contiguously in RAM (1 float32 = 4 bytes)</text>
  </g>

  <!-- Divider -->
  <line x1="40" y1="125" x2="640" y2="125" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4,4"/>

  <!-- Section 2: Virtual 2D Broadcast View -->
  <text x="40" y="150" font-size="12" font-weight="700" fill="#1f2937">2. Broadcast View: shape (2, 3), element strides (0, 1)</text>

  <!-- Matrix Grid -->
  <g transform="translate(40, 175)">
    <!-- Row 0 -->
    <rect x="0" y="0" width="70" height="45" rx="0" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="35" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">10.0</text>
    <text x="35" y="36" text-anchor="middle" font-size="8.5" fill="#6b7280">[0, 0] → slot 0</text>

    <rect x="70" y="0" width="70" height="45" rx="0" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="105" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">20.0</text>
    <text x="105" y="36" text-anchor="middle" font-size="8.5" fill="#6b7280">[0, 1] → slot 1</text>

    <rect x="140" y="0" width="70" height="45" rx="0" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="175" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">30.0</text>
    <text x="175" y="36" text-anchor="middle" font-size="8.5" fill="#6b7280">[0, 2] → slot 2</text>

    <!-- Row 1 (ACCENT: Row Stride 0) -->
    <rect x="0" y="55" width="70" height="45" rx="0" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="35" y="77" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">10.0</text>
    <text x="35" y="91" text-anchor="middle" font-size="8.5" fill="#ff8246">[1, 0] → slot 0</text>

    <rect x="70" y="55" width="70" height="45" rx="0" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="105" y="77" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">20.0</text>
    <text x="105" y="91" text-anchor="middle" font-size="8.5" fill="#ff8246">[1, 1] → slot 1</text>

    <rect x="140" y="55" width="70" height="45" rx="0" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="175" y="77" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">30.0</text>
    <text x="175" y="91" text-anchor="middle" font-size="8.5" fill="#ff8246">[1, 2] → slot 2</text>
  </g>

  <!-- Explanation Panel on Right -->
  <g transform="translate(290, 165)">
    <rect x="0" y="0" width="350" height="135" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
    <text x="20" y="25" font-size="11" font-weight="700" fill="#1f2937">The Stride-0 Offset Invariant:</text>
    <text x="20" y="48" font-size="11" font-family="monospace" fill="#1f2937">offset = (row * stride[0]) + (col * stride[1])</text>
    <text x="20" y="70" font-size="10.5" font-family="monospace" fill="#ff8246">offset = (1 * 0) + (1 * 1) = 1</text>
    <text x="20" y="95" font-size="9.5" fill="#1f2937">• Row stride = 0: row index multiplies by 0.</text>
    <text x="20" y="112" font-size="9.5" fill="#1f2937">• Memory read pointer never advances when stepping rows.</text>
    <text x="20" y="125" font-size="9.5" fill="#6b7280">• Zero extra memory allocated: 6 virtual elements in 3 physical slots.</text>
  </g>
  
  <text x="340" y="340" text-anchor="middle" font-size="9.5" fill="#6b7280">Hardware reality: Stride-0 broadcasting allows GEMM to reuse cache lines across batch rows without data copying.</text>
{FOOTER}"""
    write_svg("01_stride-zero-broadcast.svg", body)


# -----------------------------------------------------------------------------
# 3. 02_space-folding-relu.svg
# -----------------------------------------------------------------------------
def gen_02_space_folding_relu():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Left Panel: Input Space (Non-separable) -->
  <g transform="translate(40, 25)">
    <text x="100" y="15" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">Input Space (x₁, x₂)</text>
    <text x="100" y="32" text-anchor="middle" font-size="9.5" fill="#6b7280">Linearly Inseparable (XOR)</text>
    
    <!-- Coordinate Box -->
    <rect x="10" y="45" width="180" height="180" rx="2" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
    <!-- Axes -->
    <line x1="30" y1="205" x2="175" y2="205" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>
    <line x1="30" y1="205" x2="30" y2="60" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>
    <text x="170" y="218" font-size="9" fill="#6b7280">x₁</text>
    <text x="20" y="65" font-size="9" fill="#6b7280">x₂</text>

    <!-- Failed linear hyperplane -->
    <line x1="40" y1="70" x2="170" y2="200" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="4,3"/>
    <text x="115" y="125" font-size="8.5" fill="#9ca3af" transform="rotate(45, 115, 125)">No single cut succeeds</text>

    <!-- XOR Points -->
    <circle cx="50" cy="185" r="5" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
    <text x="50" y="175" text-anchor="middle" font-size="8.5" font-weight="700" fill="#1f2937">(0,0)</text>

    <circle cx="150" cy="85" r="5" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
    <text x="150" y="75" text-anchor="middle" font-size="8.5" font-weight="700" fill="#1f2937">(1,1)</text>

    <circle cx="50" cy="85" r="5" fill="#1f2937"/>
    <text x="50" y="100" text-anchor="middle" font-size="8.5" font-weight="700" fill="#1f2937">(0,1)</text>

    <circle cx="150" cy="185" r="5" fill="#1f2937"/>
    <text x="150" y="200" text-anchor="middle" font-size="8.5" font-weight="700" fill="#1f2937">(1,0)</text>
  </g>

  <!-- Center Arrow: Linear + ReLU Transformation -->
  <g transform="translate(255, 110)">
    <path d="M0 25 H40" stroke="#ff8246" stroke-width="1.5" marker-end="url(#arrow-orange)"/>
    <text x="20" y="12" text-anchor="middle" font-size="10" font-weight="700" fill="#ff8246">Layer 1</text>
    <text x="20" y="45" text-anchor="middle" font-size="9" fill="#1f2937">W₁x + b₁</text>
    <text x="20" y="60" text-anchor="middle" font-size="9" font-weight="700" fill="#ff8246">+ ReLU</text>
  </g>

  <!-- Right Panel: Hidden Folded Space (ACCENT) -->
  <g transform="translate(325, 25)">
    <text x="155" y="15" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">Folded Feature Space (h₁, h₂)</text>
    <text x="155" y="32" text-anchor="middle" font-size="9.5" fill="#ff8246">Linearly Separable via Axis Folding</text>
    
    <!-- Coordinate Box -->
    <rect x="25" y="45" width="260" height="180" rx="3" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <!-- Axes -->
    <line x1="50" y1="205" x2="265" y2="205" stroke="#1f2937" stroke-width="1" marker-end="url(#arrow)"/>
    <line x1="50" y1="205" x2="50" y2="60" stroke="#1f2937" stroke-width="1" marker-end="url(#arrow)"/>
    <text x="260" y="218" font-size="9" fill="#1f2937">h₁ = ReLU(x₁+x₂)</text>
    <text x="40" y="65" font-size="9" fill="#1f2937">h₂ = ReLU(x₁+x₂-1)</text>

    <!-- Successful Linear Separator (adjusted geometry for breathing room) -->
    <line x1="65" y1="205" x2="215" y2="80" stroke="#1f2937" stroke-width="1.8"/>
    <text x="145" y="130" font-size="8.5" font-weight="700" fill="#1f2937" transform="rotate(-40, 145, 130)">Hyperplane: h₁ - 2h₂ = 0.5</text>

    <!-- Transformed Points -->
    <circle cx="50" cy="205" r="5" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
    <text x="75" y="218" font-size="8.5" font-weight="700" fill="#1f2937">(0,0) → [0, 0]</text>

    <!-- Collapsed (0,1) and (1,0) -> (1, 0) -->
    <circle cx="150" cy="205" r="6" fill="#ff8246" stroke="#1f2937" stroke-width="1.5"/>
    <text x="150" y="193" text-anchor="middle" font-size="8.5" font-weight="700" fill="#1f2937">(0,1) &amp; (1,0) Folded to [1, 0]</text>

    <!-- (1,1) -> (2, 1) -->
    <circle cx="215" cy="125" r="5" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
    <text x="215" y="115" text-anchor="middle" font-size="8.5" font-weight="700" fill="#1f2937">(1,1) → [2, 1]</text>
  </g>

  <text x="340" y="280" text-anchor="middle" font-size="9.5" fill="#1f2937">Without ReLU, stacked affine layers collapse: W₂(W₁x + b₁) + b₂ = W_eff x + b_eff (still linear).</text>
  <text x="340" y="298" text-anchor="middle" font-size="9.5" fill="#6b7280">ReLU acts as a mechanical hinge along coordinate axes, clamping negative pre-activations to 0 and folding space.</text>
{FOOTER}"""
    write_svg("02_space-folding-relu.svg", body)


# -----------------------------------------------------------------------------
# 4. 03_variance-waterfall.svg
# -----------------------------------------------------------------------------
def gen_03_variance_waterfall():
    h = 350
    body = f"""{HEADER.format(height=h)}
  <text x="340" y="25" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937">Signal Propagation Across Depth: The Need for Scaled Initialization</text>

  <!-- Column 1: Unscaled Initialization -->
  <g transform="translate(40, 45)">
    <rect x="0" y="0" width="180" height="255" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
    <text x="90" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Unscaled: N(0, 1)</text>
    <text x="90" y="38" text-anchor="middle" font-size="9" fill="#6b7280">Var(W) = 1.0</text>
    
    <text x="15" y="70" font-size="9.5" font-weight="700" fill="#1f2937">Input (L=0):</text>
    <text x="110" y="70" font-size="9.5" fill="#1f2937">σ² = 1.0</text>
    
    <text x="15" y="115" font-size="9.5" font-weight="700" fill="#1f2937">Layer 5:</text>
    <text x="110" y="115" font-size="9.5" fill="#1f2937">σ² ≈ 10³</text>
    
    <text x="15" y="160" font-size="9.5" font-weight="700" fill="#1f2937">Layer 10:</text>
    <text x="110" y="160" font-size="9.5" fill="#1f2937">σ² ≈ 10²⁰</text>
    
    <text x="15" y="205" font-size="9.5" font-weight="700" fill="#1f2937">Layer 15:</text>
    <text x="110" y="205" font-size="9.5" fill="#1f2937">σ² ≈ 10⁴⁰</text>

    <rect x="15" y="222" width="150" height="22" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1"/>
    <text x="90" y="237" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">Explosion → NaN Overflow</text>
  </g>

  <!-- Column 2: Small Initialization -->
  <g transform="translate(250, 45)">
    <rect x="0" y="0" width="180" height="255" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
    <text x="90" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1f2937">Underscaled: N(0, 0.01)</text>
    <text x="90" y="38" text-anchor="middle" font-size="9" fill="#6b7280">Var(W) = 10⁻⁴</text>

    <text x="15" y="70" font-size="9.5" font-weight="700" fill="#1f2937">Input (L=0):</text>
    <text x="110" y="70" font-size="9.5" fill="#1f2937">σ² = 1.0</text>
    
    <text x="15" y="115" font-size="9.5" font-weight="700" fill="#1f2937">Layer 5:</text>
    <text x="110" y="115" font-size="9.5" fill="#1f2937">σ² ≈ 10⁻⁴</text>
    
    <text x="15" y="160" font-size="9.5" font-weight="700" fill="#1f2937">Layer 10:</text>
    <text x="110" y="160" font-size="9.5" fill="#1f2937">σ² ≈ 10⁻¹²</text>
    
    <text x="15" y="205" font-size="9.5" font-weight="700" fill="#1f2937">Layer 15:</text>
    <text x="110" y="205" font-size="9.5" fill="#1f2937">σ² → 0.0</text>

    <rect x="15" y="222" width="150" height="22" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1"/>
    <text x="90" y="237" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">Collapse → Vanishing Gradient</text>
  </g>

  <!-- Column 3: Kaiming He (ACCENT) -->
  <g transform="translate(460, 45)">
    <rect x="0" y="0" width="180" height="255" rx="3" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="90" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#ff8246">Kaiming He: √(2 / D_in)</text>
    <text x="90" y="38" text-anchor="middle" font-size="9" fill="#1f2937">Accounts for ReLU halving</text>

    <text x="15" y="70" font-size="9.5" font-weight="700" fill="#1f2937">Input (L=0):</text>
    <text x="110" y="70" font-size="9.5" font-weight="700" fill="#ff8246">σ² = 1.00</text>
    
    <text x="15" y="115" font-size="9.5" font-weight="700" fill="#1f2937">Layer 5:</text>
    <text x="110" y="115" font-size="9.5" fill="#1f2937">σ² ≈ 1.01</text>
    
    <text x="15" y="160" font-size="9.5" font-weight="700" fill="#1f2937">Layer 10:</text>
    <text x="110" y="160" font-size="9.5" fill="#1f2937">σ² ≈ 0.99</text>
    
    <text x="15" y="205" font-size="9.5" font-weight="700" fill="#1f2937">Layer 15:</text>
    <text x="110" y="205" font-size="9.5" fill="#1f2937">σ² ≈ 1.02</text>

    <rect x="15" y="222" width="150" height="22" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
    <text x="90" y="237" text-anchor="middle" font-size="9" font-weight="700" fill="#ff8246">Invariant Signal Preserved</text>
  </g>

  <text x="340" y="325" text-anchor="middle" font-size="9.5" fill="#6b7280">Under ReLU, half the activations are zeroed out (E[max(0, z)²] = 0.5 Var(z)). Scaling variance by 2/D_in restores balance.</text>
{FOOTER}"""
    write_svg("03_variance-waterfall.svg", body)


# -----------------------------------------------------------------------------
# 5. 04_index-gather-memory.svg
# -----------------------------------------------------------------------------
def gen_04_index_gather_memory():
    h = 340
    body = f"""{HEADER.format(height=h)}
  <!-- Top Path: Textbook One-Hot -->
  <text x="40" y="25" font-size="12" font-weight="700" fill="#1f2937">Textbook Formulation: One-Hot Multiplication</text>
  <text x="40" y="42" font-size="9.5" fill="#6b7280">L = - Σ y_c log(p_c) where y is a sparse one-hot matrix</text>

  <g transform="translate(40, 52)">
    <!-- Big One-Hot Matrix -->
    <rect x="0" y="0" width="220" height="70" rx="2" fill="#f8f9fa" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="110" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Dense One-Hot Matrix Y</text>
    <text x="110" y="40" text-anchor="middle" font-size="9" font-family="monospace" fill="#6b7280">[ 0.0, 0.0, 1.0, 0.0, ... 0.0 ]</text>
    <text x="110" y="58" text-anchor="middle" font-size="8.5" fill="#6b7280">shape (B=128, C=32,000) = 16.38 MB</text>

    <text x="240" y="40" font-size="14" font-weight="700" fill="#1f2937">×</text>

    <!-- Log-Probs Matrix -->
    <rect x="260" y="0" width="180" height="70" rx="2" fill="#f8f9fa" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="350" y="25" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Log-Probabilities</text>
    <text x="350" y="45" text-anchor="middle" font-size="9" font-family="monospace" fill="#6b7280">log_softmax(logits)</text>
    <text x="350" y="60" text-anchor="middle" font-size="8.5" fill="#6b7280">shape (128, 32,000)</text>

    <!-- Waste callout -->
    <rect x="460" y="10" width="140" height="50" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1"/>
    <text x="530" y="28" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">99.997% Waste</text>
    <text x="530" y="45" text-anchor="middle" font-size="8.5" fill="#6b7280">Multiplying zeros</text>
  </g>

  <!-- Divider -->
  <line x1="40" y1="145" x2="640" y2="145" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4,4"/>

  <!-- Bottom Path: Systems Direct Index Gather (ACCENT) -->
  <text x="40" y="172" font-size="12" font-weight="700" fill="#1f2937">Production Systems Reality: Direct Index Pointer Gather</text>
  <text x="40" y="189" font-size="9.5" fill="#ff8246">log_probs[np.arange(B), target_idx] — zero matrix materialization</text>

  <g transform="translate(40, 202)">
    <!-- Integer ID Vector -->
    <rect x="0" y="0" width="160" height="70" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="80" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Target Indices Vector</text>
    <text x="80" y="42" text-anchor="middle" font-size="11" font-family="monospace" fill="#ff8246">[ 2, 841, 19, ... ]</text>
    <text x="80" y="58" text-anchor="middle" font-size="8.5" fill="#6b7280">shape (128,) int64 = 1 KB</text>

    <path d="M170 35 H220" stroke="#ff8246" stroke-width="1.5" marker-end="url(#arrow-orange)"/>

    <!-- Direct Byte Calculation Box -->
    <rect x="230" y="0" width="370" height="70" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
    <text x="245" y="22" font-size="10" font-weight="700" fill="#1f2937">Memory Controller Direct Offset Calculation:</text>
    <text x="245" y="42" font-size="10.5" font-family="monospace" fill="#1f2937">addr_b = base_ptr + (b * C + target[b]) * 4 bytes</text>
    <text x="245" y="60" font-size="9" fill="#ff8246">Fetches exactly 1 float32 per batch sample directly from memory bus</text>
  </g>

  <text x="340" y="315" text-anchor="middle" font-size="9.5" fill="#6b7280">Memory reduction: 16,384×. Avoids allocating multi-gigabyte one-hot arrays on the GPU during training.</text>
{FOOTER}"""
    write_svg("04_index-gather-memory.svg", body)


# -----------------------------------------------------------------------------
# 6. 05_dataloader-copies-pipeline.svg
# -----------------------------------------------------------------------------
def gen_05_dataloader_copies_pipeline():
    h = 350
    body = f"""{HEADER.format(height=h)}
  <!-- Top: The 3-Copy Synchronous CPU Path -->
  <text x="40" y="25" font-size="12" font-weight="700" fill="#1f2937">TinyTorch Synchronous DataLoader: The 3-Copy Collation Path</text>
  
  <g transform="translate(40, 45)">
    <!-- Dataset Storage -->
    <rect x="0" y="0" width="110" height="60" rx="2" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
    <text x="55" y="25" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Dataset Buffer</text>
    <text x="55" y="42" text-anchor="middle" font-size="8.5" fill="#6b7280">Raw samples in RAM</text>

    <path d="M110 30 H140" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>
    <text x="125" y="22" text-anchor="middle" font-size="8" fill="#6b7280">Copy 1</text>

    <!-- Slice Tensor -->
    <rect x="150" y="0" width="120" height="60" rx="2" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
    <text x="210" y="25" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Sample Slices</text>
    <text x="210" y="42" text-anchor="middle" font-size="8.5" fill="#6b7280">__getitem__() array</text>

    <path d="M270 30 H300" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>
    <text x="285" y="22" text-anchor="middle" font-size="8" fill="#6b7280">Copy 2</text>

    <!-- Stacked contiguous array -->
    <rect x="310" y="0" width="120" height="60" rx="2" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
    <text x="370" y="25" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">np.stack() Chunk</text>
    <text x="370" y="42" text-anchor="middle" font-size="8.5" fill="#6b7280">Contiguous batch RAM</text>

    <path d="M430 30 H460" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>
    <text x="445" y="22" text-anchor="middle" font-size="8" fill="#6b7280">Copy 3</text>

    <!-- Final Tensor Wrapper -->
    <rect x="470" y="0" width="130" height="60" rx="2" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
    <text x="535" y="25" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Batch Tensor</text>
    <text x="535" y="42" text-anchor="middle" font-size="8.5" fill="#6b7280">Tensor(batch_data)</text>
  </g>

  <!-- Divider -->
  <line x1="40" y1="135" x2="640" y2="135" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4,4"/>

  <!-- Bottom: Production Pinned DMA Pipeline (ACCENT) -->
  <text x="40" y="160" font-size="12" font-weight="700" fill="#1f2937">Production Pipeline: Asynchronous Double-Buffering &amp; Pinned DMA</text>

  <g transform="translate(40, 180)">
    <!-- Multi-worker CPU -->
    <rect x="0" y="0" width="130" height="90" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="65" y="25" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Worker Processes</text>
    <text x="65" y="45" text-anchor="middle" font-size="8.5" fill="#6b7280">Multi-worker IPC</text>
    <text x="65" y="60" text-anchor="middle" font-size="8.5" fill="#6b7280">Shared RAM (/dev/shm)</text>
    <text x="65" y="75" text-anchor="middle" font-size="8.5" fill="#6b7280">Parallel decoding</text>

    <path d="M130 45 H160" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

    <!-- Pinned Host Memory -->
    <rect x="170" y="0" width="180" height="90" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="260" y="25" text-anchor="middle" font-size="10" font-weight="700" fill="#ff8246">Pinned Host Memory</text>
    <text x="260" y="45" text-anchor="middle" font-size="8.5" fill="#1f2937">cudaHostAlloc() Page-Locked</text>
    <text x="260" y="60" text-anchor="middle" font-size="8.5" fill="#6b7280">OS forbidden from swapping pages</text>
    <text x="260" y="75" text-anchor="middle" font-size="8.5" fill="#1f2937">Ready for non-blocking DMA</text>

    <!-- PCIe Bus Arrow -->
    <path d="M350 45 H410" stroke="#ff8246" stroke-width="2" marker-end="url(#arrow-orange)"/>
    <text x="380" y="35" text-anchor="middle" font-size="8" font-weight="700" fill="#ff8246">PCIe DMA</text>

    <!-- GPU HBM -->
    <rect x="420" y="0" width="180" height="90" rx="2" fill="#f8f9fa" stroke="#1f2937" stroke-width="1.2"/>
    <text x="510" y="25" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">GPU VRAM / HBM</text>
    <text x="510" y="45" text-anchor="middle" font-size="8.5" fill="#6b7280">Overlapped Execution</text>
    <text x="510" y="60" text-anchor="middle" font-size="8.5" fill="#1f2937">GPU computes Batch N</text>
    <text x="510" y="75" text-anchor="middle" font-size="8.5" fill="#6b7280">while DMA streams Batch N+1</text>
  </g>

  <text x="340" y="325" text-anchor="middle" font-size="9.5" fill="#6b7280">Pageable host memory cannot be copied directly by GPU DMA; pinning enables zero-copy background streaming.</text>
{FOOTER}"""
    write_svg("05_dataloader-copies-pipeline.svg", body)


# -----------------------------------------------------------------------------
# 7. 06_autograd-pointer-dag.svg
# -----------------------------------------------------------------------------
def gen_06_autograd_pointer_dag():
    h = 380
    body = f"""{HEADER.format(height=h)}
  <!-- Graph Structure -->
  <text x="40" y="25" font-size="12" font-weight="700" fill="#1f2937">Dynamic Tape Pointer Graph &amp; Reverse Topological Traversal</text>
  <text x="40" y="42" font-size="9.5" fill="#6b7280">Evaluating: y = x · x; L = y + x for scalar x=3.0</text>

  <g transform="translate(40, 55)">
    <!-- Leaf Tensor x (ACCENT) -->
    <rect x="0" y="70" width="130" height="85" rx="3" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
    <text x="65" y="90" text-anchor="middle" font-size="10" font-weight="700" fill="#ff8246">Leaf Tensor x</text>
    <text x="65" y="106" text-anchor="middle" font-size="8.5" fill="#6b7280">id: 0x10</text>
    <text x="65" y="122" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">data: 3.0</text>
    <text x="65" y="138" text-anchor="middle" font-size="9.5" font-weight="700" fill="#ff8246">grad: 1.0 + 6.0 = 7.0</text>
    <text x="65" y="150" text-anchor="middle" font-size="8" fill="#6b7280">_grad_fn: None</text>

    <!-- Forward Arrows from x to MulNode -->
    <path d="M130 95 H180" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>
    <path d="M130 115 H180" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- MulBackward Node -->
    <rect x="180" y="75" width="120" height="65" rx="2" fill="#f4f5f7" stroke="#1f2937" stroke-width="1.2"/>
    <text x="240" y="95" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">MulBackward</text>
    <text x="240" y="110" text-anchor="middle" font-size="8.5" fill="#6b7280">inputs: (0x10, 0x10)</text>
    <text x="240" y="125" text-anchor="middle" font-size="8.5" fill="#6b7280">saved: x_val = 3.0</text>

    <!-- Arrow MulNode to y -->
    <path d="M300 107 H330" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- Intermediate Tensor y -->
    <rect x="330" y="75" width="110" height="65" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="385" y="95" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Tensor y</text>
    <text x="385" y="110" text-anchor="middle" font-size="9" fill="#1f2937">data: 9.0</text>
    <text x="385" y="125" text-anchor="middle" font-size="9" fill="#6b7280">grad: 1.0</text>

    <!-- Arrow y to AddNode -->
    <path d="M440 107 H470" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- Direct residual path from x to AddNode -->
    <path d="M65 70 V35 H490 V75" fill="none" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,2" marker-end="url(#arrow)"/>
    <text x="270" y="28" text-anchor="middle" font-size="8.5" fill="#6b7280">Direct connection to x (Branch 2)</text>

    <!-- AddBackward Node -->
    <rect x="470" y="75" width="120" height="65" rx="2" fill="#f4f5f7" stroke="#1f2937" stroke-width="1.2"/>
    <text x="530" y="95" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">AddBackward</text>
    <text x="530" y="110" text-anchor="middle" font-size="8.5" fill="#6b7280">inputs: (0x30, 0x10)</text>
    <text x="530" y="125" text-anchor="middle" font-size="8.5" fill="#6b7280">saved: []</text>

    <!-- Arrow AddNode to Root L -->
    <path d="M530 140 V165" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- Root Tensor L -->
    <rect x="470" y="165" width="120" height="55" rx="2" fill="#f8f9fa" stroke="#1f2937" stroke-width="1.2"/>
    <text x="530" y="185" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Root Tensor L</text>
    <text x="530" y="200" text-anchor="middle" font-size="9" fill="#1f2937">data: 12.0</text>
    <text x="530" y="213" text-anchor="middle" font-size="8.5" fill="#ff8246">seed: dL/dL = 1.0</text>
  </g>

  <!-- Backward Trace Timeline Box (Cleanly wrapped) -->
  <g transform="translate(40, 290)">
    <rect x="0" y="0" width="600" height="70" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
    <text x="15" y="17" font-size="9.5" font-weight="700" fill="#1f2937">Topological Backward Execution Log:</text>
    <text x="15" y="33" font-size="9" font-family="monospace" fill="#1f2937">1. AddBackward (seed = 1.0) → dL/dy = 1.0,  dL/dx (direct) = 1.0</text>
    <text x="15" y="48" font-size="9" font-family="monospace" fill="#1f2937">2. MulBackward (in_grad = 1.0) → dL/dx (via mul) = 3.0 + 3.0 = 6.0</text>
    <text x="15" y="63" font-size="9" font-family="monospace" fill="#ff8246">3. Accumulate into Leaf x: pending[0x10] = 1.0 + 6.0 = 7.0 ✓</text>
  </g>
{FOOTER}"""
    write_svg("06_autograd-pointer-dag.svg", body)


# -----------------------------------------------------------------------------
# 8. 07_ravine-optimization.svg
# -----------------------------------------------------------------------------
def gen_07_ravine_optimization():
    h = 360
    body = f"""{HEADER.format(height=h)}
  <!-- Left Panel: Ravine Contours & Trajectories -->
  <g transform="translate(40, 25)">
    <text x="140" y="15" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">Ravine Landscape: L(w₁, w₂) = 50w₁² + 0.5w₂²</text>
    <text x="140" y="32" text-anchor="middle" font-size="9.5" fill="#6b7280">Condition Number κ = 100: Steep in w₁, Flat in w₂</text>

    <!-- Coordinate Grid -->
    <rect x="0" y="45" width="280" height="230" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
    
    <!-- Elliptical Contours -->
    <ellipse cx="140" cy="160" rx="120" ry="25" fill="none" stroke="#d1d5db" stroke-width="1"/>
    <ellipse cx="140" cy="160" rx="80" ry="16" fill="none" stroke="#d1d5db" stroke-width="1"/>
    <ellipse cx="140" cy="160" rx="40" ry="8" fill="none" stroke="#d1d5db" stroke-width="1"/>
    <ellipse cx="140" cy="160" rx="15" ry="3" fill="none" stroke="#d1d5db" stroke-width="1"/>
    <circle cx="140" cy="160" r="3" fill="#1f2937"/>
    <text x="140" y="175" text-anchor="middle" font-size="8.5" fill="#1f2937">Minimum (0, 0)</text>

    <!-- Path 1: SGD Oscillations (Gray dashed) -->
    <path d="M30 65 L45 235 L70 95 L95 215 L115 125 L125 185 L135 150" fill="none" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="3,2"/>
    <text x="45" y="85" font-size="8.5" font-weight="700" fill="#9ca3af">SGD: Violent oscillation</text>

    <!-- Path 2: AdamW Momentum (ACCENT Flameorange) -->
    <path d="M30 65 Q60 145 137 159" fill="none" stroke="#ff8246" stroke-width="2" marker-end="url(#arrow-orange)"/>
    <text x="145" y="105" text-anchor="middle" font-size="8.5" font-weight="700" fill="#ff8246">AdamW: Smooth ravine descent</text>
  </g>

  <!-- Right Panel: The 16-Byte Parameter Rule -->
  <g transform="translate(360, 25)">
    <text x="140" y="15" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">The 16-Byte Optimizer Rule</text>
    <text x="140" y="32" text-anchor="middle" font-size="9.5" fill="#6b7280">Memory Breakdown per FP32 Model Parameter</text>

    <rect x="0" y="45" width="280" height="230" rx="3" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>

    <!-- Row 1: Parameter -->
    <rect x="20" y="65" width="70" height="35" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
    <text x="55" y="87" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Weight θ</text>
    <text x="110" y="85" font-size="10" font-weight="700" fill="#1f2937">4 bytes (Float32)</text>

    <!-- Row 2: Gradient -->
    <rect x="20" y="110" width="70" height="35" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
    <text x="55" y="132" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">Grad g</text>
    <text x="110" y="130" font-size="10" font-weight="700" fill="#1f2937">4 bytes (Float32)</text>

    <!-- Row 3: 1st Moment m -->
    <rect x="20" y="155" width="70" height="35" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
    <text x="55" y="177" text-anchor="middle" font-size="10" font-weight="700" fill="#ff8246">Moment m</text>
    <text x="110" y="175" font-size="10" font-weight="700" fill="#1f2937">4 bytes (First moment)</text>

    <!-- Row 4: 2nd Moment v -->
    <rect x="20" y="200" width="70" height="35" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
    <text x="55" y="222" text-anchor="middle" font-size="10" font-weight="700" fill="#ff8246">Moment v</text>
    <text x="110" y="220" font-size="10" font-weight="700" fill="#1f2937">4 bytes (Second moment)</text>

    <!-- Total Callout -->
    <line x1="20" y1="245" x2="260" y2="245" stroke="#ff8246" stroke-width="1"/>
    <text x="140" y="263" text-anchor="middle" font-size="11" font-weight="700" fill="#ff8246">Total = 16 Bytes per Parameter</text>
  </g>

  <text x="340" y="325" text-anchor="middle" font-size="9.5" fill="#1f2937">AdamW normalizes gradient steps by √v + ε, suppressing oscillating steep directions while accelerating along flat floors.</text>
  <text x="340" y="342" text-anchor="middle" font-size="9.5" fill="#6b7280">Systems consequence: Training a 7B parameter model requires 7B × 16 bytes = 112 GB VRAM for optimizer state alone.</text>
{FOOTER}"""
    write_svg("07_ravine-optimization.svg", body)


# -----------------------------------------------------------------------------
# 9. 08_microbatch-memory-timeline.svg
# -----------------------------------------------------------------------------
def gen_08_microbatch_memory_timeline():
    h = 350
    body = f"""{HEADER.format(height=h)}
  <text x="340" y="25" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937">Micro-Batching: Effective Batch Size Without Memory Explosion</text>

  <g transform="translate(50, 45)">
    <!-- Axes -->
    <line x1="40" y1="230" x2="570" y2="230" stroke="#1f2937" stroke-width="1.2" marker-end="url(#arrow)"/>
    <line x1="40" y1="230" x2="40" y2="15" stroke="#1f2937" stroke-width="1.2" marker-end="url(#arrow)"/>
    <text x="560" y="245" font-size="9.5" fill="#1f2937">Step Time (t)</text>
    <text x="25" y="15" font-size="9.5" fill="#1f2937">VRAM</text>

    <!-- OOM Limit Line -->
    <line x1="40" y1="60" x2="550" y2="60" stroke="#1f2937" stroke-width="1.2" stroke-dasharray="4,4"/>
    <text x="545" y="52" text-anchor="end" font-size="9" font-weight="700" fill="#1f2937">GPU Hardware VRAM Ceiling (16 GB)</text>

    <!-- Path A: Full Batch B=256 Spike (OOM) -->
    <path d="M40 230 Q80 20 120 20 Q160 20 200 230" fill="none" stroke="#9ca3af" stroke-width="1.8" stroke-dasharray="4,2"/>
    <rect x="90" y="15" width="60" height="20" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1"/>
    <text x="120" y="28" text-anchor="middle" font-size="8.5" font-weight="700" fill="#1f2937">OOM! (32 GB)</text>
    <text x="180" y="90" font-size="9" fill="#9ca3af">Single Batch B=256</text>

    <!-- Path B: Micro-Batching 4x64 (ACCENT) -->
    <!-- Micro-batch 1 -->
    <path d="M40 230 Q70 120 100 120 Q130 120 160 215" fill="none" stroke="#ff8246" stroke-width="2"/>
    <text x="100" y="112" text-anchor="middle" font-size="8" font-weight="700" fill="#ff8246">μB 1 (4 GB)</text>

    <!-- Micro-batch 2 -->
    <path d="M160 215 Q190 115 220 115 Q250 115 280 210" fill="none" stroke="#ff8246" stroke-width="2"/>
    <text x="220" y="107" text-anchor="middle" font-size="8" font-weight="700" fill="#ff8246">μB 2</text>

    <!-- Micro-batch 3 -->
    <path d="M280 210 Q310 110 340 110 Q370 110 400 205" fill="none" stroke="#ff8246" stroke-width="2"/>
    <text x="340" y="102" text-anchor="middle" font-size="8" font-weight="700" fill="#ff8246">μB 3</text>

    <!-- Micro-batch 4 -->
    <path d="M400 205 Q430 105 460 105 Q490 105 520 200" fill="none" stroke="#ff8246" stroke-width="2"/>
    <text x="460" y="97" text-anchor="middle" font-size="8" font-weight="700" fill="#ff8246">μB 4</text>

    <!-- Accumulator baseline drift -->
    <line x1="40" y1="230" x2="520" y2="200" stroke="#ff8246" stroke-width="1.2" stroke-dasharray="2,2"/>
    <text x="210" y="210" font-size="8.5" fill="#ff8246">Accumulated .grad buffers</text>

    <!-- Optimizer Step -->
    <path d="M520 200 V225" stroke="#1f2937" stroke-width="2" marker-end="url(#arrow)"/>
    <text x="520" y="190" text-anchor="middle" font-size="8.5" font-weight="700" fill="#1f2937">opt.step()</text>
    <text x="520" y="245" text-anchor="middle" font-size="8" fill="#6b7280">zero_grad()</text>
  </g>

  <text x="340" y="310" text-anchor="middle" font-size="9.5" fill="#1f2937">Intermediate activations are freed immediately after backward pass of each micro-batch.</text>
  <text x="340" y="327" text-anchor="middle" font-size="9.5" fill="#6b7280">Only parameter gradients accumulate: effective batch size 256 achieved in 4 GB memory budget.</text>
{FOOTER}"""
    write_svg("08_microbatch-memory-timeline.svg", body)


# -----------------------------------------------------------------------------
# 10. milestone_01_kernel-trace.svg
# -----------------------------------------------------------------------------
def gen_milestone_01_kernel_trace():
    h = 390
    body = f"""{HEADER.format(height=h)}
  <text x="340" y="25" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937">Milestone I: The Complete End-to-End System Kernel Round-Trip</text>
  <text x="340" y="42" text-anchor="middle" font-size="9.5" fill="#6b7280">Tracing memory mutation and gradient flow through the 8 foundational modules</text>

  <!-- Forward Pipeline (Top Row, Left to Right) -->
  <g transform="translate(40, 60)">
    <!-- Node 1: Input -->
    <rect x="0" y="0" width="100" height="55" rx="2" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
    <text x="50" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">1. Batch X</text>
    <text x="50" y="38" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#6b7280">[4, 2] float32</text>

    <path d="M100 27 H125" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- Node 2: Linear 1 -->
    <rect x="125" y="0" width="100" height="55" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="175" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">2. Linear 1</text>
    <text x="175" y="38" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#6b7280">X @ W₁ + b₁</text>

    <path d="M225 27 H250" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- Node 3: ReLU -->
    <rect x="250" y="0" width="100" height="55" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="300" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">3. ReLU</text>
    <text x="300" y="38" text-anchor="middle" font-size="8.5" fill="#6b7280">Save mask</text>

    <path d="M350 27 H375" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- Node 4: Linear 2 -->
    <rect x="375" y="0" width="100" height="55" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="425" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">4. Linear 2</text>
    <text x="425" y="38" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#6b7280">H @ W₂ + b₂</text>

    <path d="M475 27 H500" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- Node 5: Sigmoid & Loss -->
    <rect x="500" y="0" width="100" height="55" rx="2" fill="#f8f9fa" stroke="#1f2937" stroke-width="1.2"/>
    <text x="550" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">5. BCE Loss</text>
    <text x="550" y="38" text-anchor="middle" font-size="8.5" fill="#6b7280">Scalar L = 0.69</text>
  </g>

  <!-- Backward Loop Arrow Down & Left -->
  <text x="590" y="132" text-anchor="middle" font-size="9" font-weight="700" fill="#ff8246">loss.backward()</text>
  <path d="M590 140 V175 H540" fill="none" stroke="#ff8246" stroke-width="1.5" marker-end="url(#arrow-orange)"/>

  <!-- Backward Pipeline (Bottom Row, Right to Left: Greyscale Standard Nodes) -->
  <g transform="translate(40, 185)">
    <!-- Node 6: dL/dSigmoid -->
    <rect x="440" y="0" width="100" height="60" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="490" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">6. Sigmoid Grad</text>
    <text x="490" y="38" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#1f2937">p - y</text>
    <text x="490" y="50" text-anchor="middle" font-size="8" fill="#6b7280">Seed = 1.0</text>

    <path d="M440 30 H415" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- Node 7: Linear 2 Grad -->
    <rect x="315" y="0" width="100" height="60" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="365" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">7. W₂, b₂ Grad</text>
    <text x="365" y="38" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#1f2937">H.T @ grad</text>
    <text x="365" y="50" text-anchor="middle" font-size="8" fill="#6b7280">Accumulate .grad</text>

    <path d="M315 30 H290" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- Node 8: ReLU Backward -->
    <rect x="190" y="0" width="100" height="60" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="240" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">8. Mask Filter</text>
    <text x="240" y="38" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#1f2937">grad * (z > 0)</text>
    <text x="240" y="50" text-anchor="middle" font-size="8" fill="#6b7280">Apply saved tape</text>

    <path d="M190 30 H165" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>

    <!-- Node 9: Linear 1 Grad -->
    <rect x="65" y="0" width="100" height="60" rx="2" fill="#f4f5f7" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="115" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">9. W₁, b₁ Grad</text>
    <text x="115" y="38" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#1f2937">X.T @ grad</text>
    <text x="115" y="50" text-anchor="middle" font-size="8" fill="#6b7280">Leaf parameters</text>
  </g>

  <!-- Step & Zero Loop (ACCENT: The State Mutation) -->
  <g transform="translate(40, 275)">
    <rect x="65" y="0" width="475" height="52" rx="3" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
    <text x="300" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#ff8246">10. In-Place State Mutation: opt.step() &amp; opt.zero_grad()</text>
    <text x="300" y="38" text-anchor="middle" font-size="9" fill="#1f2937">param.data -= lr * param.grad (mutates C-buffer in place) → sever _grad_fn tape</text>
  </g>

  <text x="340" y="355" text-anchor="middle" font-size="9.5" fill="#6b7280">The xv6 lifecycle of machine learning: allocate buffer → record DAG → contract chain rule → mutate in place → release graph.</text>
{FOOTER}"""
    write_svg("milestone_01_kernel-trace.svg", body)


def main():
    gen_00_physical_stack()
    gen_01_stride_zero_broadcast()
    gen_02_space_folding_relu()
    gen_03_variance_waterfall()
    gen_04_index_gather_memory()
    gen_05_dataloader_copies_pipeline()
    gen_06_autograd_pointer_dag()
    gen_07_ravine_optimization()
    gen_08_microbatch_memory_timeline()
    gen_milestone_01_kernel_trace()
    print("Tier 1 generation complete.")


if __name__ == "__main__":
    main()
