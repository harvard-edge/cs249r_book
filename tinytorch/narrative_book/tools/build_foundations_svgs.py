#!/usr/bin/env python3
"""
Generate Tier 1 (Foundations) Mechanical SVG Diagrams for TinyTorch Narrative Book.
Strict compliance with Vol 3 Figure 1.10 standard & tinytorch/palette.md:
- 680 width viewBox
- Pure white background (#ffffff), NO outer frame border stroke
- NO embedded canvas titles or subtitles (Quarto fig-cap owns the caption)
- Subsystem container cards with 24px header bands (rx="2")
- Palette: #ffffff, #f8fafc, #fff1e8, #f1f5f9, #9ca3af, #cbd5e1, #ff8246, #c85a17, #1f2937, #6b7280
- Max 1 accent node per diagram
- Fonts: TeX Gyre Heros, Helvetica Neue, Arial, sans-serif
- Uniform stroke weights, crisp geometry, tactile systems intuition.
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


# -----------------------------------------------------------------------------
# 1. 00_physical-stack.svg
# -----------------------------------------------------------------------------
def gen_00_physical_stack():
    h = 360
    body = f"""{HEADER.format(height=h)}
  <!-- Layer 1: User Application Code -->
  <rect x="25" y="20" width="630" height="50" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="140" height="50" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="95" y="50" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">1. USER SCRIPT</text>
  <text x="180" y="42" font-size="9" font-family="monospace" font-weight="bold" fill="#1f2937">y = x @ W + b; loss = criterion(y, target); loss.backward()</text>
  <text x="180" y="58" font-size="8.5" fill="#6b7280">Python high-level operations declaring models, losses, and training steps</text>

  <path d="M340 70 V82" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Layer 2: Tensor Object Model -->
  <rect x="25" y="82" width="630" height="50" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="82" width="140" height="50" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="95" y="112" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">2. TENSOR OBJECT</text>
  <text x="180" y="104" font-size="9" font-family="monospace" font-weight="bold" fill="#1f2937">Tensor(data, shape=(B, D), strides=(D, 1), requires_grad=True)</text>
  <text x="180" y="120" font-size="8.5" fill="#6b7280">Metadata container holding shape, byte strides, and gradient accumulator references</text>

  <path d="M340 132 V144" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

  <!-- Layer 3: Dynamic Autograd Tape (ACCENT) -->
  <rect x="25" y="144" width="630" height="52" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="25" y="144" width="140" height="52" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="95" y="175" text-anchor="middle" font-size="9.5" font-weight="700" fill="#c85a17">3. AUTOGRAD TAPE</text>
  <text x="180" y="166" font-size="9" font-family="monospace" font-weight="bold" fill="#1f2937">Directed Acyclic Graph (DAG) of Function closures</text>
  <text x="180" y="182" font-size="8.5" fill="#6b7280">Records backward closures, saved tensors, and topological reverse-execution order</text>

  <path d="M340 196 V208" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

  <!-- Layer 4: C-Contiguous DRAM Memory Allocator -->
  <rect x="25" y="208" width="630" height="50" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="208" width="140" height="50" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="95" y="238" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">4. MEMORY STORAGE</text>
  <text x="180" y="230" font-size="9" font-family="monospace" fill="#1f2937">np.ndarray: float32 flat buffer, pointer offsets, byte strides</text>
  <text x="180" y="246" font-size="8.5" fill="#6b7280">Contiguous physical DRAM buffer; stride arithmetic eliminates data replication</text>

  <path d="M340 258 V270" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Layer 5: Hardware Execution & PyTorch Bridge -->
  <rect x="25" y="270" width="630" height="50" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="270" width="140" height="50" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="95" y="300" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">5. HARDWARE EXEC</text>
  <text x="180" y="292" font-size="9" font-family="monospace" fill="#1f2937">CPU BLAS GEMM / SIMD AVX2 / CUDA Kernels (Production Bridge)</text>
  <text x="180" y="308" font-size="8.5" fill="#6b7280">Hardware execution units where numerical verification matches PyTorch bit-for-bit</text>

  <!-- Bottom Annotation -->
  <text x="340" y="344" text-anchor="middle" font-size="9" fill="#6b7280">The TinyTorch execution stack lowers Python user scripts into strided memory buffers and topological autograd tapes.</text>
{FOOTER}"""
    write_svg("00_physical-stack.svg", body)


# -----------------------------------------------------------------------------
# 2. 01_stride-zero-broadcast.svg
# -----------------------------------------------------------------------------
def gen_01_stride_zero_broadcast():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Section 1: Physical Buffer in Memory -->
  <rect x="25" y="20" width="630" height="95" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="630" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. PHYSICAL 1D BUFFER IN MEMORY (SHAPE [3], 12 BYTES)</text>
  <text x="440" y="36" font-size="8.5" font-family="monospace" fill="#6b7280">float32 = 4 bytes/element</text>

  <!-- Buffer Cells -->
  <g transform="translate(45, 54)">
    <!-- Cell 0 -->
    <rect x="0" y="10" width="85" height="40" rx="0" fill="#f8fafc" stroke="#1f2937" stroke-width="1.2"/>
    <text x="42" y="35" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">10.0</text>
    <text x="42" y="5" text-anchor="middle" font-size="9" font-family="monospace" fill="#6b7280">offset 0 (0x00)</text>

    <!-- Cell 1 -->
    <rect x="85" y="10" width="85" height="40" rx="0" fill="#f8fafc" stroke="#1f2937" stroke-width="1.2"/>
    <text x="127" y="35" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">20.0</text>
    <text x="127" y="5" text-anchor="middle" font-size="9" font-family="monospace" fill="#6b7280">offset 1 (0x04)</text>

    <!-- Cell 2 -->
    <rect x="170" y="10" width="85" height="40" rx="0" fill="#f8fafc" stroke="#1f2937" stroke-width="1.2"/>
    <text x="212" y="35" text-anchor="middle" font-size="12" font-weight="700" fill="#1f2937">30.0</text>
    <text x="212" y="5" text-anchor="middle" font-size="9" font-family="monospace" fill="#6b7280">offset 2 (0x08)</text>

    <text x="280" y="35" font-size="9" fill="#6b7280">Contiguous physical allocation: exactly 3 floats stored in RAM</text>
  </g>

  <!-- Section 2: Virtual 2D Broadcast View (ACCENT) -->
  <rect x="25" y="128" width="280" height="140" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="25" y="128" width="280" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="35" y="144" font-size="9.5" font-weight="700" fill="#c85a17">2. BROADCAST VIEW (SHAPE 2x3)</text>
  <text x="220" y="144" font-size="8.5" font-family="monospace" font-weight="bold" fill="#c85a17">strides=(0, 1)</text>

  <!-- Matrix Grid -->
  <g transform="translate(45, 162)">
    <!-- Row 0 -->
    <rect x="0" y="0" width="75" height="40" rx="0" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="37" y="20" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1f2937">10.0</text>
    <text x="37" y="32" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">[0,0]→slot 0</text>

    <rect x="75" y="0" width="75" height="40" rx="0" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="112" y="20" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1f2937">20.0</text>
    <text x="112" y="32" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">[0,1]→slot 1</text>

    <rect x="150" y="0" width="75" height="40" rx="0" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="187" y="20" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1f2937">30.0</text>
    <text x="187" y="32" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">[0,2]→slot 2</text>

    <!-- Row 1 (ACCENT: Row Stride 0) -->
    <rect x="0" y="46" width="75" height="42" rx="0" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="37" y="66" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1f2937">10.0</text>
    <text x="37" y="80" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">[1,0]→slot 0</text>

    <rect x="75" y="46" width="75" height="42" rx="0" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="112" y="66" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1f2937">20.0</text>
    <text x="112" y="80" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">[1,1]→slot 1</text>

    <rect x="150" y="46" width="75" height="42" rx="0" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="187" y="66" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1f2937">30.0</text>
    <text x="187" y="80" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">[1,2]→slot 2</text>
  </g>

  <!-- Explanation Panel on Right -->
  <rect x="320" y="128" width="335" height="140" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="320" y="128" width="335" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="330" y="144" font-size="9.5" font-weight="700" fill="#1f2937">3. THE STRIDE-0 OFFSET INVARIANT</text>

  <g transform="translate(330, 160)">
    <text x="0" y="15" font-size="9" font-family="monospace" fill="#1f2937">offset = (row * stride[0]) + (col * stride[1])</text>
    <text x="0" y="32" font-size="9.5" font-family="monospace" font-weight="bold" fill="#ff8246">offset = (1 * 0) + (1 * 1) = 1 (slot 1)</text>
    <text x="0" y="55" font-size="8.5" fill="#1f2937">• Row stride = 0: row index multiplies by zero.</text>
    <text x="0" y="70" font-size="8.5" fill="#1f2937">• Read pointer never advances when stepping rows.</text>
    <text x="0" y="86" font-size="8.5" font-weight="bold" fill="#ff8246">• Zero copy: 6 virtual elements mapped in 3 physical slots.</text>
  </g>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Stride-0 broadcasting enables virtual multi-dimensional expansion without allocating or duplicating physical memory.</text>
{FOOTER}"""
    write_svg("01_stride-zero-broadcast.svg", body)


# -----------------------------------------------------------------------------
# 3. 02_space-folding-relu.svg
# -----------------------------------------------------------------------------
def gen_02_space_folding_relu():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Left Panel: Input Space (Non-separable) -->
  <rect x="25" y="20" width="300" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="300" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. INPUT SPACE (LINEARLY INSEPARABLE XOR)</text>

  <!-- Coordinate Box -->
  <g transform="translate(70, 58)">
    <rect x="0" y="0" width="180" height="150" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <!-- Axes -->
    <line x1="20" y1="130" x2="165" y2="130" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>
    <line x1="20" y1="130" x2="20" y2="15" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>
    <text x="160" y="142" font-size="8.5" font-family="monospace" fill="#6b7280">x₁</text>
    <text x="10" y="15" font-size="8.5" font-family="monospace" fill="#6b7280">x₂</text>

    <!-- Failed linear hyperplane -->
    <line x1="30" y1="25" x2="150" y2="135" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,2"/>
    <text x="130" y="55" font-size="7.5" fill="#9ca3af" transform="rotate(42, 130, 55)">Linear failure</text>

    <!-- XOR Data Points -->
    <circle cx="35" cy="115" r="4.5" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
    <text x="45" y="125" font-size="7.5" font-family="monospace" fill="#1f2937">(0,0) [0]</text>

    <circle cx="135" cy="115" r="4.5" fill="#ff8246" stroke="#1f2937" stroke-width="1.5"/>
    <text x="135" y="105" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">(1,0) [1]</text>

    <circle cx="35" cy="35" r="4.5" fill="#ff8246" stroke="#1f2937" stroke-width="1.5"/>
    <text x="46" y="38" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">(0,1) [1]</text>

    <circle cx="135" cy="35" r="4.5" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
    <text x="145" y="45" font-size="7.5" font-family="monospace" fill="#1f2937">(1,1) [0]</text>
  </g>

  <text x="35" y="232" font-size="8.5" fill="#6b7280">No single 2D hyperplane can isolate</text>
  <text x="35" y="248" font-size="8.5" fill="#6b7280">the [1] labels from the [0] labels.</text>

  <!-- Arrow: ReLU Hinge Transformation -->
  <line x1="325" y1="147" x2="345" y2="147" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>
  <text x="335" y="138" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">ReLU</text>

  <!-- Right Panel: Folded Feature Space (ACCENT) -->
  <rect x="345" y="20" width="310" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="345" y="20" width="310" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="357" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. FOLDED FEATURE SPACE (RELU ACTIVATION)</text>

  <!-- Folded Coordinate Box -->
  <g transform="translate(390, 58)">
    <rect x="0" y="0" width="180" height="150" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
    <!-- Axes -->
    <line x1="25" y1="130" x2="165" y2="130" stroke="#1f2937" stroke-width="1.2" marker-end="url(#arrow)"/>
    <line x1="25" y1="130" x2="25" y2="15" stroke="#1f2937" stroke-width="1.2" marker-end="url(#arrow)"/>
    <text x="160" y="142" font-size="8.5" font-family="monospace" fill="#1f2937">h₁</text>
    <text x="12" y="15" font-size="8.5" font-family="monospace" fill="#1f2937">h₂</text>

    <!-- Successful Linear Separator -->
    <line x1="25" y1="35" x2="155" y2="125" stroke="#ff8246" stroke-width="2"/>
    <text x="65" y="28" font-size="8" font-weight="bold" fill="#ff8246">Hyperplane: h₁ - 2h₂ = 0.5</text>

    <!-- Transformed Points -->
    <circle cx="25" cy="130" r="4.5" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
    <text x="35" y="142" font-size="7.5" font-family="monospace" fill="#1f2937">(0,0) → [0, 0]</text>

    <!-- Collapsed (0,1) and (1,0) -> (1, 0) -->
    <circle cx="105" cy="130" r="5.5" fill="#ff8246" stroke="#1f2937" stroke-width="1.5"/>
    <text x="105" y="118" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">(0,1) &amp; (1,0) Folded to [1, 0]</text>

    <!-- (1,1) -> (2, 1) -->
    <circle cx="150" cy="70" r="4.5" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>
    <text x="150" y="88" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#1f2937">(1,1) → [2, 1]</text>
  </g>

  <text x="357" y="232" font-size="8.5" font-weight="bold" fill="#1f2937">Space-Folding Invariant:</text>
  <text x="357" y="248" font-size="8.5" fill="#6b7280">ReLU clamps negative coordinates to 0, folding the input manifold</text>
  <text x="357" y="262" font-size="8.5" fill="#6b7280">so XOR classes become cleanly linearly separable.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Without non-linear activations, deep networks collapse into a single affine transformation; ReLU acts as a mechanical hinge that folds space.</text>
{FOOTER}"""
    write_svg("02_space-folding-relu.svg", body)


# -----------------------------------------------------------------------------
# 4. 03_variance-waterfall.svg
# -----------------------------------------------------------------------------
def gen_03_variance_waterfall():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Column 1: Unscaled Initialization -->
  <rect x="25" y="20" width="195" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="195" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. UNSCALED: Var(W)=1.0</text>

  <g transform="translate(35, 60)">
    <text x="10" y="15" font-size="9" font-weight="700" fill="#1f2937">Input (L=0):</text>
    <text x="110" y="15" font-size="9" font-family="monospace" fill="#1f2937">σ² = 1.0</text>

    <text x="10" y="45" font-size="9" font-weight="700" fill="#1f2937">Layer 5:</text>
    <text x="110" y="45" font-size="9" font-family="monospace" fill="#1f2937">σ² ≈ 10³</text>

    <text x="10" y="75" font-size="9" font-weight="700" fill="#1f2937">Layer 10:</text>
    <text x="110" y="75" font-size="9" font-family="monospace" fill="#1f2937">σ² ≈ 10²⁰</text>

    <text x="10" y="105" font-size="9" font-weight="700" fill="#1f2937">Layer 15:</text>
    <text x="110" y="105" font-size="9" font-family="monospace" fill="#1f2937">σ² ≈ 10⁴⁰</text>

    <rect x="10" y="130" width="155" height="26" rx="1" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
    <text x="87" y="147" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#1f2937">Explosion → NaN Overflow</text>
  </g>

  <!-- Column 2: Small Initialization -->
  <rect x="240" y="20" width="195" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="240" y="20" width="195" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="250" y="36" font-size="9.5" font-weight="700" fill="#1f2937">2. UNDERSCALED: Var=10⁻⁴</text>

  <g transform="translate(250, 60)">
    <text x="10" y="15" font-size="9" font-weight="700" fill="#1f2937">Input (L=0):</text>
    <text x="110" y="15" font-size="9" font-family="monospace" fill="#1f2937">σ² = 1.0</text>

    <text x="10" y="45" font-size="9" font-weight="700" fill="#1f2937">Layer 5:</text>
    <text x="110" y="45" font-size="9" font-family="monospace" fill="#1f2937">σ² ≈ 10⁻⁴</text>

    <text x="10" y="75" font-size="9" font-weight="700" fill="#1f2937">Layer 10:</text>
    <text x="110" y="75" font-size="9" font-family="monospace" fill="#1f2937">σ² ≈ 10⁻¹²</text>

    <text x="10" y="105" font-size="9" font-weight="700" fill="#1f2937">Layer 15:</text>
    <text x="110" y="105" font-size="9" font-family="monospace" fill="#1f2937">σ² → 0.0</text>

    <rect x="10" y="130" width="155" height="26" rx="1" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
    <text x="87" y="147" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#1f2937">Collapse → Vanishing Grad</text>
  </g>

  <!-- Column 3: Kaiming He (ACCENT) -->
  <rect x="455" y="20" width="200" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="455" y="20" width="200" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="465" y="36" font-size="9.5" font-weight="700" fill="#c85a17">3. KAIMING: Var = 2 / D_in</text>

  <g transform="translate(465, 60)">
    <text x="10" y="15" font-size="9" font-weight="700" fill="#1f2937">Input (L=0):</text>
    <text x="110" y="15" font-size="9" font-family="monospace" font-weight="bold" fill="#ff8246">σ² = 1.00</text>

    <text x="10" y="45" font-size="9" font-weight="700" fill="#1f2937">Layer 5:</text>
    <text x="110" y="45" font-size="9" font-family="monospace" fill="#1f2937">σ² ≈ 1.01</text>

    <text x="10" y="75" font-size="9" font-weight="700" fill="#1f2937">Layer 10:</text>
    <text x="110" y="75" font-size="9" font-family="monospace" fill="#1f2937">σ² ≈ 0.99</text>

    <text x="10" y="105" font-size="9" font-weight="700" fill="#1f2937">Layer 15:</text>
    <text x="110" y="105" font-size="9" font-family="monospace" fill="#1f2937">σ² ≈ 1.02</text>

    <rect x="10" y="130" width="160" height="26" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="90" y="147" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#ff8246">Invariant Signal Preserved</text>
  </g>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Under ReLU, half the pre-activations are zeroed out; scaling variance by 2/D_in preserves unit signal variance across arbitrary network depth.</text>
{FOOTER}"""
    write_svg("03_variance-waterfall.svg", body)


# -----------------------------------------------------------------------------
# 5. 04_index-gather-memory.svg
# -----------------------------------------------------------------------------
def gen_04_index_gather_memory():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Top Path: Textbook One-Hot -->
  <rect x="25" y="20" width="630" height="115" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="630" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. TEXTBOOK FORMULATION: ONE-HOT MULTIPLICATION (16.38 MB)</text>
  <text x="440" y="36" font-size="8.5" font-family="monospace" fill="#6b7280">L = - Σ y_c log(p_c)</text>

  <g transform="translate(40, 52)">
    <!-- Big One-Hot Matrix -->
    <rect x="0" y="0" width="220" height="68" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="110" y="20" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">Dense One-Hot Matrix Y</text>
    <text x="110" y="38" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#6b7280">[ 0.0, 0.0, 1.0, 0.0, ... 0.0 ]</text>
    <text x="110" y="54" text-anchor="middle" font-size="8" fill="#6b7280">shape (B=128, C=32,000) = 16.38 MB</text>

    <text x="235" y="38" font-size="14" font-weight="700" fill="#1f2937">×</text>

    <!-- Log-Probs Matrix -->
    <rect x="255" y="0" width="180" height="68" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="345" y="20" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">Log-Probabilities</text>
    <text x="345" y="38" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#6b7280">log_softmax(logits)</text>
    <text x="345" y="54" text-anchor="middle" font-size="8" fill="#6b7280">shape (128, 32,000)</text>

    <!-- Waste callout -->
    <rect x="450" y="10" width="145" height="48" rx="1" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
    <text x="522" y="28" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">99.997% Waste</text>
    <text x="522" y="44" text-anchor="middle" font-size="8" fill="#6b7280">Multiplying zeros in DRAM</text>
  </g>

  <!-- Bottom Path: Systems Direct Index Gather (ACCENT) -->
  <rect x="25" y="150" width="630" height="125" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="25" y="150" width="630" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="35" y="166" font-size="9.5" font-weight="700" fill="#c85a17">2. PRODUCTION SYSTEMS REALITY: DIRECT INDEX GATHER (1 KB)</text>
  <text x="440" y="166" font-size="8.5" font-family="monospace" font-weight="bold" fill="#c85a17">16,384x memory reduction</text>

  <g transform="translate(40, 185)">
    <!-- Integer ID Vector -->
    <rect x="0" y="0" width="165" height="74" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="82" y="22" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">Target Indices Vector</text>
    <text x="82" y="44" text-anchor="middle" font-size="11" font-family="monospace" font-weight="bold" fill="#ff8246">[ 2, 841, 19, ... ]</text>
    <text x="82" y="62" text-anchor="middle" font-size="8" fill="#6b7280">shape (128,) int64 = 1 KB</text>

    <path d="M175 37 H220" stroke="#ff8246" stroke-width="1.5" marker-end="url(#arrow-orange)"/>

    <!-- Direct Byte Calculation Box -->
    <rect x="230" y="0" width="370" height="74" rx="1" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
    <text x="245" y="24" font-size="9.5" font-weight="700" fill="#1f2937">Memory Controller Direct Offset Calculation:</text>
    <text x="245" y="44" font-size="10" font-family="monospace" fill="#1f2937">addr_b = base_ptr + (b * C + target[b]) * 4 bytes</text>
    <text x="245" y="62" font-size="8.5" fill="#ff8246">Fetches exactly 1 float32 per batch sample directly from memory bus</text>
  </g>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Production cross-entropy bypasses multi-gigabyte one-hot matrix materialization by indexing directly into predicted log-probabilities.</text>
{FOOTER}"""
    write_svg("04_index-gather-memory.svg", body)


# -----------------------------------------------------------------------------
# 6. 05_dataloader-copies-pipeline.svg
# -----------------------------------------------------------------------------
def gen_05_dataloader_copies_pipeline():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Top: The 3-Copy Synchronous CPU Path -->
  <rect x="25" y="20" width="630" height="115" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="630" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. SYNCHRONOUS CPU PIPELINE (3 HOST COPIES)</text>
  <text x="440" y="36" font-size="8.5" fill="#6b7280">Blocks training loop during I/O</text>

  <g transform="translate(40, 56)">
    <!-- Dataset Storage -->
    <rect x="0" y="0" width="115" height="58" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="57" y="24" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">Dataset Buffer</text>
    <text x="57" y="42" text-anchor="middle" font-size="8" fill="#6b7280">Raw samples in RAM</text>

    <path d="M115 29 H145" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>
    <text x="130" y="20" text-anchor="middle" font-size="7.5" fill="#6b7280">Copy 1</text>

    <!-- Slice Tensor -->
    <rect x="150" y="0" width="125" height="58" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="212" y="24" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">Sample Slices</text>
    <text x="212" y="42" text-anchor="middle" font-size="8" fill="#6b7280">__getitem__() array</text>

    <path d="M275 29 H305" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>
    <text x="290" y="20" text-anchor="middle" font-size="7.5" fill="#6b7280">Copy 2</text>

    <!-- Stacked contiguous array -->
    <rect x="310" y="0" width="125" height="58" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="372" y="24" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">np.stack() Chunk</text>
    <text x="372" y="42" text-anchor="middle" font-size="8" fill="#6b7280">Contiguous batch RAM</text>

    <path d="M435 29 H465" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>
    <text x="450" y="20" text-anchor="middle" font-size="7.5" fill="#6b7280">Copy 3</text>

    <!-- Final Tensor Wrapper -->
    <rect x="470" y="0" width="130" height="58" rx="1" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
    <text x="535" y="24" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">Batch Tensor</text>
    <text x="535" y="42" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">Tensor(batch)</text>
  </g>

  <!-- Bottom: Production Pinned DMA Pipeline (ACCENT) -->
  <rect x="25" y="150" width="630" height="125" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="25" y="150" width="630" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="35" y="166" font-size="9.5" font-weight="700" fill="#c85a17">2. ASYNCHRONOUS PIPELINE (PINNED DMA &amp; DOUBLE-BUFFERING)</text>

  <g transform="translate(40, 185)">
    <!-- Multi-worker CPU -->
    <rect x="0" y="0" width="140" height="74" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="70" y="22" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">Worker Processes</text>
    <text x="70" y="40" text-anchor="middle" font-size="8" fill="#6b7280">Multi-worker IPC</text>
    <text x="70" y="54" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">/dev/shm shared RAM</text>

    <path d="M140 37 H170" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

    <!-- Pinned Host Memory -->
    <rect x="175" y="0" width="190" height="74" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="270" y="22" text-anchor="middle" font-size="9.5" font-weight="700" fill="#c85a17">Pinned Host Memory</text>
    <text x="270" y="40" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">cudaHostAlloc() Page-Locked</text>
    <text x="270" y="56" text-anchor="middle" font-size="8" fill="#6b7280">Non-blocking background DMA</text>

    <!-- PCIe Bus Arrow -->
    <path d="M365 37 H405" stroke="#ff8246" stroke-width="1.5" marker-end="url(#arrow-orange)"/>
    <text x="385" y="28" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">PCIe</text>

    <!-- GPU HBM -->
    <rect x="410" y="0" width="190" height="74" rx="1" fill="#ffffff" stroke="#1f2937" stroke-width="1.2"/>
    <text x="505" y="22" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1f2937">GPU VRAM / HBM</text>
    <text x="505" y="40" text-anchor="middle" font-size="8" font-weight="bold" fill="#1f2937">Overlapped Execution</text>
    <text x="505" y="56" text-anchor="middle" font-size="8" fill="#6b7280">GPU computes N while DMA streams N+1</text>
  </g>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Double-buffering and page-locked host memory allow I/O data transfers to overlap perfectly with GPU kernel execution.</text>
{FOOTER}"""
    write_svg("05_dataloader-copies-pipeline.svg", body)


# -----------------------------------------------------------------------------
# 7. 06_autograd-pointer-dag.svg
# -----------------------------------------------------------------------------
def gen_06_autograd_pointer_dag():
    h = 340
    body = f"""{HEADER.format(height=h)}
  <!-- Graph Container -->
  <rect x="25" y="20" width="630" height="215" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="630" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">DYNAMIC TAPE POINTER GRAPH: y = x · x; L = y + x (EVALUATED AT x = 3.0)</text>

  <g transform="translate(35, 52)">
    <!-- Leaf Tensor x (ACCENT) -->
    <rect x="0" y="50" width="125" height="85" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
    <rect x="0" y="50" width="125" height="20" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="62" y="64" text-anchor="middle" font-size="8.5" font-weight="700" fill="#c85a17">LEAF TENSOR x</text>
    <text x="62" y="86" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">id: 0x10</text>
    <text x="62" y="102" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#1f2937">data: 3.0</text>
    <text x="62" y="118" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#ff8246">grad: 1.0+6.0 = 7.0</text>

    <!-- Forward Arrows from x to MulNode -->
    <path d="M125 80 H175" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>
    <path d="M125 105 H175" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <!-- MulBackward Node -->
    <rect x="175" y="60" width="120" height="65" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="235" y="80" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">MulBackward</text>
    <text x="235" y="96" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">inputs: (0x10, 0x10)</text>
    <text x="235" y="110" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">saved: x = 3.0</text>

    <!-- Arrow MulNode to y -->
    <path d="M295 92 H330" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <!-- Intermediate Tensor y -->
    <rect x="330" y="60" width="110" height="65" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="385" y="80" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">Tensor y</text>
    <text x="385" y="96" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#1f2937">data: 9.0</text>
    <text x="385" y="110" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#6b7280">grad: 1.0</text>

    <!-- Arrow y to AddNode -->
    <path d="M440 92 H475" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <!-- Direct residual path from x to AddNode -->
    <path d="M62 50 V18 H495 V58" fill="none" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,2" marker-end="url(#arrow-gray)"/>
    <text x="278" y="14" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">Direct connection to x (Branch 2)</text>

    <!-- AddBackward Node -->
    <rect x="475" y="60" width="125" height="65" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="537" y="80" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">AddBackward</text>
    <text x="537" y="96" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">inputs: (0x30, 0x10)</text>
    <text x="537" y="110" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">saved: []</text>

    <!-- Arrow AddNode to Root L -->
    <path d="M537 125 V140" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <!-- Root Tensor L -->
    <rect x="475" y="142" width="125" height="46" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1"/>
    <text x="537" y="158" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">Root Tensor L</text>
    <text x="537" y="172" text-anchor="middle" font-size="8" font-family="monospace" fill="#c85a17">data: 12.0 (seed=1.0)</text>
  </g>

  <!-- Backward Trace Timeline Box -->
  <rect x="25" y="245" width="630" height="65" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="35" y="262" font-size="9" font-weight="700" fill="#1f2937">Topological Backward Traversal Order:</text>
  <text x="35" y="278" font-size="8" font-family="monospace" fill="#1f2937">1. AddBackward (seed = 1.0) → dL/dy = 1.0, dL/dx (direct) = 1.0</text>
  <text x="35" y="291" font-size="8" font-family="monospace" fill="#1f2937">2. MulBackward (dL/dy = 1.0) → dL/dx (via mul) = x + x = 3.0 + 3.0 = 6.0</text>
  <text x="35" y="303" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">3. Accumulate into Leaf x: pending[0x10] = 1.0 + 6.0 = 7.0 ✓</text>

  <!-- Bottom Annotation -->
  <text x="340" y="328" text-anchor="middle" font-size="9" fill="#6b7280">Autograd constructs a dynamic pointer DAG during forward execution and sweeps reverse topological order during backward passes.</text>
{FOOTER}"""
    write_svg("06_autograd-pointer-dag.svg", body)


# -----------------------------------------------------------------------------
# 8. 07_ravine-optimization.svg
# -----------------------------------------------------------------------------
def gen_07_ravine_optimization():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Left Panel: Ravine Contours & Trajectories -->
  <rect x="25" y="20" width="300" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="300" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. ILL-CONDITIONED RAVINE LANDSCAPE (κ = 100)</text>

  <!-- Elliptical Contours -->
  <g transform="translate(35, 54)">
    <rect x="0" y="0" width="280" height="150" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <ellipse cx="140" cy="75" rx="120" ry="25" fill="none" stroke="#d1d5db" stroke-width="1"/>
    <ellipse cx="140" cy="75" rx="80" ry="16" fill="none" stroke="#d1d5db" stroke-width="1"/>
    <ellipse cx="140" cy="75" rx="40" ry="8" fill="none" stroke="#d1d5db" stroke-width="1"/>
    <ellipse cx="140" cy="75" rx="15" ry="3" fill="none" stroke="#d1d5db" stroke-width="1"/>
    <circle cx="140" cy="75" r="3" fill="#1f2937"/>
    <text x="140" y="90" text-anchor="middle" font-size="8" fill="#1f2937">Min (0, 0)</text>

    <!-- Path 1: SGD Oscillations (Gray dashed) -->
    <path d="M30 25 L45 125 L70 35 L95 115 L115 55 L125 95 L135 70" fill="none" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="3,2"/>

    <!-- Path 2: AdamW Momentum (ACCENT Flameorange) -->
    <path d="M30 25 Q60 75 137 75" fill="none" stroke="#ff8246" stroke-width="2" marker-end="url(#arrow-orange)"/>

    <!-- Clean Legend in Top-Right Corner -->
    <text x="270" y="16" text-anchor="end" font-size="8" font-weight="bold" fill="#6b7280">SGD: Violent oscillation (dashed)</text>
    <text x="270" y="28" text-anchor="end" font-size="8" font-weight="bold" fill="#ff8246">AdamW: Smooth ravine descent (orange)</text>
  </g>

  <text x="35" y="222" font-size="8.5" fill="#6b7280">L(w₁, w₂) = 50w₁² + 0.5w₂²</text>
  <text x="35" y="238" font-size="8.5" fill="#6b7280">High curvature along w₁ causes SGD to bounce;</text>
  <text x="35" y="254" font-size="8.5" fill="#6b7280">second-moment normalization damps oscillation.</text>

  <!-- Right Panel: The 16-Byte Parameter Rule (ACCENT) -->
  <rect x="345" y="20" width="310" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="345" y="20" width="310" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="357" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. THE 16-BYTE OPTIMIZER MEMORY RULE</text>

  <g transform="translate(355, 54)">
    <!-- Row 1: Parameter -->
    <rect x="10" y="10" width="70" height="30" rx="1" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="45" y="30" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">Weight θ</text>
    <text x="95" y="30" font-size="9" font-family="monospace" fill="#1f2937">4 bytes (FP32 model parameter)</text>

    <!-- Row 2: Gradient -->
    <rect x="10" y="48" width="70" height="30" rx="1" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="45" y="68" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">Grad g</text>
    <text x="95" y="68" font-size="9" font-family="monospace" fill="#1f2937">4 bytes (FP32 gradient buffer)</text>

    <!-- Row 3: 1st Moment m -->
    <rect x="10" y="86" width="70" height="30" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
    <text x="45" y="106" text-anchor="middle" font-size="9" font-weight="bold" fill="#ff8246">Moment m</text>
    <text x="95" y="106" font-size="9" font-family="monospace" fill="#1f2937">4 bytes (Exponential moving mean)</text>

    <!-- Row 4: 2nd Moment v -->
    <rect x="10" y="124" width="70" height="30" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
    <text x="45" y="144" text-anchor="middle" font-size="9" font-weight="bold" fill="#ff8246">Moment v</text>
    <text x="95" y="144" font-size="9" font-family="monospace" fill="#1f2937">4 bytes (Moving variance)</text>

    <!-- Total Callout -->
    <line x1="10" y1="165" x2="280" y2="165" stroke="#ff8246" stroke-width="1"/>
    <text x="145" y="184" text-anchor="middle" font-size="10.5" font-family="monospace" font-weight="bold" fill="#ff8246">Total = 16 Bytes per Parameter</text>
  </g>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Adaptive optimizers dampen ravine oscillations by scaling steps by √v, requiring 16 bytes per parameter in active GPU memory.</text>
{FOOTER}"""
    write_svg("07_ravine-optimization.svg", body)


# -----------------------------------------------------------------------------
# 9. 08_microbatch-memory-timeline.svg
# -----------------------------------------------------------------------------
def gen_08_microbatch_memory_timeline():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Main Container -->
  <rect x="25" y="20" width="630" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="630" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">MICRO-BATCHING TIMELINE: EFFECTIVE BATCH SIZE WITHOUT MEMORY EXPLOSION</text>

  <g transform="translate(45, 50)">
    <!-- Axes -->
    <line x1="40" y1="170" x2="570" y2="170" stroke="#1f2937" stroke-width="1.2" marker-end="url(#arrow)"/>
    <line x1="40" y1="170" x2="40" y2="15" stroke="#1f2937" stroke-width="1.2" marker-end="url(#arrow)"/>
    <text x="560" y="184" font-size="8.5" font-family="monospace" fill="#1f2937">Time (t) -&gt;</text>
    <text x="20" y="15" font-size="8.5" font-family="monospace" fill="#1f2937">VRAM</text>

    <!-- OOM Limit Line -->
    <line x1="40" y1="50" x2="550" y2="50" stroke="#1f2937" stroke-width="1.2" stroke-dasharray="4,4"/>
    <text x="545" y="44" text-anchor="end" font-size="8.5" font-weight="bold" fill="#1f2937">GPU Hardware VRAM Ceiling (16 GB)</text>

    <!-- Path A: Full Batch B=256 Spike (OOM) -->
    <path d="M40 170 Q80 15 120 15 Q160 15 200 170" fill="none" stroke="#9ca3af" stroke-width="1.8" stroke-dasharray="4,2"/>
    <rect x="75" y="5" width="90" height="26" rx="1" fill="#f8fafc" stroke="#9ca3af" stroke-width="1"/>
    <text x="120" y="18" text-anchor="middle" font-size="8" font-weight="bold" fill="#1f2937">OOM! (32 GB)</text>
    <text x="120" y="27" text-anchor="middle" font-size="6.5" font-family="monospace" fill="#6b7280">Full Batch B=256</text>

    <!-- Path B: Micro-Batching 4x64 (ACCENT) -->
    <!-- Micro-batch 1 -->
    <path d="M40 170 Q70 85 100 85 Q130 85 160 160" fill="none" stroke="#ff8246" stroke-width="2"/>
    <text x="100" y="125" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">μB 1 (4 GB)</text>

    <!-- Micro-batch 2 -->
    <path d="M160 160 Q190 80 220 80 Q250 80 280 155" fill="none" stroke="#ff8246" stroke-width="2"/>
    <text x="220" y="125" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">μB 2</text>

    <!-- Micro-batch 3 -->
    <path d="M280 155 Q310 75 340 75 Q370 75 400 150" fill="none" stroke="#ff8246" stroke-width="2"/>
    <text x="340" y="125" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">μB 3</text>

    <!-- Micro-batch 4 -->
    <path d="M400 150 Q430 70 460 70 Q490 70 520 145" fill="none" stroke="#ff8246" stroke-width="2"/>
    <text x="460" y="125" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">μB 4</text>

    <!-- Accumulator baseline drift -->
    <line x1="40" y1="170" x2="520" y2="145" stroke="#ff8246" stroke-width="1.2" stroke-dasharray="2,2"/>

    <!-- Optimizer Step -->
    <path d="M520 145 V168" stroke="#1f2937" stroke-width="1.8" marker-end="url(#arrow)"/>
    <text x="520" y="136" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">opt.step()</text>
  </g>

  <text x="45" y="238" font-size="8.5" fill="#1f2937">Intermediate activations are freed immediately after the backward pass of each micro-batch.</text>
  <text x="45" y="254" font-size="8.5" fill="#6b7280">Only parameter gradients accumulate (dashed orange baseline): effective batch size 256 achieved in 4 GB budget.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Gradient accumulation decouples mathematical batch size from hardware VRAM constraints via sequential micro-batch activation clearing.</text>
{FOOTER}"""
    write_svg("08_microbatch-memory-timeline.svg", body)


# -----------------------------------------------------------------------------
# 10. milestone_01_kernel-trace.svg
# -----------------------------------------------------------------------------
def gen_milestone_01_kernel_trace():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Forward Pipeline Container -->
  <rect x="25" y="20" width="630" height="95" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="630" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. FORWARD PASS EXECUTION (MODULES 01-05)</text>

  <!-- Forward Pipeline Nodes -->
  <g transform="translate(35, 52)">
    <rect x="0" y="0" width="105" height="50" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="52" y="20" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">1. Batch X</text>
    <text x="52" y="36" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">[4, 2] float32</text>

    <path d="M105 25 H125" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <rect x="125" y="0" width="105" height="50" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="177" y="20" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">2. Linear 1</text>
    <text x="177" y="36" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">X @ W₁ + b₁</text>

    <path d="M230 25 H250" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <rect x="250" y="0" width="105" height="50" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="302" y="20" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">3. ReLU</text>
    <text x="302" y="36" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">Save mask</text>

    <path d="M355 25 H375" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <rect x="375" y="0" width="105" height="50" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="427" y="20" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">4. Linear 2</text>
    <text x="427" y="36" text-anchor="middle" font-size="8" font-family="monospace" fill="#6b7280">H @ W₂ + b₂</text>

    <path d="M480 25 H500" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <rect x="500" y="0" width="110" height="50" rx="1" fill="#f1f5f9" stroke="#1f2937" stroke-width="1.2"/>
    <text x="555" y="20" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">5. BCE Loss</text>
    <text x="555" y="36" text-anchor="middle" font-size="8" font-family="monospace" fill="#c85a17">Scalar L = 0.69</text>
  </g>

  <!-- Backward Pipeline Container -->
  <rect x="25" y="125" width="630" height="95" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="125" width="630" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="141" font-size="9.5" font-weight="700" fill="#1f2937">2. REVERSE AUTOGRAD SWEEP (MODULE 06)</text>

  <!-- Backward Pipeline Nodes (Right to Left) -->
  <g transform="translate(35, 157)">
    <rect x="470" y="0" width="140" height="50" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="540" y="20" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">6. Sigmoid Grad</text>
    <text x="540" y="36" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">p - y (Seed=1.0)</text>

    <path d="M470 25 H445" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <rect x="315" y="0" width="130" height="50" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="380" y="20" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">7. W₂, b₂ Grad</text>
    <text x="380" y="36" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">H.T @ grad</text>

    <path d="M315 25 H290" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <rect x="160" y="0" width="130" height="50" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="225" y="20" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">8. ReLU Mask</text>
    <text x="225" y="36" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">grad * (z &gt; 0)</text>

    <path d="M160 25 H135" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

    <rect x="0" y="0" width="135" height="50" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="67" y="20" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">9. W₁, b₁ Grad</text>
    <text x="67" y="36" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">X.T @ grad</text>
  </g>

  <!-- Step & Zero Container (ACCENT) -->
  <rect x="25" y="230" width="630" height="48" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="35" y="250" font-size="9.5" font-weight="700" fill="#c85a17">3. IN-PLACE OPTIMIZER MUTATION: opt.step() &amp; opt.zero_grad() (MODULE 07)</text>
  <text x="35" y="266" font-size="8" font-family="monospace" fill="#1f2937">param.data -= lr * param.grad (in-place memory mutation) → sever _grad_fn tape references</text>

  <!-- Bottom Annotation -->
  <text x="340" y="300" text-anchor="middle" font-size="9" fill="#6b7280">The complete TinyTorch lifecycle: forward evaluation → autograd graph recording → reverse gradient propagation → in-place optimizer step.</text>
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
