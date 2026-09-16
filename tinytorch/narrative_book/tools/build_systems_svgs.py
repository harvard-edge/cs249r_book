#!/usr/bin/env python3
"""
Generate Part III (Systems & Acceleration) and Part IV (Extensions) Margin Micro-Figures
for TinyTorch Narrative Book.
Strict compliance with Vol 3 Figure 1.10 standard & tinytorch/palette.md:
- 220 width viewBox, proportional height (120-125)
- Pure white background (#ffffff), NO outer frame border stroke
- Subsystem container cards with 20px header bands (rx="2")
- Palette: #ffffff, #f8fafc, #fff1e8, #f1f5f9, #9ca3af, #cbd5e1, #ff8246, #c85a17, #1f2937, #6b7280
- Max 1 accent node/container per diagram
- Fonts: TeX Gyre Heros, Helvetica Neue, Arial, sans-serif
- Uniform stroke weights, crisp geometry, tactile systems intuition.
- Dual-synchronized to narrative_book/ and quarto/ asset directories.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEST_DIRS = [
    REPO_ROOT / "tinytorch/narrative_book/assets/images/diagrams",
    REPO_ROOT / "tinytorch/quarto/assets/images/diagrams",
]

def write_svg(filename: str, content: str):
    for d in DEST_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Generated: {filename}")


# -----------------------------------------------------------------------------
# 1. 14_profiling-margin-roofline.svg (viewBox 0 0 220 120)
# -----------------------------------------------------------------------------
def gen_14_profiling_margin_roofline():
    h = 120
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="110" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">ROOFLINE: MEMORY VS COMPUTE</text>

  <g transform="translate(15, 26)">
    <!-- Axes -->
    <line x1="20" y1="52" x2="185" y2="52" stroke="#cbd5e1" stroke-width="1"/>
    <line x1="20" y1="8" x2="20" y2="52" stroke="#cbd5e1" stroke-width="1"/>
    <text x="185" y="61" text-anchor="end" font-size="6" font-family="monospace" fill="#9ca3af">Intensity (FLOP/B)</text>
    <text x="23" y="14" font-size="6" font-family="monospace" fill="#9ca3af">GFLOP/s</text>

    <!-- Roofline Ceilings -->
    <!-- Slanted Memory Ceiling -->
    <line x1="20" y1="52" x2="92" y2="20" stroke="#c85a17" stroke-width="1.6"/>
    <!-- Horizontal Compute Ceiling -->
    <line x1="92" y1="20" x2="185" y2="20" stroke="#1f2937" stroke-width="1.6"/>

    <!-- Ridge Point Dashed Line -->
    <line x1="92" y1="20" x2="92" y2="52" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="2 2"/>
    <circle cx="92" cy="20" r="2.5" fill="#c85a17"/>
    <text x="92" y="14" text-anchor="middle" font-size="6.5" font-weight="bold" fill="#c85a17">Ridge Point</text>

    <!-- Workload Points -->
    <!-- Attention: Memory Bound -->
    <circle cx="52" cy="38" r="3" fill="#ff8246" stroke="#c85a17" stroke-width="0.8"/>
    <text x="52" y="27" text-anchor="middle" font-size="6" font-weight="bold" fill="#c85a17">Attention</text>
    <text x="52" y="49" text-anchor="middle" font-size="5" fill="#6b7280">Mem-Bound</text>

    <!-- Dense GEMM: Compute Bound -->
    <circle cx="148" cy="20" r="3" fill="#1f2937"/>
    <text x="148" y="32" text-anchor="middle" font-size="6" font-weight="bold" fill="#1f2937">Dense GEMM</text>
    <text x="148" y="41" text-anchor="middle" font-size="5" fill="#6b7280">Compute-Bound</text>

    <!-- Summary Footnote Badge -->
    <rect x="18" y="68" width="154" height="12" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="0.6"/>
    <text x="95" y="77" text-anchor="middle" font-size="5.8" font-weight="bold" fill="#c85a17">Bottleneck shifts from bandwidth to arithmetic</text>
  </g>
</svg>
"""
    write_svg("14_profiling-margin-roofline.svg", body)


# -----------------------------------------------------------------------------
# 2. 15_quantization-margin-grid.svg (viewBox 0 0 220 120)
# -----------------------------------------------------------------------------
def gen_15_quantization_margin_grid():
    h = 120
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="110" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">AFFINE INT8 QUANTIZATION GRID</text>

  <g transform="translate(15, 26)">
    <!-- FP32 Continuous Axis -->
    <line x1="15" y1="18" x2="175" y2="18" stroke="#9ca3af" stroke-width="1"/>
    <text x="15" y="10" font-size="6.5" font-weight="bold" fill="#6b7280">FP32 Domain (x)</text>
    <line x1="25" y1="14" x2="25" y2="22" stroke="#9ca3af" stroke-width="0.8"/>
    <text x="25" y="27" text-anchor="middle" font-size="6" fill="#9ca3af">-1.0</text>
    <line x1="85" y1="14" x2="85" y2="22" stroke="#9ca3af" stroke-width="0.8"/>
    <text x="85" y="27" text-anchor="middle" font-size="6" fill="#9ca3af">0.0</text>
    <line x1="155" y1="14" x2="155" y2="22" stroke="#9ca3af" stroke-width="0.8"/>
    <text x="155" y="27" text-anchor="middle" font-size="6" fill="#9ca3af">+2.0</text>

    <!-- Point x -->
    <circle cx="125" cy="18" r="2.5" fill="#c85a17"/>
    <text x="125" y="11" text-anchor="middle" font-size="6" font-family="monospace" font-weight="bold" fill="#c85a17">x = 1.35</text>

    <!-- Mapping projection arrow -->
    <path d="M 125 21 L 125 43" stroke="#ff8246" stroke-width="1.2" stroke-dasharray="2 2"/>
    <polygon points="122.5 42, 125 46, 127.5 42" fill="#ff8246"/>
    <text x="135" y="34" font-size="5.8" font-family="monospace" fill="#c85a17">q = round(x/S) + Z</text>

    <!-- INT8 Discrete Grid Axis -->
    <line x1="15" y1="48" x2="175" y2="48" stroke="#1f2937" stroke-width="1"/>
    <text x="15" y="60" font-size="6.5" font-weight="bold" fill="#1f2937">INT8 Grid (q)</text>
    <line x1="25" y1="45" x2="25" y2="51" stroke="#1f2937" stroke-width="0.8"/>
    <text x="25" y="42" text-anchor="middle" font-size="6" fill="#6b7280">-128</text>
    <line x1="85" y1="45" x2="85" y2="51" stroke="#1f2937" stroke-width="0.8"/>
    <text x="85" y="42" text-anchor="middle" font-size="6" fill="#6b7280">Z</text>
    <line x1="155" y1="45" x2="155" y2="51" stroke="#1f2937" stroke-width="0.8"/>
    <text x="155" y="42" text-anchor="middle" font-size="6" fill="#6b7280">127</text>

    <!-- Quantized Point q -->
    <rect x="122.5" y="45.5" width="5" height="5" fill="#c85a17"/>
    <text x="125" y="60" text-anchor="middle" font-size="6" font-family="monospace" font-weight="bold" fill="#c85a17">q = 86</text>

    <!-- Footnote Badge -->
    <rect x="25" y="68" width="140" height="12" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="0.6"/>
    <text x="95" y="77" text-anchor="middle" font-size="6" font-weight="bold" fill="#c85a17">Memory footprint: 4.0× reduction</text>
  </g>
</svg>
"""
    write_svg("15_quantization-margin-grid.svg", body)


# -----------------------------------------------------------------------------
# 3. 16_compression-margin-temperature.svg (viewBox 0 0 220 120)
# -----------------------------------------------------------------------------
def gen_16_compression_margin_temperature():
    h = 120
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="110" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">DISTILLATION: SOFT TARGET ENTROPY</text>

  <g transform="translate(15, 28)">
    <!-- Left: T = 1 Hard/Sharp Target -->
    <g transform="translate(10, 3)">
      <text x="35" y="8" text-anchor="middle" font-size="7" font-weight="bold" fill="#1f2937">T = 1.0 (Hard)</text>
      <!-- Bars -->
      <rect x="5" y="32" width="10" height="3" fill="#cbd5e1"/>
      <rect x="20" y="33" width="10" height="2" fill="#cbd5e1"/>
      <rect x="35" y="12" width="10" height="23" fill="#1f2937"/>
      <rect x="50" y="33" width="10" height="2" fill="#cbd5e1"/>
      <rect x="65" y="34" width="10" height="1" fill="#cbd5e1"/>
      <!-- Base axis -->
      <line x1="0" y1="35" x2="80" y2="35" stroke="#9ca3af" stroke-width="0.8"/>
      <text x="40" y="44" text-anchor="middle" font-size="6" fill="#6b7280">p = [0.03, 0.01, 0.94, ...]</text>
      <text x="40" y="52" text-anchor="middle" font-size="5.5" fill="#9ca3af">Zero Inter-Class Signal</text>
    </g>

    <!-- Arrow Bridge -->
    <path d="M 98 22 H 108" stroke="#ff8246" stroke-width="1.2"/>
    <polygon points="107 19.5, 112 22, 107 24.5" fill="#ff8246"/>

    <!-- Right: T = 4 Softened Distribution -->
    <g transform="translate(115, 3)">
      <text x="35" y="8" text-anchor="middle" font-size="7" font-weight="bold" fill="#c85a17">T = 4.0 (Softened)</text>
      <!-- Bars with dark knowledge visible -->
      <rect x="5" y="24" width="10" height="11" fill="#fff1e8" stroke="#ff8246" stroke-width="0.8"/>
      <rect x="20" y="29" width="10" height="6" fill="#fff1e8" stroke="#ff8246" stroke-width="0.8"/>
      <rect x="35" y="17" width="10" height="18" fill="#c85a17"/>
      <rect x="50" y="27" width="10" height="8" fill="#fff1e8" stroke="#ff8246" stroke-width="0.8"/>
      <rect x="65" y="31" width="10" height="4" fill="#fff1e8" stroke="#ff8246" stroke-width="0.8"/>
      <!-- Base axis -->
      <line x1="0" y1="35" x2="80" y2="35" stroke="#9ca3af" stroke-width="0.8"/>
      <text x="40" y="44" text-anchor="middle" font-size="6" font-weight="bold" fill="#c85a17">Dark Knowledge Revealed</text>
      <text x="40" y="52" text-anchor="middle" font-size="5.5" fill="#6b7280">p = [0.22, 0.12, 0.38, ...]</text>
    </g>

    <!-- Bottom summary line -->
    <text x="100" y="74" text-anchor="middle" font-size="6.5" fill="#1f2937">Student learns relative similarities via KL divergence</text>
  </g>
</svg>
"""
    write_svg("16_compression-margin-temperature.svg", body)


# -----------------------------------------------------------------------------
# 4. 17_acceleration-margin-fusion.svg (viewBox 0 0 220 120)
# -----------------------------------------------------------------------------
def gen_17_acceleration_margin_fusion():
    h = 120
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="110" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">OPERATOR FUSION MEMORY TRAFFIC</text>

  <g transform="translate(12, 26)">
    <!-- Unfused Row -->
    <g transform="translate(0, 3)">
      <rect x="0" y="0" width="34" height="24" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
      <text x="17" y="11" text-anchor="middle" font-size="6" font-weight="bold" fill="#1f2937">DRAM</text>
      <text x="17" y="18" text-anchor="middle" font-size="4.5" fill="#6b7280">(Slow)</text>

      <!-- 8 round-trip arrows to execution cores -->
      <path d="M 38 6 H 145" stroke="#9ca3af" stroke-width="0.7" stroke-dasharray="2 2"/>
      <path d="M 38 18 H 145" stroke="#9ca3af" stroke-width="0.7" stroke-dasharray="2 2"/>

      <rect x="46" y="2" width="22" height="20" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.7"/>
      <text x="57" y="14" text-anchor="middle" font-size="5" fill="#1f2937">Op 1</text>

      <rect x="74" y="2" width="22" height="20" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.7"/>
      <text x="85" y="14" text-anchor="middle" font-size="5" fill="#1f2937">Op 2</text>

      <text x="103" y="14" text-anchor="middle" font-size="6" fill="#9ca3af">···</text>

      <rect x="112" y="2" width="22" height="20" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.7"/>
      <text x="123" y="14" text-anchor="middle" font-size="5" fill="#1f2937">Op 8</text>

      <text x="168" y="14" text-anchor="middle" font-size="5.5" font-weight="bold" fill="#6b7280">8 Roundtrips</text>
    </g>

    <!-- Fused Row -->
    <g transform="translate(0, 34)">
      <rect x="0" y="0" width="34" height="25" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
      <text x="17" y="11" text-anchor="middle" font-size="6" font-weight="bold" fill="#c85a17">DRAM</text>
      <text x="17" y="19" text-anchor="middle" font-size="4.5" fill="#c85a17">1 Read/Write</text>

      <!-- Arrow into SRAM/Register fused block -->
      <path d="M 37 12.5 H 48" stroke="#ff8246" stroke-width="1.2"/>
      <polygon points="46 10, 50 12.5, 46 15" fill="#ff8246"/>

      <!-- Fused Kernel Box -->
      <rect x="52" y="0" width="140" height="25" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
      <text x="122" y="11" text-anchor="middle" font-size="6.2" font-weight="bold" fill="#c85a17">Fused Kernel (On-Chip Registers / SRAM)</text>
      <text x="122" y="19" text-anchor="middle" font-size="5.2" fill="#6b7280">Load once → Chained arithmetic → Write once</text>
    </g>

    <!-- Summary Footnote -->
    <text x="98" y="73" text-anchor="middle" font-size="6.2" font-weight="bold" fill="#c85a17">Intermediate memory traffic reduced by 9.0×</text>
  </g>
</svg>
"""
    write_svg("17_acceleration-margin-fusion.svg", body)


# -----------------------------------------------------------------------------
# 5. 18_memoization-margin-kvcache.svg (viewBox 0 0 120)
# -----------------------------------------------------------------------------
def gen_18_memoization_margin_kvcache():
    h = 120
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="110" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">KV CACHE STATIC BUFFER</text>

  <g transform="translate(15, 26)">
    <!-- Write head indicator -->
    <g transform="translate(108, 0)">
      <text x="12" y="7" text-anchor="middle" font-size="6.2" font-weight="bold" fill="#c85a17">Write Head (t)</text>
      <path d="M 12 10 L 12 16" stroke="#ff8246" stroke-width="1.2"/>
      <polygon points="9.5 15, 12 19, 14.5 15" fill="#ff8246"/>
    </g>

    <!-- Buffer Slots Array -->
    <g transform="translate(4, 20)">
      <!-- Filled Past Cached Slots -->
      <rect x="0" y="0" width="24" height="24" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
      <text x="12" y="15" text-anchor="middle" font-size="5.8" font-family="monospace" fill="#1f2937">K₀, V₀</text>

      <rect x="26" y="0" width="24" height="24" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
      <text x="38" y="15" text-anchor="middle" font-size="5.8" font-family="monospace" fill="#1f2937">K₁, V₁</text>

      <rect x="52" y="0" width="24" height="24" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
      <text x="64" y="15" text-anchor="middle" font-size="5.8" font-family="monospace" fill="#1f2937">K₂, V₂</text>

      <rect x="78" y="0" width="24" height="24" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
      <text x="90" y="15" text-anchor="middle" font-size="5.8" font-family="monospace" fill="#1f2937">K₃, V₃</text>

      <!-- Active Write Slot (t) -->
      <rect x="105" y="0" width="26" height="24" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
      <text x="118" y="15" text-anchor="middle" font-size="5.8" font-family="monospace" font-weight="bold" fill="#c85a17">K_t, V_t</text>

      <!-- Empty / Unused Slots -->
      <rect x="134" y="0" width="22" height="24" fill="#ffffff" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="2 2"/>
      <text x="145" y="15" text-anchor="middle" font-size="5.5" fill="#cbd5e1">empty</text>

      <rect x="158" y="0" width="22" height="24" fill="#ffffff" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="2 2"/>
      <text x="169" y="15" text-anchor="middle" font-size="5.5" fill="#cbd5e1">empty</text>
    </g>

    <!-- Range Brackets -->
    <text x="55" y="54" text-anchor="middle" font-size="5.8" fill="#6b7280">Reused without recomputation</text>
    <text x="156" y="54" text-anchor="middle" font-size="5.8" fill="#9ca3af">Static max context</text>

    <!-- Bottom complexity comparison -->
    <rect x="25" y="62" width="145" height="13" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.6"/>
    <text x="97.5" y="71" text-anchor="middle" font-size="6.2" font-weight="bold" fill="#1f2937">Decode step FLOPs: O(S²) → O(1) new math</text>
  </g>
</svg>
"""
    write_svg("18_memoization-margin-kvcache.svg", body)


# -----------------------------------------------------------------------------
# 6. 19_benchmarking-margin-latency.svg (viewBox 0 0 220 120)
# -----------------------------------------------------------------------------
def gen_19_benchmarking_margin_latency():
    h = 120
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="110" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">TAIL LATENCY PERCENTILES</text>

  <g transform="translate(15, 26)">
    <!-- Axes -->
    <line x1="15" y1="50" x2="185" y2="50" stroke="#cbd5e1" stroke-width="1"/>
    <line x1="15" y1="8" x2="15" y2="50" stroke="#cbd5e1" stroke-width="1"/>
    <text x="185" y="59" text-anchor="end" font-size="6" font-family="monospace" fill="#9ca3af">Latency (ms)</text>
    <text x="18" y="12" font-size="6" font-family="monospace" fill="#9ca3af">Density</text>

    <!-- Smooth Skewed Latency Distribution Curve -->
    <path d="M 20 49 C 32 14, 48 14, 62 25 C 80 38, 120 46, 175 49" fill="none" stroke="#1f2937" stroke-width="1.6"/>

    <!-- P50 Median Line -->
    <line x1="52" y1="16" x2="52" y2="50" stroke="#9ca3af" stroke-width="0.8" stroke-dasharray="2 2"/>
    <text x="52" y="11" text-anchor="middle" font-size="6" font-weight="bold" fill="#6b7280">P50</text>

    <!-- P95 Line -->
    <line x1="100" y1="38" x2="100" y2="50" stroke="#c85a17" stroke-width="0.8" stroke-dasharray="2 2"/>
    <text x="100" y="33" text-anchor="middle" font-size="6" font-weight="bold" fill="#c85a17">P95</text>

    <!-- P99 Tail Outlier Line -->
    <line x1="145" y1="46" x2="145" y2="50" stroke="#ff8246" stroke-width="1"/>
    <circle cx="145" cy="46" r="2" fill="#ff8246"/>
    <text x="145" y="41" text-anchor="middle" font-size="6" font-weight="bold" fill="#c85a17">P99 (Tail)</text>

    <!-- Annotation badge -->
    <rect x="18" y="66" width="154" height="12" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="0.6"/>
    <text x="95" y="75" text-anchor="middle" font-size="5.8" font-weight="bold" fill="#c85a17">Warmup isolates steady state from cold JIT spikes</text>
  </g>
</svg>
"""
    write_svg("19_benchmarking-margin-latency.svg", body)


# -----------------------------------------------------------------------------
# 7. 20_capstone-margin-pareto.svg (viewBox 0 0 220 120)
# -----------------------------------------------------------------------------
def gen_20_capstone_margin_pareto():
    h = 120
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="110" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">CAPSTONE OPTIMIZATION PARETO</text>

  <g transform="translate(15, 26)">
    <!-- Axes -->
    <line x1="20" y1="52" x2="185" y2="52" stroke="#cbd5e1" stroke-width="1"/>
    <line x1="20" y1="8" x2="20" y2="52" stroke="#cbd5e1" stroke-width="1"/>
    <text x="185" y="61" text-anchor="end" font-size="6" font-family="monospace" fill="#9ca3af">Throughput (tok/s) →</text>
    <text x="23" y="13" font-size="6" font-family="monospace" fill="#9ca3af">Perplexity</text>

    <!-- Frontier Curve -->
    <path d="M 35 38 Q 75 37 115 35 T 165 33" fill="none" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="2 2"/>

    <!-- Point 1: Baseline -->
    <circle cx="35" cy="38" r="2.5" fill="#9ca3af"/>
    <text x="35" y="32" text-anchor="middle" font-size="5.2" fill="#6b7280">1. Baseline</text>
    <text x="35" y="47" text-anchor="middle" font-size="4.8" fill="#9ca3af">1.0×</text>

    <!-- Point 2: Distilled -->
    <circle cx="75" cy="37" r="2.5" fill="#6b7280"/>
    <text x="75" y="31" text-anchor="middle" font-size="5.2" fill="#6b7280">2. Distilled</text>
    <text x="75" y="47" text-anchor="middle" font-size="4.8" fill="#9ca3af">1.4×</text>

    <!-- Point 3: Quantized -->
    <circle cx="115" cy="35" r="2.5" fill="#c85a17"/>
    <text x="115" y="29" text-anchor="middle" font-size="5.2" font-weight="bold" fill="#c85a17">3. INT8</text>
    <text x="115" y="47" text-anchor="middle" font-size="4.8" fill="#c85a17">2.8×</text>

    <!-- Point 4: Fully Accelerated Capstone -->
    <circle cx="165" cy="33" r="3.5" fill="#ff8246" stroke="#c85a17" stroke-width="1"/>
    <text x="165" y="25" text-anchor="middle" font-size="5.8" font-weight="bold" fill="#c85a17">4. Fused+Cached</text>
    <text x="165" y="47" text-anchor="middle" font-size="5.2" font-weight="bold" fill="#c85a17">4.8×</text>

    <!-- Summary Badge -->
    <rect x="20" y="66" width="150" height="12" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="0.6"/>
    <text x="95" y="75" text-anchor="middle" font-size="5.8" font-weight="bold" fill="#c85a17">Total speedup: 4.8× · Footprint reduced 75%</text>
  </g>
</svg>
"""
    write_svg("20_capstone-margin-pareto.svg", body)


# -----------------------------------------------------------------------------
# 8. milestone_03-margin-budget.svg (viewBox 0 0 220 120)
# -----------------------------------------------------------------------------
def gen_milestone_03_margin_budget():
    h = 120
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="110" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">EDGE MEMORY BUDGET (512 MB)</text>

  <g transform="translate(15, 26)">
    <!-- Hardware Ceiling Line -->
    <line x1="5" y1="10" x2="185" y2="10" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="2 2"/>
    <text x="185" y="7" text-anchor="end" font-size="6" font-family="monospace" fill="#9ca3af">512 MB Hardware Ceiling</text>

    <!-- Stacked Memory Bar -->
    <g transform="translate(5, 15)">
      <!-- INT8 Weights (60%) -->
      <rect x="0" y="0" width="105" height="22" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
      <text x="52.5" y="10" text-anchor="middle" font-size="6" font-weight="bold" fill="#c85a17">INT8 Weights</text>
      <text x="52.5" y="18" text-anchor="middle" font-size="5" fill="#c85a17">307 MB (60%)</text>

      <!-- KV Cache (25%) -->
      <rect x="107" y="0" width="44" height="22" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
      <text x="129" y="10" text-anchor="middle" font-size="5.5" font-weight="bold" fill="#1f2937">KV Cache</text>
      <text x="129" y="18" text-anchor="middle" font-size="5" fill="#6b7280">128 MB</text>

      <!-- Activations (10%) -->
      <rect x="153" y="0" width="18" height="22" rx="1" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="0.8"/>
      <text x="162" y="10" text-anchor="middle" font-size="5" fill="#1f2937">Act</text>
      <text x="162" y="18" text-anchor="middle" font-size="4.5" fill="#6b7280">51M</text>

      <!-- Safety Headroom (5%) -->
      <rect x="173" y="0" width="9" height="22" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.8"/>
    </g>

    <!-- Legend and safety notes -->
    <text x="95" y="50" text-anchor="middle" font-size="5.8" fill="#6b7280">Static pre-allocation prevents mid-generation OOM spikes</text>

    <!-- Bottom summary badge -->
    <rect x="15" y="62" width="160" height="12" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="0.6"/>
    <text x="95" y="71" text-anchor="middle" font-size="5.8" font-weight="bold" fill="#c85a17">Streamed autoregression within strict embedded limit</text>
  </g>
</svg>
"""
    write_svg("milestone_03-margin-budget.svg", body)


# -----------------------------------------------------------------------------
# 9. 21_extensions-margin-systolic.svg (viewBox 0 0 220 125)
# -----------------------------------------------------------------------------
def gen_21_extensions_margin_systolic():
    h = 125
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="115" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">2D SYSTOLIC ARRAY WAVEFRONT</text>

  <g transform="translate(15, 28)">
    <!-- Diagonal wavefront guide line behind cells -->
    <line x1="45" y1="20" x2="160" y2="78" stroke="#ff8246" stroke-width="1" stroke-dasharray="2 2"/>
    <text x="166" y="52" font-size="6" font-weight="bold" fill="#c85a17">Wavefront</text>

    <!-- 3x3 PE (Processing Element) Grid -->
    <!-- Row 0 -->
    <rect x="65" y="15" width="18" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
    <text x="74" y="26" text-anchor="middle" font-size="5.8" font-weight="bold" fill="#c85a17">PE₀₀</text>

    <rect x="95" y="15" width="18" height="18" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
    <text x="104" y="26" text-anchor="middle" font-size="5.8" fill="#1f2937">PE₀₁</text>

    <rect x="125" y="15" width="18" height="18" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
    <text x="134" y="26" text-anchor="middle" font-size="5.8" fill="#1f2937">PE₀₂</text>

    <!-- Row 1 -->
    <rect x="65" y="37" width="18" height="18" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
    <text x="74" y="48" text-anchor="middle" font-size="5.8" fill="#1f2937">PE₁₀</text>

    <rect x="95" y="37" width="18" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
    <text x="104" y="48" text-anchor="middle" font-size="5.8" font-weight="bold" fill="#c85a17">PE₁₁</text>

    <rect x="125" y="37" width="18" height="18" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
    <text x="134" y="48" text-anchor="middle" font-size="5.8" fill="#1f2937">PE₁₂</text>

    <!-- Row 2 -->
    <rect x="65" y="59" width="18" height="18" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
    <text x="74" y="70" text-anchor="middle" font-size="5.8" fill="#1f2937">PE₂₀</text>

    <rect x="95" y="59" width="18" height="18" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="0.8"/>
    <text x="104" y="70" text-anchor="middle" font-size="5.8" fill="#1f2937">PE₂₁</text>

    <rect x="125" y="59" width="18" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
    <text x="134" y="70" text-anchor="middle" font-size="5.8" font-weight="bold" fill="#c85a17">PE₂₂</text>

    <!-- Streaming Inputs -->
    <!-- Input A from left -->
    <path d="M 45 24 H 58" stroke="#1f2937" stroke-width="1"/>
    <polygon points="56 22, 60 24, 56 26" fill="#1f2937"/>
    <text x="35" y="26" text-anchor="end" font-size="6" font-weight="bold" fill="#1f2937">A[i]</text>

    <!-- Input B from top -->
    <path d="M 74 4 V 11" stroke="#1f2937" stroke-width="1"/>
    <polygon points="72 10, 74 13, 76 10" fill="#1f2937"/>
    <text x="74" y="2" text-anchor="middle" font-size="6" font-weight="bold" fill="#1f2937">B[j]</text>

    <!-- Footnote: Latency formula -->
    <text x="104" y="85" text-anchor="middle" font-size="5.8" fill="#6b7280">Rhythmic lock-step: Latency = 2N - 1 cycles</text>
  </g>
</svg>
"""
    write_svg("21_extensions-margin-systolic.svg", body)


def main():
    gen_14_profiling_margin_roofline()
    gen_15_quantization_margin_grid()
    gen_16_compression_margin_temperature()
    gen_17_acceleration_margin_fusion()
    gen_18_memoization_margin_kvcache()
    gen_19_benchmarking_margin_latency()
    gen_20_capstone_margin_pareto()
    gen_milestone_03_margin_budget()
    gen_21_extensions_margin_systolic()

if __name__ == "__main__":
    main()
