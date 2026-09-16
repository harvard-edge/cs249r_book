#!/usr/bin/env python3
"""
Generate Part II (Deep Architectures) Margin Micro-Figures for TinyTorch Narrative Book.
Strict compliance with Vol 3 Figure 1.10 standard & tinytorch/palette.md:
- 220 width viewBox, proportional height (95-125)
- Pure white background (#ffffff), NO outer frame border stroke
- Subsystem container cards with 20px header bands (rx="2")
- Palette: #ffffff, #f8fafc, #fff1e8, #f1f5f9, #9ca3af, #cbd5e1, #ff8246, #c85a17, #1f2937, #6b7280
- Max 1 accent node/container per diagram
- Fonts: TeX Gyre Heros, Helvetica Neue, Arial, sans-serif
- Uniform stroke weights, crisp geometry, tactile systems intuition.
- Dual-synchronized to book/ and quarto/ asset directories.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEST_DIRS = [
    REPO_ROOT / "tinytorch/book/assets/images/diagrams",
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
# 1. 09_convolutions-margin-im2col.svg (viewBox 0 0 220 115)
# -----------------------------------------------------------------------------
def gen_09_convolutions_margin_im2col():
    h = 115
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="105" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">IM2COL MEMORY EXPANSION (3×3)</text>

  <g transform="translate(15, 33)">
    <!-- 2D Image Patch -->
    <rect x="0" y="0" width="45" height="45" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <!-- 3x3 grid inside patch -->
    <line x1="15" y1="0" x2="15" y2="45" stroke="#cbd5e1" stroke-width="0.8"/>
    <line x1="30" y1="0" x2="30" y2="45" stroke="#cbd5e1" stroke-width="0.8"/>
    <line x1="0" y1="15" x2="45" y2="15" stroke="#cbd5e1" stroke-width="0.8"/>
    <line x1="0" y1="30" x2="45" y2="30" stroke="#cbd5e1" stroke-width="0.8"/>
    <!-- Accent cell in patch -->
    <rect x="15" y="15" width="15" height="15" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
    <text x="22.5" y="56" text-anchor="middle" font-size="7" fill="#6b7280">Patch (3×3)</text>

    <!-- Arrow to column matrix -->
    <path d="M55 22.5 H75" stroke="#ff8246" stroke-width="1.2"/>
    <polygon points="73 19.5, 79 22.5, 73 25.5" fill="#ff8246"/>

    <!-- Lowered Column Vector Row -->
    <g transform="translate(85, 8)">
      <rect x="0" y="0" width="100" height="28" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
      <!-- 9 columns inside -->
      <line x1="11" y1="0" x2="11" y2="28" stroke="#ff8246" stroke-width="0.6"/>
      <line x1="22" y1="0" x2="22" y2="28" stroke="#ff8246" stroke-width="0.6"/>
      <line x1="33" y1="0" x2="33" y2="28" stroke="#ff8246" stroke-width="0.6"/>
      <line x1="44" y1="0" x2="44" y2="28" stroke="#ff8246" stroke-width="0.6"/>
      <line x1="55" y1="0" x2="55" y2="28" stroke="#ff8246" stroke-width="0.6"/>
      <line x1="66" y1="0" x2="66" y2="28" stroke="#ff8246" stroke-width="0.6"/>
      <line x1="77" y1="0" x2="77" y2="28" stroke="#ff8246" stroke-width="0.6"/>
      <line x1="88" y1="0" x2="88" y2="28" stroke="#ff8246" stroke-width="0.6"/>
      <text x="50" y="18" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#c85a17">9× Row Vector</text>
      <text x="50" y="42" text-anchor="middle" font-size="7" fill="#6b7280">1 Row per Sliding Window</text>
    </g>

    <!-- Bottom multiplier ledger -->
    <text x="95" y="66" text-anchor="middle" font-size="7.5" font-weight="700" fill="#1f2937">Memory footprint expands by K² = 9.0×</text>
  </g>
</svg>
"""
    write_svg("09_convolutions-margin-im2col.svg", body)


# -----------------------------------------------------------------------------
# 2. 10_tokenization-margin-bpe.svg (viewBox 0 0 220 115)
# -----------------------------------------------------------------------------
def gen_10_tokenization_margin_bpe():
    h = 115
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="105" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">BPE MERGE FREQUENCY DECAY</text>

  <g transform="translate(15, 30)">
    <!-- Axes -->
    <line x1="15" y1="55" x2="180" y2="55" stroke="#cbd5e1" stroke-width="1"/>
    <line x1="15" y1="5" x2="15" y2="55" stroke="#cbd5e1" stroke-width="1"/>
    <text x="180" y="66" text-anchor="end" font-size="7" font-family="monospace" fill="#9ca3af">Merge Rank</text>
    <text x="18" y="10" font-size="7" font-family="monospace" fill="#9ca3af">Freq</text>

    <!-- Zipfian power-law decay curve without right upward wiggle -->
    <path d="M 16 10 Q 25 38 48 47 T 95 52 L 175 53" fill="none" stroke="#ff8246" stroke-width="1.8"/>

    <!-- Markers for top merges -->
    <circle cx="20" cy="20" r="2.5" fill="#c85a17"/>
    <text x="24" y="18" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">"th"</text>

    <circle cx="35" cy="40" r="2.5" fill="#c85a17"/>
    <text x="39" y="37" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">"he"</text>

    <circle cx="58" cy="48" r="2" fill="#9ca3af"/>
    <text x="62" y="45" font-size="6.5" font-family="monospace" fill="#6b7280">"in"</text>

    <!-- Summary metrics -->
    <rect x="95" y="8" width="85" height="26" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="0.8"/>
    <text x="137.5" y="19" text-anchor="middle" font-size="7" font-weight="700" fill="#c85a17">Compression: 4.2×</text>
    <text x="137.5" y="29" text-anchor="middle" font-size="6.5" fill="#6b7280">Bytes → Subwords</text>
  </g>
</svg>
"""
    write_svg("10_tokenization-margin-bpe.svg", body)


# -----------------------------------------------------------------------------
# 3. 11_embeddings-margin-scatter.svg (viewBox 0 0 220 110)
# -----------------------------------------------------------------------------
def gen_11_embeddings_margin_scatter():
    h = 110
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="100" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">REPEATED TOKENS: SCATTER-ADD</text>

  <g transform="translate(15, 30)">
    <!-- Centered Sequence tokens -->
    <g transform="translate(24, 4)">
      <rect x="0" y="0" width="42" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
      <text x="21" y="12" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#c85a17">ID: 42</text>
      <text x="21" y="27" text-anchor="middle" font-size="6.5" fill="#6b7280">pos 0 (g₀)</text>

      <rect x="50" y="0" width="42" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
      <text x="71" y="12" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#1f2937">ID: 105</text>
      <text x="71" y="27" text-anchor="middle" font-size="6.5" fill="#6b7280">pos 1</text>

      <rect x="100" y="0" width="42" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
      <text x="121" y="12" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#c85a17">ID: 42</text>
      <text x="121" y="27" text-anchor="middle" font-size="6.5" fill="#6b7280">pos 2 (g₂)</text>
    </g>

    <!-- Converging gradient arcs -->
    <path d="M 45 35 Q 65 48 95 49" fill="none" stroke="#ff8246" stroke-width="1.2"/>
    <path d="M 145 35 Q 125 48 95 49" fill="none" stroke="#ff8246" stroke-width="1.2"/>

    <!-- Centered Embedding Row Accumulator -->
    <rect x="30" y="47" width="130" height="20" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="95" y="60" text-anchor="middle" font-size="7.5" font-family="monospace" font-weight="bold" fill="#c85a17">grad[42] = g₀ + g₂</text>
  </g>
</svg>
"""
    write_svg("11_embeddings-margin-scatter.svg", body)


# -----------------------------------------------------------------------------
# 4. 12_attention-margin-mask.svg (viewBox 0 0 220 120)
# -----------------------------------------------------------------------------
def gen_12_attention_margin_mask():
    h = 120
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="110" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">CAUSAL ATTENTION MASK (4×4)</text>

  <g transform="translate(25, 30)">
    <!-- 4x4 Heatmap Matrix -->
    <!-- Row 0: 0, -inf, -inf, -inf -->
    <rect x="0" y="0" width="18" height="16" fill="#fff1e8" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="9" y="11" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">0</text>
    <rect x="18" y="0" width="18" height="16" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="27" y="11" text-anchor="middle" font-size="6" font-family="monospace" fill="#9ca3af">-∞</text>
    <rect x="36" y="0" width="18" height="16" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="45" y="11" text-anchor="middle" font-size="6" font-family="monospace" fill="#9ca3af">-∞</text>
    <rect x="54" y="0" width="18" height="16" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="63" y="11" text-anchor="middle" font-size="6" font-family="monospace" fill="#9ca3af">-∞</text>

    <!-- Row 1: 0, 0, -inf, -inf -->
    <rect x="0" y="16" width="18" height="16" fill="#fff1e8" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="9" y="27" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">0</text>
    <rect x="18" y="16" width="18" height="16" fill="#fff1e8" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="27" y="27" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">0</text>
    <rect x="36" y="16" width="18" height="16" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="45" y="27" text-anchor="middle" font-size="6" font-family="monospace" fill="#9ca3af">-∞</text>
    <rect x="54" y="16" width="18" height="16" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="63" y="27" text-anchor="middle" font-size="6" font-family="monospace" fill="#9ca3af">-∞</text>

    <!-- Row 2: 0, 0, 0, -inf -->
    <rect x="0" y="32" width="18" height="16" fill="#fff1e8" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="9" y="43" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">0</text>
    <rect x="18" y="32" width="18" height="16" fill="#fff1e8" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="27" y="43" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">0</text>
    <rect x="36" y="32" width="18" height="16" fill="#fff1e8" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="45" y="43" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">0</text>
    <rect x="54" y="32" width="18" height="16" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="63" y="43" text-anchor="middle" font-size="6" font-family="monospace" fill="#9ca3af">-∞</text>

    <!-- Row 3: 0, 0, 0, 0 -->
    <rect x="0" y="48" width="18" height="16" fill="#fff1e8" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="9" y="59" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">0</text>
    <rect x="18" y="48" width="18" height="16" fill="#fff1e8" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="27" y="59" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">0</text>
    <rect x="36" y="48" width="18" height="16" fill="#fff1e8" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="45" y="59" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">0</text>
    <rect x="54" y="48" width="18" height="16" fill="#fff1e8" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="63" y="59" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">0</text>

    <!-- Explanatory legend -->
    <g transform="translate(85, 10)">
      <rect x="0" y="0" width="85" height="48" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.8"/>
      <text x="42" y="14" text-anchor="middle" font-size="7" font-weight="700" fill="#1f2937">Softmax Effect</text>
      <text x="42" y="27" text-anchor="middle" font-size="6.5" font-family="monospace" fill="#c85a17">e^0 = 1 (Active)</text>
      <text x="42" y="40" text-anchor="middle" font-size="6.5" font-family="monospace" fill="#6b7280">e^(-∞) = 0 (Zeroed)</text>
    </g>
  </g>
</svg>
"""
    write_svg("12_attention-margin-mask.svg", body)


# -----------------------------------------------------------------------------
# 5. 13_transformers-margin-pre-ln.svg (viewBox 0 0 220 120)
# -----------------------------------------------------------------------------
def gen_13_transformers_margin_pre_ln():
    h = 120
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="110" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">PRE-LN RESIDUAL HIGHWAY</text>

  <g transform="translate(10, 30)">
    <!-- Main input node -->
    <rect x="5" y="24" width="28" height="22" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="19" y="38" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">x</text>

    <!-- Clean Skip Connection (ACCENT HIGHWAY) -->
    <path d="M 33 35 H 147" fill="none" stroke="#ff8246" stroke-width="2"/>
    <text x="90" y="30" text-anchor="middle" font-size="7" font-weight="700" fill="#c85a17">Clean Skip (dL/dx = dL/dy + ...)</text>

    <!-- Sub-layer branch: LN -> SubLayer -->
    <path d="M 19 46 V 65 H 38" fill="none" stroke="#9ca3af" stroke-width="1"/>
    <rect x="38" y="54" width="36" height="20" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="56" y="67" text-anchor="middle" font-size="6.5" font-family="monospace" fill="#1f2937">LN(x)</text>

    <path d="M 74 65 H 88" stroke="#9ca3af" stroke-width="1"/>
    <rect x="88" y="54" width="44" height="20" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.8"/>
    <text x="110" y="67" text-anchor="middle" font-size="6.5" font-family="monospace" fill="#1f2937">SubLayer</text>

    <!-- Vertical route perfectly aligning with addition circle center cx=155 -->
    <path d="M 132 65 H 155 V 43" fill="none" stroke="#9ca3af" stroke-width="1"/>

    <!-- Addition node -->
    <circle cx="155" cy="35" r="8" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
    <text x="155" y="38" text-anchor="middle" font-size="9" font-weight="bold" fill="#c85a17">+</text>

    <!-- Output y -->
    <path d="M 163 35 H 178" stroke="#ff8246" stroke-width="1.5"/>
    <text x="183" y="38" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">y</text>
  </g>
</svg>
"""
    write_svg("13_transformers-margin-pre-ln.svg", body)


# -----------------------------------------------------------------------------
# 6. milestone_02-margin-temperature.svg (viewBox 0 0 220 115)
# -----------------------------------------------------------------------------
def gen_milestone_02_margin_temperature():
    h = 115
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="{h}" viewBox="0 0 220 {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="220" height="{h}" fill="#ffffff"/>
  <rect x="5" y="5" width="210" height="105" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8.5" font-weight="700" fill="#1f2937">AUTOREGRESSIVE TEMPERATURE (T)</text>

  <g transform="translate(15, 30)">
    <!-- Axes -->
    <line x1="15" y1="55" x2="180" y2="55" stroke="#cbd5e1" stroke-width="1"/>
    <line x1="15" y1="5" x2="15" y2="55" stroke="#cbd5e1" stroke-width="1"/>
    <text x="180" y="66" text-anchor="end" font-size="7" font-family="monospace" fill="#9ca3af">Tokens</text>
    <text x="18" y="10" font-size="7" font-family="monospace" fill="#9ca3af">P(w)</text>

    <!-- T = 0.5 (Sharp peak, argmax-like) -->
    <path d="M 20 55 L 75 55 L 95 10 L 115 55 L 180 55" fill="none" stroke="#ff8246" stroke-width="2"/>
    <text x="95" y="7" text-anchor="middle" font-size="7" font-family="monospace" font-weight="bold" fill="#c85a17">T=0.5 (sharp)</text>

    <!-- T = 1.0 (Baseline) -->
    <path d="M 20 55 Q 60 55 80 40 Q 95 24 110 40 Q 130 55 180 55" fill="none" stroke="#1f2937" stroke-width="1.2" stroke-dasharray="2,2"/>
    <text x="135" y="32" font-size="6.5" fill="#1f2937">T=1.0 (standard)</text>

    <!-- T = 2.0 (Flattened/uniform) -->
    <line x1="20" y1="46" x2="180" y2="46" stroke="#9ca3af" stroke-width="1.2"/>
    <text x="145" y="44" font-size="6.5" fill="#6b7280">T=2.0 (uniform)</text>
  </g>
</svg>
"""
    write_svg("milestone_02-margin-temperature.svg", body)


def main():
    print("Generating Part II (Deep Architectures) Margin Micro-Figures...")
    gen_09_convolutions_margin_im2col()
    gen_10_tokenization_margin_bpe()
    gen_11_embeddings_margin_scatter()
    gen_12_attention_margin_mask()
    gen_13_transformers_margin_pre_ln()
    gen_milestone_02_margin_temperature()
    print("All Part II SVGs generated successfully.")


if __name__ == "__main__":
    main()
