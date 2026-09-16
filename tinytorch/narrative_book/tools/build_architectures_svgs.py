#!/usr/bin/env python3
"""
Generator for TinyTorch Tier 2 (Architectures) Mechanical SVG Diagrams.
Adheres strictly to Vol 3 Figure 1.10 standard & tinytorch/palette.md:
- 680px viewBox width
- Height sized to content + breathing room
- Canvas base is pure white (#ffffff), NO outer frame border stroke
- NO embedded canvas titles or subtitles (Quarto fig-cap owns the caption)
- Subsystem container cards with 24px header bands (rx="2")
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
# 1. Ch 09: 09_im2col-lowering-gemm.svg
# -------------------------------------------------------------
def gen_09_im2col():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Left: Spatial Input Feature Map -->
  <rect x="25" y="20" width="180" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="180" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. SPATIAL INPUT TENSOR</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">Shape: (H=4, W=4), K=2×2, S=1</text>

  <!-- 4x4 Grid -->
  <!-- Row 0 -->
  <rect x="37" y="74" width="32" height="28" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="53" y="92" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#ff8246">x00</text>
  <rect x="73" y="74" width="32" height="28" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="89" y="92" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#ff8246">x01</text>
  <rect x="109" y="74" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="125" y="92" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x02</text>
  <rect x="145" y="74" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="161" y="92" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x03</text>

  <!-- Row 1 -->
  <rect x="37" y="106" width="32" height="28" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="53" y="124" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#ff8246">x10</text>
  <rect x="73" y="106" width="32" height="28" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="89" y="124" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="bold" fill="#ff8246">x11</text>
  <rect x="109" y="106" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="125" y="124" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x12</text>
  <rect x="145" y="106" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="161" y="124" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x13</text>

  <!-- Row 2 -->
  <rect x="37" y="138" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="53" y="156" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x20</text>
  <rect x="73" y="138" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="89" y="156" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x21</text>
  <rect x="109" y="138" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="125" y="156" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x22</text>
  <rect x="145" y="138" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="161" y="156" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x23</text>

  <!-- Row 3 -->
  <rect x="37" y="170" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="53" y="188" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x30</text>
  <rect x="73" y="170" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="89" y="188" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x31</text>
  <rect x="109" y="170" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="125" y="188" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x32</text>
  <rect x="145" y="170" width="32" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="161" y="188" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#374151">x33</text>

  <text x="35" y="222" font-size="8.5" fill="#6b7280">Patch 0: [x00, x01, x10, x11]</text>
  <text x="35" y="238" font-size="8.5" fill="#6b7280">Patch 1: [x01, x02, x11, x12] (overlap)</text>
  <text x="35" y="254" font-size="8.5" fill="#6b7280">Stride 1: 9 receptive fields</text>

  <!-- Arrow: im2col lowering -->
  <line x1="205" y1="145" x2="225" y2="145" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>
  <text x="215" y="136" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">im2col</text>

  <!-- Center: Lowered 2D Matrix (ACCENT) -->
  <rect x="230" y="20" width="220" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="230" y="20" width="220" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="242" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. LOWERED 2D MATRIX (X_COL)</text>
  <text x="242" y="58" font-size="8.5" fill="#6b7280">Shape: (N_patches=9, K_len=4) in DRAM</text>

  <!-- Unrolled rows -->
  <rect x="242" y="74" width="36" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="260" y="87" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x00</text>
  <rect x="281" y="74" width="36" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="299" y="87" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x01</text>
  <rect x="320" y="74" width="36" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="338" y="87" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x10</text>
  <rect x="359" y="74" width="36" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="377" y="87" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x11</text>
  <text x="408" y="87" font-size="8" fill="#6b7280">P0</text>

  <rect x="242" y="97" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="260" y="110" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x01</text>
  <rect x="281" y="97" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="299" y="110" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x02</text>
  <rect x="320" y="97" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="338" y="110" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x11</text>
  <rect x="359" y="97" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="377" y="110" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x12</text>
  <text x="408" y="110" font-size="8" fill="#6b7280">P1</text>

  <rect x="242" y="120" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="260" y="133" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x02</text>
  <rect x="281" y="120" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="299" y="133" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x03</text>
  <rect x="320" y="120" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="338" y="133" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x12</text>
  <rect x="359" y="120" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="377" y="133" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x13</text>
  <text x="408" y="133" font-size="8" fill="#6b7280">P2</text>

  <text x="314" y="157" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">... 6 more patch rows ...</text>

  <rect x="242" y="169" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="260" y="182" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x22</text>
  <rect x="281" y="169" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="299" y="182" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x23</text>
  <rect x="320" y="169" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="338" y="182" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x32</text>
  <rect x="359" y="169" width="36" height="18" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="377" y="182" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x33</text>
  <text x="408" y="182" font-size="8" fill="#6b7280">P8</text>

  <text x="242" y="214" font-size="8.5" fill="#6b7280">Memory replication trade-off:</text>
  <text x="242" y="228" font-size="8.5" fill="#6b7280">Overlapping values copied 4×,</text>
  <text x="242" y="242" font-size="8.5" fill="#6b7280">enabling cache-line aligned GEMM.</text>
  <text x="242" y="256" font-size="8.5" font-weight="bold" fill="#ff8246">Yields 10x-50x speedup over loops.</text>

  <!-- Arrow to GEMM -->
  <line x1="450" y1="145" x2="470" y2="145" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Right: BLAS GEMM Multiply & Reshape -->
  <rect x="475" y="20" width="180" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="475" y="20" width="180" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="487" y="36" font-size="9.5" font-weight="700" fill="#1f2937">3. BLAS GEMM &amp; RESHAPE</text>
  <text x="487" y="58" font-size="8.5" fill="#6b7280">Kernel: (C_out=2, K_len=4)</text>

  <!-- Weight matrix row -->
  <rect x="487" y="74" width="156" height="38" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="495" y="90" font-size="8" font-family="monospace" fill="#1f2937">W_row: (2, 4) flattened</text>
  <text x="495" y="104" font-size="8" font-family="monospace" fill="#1f2937">Out_2D = X_col @ W_row.T</text>

  <line x1="565" y1="120" x2="565" y2="135" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Output 2D & Reshape -->
  <rect x="487" y="142" width="156" height="38" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="495" y="158" font-size="8" font-family="monospace" fill="#1f2937">Out_2D: (9, 2)</text>
  <text x="495" y="172" font-size="8" font-family="monospace" fill="#6b7280">9 patches x 2 filters</text>

  <line x1="565" y1="188" x2="565" y2="203" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <rect x="487" y="210" width="156" height="42" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1"/>
  <text x="495" y="226" font-size="8.5" font-family="monospace" font-weight="bold" fill="#1f2937">.reshape(3, 3, 2)</text>
  <text x="495" y="242" font-size="8.5" fill="#6b7280">Final Tensor (H=3, W=3, C=2)</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">The im2col lowering transforms spatial 2D sliding windows into contiguous row vectors, trading memory for peak BLAS GEMM FLOPs.</text>
{FOOTER}"""
    write_svg("09_im2col-lowering-gemm.svg", body)


# -------------------------------------------------------------
# 2. Ch 10: 10_bpe-merge-collapse.svg
# -------------------------------------------------------------
def gen_10_bpe():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Left: Raw Input Stream & Frequencies -->
  <rect x="25" y="20" width="185" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="185" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. CORPUS FREQUENCIES</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">Word corpus with end token &lt;/w&gt;</text>

  <rect x="35" y="74" width="165" height="42" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="43" y="91" font-size="8.5" font-family="monospace" fill="#1f2937">'l o w e s t &lt;/w&gt;' : 5</text>
  <text x="43" y="106" font-size="8.5" font-family="monospace" fill="#1f2937">'n e w e s t &lt;/w&gt;' : 6</text>

  <rect x="35" y="124" width="165" height="42" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="43" y="141" font-size="8.5" font-family="monospace" fill="#1f2937">'w i d e s t &lt;/w&gt;' : 3</text>
  <text x="43" y="156" font-size="8.5" font-family="monospace" fill="#1f2937">'s o f t e s t &lt;/w&gt;' : 2</text>

  <text x="35" y="188" font-size="8.5" font-weight="bold" fill="#1f2937">Initial Alphabet (Base Vocab):</text>
  <text x="35" y="203" font-size="8" font-family="monospace" fill="#374151">['d', 'e', 'f', 'l', 's', ...]</text>
  <text x="35" y="226" font-size="8.5" fill="#6b7280">Sequence length = raw chars.</text>
  <text x="35" y="242" font-size="8.5" fill="#6b7280">Maximum byte fragmentation.</text>
  <text x="35" y="258" font-size="8.5" fill="#6b7280">Context window exhausted rapidly.</text>

  <!-- Arrow -->
  <line x1="210" y1="147" x2="230" y2="147" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Center: Iterative Merge Sieve (ACCENT) -->
  <rect x="235" y="20" width="215" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="235" y="20" width="215" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="247" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. GREEDY MERGE SIEVE</text>
  <text x="247" y="58" font-size="8.5" fill="#6b7280">Rank-ordered frequency collapse</text>

  <!-- Step 1 -->
  <rect x="247" y="74" width="191" height="46" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="255" y="90" font-size="8" font-family="monospace" fill="#1f2937">Iteration 1: Count bigrams</text>
  <text x="255" y="104" font-size="8.5" font-family="monospace" font-weight="bold" fill="#ff8246">Pair ('e', 's') freq = 16 (MAX)</text>
  <text x="255" y="115" font-size="8" font-family="monospace" fill="#6b7280">Rule #1: ('e', 's') -&gt; 'es'</text>

  <line x1="342" y1="126" x2="342" y2="140" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

  <!-- Step 2 -->
  <rect x="247" y="146" width="191" height="46" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="255" y="162" font-size="8" font-family="monospace" fill="#1f2937">Iteration 2: Re-count</text>
  <text x="255" y="176" font-size="8.5" font-family="monospace" font-weight="bold" fill="#ff8246">Pair ('es', 't') freq = 16 (MAX)</text>
  <text x="255" y="187" font-size="8" font-family="monospace" fill="#6b7280">Rule #2: ('es', 't') -&gt; 'est'</text>

  <text x="247" y="214" font-size="8.5" fill="#6b7280">Merge table stores exact pair rank.</text>
  <text x="247" y="228" font-size="8.5" fill="#6b7280">High-frequency subwords collapse first,</text>
  <text x="247" y="242" font-size="8.5" fill="#6b7280">compressing token sequences by ~3x</text>
  <text x="247" y="256" font-size="8.5" font-weight="bold" fill="#ff8246">without out-of-vocabulary errors.</text>

  <!-- Arrow -->
  <line x1="450" y1="147" x2="470" y2="147" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Right: Subword Representation -->
  <rect x="475" y="20" width="180" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="475" y="20" width="180" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="487" y="36" font-size="9.5" font-weight="700" fill="#1f2937">3. SUBWORD VOCABULARY</text>
  <text x="487" y="58" font-size="8.5" fill="#6b7280">Fixed vocabulary token IDs</text>

  <rect x="487" y="74" width="156" height="42" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="495" y="91" font-size="8" font-family="monospace" fill="#1f2937">'lowest' -&gt; ['low', 'est']</text>
  <text x="495" y="106" font-size="8.5" font-family="monospace" font-weight="bold" fill="#1f2937">IDs: [104, 302]</text>

  <rect x="487" y="124" width="156" height="42" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="495" y="141" font-size="8" font-family="monospace" fill="#1f2937">'newest' -&gt; ['new', 'est']</text>
  <text x="495" y="156" font-size="8.5" font-family="monospace" font-weight="bold" fill="#1f2937">IDs: [215, 302]</text>

  <text x="487" y="188" font-size="8.5" font-weight="bold" fill="#1f2937">Graceful OOV Handling:</text>
  <text x="487" y="204" font-size="8.5" fill="#6b7280">Unseen words decompose</text>
  <text x="487" y="220" font-size="8.5" fill="#6b7280">into known subword tiles</text>
  <text x="487" y="236" font-size="8.5" fill="#6b7280">or individual byte tokens,</text>
  <text x="487" y="252" font-size="8.5" font-weight="bold" fill="#1f2937">never throwing [UNK].</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">BPE constructs a greedy subword vocabulary by iteratively contracting the most frequent adjacent bigrams into unified tokens.</text>
{FOOTER}"""
    write_svg("10_bpe-merge-collapse.svg", body)


# -------------------------------------------------------------
# 3. Ch 11: 11_embedding-gather-scatter.svg
# -------------------------------------------------------------
def gen_11_embedding():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Left: Forward Gather O(1) -->
  <rect x="25" y="20" width="295" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="295" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. FORWARD: POINTER GATHER (O(1))</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">Batch token indices: [42, 7, 42] (len=3, vocab=50k, dim=64)</text>

  <!-- Weight Table -->
  <rect x="35" y="74" width="125" height="118" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="43" y="90" font-size="8" font-weight="bold" fill="#1f2937">Weight Matrix W</text>
  <rect x="41" y="98" width="113" height="16" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="45" y="110" font-size="7.5" font-family="monospace" fill="#6b7280">row 0: [...]</text>
  <rect x="41" y="118" width="113" height="17" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="45" y="130" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">row 7: [0.12, -0.4...]</text>
  <rect x="41" y="139" width="113" height="17" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="45" y="151" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">row 42: [0.85, 0.31...]</text>
  <text x="45" y="178" font-size="7.5" font-family="monospace" fill="#9ca3af">... row 49999 [...]</text>

  <!-- Gather lines -->
  <line x1="160" y1="147" x2="190" y2="112" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>
  <line x1="160" y1="126" x2="190" y2="135" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>
  <line x1="160" y1="147" x2="190" y2="158" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Output Tensor -->
  <rect x="195" y="74" width="115" height="118" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="203" y="90" font-size="8" font-weight="bold" fill="#1f2937">Output Tensor Y</text>
  <rect x="201" y="98" width="103" height="18" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="205" y="111" font-size="8" font-family="monospace" fill="#1f2937">Y[0] &lt;- W[42]</text>
  <rect x="201" y="122" width="103" height="18" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="205" y="135" font-size="8" font-family="monospace" fill="#1f2937">Y[1] &lt;- W[7]</text>
  <rect x="201" y="146" width="103" height="18" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="205" y="159" font-size="8" font-family="monospace" fill="#1f2937">Y[2] &lt;- W[42]</text>

  <text x="35" y="214" font-size="8.5" fill="#6b7280">Zero FLOPs: pure memory bus copy.</text>
  <text x="35" y="228" font-size="8.5" fill="#6b7280">Bypasses 50,000-wide one-hot GEMM,</text>
  <text x="35" y="242" font-size="8.5" fill="#6b7280">reducing DRAM bandwidth by 16,000x.</text>

  <!-- Right: Backward Scatter-Add (ACCENT) -->
  <rect x="335" y="20" width="320" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="335" y="20" width="320" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="347" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. BACKWARD: ATOMIC SCATTER-ADD</text>
  <text x="347" y="58" font-size="8.5" fill="#6b7280">Grad Output dL/dY arrives for indices [42, 7, 42]</text>

  <!-- Incoming grads -->
  <rect x="347" y="74" width="110" height="118" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="355" y="90" font-size="8" font-weight="bold" fill="#1f2937">Grad Output dL/dY</text>
  <rect x="353" y="98" width="98" height="18" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="357" y="111" font-size="7.5" font-family="monospace" fill="#1f2937">dY[0] (for 42)</text>
  <rect x="353" y="122" width="98" height="18" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="357" y="135" font-size="7.5" font-family="monospace" fill="#1f2937">dY[1] (for 7)</text>
  <rect x="353" y="146" width="98" height="18" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="357" y="159" font-size="7.5" font-family="monospace" fill="#1f2937">dY[2] (for 42)</text>

  <!-- Scatter arrows -->
  <line x1="462" y1="110" x2="495" y2="152" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>
  <line x1="462" y1="131" x2="495" y2="131" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>
  <line x1="462" y1="155" x2="495" y2="155" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

  <!-- Grad Weight Table Accumulation -->
  <rect x="500" y="74" width="145" height="118" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="508" y="90" font-size="8" font-weight="bold" fill="#1f2937">Grad Weight dL/dW</text>
  <rect x="506" y="98" width="133" height="16" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="510" y="110" font-size="7.5" font-family="monospace" fill="#6b7280">row 0: 0.0 (untouched)</text>
  <rect x="506" y="118" width="133" height="17" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="510" y="130" font-size="7.5" font-family="monospace" fill="#1f2937">row 7: += dY[1]</text>
  <rect x="506" y="139" width="133" height="20" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="510" y="153" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">row 42: += dY[0] + dY[2]</text>
  <text x="510" y="178" font-size="7.5" font-family="monospace" fill="#9ca3af">... sparse zeros ...</text>

  <text x="347" y="214" font-size="8.5" font-weight="bold" fill="#1f2937">THE DUPLICATE INDEX TRAP:</text>
  <text x="347" y="228" font-size="8.5" fill="#6b7280">Naive `dW[idx] = dY` overwrites row 42, dropping dY[0]!</text>
  <text x="347" y="242" font-size="8.5" fill="#6b7280">Must use `np.add.at(dW, idx, dY)` for atomic accumulation.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Forward embedding lookup operates via zero-FLOP pointer indexing; backward propagation requires atomic scatter-add to handle duplicate tokens.</text>
{FOOTER}"""
    write_svg("11_embedding-gather-scatter.svg", body)


# -------------------------------------------------------------
# 4. Ch 12: 12_causal-attention-engine.svg
# -------------------------------------------------------------
def gen_12_attention():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Left: Raw Scores -->
  <rect x="25" y="20" width="185" height="265" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="185" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. RAW SCORES (Q · Kᵀ / √d_k)</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">S = Q @ K.T / sqrt(d_k) [4, 4]</text>

  <!-- 4x4 Grid -->
  <rect x="42" y="74" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="58" y="91" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.2</text>
  <rect x="78" y="74" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="94" y="91" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.4</text>
  <rect x="114" y="74" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="130" y="91" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.8</text>
  <rect x="150" y="74" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="166" y="91" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.9</text>

  <rect x="42" y="104" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="58" y="121" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.8</text>
  <rect x="78" y="104" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="94" y="121" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">2.1</text>
  <rect x="114" y="104" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="130" y="121" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.3</text>
  <rect x="150" y="104" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="166" y="121" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.5</text>

  <rect x="42" y="134" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="58" y="151" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.5</text>
  <rect x="78" y="134" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="94" y="151" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.7</text>
  <rect x="114" y="134" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="130" y="151" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">2.4</text>
  <rect x="150" y="134" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="166" y="151" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.1</text>

  <rect x="42" y="164" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="58" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.3</text>
  <rect x="78" y="164" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="94" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.1</text>
  <rect x="114" y="164" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="130" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.9</text>
  <rect x="150" y="164" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="166" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">3.2</text>

  <text x="35" y="214" font-size="8.5" fill="#6b7280">Upper triangle contains illegal</text>
  <text x="35" y="228" font-size="8.5" fill="#6b7280">lookahead scores: token t attending</text>
  <text x="35" y="242" font-size="8.5" fill="#6b7280">to future tokens t+1, t+2...</text>
  <text x="35" y="256" font-size="8.5" font-weight="bold" fill="#1f2937">Must be causally zeroed.</text>

  <!-- Arrow -->
  <line x1="210" y1="152" x2="230" y2="152" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Center: Causal Masking -->
  <rect x="235" y="20" width="190" height="265" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="235" y="20" width="190" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="247" y="36" font-size="9.5" font-weight="700" fill="#1f2937">2. CAUSAL MASK (-inf)</text>
  <text x="247" y="58" font-size="8.5" fill="#6b7280">Upper triangle clamped to -1e9</text>

  <!-- Masked Grid -->
  <rect x="249" y="74" width="34" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="266" y="91" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.2</text>
  <rect x="287" y="74" width="34" height="26" rx="1" fill="#e5e7eb" stroke="#cbd5e1" stroke-width="1"/><text x="304" y="91" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>
  <rect x="325" y="74" width="34" height="26" rx="1" fill="#e5e7eb" stroke="#cbd5e1" stroke-width="1"/><text x="342" y="91" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>
  <rect x="363" y="74" width="34" height="26" rx="1" fill="#e5e7eb" stroke="#cbd5e1" stroke-width="1"/><text x="380" y="91" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>

  <rect x="249" y="104" width="34" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="266" y="121" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.8</text>
  <rect x="287" y="104" width="34" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="304" y="121" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">2.1</text>
  <rect x="325" y="104" width="34" height="26" rx="1" fill="#e5e7eb" stroke="#cbd5e1" stroke-width="1"/><text x="342" y="121" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>
  <rect x="363" y="104" width="34" height="26" rx="1" fill="#e5e7eb" stroke="#cbd5e1" stroke-width="1"/><text x="380" y="121" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>

  <rect x="249" y="134" width="34" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="266" y="151" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.5</text>
  <rect x="287" y="134" width="34" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="304" y="151" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.7</text>
  <rect x="325" y="134" width="34" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="342" y="151" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">2.4</text>
  <rect x="363" y="134" width="34" height="26" rx="1" fill="#e5e7eb" stroke="#cbd5e1" stroke-width="1"/><text x="380" y="151" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>

  <rect x="249" y="164" width="34" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="266" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.3</text>
  <rect x="287" y="164" width="34" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="304" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.1</text>
  <rect x="325" y="164" width="34" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="342" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.9</text>
  <rect x="363" y="164" width="34" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="380" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">3.2</text>

  <text x="247" y="214" font-size="8.5" fill="#6b7280">Lower triangle preserved.</text>
  <text x="247" y="228" font-size="8.5" fill="#6b7280">In float16, use -1e4 to avoid</text>
  <text x="247" y="242" font-size="8.5" fill="#6b7280">NaN underflow in exp().</text>

  <!-- Arrow -->
  <line x1="425" y1="152" x2="445" y2="152" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>

  <!-- Right: Softmax Row Normalization (ACCENT) -->
  <rect x="450" y="20" width="205" height="265" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="450" y="20" width="205" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="462" y="36" font-size="9.5" font-weight="700" fill="#c85a17">3. SOFTMAX WEIGHTS (A)</text>
  <text x="462" y="58" font-size="8.5" fill="#6b7280">exp(-inf)=0; each row sums to 1.0</text>

  <!-- Probability Grid -->
  <rect x="462" y="74" width="32" height="26" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/><text x="478" y="91" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">1.00</text>
  <rect x="498" y="74" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="514" y="91" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <rect x="534" y="74" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="550" y="91" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <rect x="570" y="74" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="586" y="91" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <text x="612" y="91" font-size="8" fill="#6b7280">T0</text>

  <rect x="462" y="104" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="478" y="121" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.21</text>
  <rect x="498" y="104" width="32" height="26" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/><text x="514" y="121" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">0.79</text>
  <rect x="534" y="104" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="550" y="121" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <rect x="570" y="104" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="586" y="121" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <text x="612" y="121" font-size="8" fill="#6b7280">T1</text>

  <rect x="462" y="134" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="478" y="151" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.27</text>
  <rect x="498" y="134" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="514" y="151" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.12</text>
  <rect x="534" y="134" width="32" height="26" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/><text x="550" y="151" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">0.61</text>
  <rect x="570" y="134" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="586" y="151" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <text x="612" y="151" font-size="8" fill="#6b7280">T2</text>

  <rect x="462" y="164" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="478" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.04</text>
  <rect x="498" y="164" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="514" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.10</text>
  <rect x="534" y="164" width="32" height="26" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/><text x="550" y="181" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.08</text>
  <rect x="570" y="164" width="32" height="26" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/><text x="586" y="181" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">0.78</text>
  <text x="612" y="181" font-size="8" fill="#6b7280">T3</text>

  <text x="462" y="214" font-size="8.5" fill="#6b7280">Final Context: Out = A @ V</text>
  <text x="462" y="228" font-size="8.5" fill="#6b7280">Zero information leakage from future</text>
  <text x="462" y="242" font-size="8.5" fill="#6b7280">positions guarantees causal time-order</text>
  <text x="462" y="256" font-size="8.5" font-weight="bold" fill="#ff8246">during autoregressive decoding.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="306" text-anchor="middle" font-size="9" fill="#6b7280">Causal attention enforces thermodynamic time-causality by injecting -∞ into upper-triangular affinity scores prior to row-wise softmax normalization.</text>
{FOOTER}"""
    write_svg("12_causal-attention-engine.svg", body)


# -------------------------------------------------------------
# 5. Ch 13: 13_residual-stream-bus.svg
# -------------------------------------------------------------
def gen_13_transformer():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Central Highway Container (ACCENT) -->
  <rect x="25" y="20" width="630" height="90" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="25" y="20" width="630" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#c85a17">RESIDUAL HIGHWAY STATE BUS: [BATCH, SEQ, D_MODEL]</text>
  <text x="430" y="36" font-size="8.5" font-family="monospace" font-weight="bold" fill="#c85a17">x_(l+1) = x_l + F_attn + F_mlp</text>

  <!-- State x_l Node -->
  <rect x="40" y="58" width="60" height="34" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="70" y="79" text-anchor="middle" font-size="9.5" font-family="monospace" font-weight="bold" fill="#ff8246">x_l</text>

  <!-- Flow along highway to First Sum -->
  <line x1="100" y1="75" x2="238" y2="75" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>
  <text x="169" y="70" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">Identity Path</text>

  <!-- Addition Node 1 -->
  <circle cx="250" cy="75" r="10" fill="#ffffff" stroke="#ff8246" stroke-width="1.5"/>
  <text x="246" y="79" font-size="11" font-family="monospace" font-weight="bold" fill="#ff8246">+</text>

  <!-- Flow along highway to Second Sum -->
  <line x1="260" y1="75" x2="458" y2="75" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>
  <text x="355" y="70" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">x'_l = x_l + delta_attn</text>

  <!-- Addition Node 2 -->
  <circle cx="470" cy="75" r="10" fill="#ffffff" stroke="#ff8246" stroke-width="1.5"/>
  <text x="466" y="79" font-size="11" font-family="monospace" font-weight="bold" fill="#ff8246">+</text>

  <!-- Flow to State x_(l+1) Node -->
  <line x1="480" y1="75" x2="568" y2="75" stroke="#ff8246" stroke-width="1.2" marker-end="url(#arrow-orange)"/>
  <rect x="570" y="58" width="70" height="34" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="605" y="79" text-anchor="middle" font-size="9.5" font-family="monospace" font-weight="bold" fill="#ff8246">x_(l+1)</text>

  <!-- Sublayer 1: Multi-Head Attention (Pre-LN) -->
  <rect x="135" y="125" width="200" height="82" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="135" y="125" width="200" height="22" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="145" y="140" font-size="9" font-weight="700" fill="#1f2937">SUBLAYER 1: ATTENTION OFF-RAMP</text>
  <text x="145" y="162" font-size="8" font-family="monospace" fill="#1f2937">1. h = LayerNorm(x_l)</text>
  <text x="145" y="176" font-size="8" font-family="monospace" fill="#1f2937">2. delta_attn = MHA(h)</text>
  <text x="145" y="196" font-size="8" fill="#6b7280">Reads stream, emits delta update</text>

  <!-- Tap off highway to LN, and add back -->
  <polyline points="70,92 70,166 135,166" fill="none" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>
  <line x1="250" y1="125" x2="250" y2="87" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Sublayer 2: Feed-Forward Network (MLP) -->
  <rect x="365" y="125" width="200" height="82" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="365" y="125" width="200" height="22" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="375" y="140" font-size="9" font-weight="700" fill="#1f2937">SUBLAYER 2: MLP OFF-RAMP</text>
  <text x="375" y="162" font-size="8" font-family="monospace" fill="#1f2937">1. h' = LayerNorm(x'_l)</text>
  <text x="375" y="176" font-size="8" font-family="monospace" fill="#1f2937">2. delta_mlp = FFN(h')</text>
  <text x="375" y="196" font-size="8" fill="#6b7280">Expanded dimension: 4 * d_model</text>

  <!-- Tap off highway to MLP, and add back -->
  <polyline points="330,75 330,166 365,166" fill="none" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>
  <line x1="470" y1="125" x2="470" y2="87" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow-gray)"/>

  <!-- Bottom: Gradient Highway Identity Backprop -->
  <rect x="25" y="222" width="630" height="58" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="35" y="240" font-size="9.5" font-weight="700" fill="#1f2937">Gradient Identity Highway: dL / dx_0 = dL / dx_L * ( I + sum( dF_l / dx_l ) )</text>
  <text x="35" y="256" font-size="8.5" fill="#6b7280">Because of addition (+), the identity matrix I carries the error signal directly back to token embeddings without exponential decay,</text>
  <text x="35" y="270" font-size="8.5" fill="#6b7280">enabling successful convergence in 100+ layer architectures.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="298" text-anchor="middle" font-size="9" fill="#6b7280">Transformer residual connections act as a linear communication bus; sub-layers read from and write additive deltas to the continuous stream.</text>
{FOOTER}"""
    write_svg("13_residual-stream-bus.svg", body)


# -------------------------------------------------------------
# 6. Milestone 02: milestone_02_teacher-forcing-grid.svg
# -------------------------------------------------------------
def gen_milestone_02():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Left Half: Training with Teacher Forcing -->
  <rect x="25" y="20" width="300" height="255" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="25" y="20" width="300" height="24" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="35" y="36" font-size="9.5" font-weight="700" fill="#1f2937">1. TRAINING: PARALLEL TEACHER FORCING</text>
  <text x="35" y="58" font-size="8.5" fill="#6b7280">Target sequence known in advance; O(1) forward steps</text>

  <!-- Input sequence -->
  <rect x="35" y="74" width="280" height="32" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="43" y="94" font-size="8.5" font-family="monospace" fill="#1f2937">Inputs X:  ['The', 'quick', 'brown', 'fox']</text>

  <!-- Single Forward Pass Block -->
  <rect x="35" y="114" width="280" height="38" rx="1" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1"/>
  <text x="43" y="130" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">Parallel Forward: Causal Mask (Seq=4)</text>
  <text x="43" y="144" font-size="8" fill="#6b7280">All 4 positions evaluated simultaneously in 1 kernel call</text>

  <!-- Shifted Targets and Loss -->
  <rect x="35" y="160" width="280" height="42" rx="1" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
  <text x="43" y="176" font-size="8" font-family="monospace" fill="#1f2937">Targets Y: ['quick', 'brown', 'fox', '&lt;eos&gt;'] (+1)</text>
  <text x="43" y="192" font-size="7.5" font-family="monospace" fill="#6b7280">Loss = CrossEntropy(Logits.view(-1, V), Y.view(-1))</text>

  <text x="35" y="222" font-size="8.5" font-weight="bold" fill="#1f2937">Parallel Efficiency:</text>
  <text x="35" y="238" font-size="8.5" fill="#6b7280">Hardware tensor cores run at 100% saturation.</text>
  <text x="35" y="254" font-size="8.5" fill="#6b7280">Ground truth fed at each step regardless of errors.</text>

  <!-- Right Half: Inference Autoregressive Loop (ACCENT) -->
  <rect x="345" y="20" width="310" height="255" rx="2" fill="#ffffff" stroke="#ff8246" stroke-width="1.2"/>
  <rect x="345" y="20" width="310" height="24" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="357" y="36" font-size="9.5" font-weight="700" fill="#c85a17">2. INFERENCE: SEQUENTIAL GENERATION</text>
  <text x="357" y="58" font-size="8.5" fill="#6b7280">Tokens sampled sequentially; O(N) iterative steps</text>

  <!-- Step 0 -->
  <rect x="357" y="74" width="286" height="26" rx="1" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
  <text x="365" y="91" font-size="8" font-family="monospace" fill="#1f2937">Step 1: Input ['The'] -&gt; Sample 'quick'</text>

  <!-- Step 1 -->
  <rect x="357" y="106" width="286" height="26" rx="1" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
  <text x="365" y="123" font-size="8" font-family="monospace" fill="#1f2937">Step 2: Input ['The', 'quick'] -&gt; Sample 'brown'</text>

  <!-- Step 2 -->
  <rect x="357" y="138" width="286" height="26" rx="1" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
  <text x="365" y="155" font-size="8" font-family="monospace" fill="#1f2937">Step 3: Input ['The', 'quick', 'brown'] -&gt; Sample 'fox'</text>

  <!-- Step 3 Loop -->
  <rect x="357" y="170" width="286" height="26" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="365" y="187" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">Step 4: Input [..., 'fox'] -&gt; Sample '&lt;eos&gt;' (STOP)</text>

  <text x="357" y="216" font-size="8.5" font-weight="bold" fill="#1f2937">Memory-Bound Bottleneck:</text>
  <text x="357" y="232" font-size="8.5" fill="#6b7280">Batch size = 1 token per step. Arithmetic intensity R &lt;&lt; 1.</text>
  <text x="357" y="248" font-size="8.5" fill="#6b7280">Requires KV Caching (Module 18) to avoid O(N²) recomputation.</text>

  <!-- Bottom Annotation -->
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#6b7280">Training leverages parallel teacher forcing across all sequence tokens; inference requires iterative autoregressive decoding.</text>
{FOOTER}"""
    write_svg("milestone_02_teacher-forcing-grid.svg", body)


if __name__ == "__main__":
    gen_09_im2col()
    gen_10_bpe()
    gen_11_embedding()
    gen_12_attention()
    gen_13_transformer()
    gen_milestone_02()
    print("Tier 2 Mechanical SVGs generated successfully.")
