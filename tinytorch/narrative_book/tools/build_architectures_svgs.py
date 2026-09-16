#!/usr/bin/env python3
"""
Generator for TinyTorch Tier 2 (Architectures) Mechanical SVG Diagrams.
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
# 1. Ch 09: 09_im2col-lowering-gemm.svg
# -------------------------------------------------------------
def gen_09_im2col():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">SPATIAL CONVOLUTION TO BLAS GEMM: THE IM2COL LOWERING</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Lowering overlapping 2D receptive fields into a 2D dense matrix: trading memory overhead for peak hardware FLOPs</text>

  <!-- Left: Spatial Input Feature Map -->
  <rect x="35" y="65" width="165" height="225" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">1. Spatial Input Tensor</text>
  <text x="45" y="96" font-size="8.5" fill="#6b7280">Shape: (H=4, W=4), Kernel: 2x2, S=1</text>

  <!-- 4x4 Grid -->
  <!-- Row 0 -->
  <rect x="47" y="112" width="28" height="28" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="61" y="130" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x00</text>
  <rect x="78" y="112" width="28" height="28" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="92" y="130" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x01</text>
  <rect x="109" y="112" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="123" y="130" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x02</text>
  <rect x="140" y="112" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="154" y="130" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x03</text>

  <!-- Row 1 -->
  <rect x="47" y="143" width="28" height="28" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="61" y="161" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x10</text>
  <rect x="78" y="143" width="28" height="28" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="92" y="161" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x11</text>
  <rect x="109" y="143" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="123" y="161" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x12</text>
  <rect x="140" y="143" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="154" y="161" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x13</text>

  <!-- Row 2 -->
  <rect x="47" y="174" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="61" y="192" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x20</text>
  <rect x="78" y="174" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="92" y="192" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x21</text>
  <rect x="109" y="174" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="123" y="192" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x22</text>
  <rect x="140" y="174" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="154" y="192" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x23</text>

  <!-- Row 3 -->
  <rect x="47" y="205" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="61" y="223" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x30</text>
  <rect x="78" y="205" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="92" y="223" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x31</text>
  <rect x="109" y="205" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="123" y="223" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x32</text>
  <rect x="140" y="205" width="28" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="154" y="223" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x33</text>

  <text x="45" y="254" font-size="8" fill="#6b7280">Patch 0: [x00, x01, x10, x11]</text>
  <text x="45" y="268" font-size="8" fill="#6b7280">Patch 1: [x01, x02, x11, x12] (overlap!)</text>

  <!-- Arrow: im2col lowering -->
  <line x1="200" y1="177" x2="218" y2="177" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>
  <text x="210" y="168" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">im2col</text>

  <!-- Center: Lowered 2D Matrix (ACCENT) -->
  <rect x="225" y="65" width="220" height="225" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="237" y="83" font-size="10" font-weight="700" fill="#1f2937">2. Lowered 2D Matrix (X_col)</text>
  <text x="237" y="96" font-size="8.5" fill="#6b7280">Shape: (N_patches=9, K_h*K_w=4) in DRAM</text>

  <!-- Unrolled rows -->
  <rect x="237" y="112" width="36" height="18" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="255" y="125" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x00</text>
  <rect x="276" y="112" width="36" height="18" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="294" y="125" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x01</text>
  <rect x="315" y="112" width="36" height="18" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="333" y="125" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x10</text>
  <rect x="354" y="112" width="36" height="18" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="372" y="125" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x11</text>
  <text x="400" y="125" font-size="8" fill="#6b7280">P0</text>

  <rect x="237" y="133" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="255" y="146" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x01</text>
  <rect x="276" y="133" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="294" y="146" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x02</text>
  <rect x="315" y="133" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="333" y="146" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x11</text>
  <rect x="354" y="133" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="372" y="146" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x12</text>
  <text x="400" y="146" font-size="8" fill="#6b7280">P1</text>

  <rect x="237" y="154" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="255" y="167" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x02</text>
  <rect x="276" y="154" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="294" y="167" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x03</text>
  <rect x="315" y="154" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="333" y="167" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x12</text>
  <rect x="354" y="154" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="372" y="167" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x13</text>
  <text x="400" y="167" font-size="8" fill="#6b7280">P2</text>

  <text x="314" y="188" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">... 6 more patch rows ...</text>

  <rect x="237" y="198" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="255" y="211" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x22</text>
  <rect x="276" y="198" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="294" y="211" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x23</text>
  <rect x="315" y="198" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="333" y="211" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x32</text>
  <rect x="354" y="198" width="36" height="18" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="372" y="211" text-anchor="middle" font-size="8" font-family="monospace" fill="#374151">x33</text>
  <text x="400" y="211" font-size="8" fill="#6b7280">P8</text>

  <text x="237" y="242" font-size="8.5" fill="#6b7280">Redundant copies: x01 replicated 4x</text>
  <text x="237" y="256" font-size="8.5" fill="#6b7280">Enables cache-line aligned GEMM,</text>
  <text x="237" y="270" font-size="8.5" fill="#6b7280">delivering 10x-50x speedup over loops.</text>

  <!-- Arrow to GEMM -->
  <line x1="445" y1="177" x2="465" y2="177" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Right: BLAS GEMM Multiply & Reshape -->
  <rect x="475" y="65" width="170" height="225" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="487" y="83" font-size="10" font-weight="700" fill="#1f2937">3. GEMM &amp; Reshape</text>
  <text x="487" y="96" font-size="8.5" fill="#6b7280">Kernel: (C_out=2, K_len=4)</text>

  <!-- Weight matrix row -->
  <rect x="487" y="112" width="146" height="34" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="495" y="126" font-size="8" font-family="monospace" fill="#1f2937">W_row: (2, 4) flattened</text>
  <text x="495" y="138" font-size="8" font-family="monospace" fill="#1f2937">Out_2D = X_col @ W_row.T</text>

  <line x1="560" y1="152" x2="560" y2="170" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Output 2D & Reshape -->
  <rect x="487" y="174" width="146" height="34" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="495" y="188" font-size="8" font-family="monospace" fill="#1f2937">Out_2D: (9, 2)</text>
  <text x="495" y="200" font-size="8" font-family="monospace" fill="#6b7280">9 patches x 2 filters</text>

  <line x1="560" y1="214" x2="560" y2="232" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <rect x="487" y="236" width="146" height="38" rx="1" fill="#f8f9fa" stroke="#d1d5db" stroke-width="1"/>
  <text x="495" y="250" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">.reshape(3, 3, 2)</text>
  <text x="495" y="264" font-size="8.5" fill="#6b7280">Final Tensor (H_out=3, W_out=3)</text>
{FOOTER}"""
    write_svg("09_im2col-lowering-gemm.svg", body)


# -------------------------------------------------------------
# 2. Ch 10: 10_bpe-merge-collapse.svg
# -------------------------------------------------------------
def gen_10_bpe():
    h = 310
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="280" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">BYTE-PAIR ENCODING (BPE): STATISTICAL PAIR COLLAPSE</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Iterative greedy vocabulary expansion: transforming raw unicode character sequences into subword tokens</text>

  <!-- Left: Raw Input Stream & Frequencies -->
  <rect x="35" y="65" width="180" height="215" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">1. Corpus Frequency Table</text>
  <text x="45" y="96" font-size="8.5" fill="#6b7280">Word corpus with end token &lt;/w&gt;</text>

  <rect x="45" y="112" width="160" height="38" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="52" y="127" font-size="8" font-family="monospace" fill="#1f2937">'l o w e s t &lt;/w&gt;' : 5</text>
  <text x="52" y="141" font-size="8" font-family="monospace" fill="#1f2937">'n e w e s t &lt;/w&gt;' : 6</text>

  <rect x="45" y="156" width="160" height="38" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="52" y="171" font-size="8" font-family="monospace" fill="#1f2937">'w i d e s t &lt;/w&gt;' : 3</text>
  <text x="52" y="185" font-size="8" font-family="monospace" fill="#1f2937">'s o f t e s t &lt;/w&gt;' : 2</text>

  <text x="45" y="216" font-size="8.5" font-weight="bold" fill="#1f2937">Initial Alphabet (Base Vocab):</text>
  <text x="45" y="230" font-size="7.5" font-family="monospace" fill="#374151">['d', 'e', 'f', 'i', 'l', 'n', 'o', 's', ...]</text>
  <text x="45" y="248" font-size="8" fill="#6b7280">Sequence length = sum of characters</text>
  <text x="45" y="262" font-size="8" fill="#6b7280">Maximum byte fragmentation</text>

  <!-- Arrow -->
  <line x1="215" y1="172" x2="235" y2="172" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Center: Iterative Merge Sieve (ACCENT) -->
  <rect x="245" y="65" width="205" height="215" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="257" y="83" font-size="10" font-weight="700" fill="#1f2937">2. Greedy Bigram Merge Sieve</text>
  <text x="257" y="96" font-size="8.5" fill="#6b7280">Rank-ordered frequency reduction</text>

  <!-- Step 1 -->
  <rect x="257" y="112" width="181" height="42" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="264" y="126" font-size="8" font-family="monospace" fill="#1f2937">Iteration 1: Count bigrams</text>
  <text x="264" y="139" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">Pair ('e', 's') freq = 16 (MAX)</text>
  <text x="264" y="150" font-size="7.5" font-family="monospace" fill="#6b7280">Rule #1: ('e', 's') -&gt; 'es'</text>

  <line x1="347" y1="158" x2="347" y2="170" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>

  <!-- Step 2 -->
  <rect x="257" y="174" width="181" height="42" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="264" y="188" font-size="8" font-family="monospace" fill="#1f2937">Iteration 2: Re-count</text>
  <text x="264" y="201" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">Pair ('es', 't') freq = 16 (MAX)</text>
  <text x="264" y="212" font-size="7.5" font-family="monospace" fill="#6b7280">Rule #2: ('es', 't') -&gt; 'est'</text>

  <text x="257" y="240" font-size="8.5" fill="#6b7280">Merge table stores exact pair priority.</text>
  <text x="257" y="254" font-size="8.5" fill="#6b7280">High-frequency subwords collapse first,</text>
  <text x="257" y="268" font-size="8.5" fill="#6b7280">compressing token sequences by ~3x.</text>

  <!-- Arrow -->
  <line x1="450" y1="172" x2="475" y2="172" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Right: Subword Representation -->
  <rect x="485" y="65" width="160" height="215" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="495" y="83" font-size="10" font-weight="700" fill="#1f2937">3. Subword Representation</text>
  <text x="495" y="96" font-size="8.5" fill="#6b7280">Fixed vocabulary token IDs</text>

  <rect x="495" y="112" width="140" height="38" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="502" y="127" font-size="8" font-family="monospace" fill="#1f2937">'lowest' -&gt; ['low', 'est']</text>
  <text x="502" y="141" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">IDs: [104, 302]</text>

  <rect x="495" y="156" width="140" height="38" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="502" y="171" font-size="8" font-family="monospace" fill="#1f2937">'newest' -&gt; ['new', 'est']</text>
  <text x="502" y="185" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">IDs: [215, 302]</text>

  <text x="495" y="216" font-size="8.5" font-weight="bold" fill="#1f2937">Graceful OOV handling:</text>
  <text x="495" y="230" font-size="8" fill="#6b7280">Unseen words decompose into</text>
  <text x="495" y="244" font-size="8" fill="#6b7280">known subwords or bytes,</text>
  <text x="495" y="258" font-size="8" fill="#6b7280">never throwing [UNK] errors.</text>
{FOOTER}"""
    write_svg("10_bpe-merge-collapse.svg", body)


# -------------------------------------------------------------
# 3. Ch 11: 11_embedding-gather-scatter.svg
# -------------------------------------------------------------
def gen_11_embedding():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">EMBEDDING LOOKUP: FORWARD GATHER VS BACKWARD SCATTER-ADD</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Why embedding is not matrix multiplication: O(1) memory pointers forward, atomic scatter accumulation backward</text>

  <!-- Left: Forward Gather O(1) -->
  <rect x="35" y="65" width="280" height="225" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">1. Forward Pass: Pointer Gather (Y = W[idx])</text>
  <text x="45" y="96" font-size="8.5" fill="#6b7280">Batch token indices: [42, 7, 42] (len=3, vocab=50000, dim=64)</text>

  <!-- Weight Table -->
  <rect x="45" y="110" width="115" height="110" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="52" y="124" font-size="8" font-weight="bold" fill="#1f2937">Weight Matrix W</text>
  <rect x="49" y="130" width="107" height="15" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="53" y="141" font-size="7.5" font-family="monospace" fill="#6b7280">row 0: [...]</text>
  <rect x="49" y="148" width="107" height="16" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="53" y="160" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">row 7: [0.12, -0.4...]</text>
  <rect x="49" y="167" width="107" height="16" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="53" y="179" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">row 42: [0.85, 0.31...]</text>
  <text x="53" y="206" font-size="7.5" font-family="monospace" fill="#9ca3af">... row 49999 [...]</text>

  <!-- Gather lines -->
  <line x1="160" y1="175" x2="195" y2="140" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>
  <line x1="160" y1="156" x2="195" y2="162" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>
  <line x1="160" y1="175" x2="195" y2="185" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Output Tensor -->
  <rect x="200" y="110" width="105" height="110" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="206" y="124" font-size="8" font-weight="bold" fill="#1f2937">Output Tensor Y</text>
  <rect x="204" y="131" width="97" height="16" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="208" y="143" font-size="7.5" font-family="monospace" fill="#1f2937">Y[0] &lt;- W[42]</text>
  <rect x="204" y="153" width="97" height="16" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="208" y="165" font-size="7.5" font-family="monospace" fill="#1f2937">Y[1] &lt;- W[7]</text>
  <rect x="204" y="175" width="97" height="16" rx="1" fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>
  <text x="208" y="187" font-size="7.5" font-family="monospace" fill="#1f2937">Y[2] &lt;- W[42]</text>

  <text x="45" y="240" font-size="8.5" fill="#6b7280">Zero FLOPs: pure memory bus copy.</text>
  <text x="45" y="254" font-size="8.5" fill="#6b7280">Bypasses 50,000-wide one-hot GEMM,</text>
  <text x="45" y="268" font-size="8.5" fill="#6b7280">reducing DRAM bandwidth by 16,000x.</text>

  <!-- Right: Backward Scatter-Add (ACCENT) -->
  <rect x="330" y="65" width="315" height="225" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="342" y="83" font-size="10" font-weight="700" fill="#1f2937">2. Backward Pass: Atomic Scatter-Add</text>
  <text x="342" y="96" font-size="8.5" fill="#6b7280">Grad Output dL/dY arrives for indices [42, 7, 42]</text>

  <!-- Incoming grads -->
  <rect x="342" y="110" width="105" height="110" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="348" y="124" font-size="8" font-weight="bold" fill="#1f2937">Grad Output dL/dY</text>
  <rect x="346" y="131" width="97" height="16" rx="1" fill="#f4f5f7" stroke="#e5e7eb" stroke-width="1"/>
  <text x="350" y="143" font-size="7.5" font-family="monospace" fill="#1f2937">dY[0] (for idx 42)</text>
  <rect x="346" y="153" width="97" height="16" rx="1" fill="#f4f5f7" stroke="#e5e7eb" stroke-width="1"/>
  <text x="350" y="165" font-size="7.5" font-family="monospace" fill="#1f2937">dY[1] (for idx 7)</text>
  <rect x="346" y="175" width="97" height="16" rx="1" fill="#f4f5f7" stroke="#e5e7eb" stroke-width="1"/>
  <text x="350" y="187" font-size="7.5" font-family="monospace" fill="#1f2937">dY[2] (for idx 42)</text>

  <!-- Scatter arrows -->
  <line x1="450" y1="140" x2="492" y2="175" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>
  <line x1="450" y1="161" x2="492" y2="157" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>
  <line x1="450" y1="183" x2="492" y2="178" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>

  <!-- Grad Weight Table Accumulation -->
  <rect x="495" y="110" width="138" height="110" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="502" y="124" font-size="8" font-weight="bold" fill="#1f2937">Grad Weight dL/dW</text>
  <rect x="499" y="131" width="130" height="15" rx="1" fill="#f4f5f7" stroke="#e5e7eb" stroke-width="1"/>
  <text x="503" y="142" font-size="7.5" font-family="monospace" fill="#6b7280">row 0: 0.0 (untouched)</text>
  <rect x="499" y="150" width="130" height="16" rx="1" fill="#f4f5f7" stroke="#e5e7eb" stroke-width="1"/>
  <text x="503" y="162" font-size="7.5" font-family="monospace" fill="#1f2937">row 7: += dY[1]</text>
  <rect x="499" y="170" width="130" height="18" rx="1" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="503" y="183" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">row 42: += dY[0] + dY[2]</text>
  <text x="503" y="206" font-size="7.5" font-family="monospace" fill="#9ca3af">... sparse zeros ...</text>

  <text x="342" y="240" font-size="8.5" font-weight="bold" fill="#1f2937">THE DUPLICATE INDEX TRAP:</text>
  <text x="342" y="254" font-size="8" fill="#6b7280">Naive `dW[idx] = dY` overwrites row 42, dropping dY[0]!</text>
  <text x="342" y="268" font-size="8" fill="#6b7280">Must use `np.add.at(dW, idx, dY)` for atomic accumulation.</text>
{FOOTER}"""
    write_svg("11_embedding-gather-scatter.svg", body)


# -------------------------------------------------------------
# 4. Ch 12: 12_causal-attention-engine.svg
# -------------------------------------------------------------
def gen_12_attention():
    h = 340
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="310" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">CAUSAL ATTENTION ENGINE: MASKING &amp; SOFTMAX NORMALIZATION</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Autoregressive masking: enforcing thermodynamic time causality in the full token-to-token affinity matrix</text>

  <!-- Left: Raw Scores -->
  <rect x="35" y="65" width="175" height="240" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">1. Raw Attention Scores</text>
  <text x="45" y="96" font-size="8.5" fill="#6b7280">S = Q @ K.T / sqrt(d_k) [Seq=4, Seq=4]</text>

  <!-- 4x4 Grid -->
  <rect x="47" y="112" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="61" y="128" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.2</text>
  <rect x="78" y="112" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="92" y="128" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.4</text>
  <rect x="109" y="112" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="123" y="128" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.8</text>
  <rect x="140" y="112" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="154" y="128" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.9</text>

  <rect x="47" y="140" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="61" y="156" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.8</text>
  <rect x="78" y="140" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="92" y="156" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">2.1</text>
  <rect x="109" y="140" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="123" y="156" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.3</text>
  <rect x="140" y="140" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="154" y="156" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.5</text>

  <rect x="47" y="168" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="61" y="184" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.5</text>
  <rect x="78" y="168" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="92" y="184" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.7</text>
  <rect x="109" y="168" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="123" y="184" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">2.4</text>
  <rect x="140" y="168" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="154" y="184" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.1</text>

  <rect x="47" y="196" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="61" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.3</text>
  <rect x="78" y="196" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="92" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.1</text>
  <rect x="109" y="196" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="123" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.9</text>
  <rect x="140" y="196" width="28" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="154" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">3.2</text>

  <text x="45" y="244" font-size="8.5" fill="#6b7280">Upper triangle contains illegal</text>
  <text x="45" y="258" font-size="8.5" fill="#6b7280">lookahead scores: token t attending</text>
  <text x="45" y="272" font-size="8.5" fill="#6b7280">to future tokens t+1, t+2...</text>

  <!-- Arrow -->
  <line x1="210" y1="180" x2="230" y2="180" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Center: Causal Masking -->
  <rect x="235" y="65" width="180" height="240" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="247" y="83" font-size="10" font-weight="700" fill="#1f2937">2. Add Causal Mask (-inf)</text>
  <text x="247" y="96" font-size="8.5" fill="#6b7280">Upper triangle clamped to -1e9</text>

  <!-- Masked Grid -->
  <rect x="247" y="112" width="30" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="262" y="128" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.2</text>
  <rect x="280" y="112" width="30" height="25" rx="1" fill="#e5e7eb" stroke="#d1d5db" stroke-width="1"/><text x="295" y="128" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>
  <rect x="313" y="112" width="30" height="25" rx="1" fill="#e5e7eb" stroke="#d1d5db" stroke-width="1"/><text x="328" y="128" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>
  <rect x="346" y="112" width="30" height="25" rx="1" fill="#e5e7eb" stroke="#d1d5db" stroke-width="1"/><text x="361" y="128" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>

  <rect x="247" y="140" width="30" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="262" y="156" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.8</text>
  <rect x="280" y="140" width="30" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="295" y="156" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">2.1</text>
  <rect x="313" y="140" width="30" height="25" rx="1" fill="#e5e7eb" stroke="#d1d5db" stroke-width="1"/><text x="328" y="156" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>
  <rect x="346" y="140" width="30" height="25" rx="1" fill="#e5e7eb" stroke="#d1d5db" stroke-width="1"/><text x="361" y="156" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>

  <rect x="247" y="168" width="30" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="262" y="184" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.5</text>
  <rect x="280" y="168" width="30" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="295" y="184" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.7</text>
  <rect x="313" y="168" width="30" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="328" y="184" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">2.4</text>
  <rect x="346" y="168" width="30" height="25" rx="1" fill="#e5e7eb" stroke="#d1d5db" stroke-width="1"/><text x="361" y="184" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#9ca3af">-inf</text>

  <rect x="247" y="196" width="30" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="262" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.3</text>
  <rect x="280" y="196" width="30" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="295" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">1.1</text>
  <rect x="313" y="196" width="30" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="328" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.9</text>
  <rect x="346" y="196" width="30" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="361" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">3.2</text>

  <text x="247" y="244" font-size="8.5" fill="#6b7280">Lower triangle preserved.</text>
  <text x="247" y="258" font-size="8.5" fill="#6b7280">In float16, use -1e4 to avoid</text>
  <text x="247" y="272" font-size="8.5" fill="#6b7280">NaN underflow in exp().</text>

  <!-- Arrow -->
  <line x1="415" y1="180" x2="435" y2="180" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>

  <!-- Right: Softmax Row Normalization (ACCENT) -->
  <rect x="440" y="65" width="205" height="240" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="452" y="83" font-size="10" font-weight="700" fill="#1f2937">3. Normalized Weights (Softmax)</text>
  <text x="452" y="96" font-size="8.5" fill="#6b7280">exp(-inf)=0; each row sums to 1.0</text>

  <!-- Probability Grid -->
  <rect x="455" y="112" width="32" height="25" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/><text x="471" y="128" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">1.00</text>
  <rect x="490" y="112" width="32" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="506" y="128" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <rect x="525" y="112" width="32" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="541" y="128" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <rect x="560" y="112" width="32" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="576" y="128" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <text x="600" y="128" font-size="8" fill="#6b7280">T0</text>

  <rect x="455" y="140" width="32" height="25" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/><text x="471" y="156" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.21</text>
  <rect x="490" y="140" width="32" height="25" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/><text x="506" y="156" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">0.79</text>
  <rect x="525" y="140" width="32" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="541" y="156" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <rect x="560" y="140" width="32" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="576" y="156" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <text x="600" y="156" font-size="8" fill="#6b7280">T1</text>

  <rect x="455" y="168" width="32" height="25" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/><text x="471" y="184" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.27</text>
  <rect x="490" y="168" width="32" height="25" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/><text x="506" y="184" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.12</text>
  <rect x="525" y="168" width="32" height="25" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/><text x="541" y="184" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">0.61</text>
  <rect x="560" y="168" width="32" height="25" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/><text x="576" y="184" text-anchor="middle" font-size="8" font-family="monospace" fill="#9ca3af">0.00</text>
  <text x="600" y="184" font-size="8" fill="#6b7280">T2</text>

  <rect x="455" y="196" width="32" height="25" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/><text x="471" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.04</text>
  <rect x="490" y="196" width="32" height="25" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/><text x="506" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.10</text>
  <rect x="525" y="196" width="32" height="25" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/><text x="541" y="212" text-anchor="middle" font-size="8" font-family="monospace" fill="#1f2937">0.08</text>
  <rect x="560" y="196" width="32" height="25" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/><text x="576" y="212" text-anchor="middle" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">0.78</text>
  <text x="600" y="212" font-size="8" fill="#6b7280">T3</text>

  <text x="452" y="244" font-size="8.5" fill="#6b7280">Final Context: Out = A @ V</text>
  <text x="452" y="258" font-size="8.5" fill="#6b7280">Zero information leakage from future</text>
  <text x="452" y="272" font-size="8.5" fill="#6b7280">positions guarantees causal generation.</text>
{FOOTER}"""
    write_svg("12_causal-attention-engine.svg", body)


# -------------------------------------------------------------
# 5. Ch 13: 13_residual-stream-bus.svg
# -------------------------------------------------------------
def gen_13_transformer():
    h = 320
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="290" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">TRANSFORMER RESIDUAL STREAM: THE CENTRAL COMMUNICATION BUS</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Sub-layers read from and add delta updates to a continuous linear highway, eliminating vanishing gradients</text>

  <!-- Central Highway (ACCENT) -->
  <rect x="35" y="65" width="610" height="68" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.2"/>
  <text x="45" y="80" font-size="9.5" font-weight="700" fill="#1f2937">Residual Highway State Bus: [Batch, Seq, d_model]</text>
  <text x="440" y="80" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">x_(l+1) = x_l + F_attn + F_mlp</text>

  <!-- State x_l Node -->
  <rect x="45" y="90" width="65" height="30" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="77" y="109" text-anchor="middle" font-size="9" font-family="monospace" font-weight="bold" fill="#ff8246">x_l</text>

  <!-- Flow along highway to First Sum -->
  <line x1="110" y1="105" x2="238" y2="105" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>
  <text x="174" y="100" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">Identity Path</text>

  <!-- Addition Node 1 -->
  <circle cx="250" cy="105" r="10" fill="#ffffff" stroke="#ff8246" stroke-width="1.5"/>
  <text x="247" y="109" font-size="11" font-family="monospace" font-weight="bold" fill="#ff8246">+</text>

  <!-- Flow along highway to Second Sum -->
  <line x1="260" y1="105" x2="458" y2="105" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>
  <text x="355" y="100" text-anchor="middle" font-size="7.5" font-family="monospace" fill="#6b7280">x'_l = x_l + delta_attn</text>

  <!-- Addition Node 2 -->
  <circle cx="470" cy="105" r="10" fill="#ffffff" stroke="#ff8246" stroke-width="1.5"/>
  <text x="467" y="109" font-size="11" font-family="monospace" font-weight="bold" fill="#ff8246">+</text>

  <!-- Flow to State x_(l+1) Node -->
  <line x1="480" y1="105" x2="548" y2="105" stroke="#ff8246" stroke-width="1" marker-end="url(#arrow-orange)"/>
  <rect x="550" y="90" width="85" height="30" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="592" y="109" text-anchor="middle" font-size="9" font-family="monospace" font-weight="bold" fill="#ff8246">x_(l+1)</text>

  <!-- Sublayer 1: Multi-Head Attention (Pre-LN) -->
  <rect x="145" y="152" width="180" height="78" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="155" y="169" font-size="9.5" font-weight="700" fill="#1f2937">Sublayer 1: Attention Off-ramp</text>
  <text x="155" y="184" font-size="8" font-family="monospace" fill="#1f2937">1. h = LayerNorm(x_l)</text>
  <text x="155" y="198" font-size="8" font-family="monospace" fill="#1f2937">2. delta_attn = MHA(h)</text>
  <text x="155" y="217" font-size="8" fill="#6b7280">Reads stream, emits delta update</text>

  <!-- Tap off highway to LN, and add back -->
  <polyline points="77,120 77,191 145,191" fill="none" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>
  <line x1="250" y1="152" x2="250" y2="117" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Sublayer 2: Feed-Forward Network (MLP) -->
  <rect x="365" y="152" width="180" height="78" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="375" y="169" font-size="9.5" font-weight="700" fill="#1f2937">Sublayer 2: MLP Off-ramp</text>
  <text x="375" y="184" font-size="8" font-family="monospace" fill="#1f2937">1. h' = LayerNorm(x'_l)</text>
  <text x="375" y="198" font-size="8" font-family="monospace" fill="#1f2937">2. delta_mlp = FFN(h')</text>
  <text x="375" y="217" font-size="8" fill="#6b7280">Expanded dimension: 4 * d_model</text>

  <!-- Tap off highway to MLP, and add back -->
  <polyline points="330,105 330,191 365,191" fill="none" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>
  <line x1="470" y1="152" x2="470" y2="117" stroke="#9ca3af" stroke-width="1" marker-end="url(#arrow-gray)"/>

  <!-- Bottom: Gradient Highway Identity Backprop -->
  <rect x="35" y="246" width="610" height="46" rx="2" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="45" y="262" font-size="9.5" font-weight="700" fill="#1f2937">Gradient Identity Highway: dL / dx_0 = dL / dx_L * ( I + sum( dF_l / dx_l ) )</text>
  <text x="45" y="278" font-size="8.5" fill="#6b7280">Because of the addition operator (+), the identity matrix I carries the error signal directly back to token embeddings without decay.</text>
{FOOTER}"""
    write_svg("13_residual-stream-bus.svg", body)


# -------------------------------------------------------------
# 6. Milestone 02: milestone_02_teacher-forcing-grid.svg
# -------------------------------------------------------------
def gen_milestone_02():
    h = 330
    body = f"""{HEADER.format(height=h)}
  <!-- Outer Frame -->
  <rect x="20" y="15" width="640" height="300" rx="3" fill="#f8f9fa" stroke="#e5e7eb" stroke-width="1.2"/>
  <text x="35" y="36" font-size="11" font-weight="700" fill="#1f2937">TRAINING VS INFERENCE: TEACHER FORCING VS AUTOREGRESSIVE LOOP</text>
  <text x="35" y="49" font-size="9" fill="#6b7280">Why training processes sequences in parallel O(1) time, while inference requires sequential O(N) iterative decoding</text>

  <!-- Left Half: Training with Teacher Forcing -->
  <rect x="35" y="65" width="295" height="235" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>
  <text x="45" y="83" font-size="10" font-weight="700" fill="#1f2937">1. Training: Parallel Teacher Forcing (O(1) Steps)</text>
  <text x="45" y="96" font-size="8.5" fill="#6b7280">Entire target sequence known in advance; causal mask prevents leakage</text>

  <!-- Input sequence -->
  <rect x="45" y="112" width="275" height="28" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="53" y="129" font-size="8" font-family="monospace" fill="#1f2937">Inputs X:  ['The', 'quick', 'brown', 'fox']</text>

  <!-- Single Forward Pass Block -->
  <rect x="45" y="146" width="275" height="35" rx="1" fill="#f8f9fa" stroke="#d1d5db" stroke-width="1"/>
  <text x="53" y="161" font-size="8" font-family="monospace" font-weight="bold" fill="#1f2937">Parallel Forward: Causal Transformer (Seq=4)</text>
  <text x="53" y="174" font-size="8" fill="#6b7280">All 4 positions evaluated simultaneously in 1 kernel call</text>

  <!-- Shifted Targets and Loss -->
  <rect x="45" y="187" width="275" height="42" rx="1" fill="#f4f5f7" stroke="#d1d5db" stroke-width="1"/>
  <text x="53" y="202" font-size="8" font-family="monospace" fill="#1f2937">Targets Y: ['quick', 'brown', 'fox', '&lt;eos&gt;'] (+1)</text>
  <text x="53" y="218" font-size="7.5" font-family="monospace" fill="#6b7280">Loss = CrossEntropy(Logits.view(-1, V), Y.view(-1))</text>

  <text x="45" y="248" font-size="8.5" font-weight="bold" fill="#1f2937">Parallel Efficiency:</text>
  <text x="45" y="262" font-size="8" fill="#6b7280">Hardware tensor cores run at 100% saturation.</text>
  <text x="45" y="276" font-size="8" fill="#6b7280">Ground truth tokens fed at every step regardless of mistakes.</text>

  <!-- Right Half: Inference Autoregressive Loop (ACCENT) -->
  <rect x="345" y="65" width="300" height="235" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1"/>
  <text x="357" y="83" font-size="10" font-weight="700" fill="#1f2937">2. Inference: Sequential Generation (O(N) Steps)</text>
  <text x="357" y="96" font-size="8.5" fill="#6b7280">Each token must be sampled before becoming input to next step</text>

  <!-- Step 0 -->
  <rect x="357" y="112" width="276" height="26" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="365" y="128" font-size="8" font-family="monospace" fill="#1f2937">Step 1: Input ['The'] -&gt; Sample 'quick'</text>

  <!-- Step 1 -->
  <rect x="357" y="144" width="276" height="26" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="365" y="160" font-size="8" font-family="monospace" fill="#1f2937">Step 2: Input ['The', 'quick'] -&gt; Sample 'brown'</text>

  <!-- Step 2 -->
  <rect x="357" y="176" width="276" height="26" rx="1" fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>
  <text x="365" y="192" font-size="8" font-family="monospace" fill="#1f2937">Step 3: Input ['The', 'quick', 'brown'] -&gt; Sample 'fox'</text>

  <!-- Step 3 Loop -->
  <rect x="357" y="208" width="276" height="26" rx="1" fill="#ffffff" stroke="#ff8246" stroke-width="1"/>
  <text x="365" y="224" font-size="8" font-family="monospace" font-weight="bold" fill="#ff8246">Step 4: Input [..., 'fox'] -&gt; Sample '&lt;eos&gt;' (STOP)</text>

  <text x="357" y="250" font-size="8.5" font-weight="bold" fill="#1f2937">Memory-Bound Bottleneck:</text>
  <text x="357" y="264" font-size="8" fill="#6b7280">Batch size = 1 token per step. Compute intensity R &lt;&lt; 1.</text>
  <text x="357" y="278" font-size="8" fill="#6b7280">Requires KV Caching (Module 18) to avoid O(N^2) quadratic recomputation.</text>
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
