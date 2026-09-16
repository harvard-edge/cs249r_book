#!/usr/bin/env python3
"""
Generate TinyTorch Framework Master Blueprint and "You Are Here" Navigation SVGs.
Follows Vol 3 styling standards and palette.md.
"""
from pathlib import Path
import subprocess

DEST_DIRS = [
    Path("tinytorch/narrative_book/assets/images/diagrams"),
    Path("tinytorch/quarto/assets/images/diagrams"),
]

def write_and_convert(name: str, content: str):
    for d in DEST_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        svg_path = d / f"{name}.svg"
        pdf_path = d / f"{name}.pdf"
        svg_path.write_text(content, encoding="utf-8")
        subprocess.run(
            ["rsvg-convert", "-f", "pdf", "--keep-aspect-ratio", str(svg_path), "-o", str(pdf_path)],
            check=True
        )
    print(f"Generated and converted: {name}.svg -> .pdf")

def gen_01_framework_you_are_here():
    w, h = 220, 165
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="{w}" height="{h}" fill="#ffffff"/>
  <!-- Outer Card -->
  <rect x="5" y="5" width="210" height="155" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8" font-weight="700" fill="#1f2937">FRAMEWORK BLUEPRINT: YOU ARE HERE</text>

  <!-- Level 1: Active Stage (Chapter 1) -->
  <g transform="translate(15, 32)">
    <rect x="0" y="0" width="190" height="34" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.6"/>
    <text x="8" y="14" font-size="8" font-weight="700" fill="#c85a17">&#9679; 01. TENSORS &amp; STRIDES</text>
    <text x="182" y="14" text-anchor="end" font-size="7" font-weight="bold" fill="#ff8246">ACTIVE</text>
    <text x="8" y="27" font-size="7" font-family="monospace" fill="#7c2d12">Flat 1D DRAM &#8226; Strided Indexing &#8226; Views</text>
  </g>

  <!-- Arrow down -->
  <line x1="110" y1="67" x2="110" y2="74" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="2,2"/>
  <polygon points="108,74 112,74 110,77" fill="#9ca3af"/>

  <!-- Level 2: Core Engine (Chapters 02-08) -->
  <g transform="translate(15, 78)">
    <rect x="0" y="0" width="190" height="32" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="8" y="13" font-size="7.5" font-weight="700" fill="#475569">02–08. THE CORE ENGINE</text>
    <text x="8" y="25" font-size="6.8" font-family="monospace" fill="#64748b">Activations &#8226; Layers &#8226; Autograd &#8226; SGD/AdamW</text>
  </g>

  <!-- Arrow down -->
  <line x1="110" y1="111" x2="110" y2="118" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="2,2"/>
  <polygon points="108,118 112,118 110,121" fill="#9ca3af"/>

  <!-- Level 3: Architectures & Acceleration (Chapters 09-21) -->
  <g transform="translate(15, 122)">
    <rect x="0" y="0" width="190" height="32" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="8" y="13" font-size="7.5" font-weight="700" fill="#475569">09–21. ARCHITECTURES &amp; SYSTEMS</text>
    <text x="8" y="25" font-size="6.8" font-family="monospace" fill="#64748b">TinyGPT &#8226; INT8 &#8226; Fusion &#8226; KV Cache &#8226; Silicon</text>
  </g>
</svg>
"""
    write_and_convert("01_framework-you-are-here", svg)


def gen_00_framework_datapath_master():
    w, h = 680, 240
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="{w}" height="{h}" fill="#ffffff"/>
  <defs>
    <marker id="dp-arrow" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#1f2937"/>
    </marker>
    <marker id="dp-orange-arrow" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#ff8246"/>
    </marker>
  </defs>

  <!-- Header Banner: TinyTorch Framework Datapath Architecture -->
  <rect x="15" y="10" width="650" height="28" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="30" y="28" font-size="11" font-weight="700" fill="#1f2937">THE TINYTORCH EXECUTION DATAPATH &amp; INVARIANT PIPELINE</text>
  <text x="645" y="28" text-anchor="end" font-size="9" font-family="monospace" fill="#6b7280">Track the Byte &#8226; Close the Loop</text>

  <!-- STAGE 1: Physical Memory & Tensor Layer -->
  <g transform="translate(15, 48)">
    <rect x="0" y="0" width="150" height="175" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
    <rect x="0" y="0" width="150" height="22" rx="2" fill="#ff8246"/>
    <text x="75" y="15" text-anchor="middle" font-size="9" font-weight="700" fill="#ffffff">1. STORAGE &amp; TENSORS</text>
    
    <rect x="10" y="32" width="130" height="36" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="47" font-size="8" font-weight="700" fill="#1f2937">Flat DRAM Buffer</text>
    <text x="18" y="60" font-size="7.5" font-family="monospace" fill="#6b7280">float32 continuous</text>

    <line x1="75" y1="69" x2="75" y2="78" stroke="#ff8246" stroke-width="1.2" marker-end="url(#dp-orange-arrow)"/>

    <rect x="10" y="80" width="130" height="42" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="95" font-size="8" font-weight="700" fill="#1f2937">Strided Tensor</text>
    <text x="18" y="107" font-size="7.2" font-family="monospace" fill="#6b7280">shape, strides, offset</text>
    <text x="18" y="117" font-size="6.8" font-family="monospace" fill="#c85a17">Zero-copy views</text>

    <line x1="75" y1="123" x2="75" y2="132" stroke="#ff8246" stroke-width="1.2" marker-end="url(#dp-orange-arrow)"/>

    <rect x="10" y="134" width="130" height="32" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="148" font-size="7.8" font-weight="700" fill="#1f2937">Function.apply</text>
    <text x="18" y="159" font-size="7" font-family="monospace" fill="#6b7280">Dispatch &amp; fresh wrapper</text>
  </g>

  <!-- Arrow 1 -> 2 -->
  <line x1="165" y1="135" x2="180" y2="135" stroke="#1f2937" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- STAGE 2: Forward Compute & Loss -->
  <g transform="translate(180, 48)">
    <rect x="0" y="0" width="150" height="175" rx="2" fill="#f8fafc" stroke="#9ca3af" stroke-width="1.2"/>
    <rect x="0" y="0" width="150" height="22" rx="2" fill="#e2e8f0"/>
    <text x="75" y="15" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">2. FORWARD &amp; LOSS</text>

    <rect x="10" y="32" width="130" height="36" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="47" font-size="8" font-weight="700" fill="#1f2937">Activations (Ch 02)</text>
    <text x="18" y="60" font-size="7.5" font-family="monospace" fill="#6b7280">ReLU, Sigmoid, GELU</text>

    <line x1="75" y1="69" x2="75" y2="78" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="80" width="130" height="42" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="95" font-size="8" font-weight="700" fill="#1f2937">Layers (Ch 03, 05)</text>
    <text x="18" y="107" font-size="7.2" font-family="monospace" fill="#6b7280">Linear(W, b), DataLoader</text>
    <text x="18" y="117" font-size="6.8" font-family="monospace" fill="#475569">Batches &amp; Collation</text>

    <line x1="75" y1="123" x2="75" y2="132" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="134" width="130" height="32" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="148" font-size="7.8" font-weight="700" fill="#1f2937">Loss Engine (Ch 04)</text>
    <text x="18" y="159" font-size="7" font-family="monospace" fill="#6b7280">Cross-Entropy LogSumExp</text>
  </g>

  <!-- Arrow 2 -> 3 -->
  <line x1="330" y1="135" x2="345" y2="135" stroke="#1f2937" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- STAGE 3: Autograd & Learning Loop -->
  <g transform="translate(345, 48)">
    <rect x="0" y="0" width="150" height="175" rx="2" fill="#f8fafc" stroke="#9ca3af" stroke-width="1.2"/>
    <rect x="0" y="0" width="150" height="22" rx="2" fill="#e2e8f0"/>
    <text x="75" y="15" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">3. BACKPROP &amp; STEP</text>

    <rect x="10" y="32" width="130" height="36" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="47" font-size="8" font-weight="700" fill="#1f2937">Autograd Tape (Ch 06)</text>
    <text x="18" y="60" font-size="7.5" font-family="monospace" fill="#6b7280">DAG Topological Sort</text>

    <line x1="75" y1="69" x2="75" y2="78" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="80" width="130" height="42" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="95" font-size="8" font-weight="700" fill="#1f2937">Optimizers (Ch 07)</text>
    <text x="18" y="107" font-size="7.2" font-family="monospace" fill="#6b7280">SGD &#8226; AdamW (m_t, v_t)</text>
    <text x="18" y="117" font-size="6.8" font-family="monospace" fill="#475569">Decoupled Weight Decay</text>

    <line x1="75" y1="123" x2="75" y2="132" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="134" width="130" height="32" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="148" font-size="7.8" font-weight="700" fill="#1f2937">Training Loop (Ch 08)</text>
    <text x="18" y="159" font-size="7" font-family="monospace" fill="#6b7280">Zero &#8226; Fwd &#8226; Back &#8226; Step</text>
  </g>

  <!-- Arrow 3 -> 4 -->
  <line x1="495" y1="135" x2="510" y2="135" stroke="#1f2937" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- STAGE 4: Model Scale & Systems Acceleration -->
  <g transform="translate(510, 48)">
    <rect x="0" y="0" width="155" height="175" rx="2" fill="#f8fafc" stroke="#9ca3af" stroke-width="1.2"/>
    <rect x="0" y="0" width="155" height="22" rx="2" fill="#e2e8f0"/>
    <text x="77" y="15" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">4. SCALE &amp; SYSTEMS</text>

    <rect x="10" y="32" width="135" height="36" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="47" font-size="8" font-weight="700" fill="#1f2937">TinyGPT (Ch 10–13)</text>
    <text x="18" y="60" font-size="7.5" font-family="monospace" fill="#6b7280">BPE &#8226; Causal Attention</text>

    <line x1="77" y1="69" x2="77" y2="78" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="80" width="135" height="42" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="95" font-size="8" font-weight="700" fill="#1f2937">Acceleration (14–18)</text>
    <text x="18" y="107" font-size="7.2" font-family="monospace" fill="#6b7280">Roofline &#8226; INT8 &#8226; Fusion</text>
    <text x="18" y="117" font-size="6.8" font-family="monospace" fill="#475569">Static KV Cache O(1)</text>

    <line x1="77" y1="123" x2="77" y2="132" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="134" width="135" height="32" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="148" font-size="7.8" font-weight="700" fill="#1f2937">Silicon &amp; Stack (21)</text>
    <text x="18" y="159" font-size="7" font-family="monospace" fill="#6b7280">Systolic Arrays &#8226; Triton</text>
  </g>
</svg>
"""
    write_and_convert("00_framework-datapath-master", svg)

if __name__ == "__main__":
    gen_01_framework_you_are_here()
    gen_00_framework_datapath_master()
