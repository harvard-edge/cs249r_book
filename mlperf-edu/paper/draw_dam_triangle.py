#!/usr/bin/env python3
"""Generate DAM Venn diagram for MLPerf EDU companion paper (Figure 4).
Modeled directly after the Chapter 1 / Volume 1 DAM Venn diagram (dam_venn.svg).
Optimized for publication-quality single-column and compact two-column formats.
"""

from pathlib import Path
import shutil
import subprocess

SVG_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 615" width="640" height="615" font-family="-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif">
  <defs>
    <style>
      .circle-data { fill: rgba(163, 31, 52, 0.08); stroke: #a31f34; stroke-width: 2.2; }
      .circle-algo { fill: rgba(0, 76, 151, 0.08); stroke: #004c97; stroke-width: 2.2; }
      .circle-mach { fill: rgba(46, 139, 87, 0.08); stroke: #2e8b57; stroke-width: 2.2; }

      .label-main-data { font-size: 20px; font-weight: 700; fill: #a31f34; letter-spacing: 0.3px; }
      .label-main-algo { font-size: 20px; font-weight: 700; fill: #004c97; letter-spacing: 0.3px; }
      .label-main-mach { font-size: 20px; font-weight: 700; fill: #2e8b57; letter-spacing: 0.3px; }

      .label-sub-data { font-size: 11px; font-weight: 600; fill: #a31f34; }
      .label-sub-algo { font-size: 11px; font-weight: 600; fill: #004c97; }
      .label-sub-mach { font-size: 11px; font-weight: 600; fill: #2e8b57; }

      .label-item { font-size: 9.5px; fill: #334155; }
      .label-item-finding { font-size: 9px; fill: #64748b; font-style: italic; }

      .label-inter-title { font-size: 12px; font-weight: 700; fill: #334155; letter-spacing: 0.5px; }
      .label-inter-topic { font-size: 10.5px; font-weight: 700; fill: #5a2d82; }
      .label-inter-ref { font-size: 9px; fill: #7c3aed; font-weight: 600; }
      .label-inter-item { font-size: 8.5px; fill: #1e293b; }
      .label-inter-finding { font-size: 8.5px; fill: #5a2d82; font-style: italic; }

      .label-center-title { font-size: 13px; font-weight: 800; fill: #a31f34; letter-spacing: 0.6px; }
      .label-center-principle { font-size: 10.5px; font-weight: 700; fill: #a31f34; }
      .label-center-ref { font-size: 9px; font-weight: 600; fill: #a31f34; }
      .label-center-math { font-size: 10.5px; font-weight: 700; fill: #b45309; }
      .label-center-focus { font-size: 9.5px; font-weight: 700; fill: #0f172a; }

      .label-footer { font-size: 10px; fill: #64748b; font-style: italic; }
      
      .halo { paint-order: stroke fill; stroke: #ffffff; stroke-width: 3.5px; stroke-linejoin: round; }
    </style>
  </defs>

  <!-- Crisp white background -->
  <rect width="640" height="615" fill="#ffffff" />

  <!-- Three main circles (Verified equilateral geometry: L=220, r=195, tight bounds) -->
  <!-- Data (D) - top center -->
  <circle cx="320" cy="193" r="195" class="circle-data" />
  <!-- Algorithm (A) - bottom-left -->
  <circle cx="210" cy="384" r="195" class="circle-algo" />
  <!-- Machine (M) - bottom-right -->
  <circle cx="430" cy="384" r="195" class="circle-mach" />

  <!-- ================================================================= -->
  <!-- 1. PURE REGION: Data (D) [Top Center]                             -->
  <!-- ================================================================= -->
  <g class="halo">
    <text x="320" y="68" text-anchor="middle" class="label-main-data">Data (D)</text>
    <text x="320" y="86" text-anchor="middle" class="label-sub-data">Sample Pruning &amp; Scaling  [§6.1]</text>
    <text x="320" y="108" text-anchor="middle" class="label-item">Sample budget scaling (100% &#x2192; 10%)</text>
    <text x="320" y="124" text-anchor="middle" class="label-item">Universal task accuracy floors</text>
    <text x="320" y="140" text-anchor="middle" class="label-item-finding">Linear execution time scaling</text>
  </g>

  <!-- ================================================================= -->
  <!-- 2. PURE REGION: Algorithm (A) [Bottom-Left]                       -->
  <!-- ================================================================= -->
  <g class="halo">
    <text x="120" y="420" text-anchor="middle" class="label-main-algo">Algorithm (A)</text>
    <text x="120" y="438" text-anchor="middle" class="label-sub-algo">Precision &amp; Capacity  [§6.1]</text>
    <text x="120" y="458" text-anchor="middle" class="label-item">Formats: FP16, BF16, INT8</text>
    <text x="120" y="474" text-anchor="middle" class="label-item">Format bifurcation under gates</text>
    <text x="120" y="488" text-anchor="middle" class="label-item-finding">Model parameter scaling</text>
  </g>

  <!-- ================================================================= -->
  <!-- 3. PURE REGION: Machine (M) [Bottom-Right]                        -->
  <!-- ================================================================= -->
  <g class="halo">
    <text x="520" y="420" text-anchor="middle" class="label-main-mach">Machine (M)</text>
    <text x="520" y="438" text-anchor="middle" class="label-sub-mach">Hardware Architecture  [§6.1]</text>
    <text x="520" y="458" text-anchor="middle" class="label-item">Unified memory SoC (CPU vs. MPS)</text>
    <text x="520" y="474" text-anchor="middle" class="label-item">100&#xd7; acceleration spread</text>
    <text x="520" y="488" text-anchor="middle" class="label-item-finding">Roofline bounds &amp; limits</text>
  </g>

  <!-- ================================================================= -->
  <!-- 4. PAIRWISE: D ∩ A [Top-Left Overlap]                            -->
  <!-- ================================================================= -->
  <g class="halo">
    <text x="205" y="222" text-anchor="middle" class="label-inter-title">D &#x2229; A</text>
    <text x="205" y="236" text-anchor="middle" class="label-inter-topic">Floors vs. Capacity</text>
    <text x="205" y="249" text-anchor="middle" class="label-inter-ref">[§6.2]</text>
    <text x="205" y="264" text-anchor="middle" class="label-inter-item">Model capacity vs. budget</text>
    <text x="205" y="278" text-anchor="middle" class="label-inter-finding">Universal accuracy floor</text>
  </g>

  <!-- ================================================================= -->
  <!-- 5. PAIRWISE: D ∩ M [Top-Right Overlap]                           -->
  <!-- ================================================================= -->
  <g class="halo">
    <text x="435" y="222" text-anchor="middle" class="label-inter-title">D &#x2229; M</text>
    <text x="435" y="236" text-anchor="middle" class="label-inter-topic">Working Set vs. Bus</text>
    <text x="435" y="249" text-anchor="middle" class="label-inter-ref">[§6.2]</text>
    <text x="435" y="264" text-anchor="middle" class="label-inter-item">Cache spills (&gt;8 MB L2)</text>
    <text x="435" y="278" text-anchor="middle" class="label-inter-finding">Invariant 2.38&#xd7; speedup</text>
  </g>

  <!-- ================================================================= -->
  <!-- 6. PAIRWISE: A ∩ M [Bottom Center Overlap]                       -->
  <!-- ================================================================= -->
  <g class="halo">
    <text x="320" y="418" text-anchor="middle" class="label-inter-title">A &#x2229; M</text>
    <text x="320" y="432" text-anchor="middle" class="label-inter-topic">Quantization &amp; Kernels</text>
    <text x="320" y="444" text-anchor="middle" class="label-inter-ref">[§6.2]</text>
    <text x="320" y="458" text-anchor="middle" class="label-inter-item">INT8 kernel alignment</text>
    <text x="320" y="471" text-anchor="middle" class="label-inter-item">Dynamic activation cost</text>
    <text x="320" y="484" text-anchor="middle" class="label-inter-finding">11&#xd7; CPU Fallback</text>
  </g>

  <!-- ================================================================= -->
  <!-- 7. THREE-WAY INTERSECTION: D ∩ A ∩ M [Center Core]              -->
  <!-- ================================================================= -->
  <g class="halo">
    <text x="320" y="308" text-anchor="middle" class="label-center-title">D &#x2229; A &#x2229; M</text>
    <text x="320" y="323" text-anchor="middle" class="label-center-principle">Non-Decomposability</text>
    <text x="320" y="335" text-anchor="middle" class="label-center-ref">[§6.3]</text>
    <text x="320" y="350" text-anchor="middle" class="label-center-math">(D, A, M)* &#x2260; (D*, A*, M*)</text>
    <text x="320" y="365" text-anchor="middle" class="label-center-focus">2&#xd7;2&#xd7;2 Factorial Co-Design</text>
  </g>

  <!-- ================================================================= -->
  <!-- Bottom Takeaway Footer                                            -->
  <!-- ================================================================= -->
  <line x1="80" y1="572" x2="560" y2="572" stroke="#cbd5e1" stroke-width="0.8" />
  <text x="320" y="595" text-anchor="middle" class="label-footer">Where Data, Algorithm, and Machine converge — non-decomposable co-design.</text>
</svg>
"""

def generate_dam_venn(output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = output_dir / "fig_dam_triangle.svg"
    pdf_path = output_dir / "fig_dam_triangle.pdf"
    png_path = output_dir / "fig_dam_triangle.png"

    # Write SVG source
    svg_path.write_text(SVG_CONTENT, encoding="utf-8")
    print(f"Written: {svg_path}")

    # Render PDF and PNG using rsvg-convert if available
    rsvg_cmd = shutil.which("rsvg-convert")
    if rsvg_cmd:
        subprocess.run([rsvg_cmd, "-f", "pdf", str(svg_path), "-o", str(pdf_path)], check=True)
        print(f"Generated PDF: {pdf_path}")
        subprocess.run([rsvg_cmd, "-w", "1400", str(svg_path), "-o", str(png_path)], check=True)
        print(f"Generated PNG: {png_path}")
    else:
        inkscape_cmd = shutil.which("inkscape")
        if inkscape_cmd:
            subprocess.run([inkscape_cmd, str(svg_path), "--export-filename", str(pdf_path)], check=True)
            subprocess.run([inkscape_cmd, str(svg_path), "--export-filename", str(png_path), "--export-width=1400"], check=True)
        else:
            raise RuntimeError("Neither rsvg-convert nor inkscape found to convert SVG to PDF/PNG.")

if __name__ == "__main__":
    generate_dam_venn()
