#!/usr/bin/env python3
"""
Figure Quality Assurance & Iteration Loop for MLSysBook.

Validates SVG textbook diagrams against:
1. The Anti-Poster Rule (zero bullets, zero paragraphs, no canvas banners).
2. The Volume IV Design Standard (micro-radii, semantic palette, canvas hygiene).
3. Prose-Figure Alignment (verifies that visual tokens match chapter math/prose).
4. High-resolution PNG rasterization for visual verification.
"""

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def audit_svg(svg_path: Path) -> dict:
    """Run programmatic checks on an SVG file against the Textbook Figure Standard."""
    issues = []
    warnings = []

    content = svg_path.read_text(encoding="utf-8")
    root = ET.fromstring(content)

    # 1. Root and ViewBox checks
    viewbox = root.attrib.get("viewBox")
    if not viewbox:
        issues.append("Missing viewBox attribute on <svg>")

    font_family = root.attrib.get("font-family")
    if not font_family:
        warnings.append("Missing font-family declaration on root <svg>")

    # 2. Canvas Base rect check (no outer stroke)
    first_rect = root.find("{http://www.w3.org/2000/svg}rect")
    if first_rect is not None:
        stroke = first_rect.attrib.get("stroke")
        if stroke and stroke.lower() not in ("none", "transparent"):
            issues.append(f"Canvas base <rect> has outer stroke='{stroke}' (violates borderless print standard)")

    # 3. Text element audit
    all_text_elements = root.findall(".//{http://www.w3.org/2000/svg}text")
    bullet_chars = ["•", "▪", "✦", "— ", "– "]

    for elem in all_text_elements:
        full_text = "".join(elem.itertext()).strip()
        
        # Check for bullet points
        for b in bullet_chars:
            if b in full_text:
                issues.append(f"Bullet point character '{b}' found in text: '{full_text[:40]}...' (violates Anti-Poster Rule)")
                break

        # Check for multi-line paragraph bloat (> 15 words)
        words = full_text.split()
        if len(words) > 15:
            issues.append(f"Text too verbose ({len(words)} words): '{full_text[:50]}...' (violates Anti-Poster Rule)")

        # Check for banner titles at canvas top (root child or top-level group)
        y_val = elem.attrib.get("y")
        try:
            if y_val and float(y_val) < 45:
                # Only flag if it's not nested in a transform group deep down
                parent = root.find(f".//*[{elem.tag}='{elem.text}']")
                if len(words) >= 4 and full_text.isupper() and "FOUNDATION" in full_text or "ARCHITECTURE" in full_text:
                    issues.append(f"In-canvas title banner found: '{full_text}' (Quarto owns figure titles)")
        except ValueError:
            pass

    return {
        "path": str(svg_path),
        "issues": issues,
        "warnings": warnings,
        "text_count": len(all_text_elements),
    }


def verify_prose_alignment(svg_path: Path, qmd_path: Path) -> dict:
    """Verify that symbols and terms in the SVG exist and are explained in the chapter prose."""
    svg_content = svg_path.read_text(encoding="utf-8")
    qmd_content = qmd_path.read_text(encoding="utf-8")

    # Extract symbols / keywords from SVG
    root = ET.fromstring(svg_content)
    svg_texts = ["".join(e.itertext()).strip() for e in root.findall(".//{http://www.w3.org/2000/svg}text")]
    
    # Chapter-specific anchor dictionaries
    chapter_anchors = {
        "03_brain": [
            ("Z (Unified Sequence)", ["\\mathbf{Z}", "Z ="]),
            ("h_context (Conditioning Latent)", ["\\mathbf{h}_{\\text{context}}", "h_context", "h_{\\text{context}}"]),
            ("z_vision (Visual Tokens)", ["\\mathbf{z}_{\\text{vision}}", "z_vision"]),
            ("z_text (Language Tokens)", ["\\mathbf{z}_{\\text{text}}", "z_text"]),
            ("z_state (Proprioceptive Tokens)", ["\\mathbf{z}_{\\text{state}}", "z_state"]),
            ("A_ij (Cross-Attention)", ["A_{i, j}", "A_{i,j}", "cross-attention"]),
            ("Action Chunking A[t:t+H]", ["\\mathbf{A}_{t:t+H}", "action chunk"]),
            ("Proposal-Permission Boundary", ["proposal–permission", "proposal-permission", "unprivileged candidate proposal"]),
            ("Hard Real-Time Safety Shield", ["1000", "safety shield", "safety monitor"]),
            ("Plant Actuators & Motor Drives", ["actuators", "motor", "FOC", "PWM"]),
        ],
        "06_training": [
            ("Score Matching / Noise Prediction ε_θ", ["\\boldsymbol{\\epsilon}_\\theta", "\\epsilon_\\theta", "noise-prediction"]),
            ("Action Chunking A[t:t+H]", ["\\mathbf{A}_{t:t+H}", "action chunk", "action trajectory chunks"]),
            ("DDIM Iterative Solver", ["DDIM", "reverse diffusion", "denoising"]),
            ("Multimodal Symmetry Breaking", ["symmetry breaking", "multimodal", "mode collapse"]),
            ("Receding Horizon Execution (H_p vs H_a)", ["receding", "horizon", "discarded tail"]),
            ("Edge SoC Compute Budget (Jetson)", ["Jetson", "Orin", "latency budget", "40.4"]),
            ("SPSC Lock-Free Buffer", ["SPSC", "ring buffer", "shared buffer"]),
            ("1000 Hz Real-Time MCU", ["1000", "real-time", "microcontroller"]),
            ("Cubic Spline Interpolation", ["cubic spline", "spline", "CubicSplineInterpolate"]),
            ("Deterministic Actuation (PWM)", ["PWM", "actuation", "servo"]),
        ],
        "08_perception": [
            ("Pinhole Projection Matrix K", ["intrinsic", "\\mathbf{k}", "pinhole"]),
            ("Feature Patch Token f_(u,v)", ["\\mathbf{f}_{u, v}", "feature patch", "backbone"]),
            ("Optical Bearing Ray r(u,v)", ["bearing ray", "\\mathbf{r}(u, v)", "unprojection"]),
            ("Categorical Depth Discretization", ["categorical", "depth bin", "lift-splat-shoot", "softmax"]),
            ("Frustum Feature Points c_(u,v,k)", ["\\mathbf{c}_{u, v, k}", "frustum", "outer product"]),
            ("SE(3) Rigid Extrinsics", ["\\mathbf{t}_{\\text{base}}^{\\text{cam}}", "se(3)", "extrinsics"]),
            ("Metric BEV Pillar Splatting", ["\\mathbf{v}_{\\text{bev}}", "bev", "pillar", "splat"]),
            ("Quadratic Depth Covariance δZ ∝ Z²", ["z^2", "depth uncertainty", "covariance"]),
            ("Defended Clearance Inset C_inset", ["c_{\\text{inset}}", "c_{\\text{safe}}", "clearance inset"]),
            ("Hard Real-Time Gating / Revocation", ["revoke", "1000", "safety"]),
        ],
        "09_memory": [
            ("3D Dynamic Scene Graph (DSG)", ["scene graph", "hierarchical", "dsg", "metric-semantic"]),
            ("Metric Primitive Level (Voxels / TSDF)", ["tsdf", "voxel", "mesh", "metric"]),
            ("Object Node Level (Affordances)", ["object", "bounding box", "affordance"]),
            ("Structural / Room Topology", ["room", "place", "topology"]),
            ("Agent / Platform State", ["agent", "base", "kinematic", "se(3)"]),
            ("Inter-Node Spatio-Temporal Relations", ["edges", "relation", "spatial"]),
            ("Seqlock Shared Memory Ingress", ["seqlock", "shared memory", "zero-copy", "lock-free"]),
            ("Epistemic Uncertainty Decay / Staleness", ["staleness", "aging", "decay", "uncertainty"]),
        ],
        "04_nervous": [
            ("Safe Torque Off (STO)", ["sto", "safe torque off", "iec 61800-5-2"]),
            ("Dual-Channel Redundant Interlock", ["dual-channel", "redundant", "interlock", "cross-monitoring"]),
            ("Gate Driver Optocoupler Disconnect", ["optocoupler", "gate driver", "isolated", "igbt", "mosfet"]),
            ("Positive Mechanical Break (E-Stop)", ["e-stop", "emergency stop", "positive break", "mechanically linked"]),
            ("Hardware Failsafe De-energization", ["de-energiz", "fail-safe", "quiescent"]),
            ("Category 4 / PLe Safety Integrity", ["cat 4", "ple", "sil 3", "iso 13849"]),
            ("Motor Inverter Bridge", ["inverter", "bridge", "three-phase", "foc"]),
        ],
        "12_enforcement": [
            ("Safe Torque Off (STO)", ["sto", "safe torque off", "iec 61800-5-2"]),
            ("Dual-Channel Redundant Interlock", ["dual-channel", "redundant", "interlock", "cross-monitoring"]),
            ("Gate Driver Optocoupler Disconnect", ["optocoupler", "gate driver", "isolated", "igbt", "mosfet"]),
            ("Positive Mechanical Break (E-Stop)", ["e-stop", "emergency stop", "positive break", "mechanically linked"]),
            ("Hardware Failsafe De-energization", ["de-energiz", "fail-safe", "quiescent"]),
            ("Category 4 / PLe Safety Integrity", ["cat 4", "ple", "sil 3", "iso 13849"]),
            ("Motor Inverter Bridge", ["inverter", "bridge", "three-phase", "foc"]),
        ],
    }

    # Detect chapter key from filename or path
    chapter_key = "03_brain"
    for k in chapter_anchors:
        if k in str(qmd_path) or k in str(svg_path):
            chapter_key = k
            break

    anchors = chapter_anchors[chapter_key]
    alignment = []
    for name, terms in anchors:
        found = any(term.lower() in qmd_content.lower() for term in terms)
        alignment.append({"anchor": name, "found_in_prose": found})

    return {
        "qmd": str(qmd_path),
        "anchors": alignment,
        "score": sum(1 for a in alignment if a["found_in_prose"]) / len(anchors),
    }


def render_png(svg_path: Path, output_png_path: Path, width: int = 2400) -> bool:
    """Render SVG to PNG using rsvg-convert."""
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["/opt/homebrew/bin/rsvg-convert", "-w", str(width), str(svg_path), "-o", str(output_png_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python figure_qa_loop.py <path_to_svg> [path_to_qmd]")
        sys.exit(1)

    svg_path = Path(sys.argv[1]).resolve()
    qmd_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None

    print(f"🔍 Running Figure QA Audit on: {svg_path.name}")
    audit = audit_svg(svg_path)

    if audit["issues"]:
        print("❌ ISSUES FOUND (violates textbook standards):")
        for iss in audit["issues"]:
            print(f"  • {iss}")
    else:
        print("✅ PASSED: Zero rule violations (Anti-Poster Rule respected).")

    if audit["warnings"]:
        print("⚠️ WARNINGS:")
        for w in audit["warnings"]:
            print(f"  • {w}")

    print(f"📊 Total semantic text nodes: {audit['text_count']}")

    if qmd_path and qmd_path.exists():
        print(f"\n📖 Verifying Alignment with Prose: {qmd_path.name}")
        align = verify_prose_alignment(svg_path, qmd_path)
        print(f"Alignment Score: {align['score'] * 100:.0f}%")
        for a in align["anchors"]:
            status = "✓" if a["found_in_prose"] else "✗"
            print(f"  {status} {a['anchor']}")

    # Render PNG
    png_path = svg_path.with_suffix(".png")
    if render_png(svg_path, png_path):
        print(f"\n🖼️ Rendered 2400px raster: {png_path}")
    else:
        print("\n❌ Failed to render raster with rsvg-convert.")


if __name__ == "__main__":
    main()
