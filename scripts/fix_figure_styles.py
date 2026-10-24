#!/usr/bin/env python3
"""
fix_figure_styles.py
====================
Patches styling anomalies across Volume IV figures:
1. Removes outer wrapper border strokes on canvas base rectangles (violates borderless page rule).
2. Adds standard root font-family to SVGs containing text that lack it.
3. Fixes bubbly container radii (rx > 6) on state machine boxes to rx="4".
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VOL4_DIR = REPO_ROOT / "books" / "vol4"

FONT_DECL = 'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif"'

def fix_outer_canvas_strokes():
    files_to_fix = [
        VOL4_DIR / "01_boundary/images/svg/chk_boundary_contact_stiffness.svg",
        VOL4_DIR / "01_boundary/images/svg/chk_boundary_latency_budget.svg",
        VOL4_DIR / "02_body/images/svg/chk_body_inertia_matching.svg",
        VOL4_DIR / "02_body/images/svg/chk_body_thermal_stall.svg",
        VOL4_DIR / "03_brain/images/svg/chk_brain_chunk_horizon_budget.svg",
        VOL4_DIR / "03_brain/images/svg/psp_brain_humanoid_tipping_latency.svg",
        VOL4_DIR / "04_nervous/images/svg/chk_nervous_seqlock_collision.svg",
        VOL4_DIR / "04_nervous/images/svg/chk_nervous_silicon_partitioning.svg",
        VOL4_DIR / "06_training/images/svg/chk_training_diffusion_denoising_budget.svg",
        VOL4_DIR / "06_training/images/svg/chk_training_residual_bounded_authority.svg",
        VOL4_DIR / "13_placement/images/svg/chk_placement_thermal_throttling_hysteresis.svg",
        VOL4_DIR / "16_release/images/svg/dfn_envelope_trichotomy.svg",
    ]

    count = 0
    for p in files_to_fix:
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8")
        orig = content

        # Replace stroke on base rect
        # e.g. <rect width="540" height="180" fill="#ffffff" rx="4" stroke="#e2e8f0" stroke-width="1"/>
        # -> <rect width="540" height="180" fill="#ffffff"/>
        content = re.sub(
            r'(<rect\b[^>]*?\bfill=[\'"](?:#ffffff|#fff|white)[\'"][^>]*?)\s+stroke=[\'"]#e2e8f0[\'"]\s+stroke-width=[\'"]1[\'"]',
            r'\1',
            content,
            flags=re.IGNORECASE
        )
        # Also remove rx="4" on full canvas rect if it's the base canvas
        content = re.sub(
            r'(<rect\b[^>]*?)\s+rx=[\'"]4[\'"](\s+fill=[\'"](?:#ffffff|#fff|white)[\'"])',
            r'\1\2',
            content,
            flags=re.IGNORECASE
        )

        if content != orig:
            p.write_text(content, encoding="utf-8")
            count += 1
            print(f"  ✓ Removed outer canvas border stroke in {p.relative_to(VOL4_DIR)}")

    print(f"Fixed outer canvas border stroke in {count} files.")


def fix_missing_root_fonts():
    svgs = sorted(list(VOL4_DIR.glob("**/images/svg/*.svg")))
    count = 0

    for s in svgs:
        content = s.read_text(encoding="utf-8")
        if "<text" not in content:
            continue

        root_m = re.search(r"<svg\b([^>]*)>", content)
        if not root_m:
            continue

        attrs = root_m.group(1)
        if "font-family=" in attrs:
            continue

        # Add font-family to root <svg>
        new_tag = f"<svg{attrs} {FONT_DECL}>"
        content = content[:root_m.start()] + new_tag + content[root_m.end():]
        s.write_text(content, encoding="utf-8")
        count += 1
        print(f"  ✓ Added root font-family to {s.relative_to(VOL4_DIR)}")

    print(f"Added root font-family to {count} SVGs.")


def fix_bubbly_radii():
    p = VOL4_DIR / "09_memory/images/svg/fig09_spatial_belief_state_machine.svg"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        orig = content
        # Change rx="8" to rx="4" on state boxes
        content = content.replace('rx="8"', 'rx="4"')
        if content != orig:
            p.write_text(content, encoding="utf-8")
            print(f"  ✓ Fixed bubbly rx in {p.relative_to(VOL4_DIR)}")


def main():
    print("Fixing figure styling anomalies across Volume IV...")
    fix_outer_canvas_strokes()
    fix_missing_root_fonts()
    fix_bubbly_radii()
    print("✅ All figure styling fixes applied!")

if __name__ == "__main__":
    main()
