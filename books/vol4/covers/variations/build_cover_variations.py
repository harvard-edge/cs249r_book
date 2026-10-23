#!/usr/bin/env python3
"""
Composite Volume IV (Physical AI) cover artwork variations onto the canonical
hardcover template (matching cover-hardcover-book-vol4.png, 867x1110) and
the high-resolution title template (matching cover-image-title-vol4.png, 1728x2160)
with seamless paper-tone multiply blending.
"""

from pathlib import Path
from PIL import Image
import numpy as np

VAR_DIR = Path(__file__).resolve().parent
COVERS_DIR = VAR_DIR.parent

VARIATIONS = [
    ("v01_causal_boundary_gate", "1. Causal Boundary Gate", "Ch 1: Digital graph crossing physical kinetic iris"),
    ("v02_spa_triad_ring", "2. Sense-Plan-Act Triad", "Ch 1-13: Three interlocked sensorimotor sectors"),
    ("v03_mobile_manipulator_envelope", "3. Reachable Workspace Shell", "Ch 2 & 12: Articulated arm inside safety bubble"),
    ("v04_dexterous_contact_sphere", "4. Dexterous Contact Closure", "Ch 2 & 10: Three-finger grasp on geodesic sphere"),
    ("v05_control_barrier_shield", "5. Control Barrier Manifold", "Ch 12 & 14: Simplex runtime safety shield"),
    ("v06_dual_clock_escapement", "6. Dual-Clock Escapement", "Ch 3, 4 & 13: Async ML brain vs kHz control gear"),
    ("v07_humanoid_crucible", "7. Whole-Body Kinetic Torus", "Ch 2 & 18: Double-helix tendon & origami ring"),
    ("v08_sim_to_real_duality", "8. Sim-to-Real Duality", "Ch 5, 6 & 7: Digital twin wireframe to physical linkage"),
    ("v09_sensorimotor_wavefront", "9. Sensorimotor Wavefront", "Ch 8, 9 & 11: Perception cone to trajectory fan"),
    ("v10_four_laws_armillary", "10. Four Laws Armillary", "Ch 1 & 15: Concentric gimbals (Conservation/Delay/Noise)"),
    ("v11_articulated_cantilever_reach", "11. Cantilever Manipulator Reach", "Non-circular: Ascending 6-DOF arm & origami fan"),
    ("v12_causal_horizon_prism", "12. Causal Boundary Diptych", "Non-circular: Graph lattice crossing vertical gate"),
    ("v13_inverted_pendulum_equilibrium", "13. Inverted Pendulum Balance", "Non-circular: Dynamic equilibrium & safety cones"),
    ("v14_bimanual_convergence_bridge", "14. Bimanual Dexterous Bridge", "Non-circular: Opposing arms grasping crystal core"),
    ("v15_kinodynamic_wing_wavefront", "15. Kinodynamic Wing Wavefront", "Non-circular: Swept S-curve origami trajectory wing"),
    ("v16_tripod_parallel_delta", "16. Parallel Delta Tripod", "Non-circular: Three-legged kinematic pyramid"),
    ("v17_swave_spine_crystal_grasp", "17. S-Wave Vertebral Grasp (15+11A)", "65% #15 + 35% #11: Origami S-wing with vertebral spine"),
    ("v18_cantilever_origami_arm_grasp", "18. Origami-Wrapped Cantilever Arm (15+11B)", "50% #15 + 50% #11: Rotary linkage arm in pleated wings"),
    ("v19_swave_dexterous_wing_grasp", "19. Three-Finger Wing Trajectory (15+11C)", "55% #15 + 45% #11: Sweeping wing + 3-finger hand"),
    ("v20_braided_origami_manipulator", "20. Braided Origami & Titanium Arm (15+11D)", "45% #15 + 55% #11: Intertwined ribbon & actuator arm"),
]


def clean_art_to_pure_white(art_img: Image.Image) -> Image.Image:
    """Ensure the outer studio border of the 1024x1024 art is pure #FFFFFF."""
    arr = np.array(art_img.convert("RGB")).astype(np.float32)
    h, w, _ = arr.shape
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    sat = arr.max(axis=2) - arr.min(axis=2)

    bg_strength = np.clip((luma - 246.0) / 7.0, 0.0, 1.0) * np.clip((8.0 - sat) / 8.0, 0.0, 1.0)
    arr = arr * (1.0 - bg_strength[:, :, None]) + 255.0 * bg_strength[:, :, None]

    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    margin = 42.0
    d_edge = np.minimum(np.minimum(xs, w - 1 - xs), np.minimum(ys, h - 1 - ys)).astype(np.float32)
    edge_t = np.clip(d_edge / margin, 0.0, 1.0)
    edge_alpha = edge_t * edge_t * (3.0 - 2.0 * edge_t)
    arr = arr * edge_alpha[:, :, None] + 255.0 * (1.0 - edge_alpha[:, :, None])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def build_clean_hardcover_paper_canvas(orig_hc: np.ndarray) -> np.ndarray:
    """Build a clean 867x1110 hardcover template with continuous paper tone across the art region."""
    canvas = orig_hc.copy().astype(np.float32)
    for y in range(0, 745):
        left_rgb = np.median(canvas[y, 65:125, :], axis=0)
        row_fill = np.tile(left_rgb[None, :], (867 - 65, 1))
        if y >= 715:
            t = (y - 715) / 30.0
            row_fill = row_fill * (1.0 - t) + canvas[y, 65:867, :] * t
        canvas[y, 65:867, :] = row_fill
    return np.clip(canvas, 0, 255).astype(np.uint8)


def build_clean_title_canvas(orig_title: np.ndarray) -> np.ndarray:
    """Build a clean 1728x2160 title template with pure white background above the typography."""
    canvas = orig_title.copy()
    canvas[0:1468, :, :] = 255
    canvas[1468:1520, 480:, :] = 255
    return canvas


def composite_art_onto_paper(paper_canvas: np.ndarray, art_img: Image.Image, x0: int, y0: int, target_w: int) -> Image.Image:
    """Composite a white-background art image onto a paper canvas with zero bounding-box seam."""
    white_art = clean_art_to_pure_white(art_img)
    art_resized = np.array(white_art.resize((target_w, target_w), Image.Resampling.LANCZOS)).astype(np.float32)

    out = paper_canvas.copy().astype(np.float32)
    patch_bg = out[y0:y0 + target_w, x0:x0 + target_w, :]

    luma = 0.299 * art_resized[:, :, 0] + 0.587 * art_resized[:, :, 1] + 0.114 * art_resized[:, :, 2]
    sat = art_resized.max(axis=2) - art_resized.min(axis=2)

    mult_patch = (art_resized / 255.0) * patch_bg
    solid_weight = np.clip((235.0 - luma) / 25.0 + sat / 18.0, 0.0, 1.0)
    blended = mult_patch * (1.0 - solid_weight[:, :, None]) + art_resized * solid_weight[:, :, None]

    out[y0:y0 + target_w, x0:x0 + target_w, :] = blended
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    base_hc = build_clean_hardcover_paper_canvas(
        np.array(Image.open(COVERS_DIR / "cover-hardcover-book-vol4.png").convert("RGB"))
    )
    base_title = build_clean_title_canvas(
        np.array(Image.open(COVERS_DIR / "cover-image-title-vol4.png").convert("RGB"))
    )

    for stem, title, subtitle in VARIATIONS:
        art_path = VAR_DIR / f"{stem}_art.png"
        if not art_path.exists():
            continue
        art_img = Image.open(art_path)
        hc_img = composite_art_onto_paper(base_hc, art_img, x0=140, y0=5, target_w=650)
        title_img = composite_art_onto_paper(base_title, art_img, x0=235, y0=155, target_w=1258)

        hc_img.save(VAR_DIR / f"{stem}_hardcover.png", "PNG", optimize=True)
        hc_img.save(VAR_DIR / f"{stem}_hardcover.webp", "WEBP", quality=88, method=6)
        title_img.save(VAR_DIR / f"{stem}_title.png", "PNG", optimize=True)
        print(f"Built {stem}")


if __name__ == "__main__":
    main()
