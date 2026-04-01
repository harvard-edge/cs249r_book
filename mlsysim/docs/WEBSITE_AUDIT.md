# MLSys-im Website Audit

**Date:** 2026-04-01
**Auditor:** Claude Code
**Scope:** Landing page, Getting Started, For Students, sidebar config

---

## What's Good

- **Positioning is on-target.** The subtitle ("Build intuition for ML system performance, cost, and carbon -- from first principles") and the footer ("First-principles ML systems modeling") align well with the "first-principles analytical calculator" identity. The Open Graph and Twitter Card metadata reinforce this.
- **Site structure is well-organized.** The sidebar groups content logically: Get Started, Tutorials (with Foundations/Optimization/Scale subsections), The Zoo, For Your Role, Concepts, Build & Extend, API Reference, Project.
- **For Students page is solid.** Clear learning path table, good FAQ, accurate solver-to-textbook mapping, correct prerequisites statement. The "Predict Before You Compute" callout is excellent pedagogy.
- **Getting Started page is thorough.** Covers installation, first analysis, CLI, custom models, HuggingFace import, YAML-based evaluation, troubleshooting, and efficiency tuning. The Pint units callout is helpful.
- **All internal page references resolve.** Every qmd file linked from the sidebar and inline links exists on disk.
- **Interactive Apps section is correctly commented out** in the sidebar (lines 105-114 of `_quarto.yml`) with a clear note about re-enabling when Marimo export is wired in.

---

## What Needs Fixing

### Critical: Landing Page Output Is Wrong

The hero "See it in action" section (lines 56-87 of `index.qmd`) shows hardcoded output that does not match the current engine:

| Field | Landing Page Claims | Actual (`Engine.solve(ResNet50, A100, bs=1, fp16)`) |
|-------|--------------------|----------------------------------------------------|
| Bottleneck | Memory Bound | **Compute** |
| Latency | 0.42 ms | **0.57 ms** |
| Throughput | 2,381 img/s | **1,762 img/s** |
| MFU | 12.4% | **4.6%** |
| Memory | 0.10 GB / 80 GB | Not shown in current profile |
| AI (FLOP/B) | 4.2 | Not shown in current profile |

**Priority: HIGH.** This is the first thing users see. The bottleneck classification is outright wrong, which undermines credibility for a tool that claims to do exactly this analysis.

### Critical: Getting Started Output Is Wrong

The "Your First Analysis" section (lines 49-64 of `getting-started.qmd`) shows inline comments claiming:

- `profile.bottleneck` -> `'Memory Bound'` (actual: `'Compute'`)
- `profile.latency` -> `0.34 ms` (actual: `0.57 ms`)
- `profile.throughput` -> `2941 samples/sec` (actual: `1762 1/s`)

**Priority: HIGH.** Students running this code will see different output than documented, causing immediate confusion.

### Moderate: `sustainability_lab.py` Used Dict Access on Result Object

The example at `mlsysim/examples/sustainability_lab.py` used `impact['total_energy_kwh']` and `impact['carbon_footprint_kg']` (dict-style access), but `SustainabilityResult` is a Pydantic model requiring attribute access (`.total_energy_kwh`, `.carbon_footprint_kg`).

**Status: FIXED** in this audit. Changed to attribute access.

### Moderate: Stale Installed Version Causes Example Failures

Running examples as scripts (e.g., `python3 mlsysim/examples/hardware_comparison.py`) from outside the repo root fails because the system-installed mlsysim (site-packages) is an older version missing newer registry entries like `Hardware.Tiny.nRF52840`. This affects `hardware_comparison.py`, `sustainability_lab.py`, and `custom_design.py`.

**Recommendation:** Add a note to the examples README or docstrings about running from the repo root, or add `pip install -e .` instructions. Alternatively, ensure the PyPI package stays in sync.

### Minor: Website Title Inconsistency

- `_quarto.yml` line 15: `"MLSys·IM - Machine Learning Systems"` (IM uppercase)
- Landing page title: `"MLSys·im"` (im lowercase)
- Open Graph: `"MLSys·IM"` (IM uppercase)
- Footer: `"MLSys·im"` (im lowercase)

Pick one casing and use it consistently. The lowercase `MLSys·im` appears more often and reads better.

### Minor: Landing Page Stats May Be Stale

The "Model every constraint" section claims 21 solvers, 22 system walls, 6 constraint domains. These should be verified against the current `core/solver.py` module. The `_quarto.yml` quartodoc section lists ~23 solver classes, suggesting the count may be slightly off.

### Minor: Colab/Binder Launch Buttons Referenced But Not Implemented

Getting Started (line 39-41) mentions "Google Colab or Binder" launch buttons at the top of tutorials, but these do not appear to be implemented yet. Either add them or soften the language to "will be available."

---

## Priority Recommendations

1. **Fix the hero output block** on the landing page to match current `Engine.solve()` results, or change the example to a workload that IS memory-bound at batch_size=1 (e.g., Llama3_8B on H100, which is memory-bound).
2. **Fix the Getting Started inline output comments** to match current engine results.
3. **Standardize the product name casing** to `MLSys·im` everywhere.
4. **Verify solver/wall counts** in the landing page stats section.
5. **Soften or remove Colab/Binder language** until launch buttons are actually wired in.
6. **Re-install the local package** (`pip install -e .`) so examples work regardless of working directory.
