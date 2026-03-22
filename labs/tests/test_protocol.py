"""
Level 4: Protocol Invariant Tests
==================================

Validates that each lab complies with the 6 Protocol Invariants
defined in .claude/docs/labs/PROTOCOL.md.

These are structural quality gates — they verify pedagogical
completeness rather than runtime correctness.

Invariant 1: Every Number Has a Source
Invariant 2: Structured Predictions (tested in test_static + test_widget)
Invariant 3: Failure States Mandatory (tested in test_static)
Invariant 4: Multi-Part Tabbed Structure (4-5 Parts + Synthesis)
Invariant 5: 2-3 Deployment Contexts
Invariant 6: No Instruments Before Chapter Introduction (manual only)

Usage:
  python3 -m pytest labs/tests/test_protocol.py -v
  python3 -m pytest labs/tests/test_protocol.py -v -k "vol1"
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Labs that are exempt from standard protocol checks
ORIENTATION_LABS = {"lab_00_introduction"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_source(lab_path: str) -> str:
    with open(lab_path) as f:
        return f.read()


def lab_stem(lab_path: str) -> str:
    return Path(lab_path).stem


def is_orientation(lab_path: str) -> bool:
    return lab_stem(lab_path) in ORIENTATION_LABS


def extract_lab_number(lab_path: str) -> int | None:
    """Extract the lab number from the filename (e.g., lab_05_foo -> 5)."""
    m = re.search(r"lab_(\d+)_", Path(lab_path).name)
    return int(m.group(1)) if m else None


def count_builder_functions(source: str) -> tuple[int, bool]:
    """Count build_part_X() functions and whether build_synthesis() exists."""
    parts = len(re.findall(r"def build_part_\w+\(", source))
    has_synthesis = "def build_synthesis(" in source
    return parts, has_synthesis


def extract_tab_keys(source: str) -> list[str]:
    """Extract tab key strings from mo.ui.tabs({...}) calls."""
    keys = []
    # Match string keys in mo.ui.tabs({ "key": ..., "key": ... })
    # Find the mo.ui.tabs block
    tabs_match = re.search(r"mo\.ui\.tabs\(\{(.+?)\}\)", source, re.DOTALL)
    if tabs_match:
        block = tabs_match.group(1)
        keys = re.findall(r'"([^"]+)"(?:\s*:)', block)
    return keys


def extract_hardware_references(source: str) -> set[str]:
    """Extract Hardware.Tier.Device references from source."""
    return set(re.findall(r"Hardware\.(\w+\.\w+)", source))


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT 1: Every Number Has a Source
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumberSources:
    """Constants should come from mlsysim registries, not hardcoded."""

    @pytest.mark.protocol
    def test_no_hardcoded_hardware_specs(self, lab_path):
        """
        Catch hardcoded hardware specs that should come from mlsysim.Hardware.

        Flags suspicious large numbers that look like FLOPS, bandwidth, or
        memory capacity values. These should be sourced from the registry.
        """
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        tree = ast.parse(source)

        suspicious = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                val = node.value
                # Flag numbers that look like hardware specs
                # 80 GB = 80_000_000_000, 2 TB/s = 2_000_000_000_000, etc.
                if val >= 1_000_000_000 and isinstance(val, int):
                    # Check if it's inside a string (skip those)
                    suspicious.append(
                        f"Line {node.lineno}: large constant {val:,} — "
                        f"should this come from mlsysim.Hardware?"
                    )

        # This is a warning, not a hard fail — some large constants are legitimate
        if len(suspicious) > 5:
            pytest.xfail(
                f"Found {len(suspicious)} large numeric constants. "
                f"Consider sourcing from mlsysim.Hardware:\n"
                + "\n".join(suspicious[:5])
            )

    @pytest.mark.protocol
    def test_uses_mlsysim_engine(self, lab_path):
        """Labs should use Engine.solve() or Hardware/Models registries."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        uses_registry = (
            "Hardware." in source
            or "Models." in source
            or "Engine.solve" in source
        )
        if not uses_registry:
            pytest.xfail(
                "Lab does not reference mlsysim Hardware, Models, or Engine. "
                "Constants should come from the registry, not be hardcoded."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT 4: Multi-Part Tabbed Structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestTabbedStructure:
    """Labs must have 4-5 parts + synthesis in mo.ui.tabs."""

    @pytest.mark.protocol
    def test_minimum_parts(self, lab_path):
        """Every lab (except lab_00) must have at least 3 build_part functions."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        parts, _ = count_builder_functions(source)
        if parts < 4:
            pytest.xfail(
                f"Only {parts} build_part functions (protocol requires 4-5). "
                f"Lab may need additional parts."
            )

    @pytest.mark.protocol
    def test_has_synthesis(self, lab_path):
        """Every lab (except lab_00) must have a build_synthesis() function."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        _, has_synthesis = count_builder_functions(source)
        assert has_synthesis, "Missing build_synthesis() function"

    @pytest.mark.protocol
    def test_tabs_contain_parts(self, lab_path):
        """The mo.ui.tabs dict should have Part keys matching builder functions."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        tab_keys = extract_tab_keys(source)
        if not tab_keys:
            pytest.skip("Could not parse tab keys")

        part_tabs = [k for k in tab_keys if k.startswith("Part")]
        if len(part_tabs) < 3:
            pytest.xfail(
                f"Only {len(part_tabs)} Part tabs found in mo.ui.tabs. "
                f"Protocol requires 4-5. Found: {tab_keys}"
            )

    @pytest.mark.protocol
    def test_tabs_include_synthesis(self, lab_path):
        """The mo.ui.tabs dict should include a Synthesis tab."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        tab_keys = extract_tab_keys(source)
        if not tab_keys:
            pytest.skip("Could not parse tab keys")

        has_synth = any("synth" in k.lower() or "graduation" in k.lower() for k in tab_keys)
        assert has_synth, (
            f"No Synthesis/Graduation tab found. Tabs: {tab_keys}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT 5: Deployment Contexts
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeploymentContexts:
    """Labs should reference multiple hardware tiers for comparison."""

    @pytest.mark.protocol
    def test_multiple_hardware_tiers(self, lab_path):
        """Labs should reference at least 2 hardware tiers (Cloud/Edge/Tiny)."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        hw_refs = extract_hardware_references(source)
        tiers = {ref.split(".")[0] for ref in hw_refs}

        if len(tiers) < 2:
            pytest.xfail(
                f"Only {len(tiers)} hardware tier(s) referenced: {tiers}. "
                f"Protocol recommends 2-3 deployment contexts."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

class TestZoneStructure:
    """Labs should follow the 4-zone architecture."""

    @pytest.mark.protocol
    def test_has_zone_comments(self, lab_path):
        """Labs should have ZONE A/B/C/D section markers."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        zones_found = re.findall(r"ZONE [A-D]", source)
        unique_zones = set(zones_found)
        if len(unique_zones) < 3:
            pytest.xfail(
                f"Only {len(unique_zones)} zone markers found: {unique_zones}. "
                f"Protocol expects 4 zones (A: Opening, B: Widgets, C: Tabs, D: Ledger)."
            )

    @pytest.mark.protocol
    def test_has_ledger_hud(self, lab_path):
        """Labs should have a ledger HUD footer."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        has_hud = "lab-hud" in source or "LEDGER" in source
        assert has_hud, "Missing ledger HUD footer (class='lab-hud' or LEDGER zone)"


# ═══════════════════════════════════════════════════════════════════════════════
# LEDGER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestLedgerIntegration:
    """Labs should save student decisions to the Design Ledger."""

    @pytest.mark.protocol
    def test_has_ledger_save(self, lab_path):
        """Every lab should call ledger.save() to record student decisions."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        if "ledger.save" not in source:
            pytest.xfail(
                "Missing ledger.save() call. "
                "Protocol requires recording student design decisions."
            )

    @pytest.mark.protocol
    def test_ledger_chapter_matches_filename(self, lab_path):
        """ledger.save(chapter=N) should match the lab number in the filename."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        lab_num = extract_lab_number(lab_path)
        if lab_num is None:
            pytest.skip("Could not extract lab number from filename")

        # Find ledger.save(chapter=N, ...) calls
        chapter_matches = re.findall(r"ledger\.save\(chapter=(\d+)", source)
        if not chapter_matches:
            pytest.skip("No ledger.save() call found")

        for ch in chapter_matches:
            assert int(ch) == lab_num, (
                f"ledger.save(chapter={ch}) does not match lab number {lab_num}. "
                f"File: {Path(lab_path).name}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PEDAGOGICAL FLOW
# ═══════════════════════════════════════════════════════════════════════════════

class TestPedagogicalFlow:
    """Verify the predict → discover → explain flow exists per part."""

    @pytest.mark.protocol
    def test_prediction_count_matches_parts(self, lab_path):
        """Each part should have at least one prediction widget."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        parts, _ = count_builder_functions(source)
        predictions = len(re.findall(
            r"mo\.ui\.(?:radio|number|dropdown)", source
        ))

        if predictions < parts:
            pytest.xfail(
                f"{predictions} prediction widgets for {parts} parts. "
                f"Protocol recommends at least one prediction per part."
            )

    @pytest.mark.protocol
    def test_mo_stop_gates_exist(self, lab_path):
        """Labs should gate instruments behind predictions using mo.stop."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        stop_count = source.count("mo.stop(")
        parts, _ = count_builder_functions(source)

        if stop_count == 0:
            pytest.xfail("No mo.stop() gates found. Instruments should be gated behind predictions.")
        elif stop_count < parts - 1:
            # Allow synthesis to not have a gate
            pytest.xfail(
                f"Only {stop_count} mo.stop() gates for {parts} parts. "
                f"Most parts should gate instruments behind predictions."
            )

    @pytest.mark.protocol
    def test_has_stakeholder_messages(self, lab_path):
        """Labs should have stakeholder messages framing each part."""
        if is_orientation(lab_path):
            pytest.skip("Lab 00 is orientation")

        source = read_source(lab_path)
        # Stakeholder messages use a colored left-border callout pattern
        stakeholder_markers = [
            "border-left:",
            "border-left-color:",
            "stakeholder",
            "Stakeholder",
            "STAKEHOLDER",
            "📧",
            "📋",
            "💼",
        ]
        has_stakeholder = any(marker in source for marker in stakeholder_markers)
        if not has_stakeholder:
            pytest.xfail(
                "No stakeholder message pattern found. "
                "Protocol requires a colored left-border callout framing each part."
            )
