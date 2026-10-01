"""Tests for the running machine's data sheet and card (mlsysim.embodied.datasheet).

The sheet feeds the Volume IV machine card (Chapter 1) and the full data sheet
(Systems appendix). These tests keep it complete, correctly categorized, and
tied to the registry values it prints.
"""

from __future__ import annotations

import pytest

from mlsysim import Embodied
from mlsysim.core.units import ureg
from mlsysim.embodied.datasheet import (
    CARD_KEYS,
    CATEGORIES,
    NEEDS_SOURCE,
    OWNER_PROVENANCE_CATEGORY,
    PROVENANCE_CATEGORY,
    SPECS,
    SUBSYSTEMS,
    data_sheet_lookup,
    format_value,
    machine_card,
    machine_data_sheet,
    resolve_field,
)
from mlsysim.embodied.types import EmbodiedSiteScenario, MobileManipulatorPlatform

AISLE = Embodied.Scenario.WarehouseAisle
RM = AISLE.machine

# Fields that are not numeric inputs of any calculation, with the reason.
_NOT_INPUTS = {
    "RM.name", "RM.archetype", "RM.metadata",
    # Composite components: their used fields are listed individually.
    "RM.base", "RM.arm", "RM.brain_soc", "RM.permission_mcu", "RM.nav_camera",
    "RM.wrist_camera", "RM.lidar", "RM.imu", "RM.intent_model", "RM.chunk_model",
    "RM.drive_unit", "RM.safety_scanner",
    # A decoding mode, not a number; the intent-latency row carries its effect.
    "RM.intent_parallel_decode",
    # Legacy 24 V control-rail fields that no calculation reads since the
    # 48 V drive bus (2026-09-28, M6 and pass R).
    "RM.control_rail_nominal", "RM.control_rail_resistance",
    "AISLE.name", "AISLE.machine", "AISLE.metadata",
}
# Datasheet values the registry stores as plain numbers with the source named in
# a comment (they cannot carry provenance until the registry wraps them).
_COMMENT_SOURCED = {"eff_conv"}  # MEANWELL_DDR240 typical efficiency
# Configured settings stored on a datasheet-sourced part: the datasheet bounds
# the range, but the stored value is chosen, so the row says Chosen (2026-09-29).
_CONFIGURED_SETTINGS = {"t_exposure"}
# Registry aliases: the sheet lists the scenario's name for the same value.
_ALIASES = {"RM.fieldbus_cycle": "AISLE.t_bus"}


def test_keys_unique_and_categories_known():
    keys = [s.key for s in SPECS]
    assert len(keys) == len(set(keys))
    for spec in SPECS:
        assert spec.category in CATEGORIES, spec.key
        assert spec.subsystem in SUBSYSTEMS, spec.key


def test_every_scalar_field_of_machine_and_scenario_is_listed():
    listed = {s.field for s in SPECS}
    for prefix, model in (("RM", MobileManipulatorPlatform), ("AISLE", EmbodiedSiteScenario)):
        for name in model.model_fields:
            path = f"{prefix}.{name}"
            if path in _NOT_INPUTS:
                continue
            assert _ALIASES.get(path, path) in listed, f"{path} missing from the data sheet"


def test_rows_resolve_and_match_the_registry():
    rows = machine_data_sheet()
    assert len(rows) == len(SPECS)
    for spec, row in zip(SPECS, rows):
        if spec.compute is None:
            leaf = resolve_field(spec.field, RM, AISLE)
            assert row.value is leaf or row.value == leaf, spec.key
        assert row.value_str, spec.key


def test_sourced_leaves_carry_the_category_of_their_provenance():
    for spec in SPECS:
        if spec.compute is not None:
            continue
        if spec.key in _CONFIGURED_SETTINGS:
            assert spec.category == "Chosen", spec.key
            continue
        root, *attrs = spec.field.split(".")
        owner = resolve_field(".".join([root, *attrs[:-1]]), RM, AISLE)
        leaf = getattr(owner, attrs[-1])
        prov = getattr(leaf, "provenance", None)
        if prov is not None:
            assert spec.category == PROVENANCE_CATEGORY[prov.kind.value], spec.key
            continue
        meta = getattr(owner, "metadata", None)
        owner_prov = getattr(meta, "provenance", None) if meta is not None else None
        is_field = attrs[-1] in getattr(type(owner), "model_fields", {})
        if (owner is not RM and owner is not AISLE and is_field and owner_prov is not None
                and owner_prov.kind.value in OWNER_PROVENANCE_CATEGORY):
            assert spec.category == OWNER_PROVENANCE_CATEGORY[owner_prov.kind.value], spec.key


def test_unsourced_scenario_values_are_never_labeled_measured_or_datasheet():
    # The machine and site scenario are estimates: only sourced leaves may claim
    # Measured or Datasheet evidence.
    for spec in SPECS:
        if spec.compute is not None or not spec.field.startswith(("RM.", "AISLE.")):
            continue
        root, *attrs = spec.field.split(".")
        owner = resolve_field(".".join([root, *attrs[:-1]]), RM, AISLE)
        if owner is not RM and owner is not AISLE:
            continue
        leaf = getattr(owner, attrs[-1])
        if leaf is None:
            continue  # a needs-source row names the evidence that would close it
        if getattr(leaf, "provenance", None) is None and spec.key not in _COMMENT_SOURCED:
            assert spec.category not in ("Measured", "Datasheet"), spec.key


def test_missing_values_say_needs_source():
    for row in machine_data_sheet():
        if row.value is None:
            assert row.value_str == NEEDS_SOURCE
            assert row.note.lower().startswith("needs source"), row.key
    lookup = data_sheet_lookup()
    assert lookup["cg_height"].value is None
    assert lookup["rotor_inertia"].value is None


def test_card_rows_and_values():
    card = machine_card()
    assert [r.key for r in card] == list(CARD_KEYS)
    got = {r.key: r for r in card}
    assert got["m_loaded"].value.to("kg").magnitude == pytest.approx(368.0)
    assert got["v_drive"].value_str == "1.5 m/s"
    assert got["v_aisle"].value_str == "1.3 m/s"
    assert got["a_brake"].value.to("m/s**2").magnitude == pytest.approx(2.0)
    assert got["j_brake"].value.to("m/s**3").magnitude == pytest.approx(64.0 / 9.0)
    assert got["d_clear"].value_str == "1.10 m"
    assert got["tau_delay"].value.to("ms").magnitude == pytest.approx(133.6)
    assert got["t_lease"].value_str == "60 ms"
    assert got["t_step"].value_str == "25 ms"
    assert got["t_period"].value_str == "50 ms"
    assert got["t_intent"].value_str == "336 ms"
    assert got["f_intent"].value_str == "2 Hz"
    assert got["t_takeover_in"].value_str == "605 ms"
    assert got["t_takeover_out"].value_str == "4.47 s"
    assert got["a_brake"].category == "Illustrative"
    assert got["v_aisle"].category == "Chosen"
    assert got["tau_delay"].category == "Derived"


def test_braking_premise_rows():
    lookup = data_sheet_lookup()
    assert lookup["mu_req"].value == pytest.approx(0.2039, abs=1e-4)
    assert lookup["mu_req_rated"].value_str == "0.250"
    assert lookup["load_share_max"].value_str == "0.815"


def test_format_value_rounds_half_up_and_refuses_bad_input():
    assert format_value(4.465 * ureg.second, "s", 2) == "4.47 s"
    assert format_value(1.0 * ureg.volt, "V", 2) == "1 V"
    assert format_value(None, "m", None) == NEEDS_SOURCE
    with pytest.raises(ValueError):
        format_value(1.0 * ureg.meter, None, 1)
    with pytest.raises(TypeError):
        format_value(True, None, None)
