from __future__ import annotations

from mlsysbook_labs import (
    adaptation_storage,
    edge_device_profile,
    energy_drain,
    federated_communication,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    training_memory_breakdown,
)


def _edge_profile(track_id: str):
    profile = get_track_profile(track_id)
    variant = get_lab_track_variant("v2_11_edge_thermodynamics", track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    return edge_device_profile(profile, variant, hardware)


def test_edge_device_profile_uses_mlsysim_hardware_capacity():
    iphone = _edge_profile("iphone")
    oura = _edge_profile("oura_ring")

    assert iphone.memory_capacity_mb == 8000.0
    assert iphone.energy_budget_wh == 15.0
    assert oura.memory_capacity_mb == 2.0
    assert oura.energy_budget_wh == 0.06
    assert oura.available_memory_mb < iphone.available_memory_mb


def test_training_memory_breakdown_strategy_changes_feasibility():
    profile = _edge_profile("iphone")

    full = training_memory_breakdown(
        params_m=profile.default_model_params_m,
        batch_size=profile.default_batch_size,
        strategy="full",
        available_memory_mb=profile.available_memory_mb,
    )
    lora = training_memory_breakdown(
        params_m=profile.default_model_params_m,
        batch_size=profile.default_batch_size,
        strategy="lora",
        available_memory_mb=profile.available_memory_mb,
    )

    assert full.total_mb > lora.total_mb
    assert full.amplification > lora.amplification
    assert isinstance(lora.fits_memory, bool)


def test_adaptation_storage_lora_is_smaller_than_full_tuning():
    result = adaptation_storage(contexts=10, model_mb=40.0)

    assert result.full_total_mb > result.lora_total_mb
    assert result.lora_total_mb > result.bias_total_mb
    assert result.lora_savings_ratio > 5.0


def test_energy_drain_accelerator_uses_less_budget_than_cpu():
    profile = _edge_profile("iphone")

    cpu = energy_drain(profile, target="cpu")
    npu = energy_drain(profile, target="npu")

    assert cpu.energy_wh > npu.energy_wh
    assert cpu.budget_used_pct > npu.budget_used_pct
    assert npu.label == profile.accelerator_label


def test_federated_communication_compression_reduces_total_bytes():
    profile = _edge_profile("robotaxi")

    raw = federated_communication(profile, beta=0.5, local_epochs=3, compression="none")
    int8 = federated_communication(profile, beta=0.5, local_epochs=3, compression="int8")

    assert raw.noniid_rounds > raw.iid_rounds
    assert int8.compressed_bytes_per_round_mb < raw.compressed_bytes_per_round_mb
    assert int8.total_communication_mb < raw.total_communication_mb
