from __future__ import annotations

from mlsysbook_labs import (
    data_pipeline_profile,
    evaluate_pipeline,
    get_lab_track_variant,
    get_track_profile,
    movement_frontier,
    pipeline_architecture,
    resolve_mlsysim_ref,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_04_data_gravity", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return data_pipeline_profile(track, variant, hardware, model)


def test_data_pipeline_profiles_resolve_refs_and_strategies():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref.startswith("Hardware.")
        assert profile.model_ref.startswith("Models.")
        assert profile.data_source
        assert profile.data_rate_mb_s > 0
        assert profile.strategies
        assert profile.retention_options


def test_pipeline_bottlenecks_are_track_specific():
    bottlenecks = {}
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)
        result = evaluate_pipeline(profile, sample_multiplier=profile.default_sample_multiplier)
        bottlenecks[track_id] = result.bottleneck_stage

        assert result.daily_raw_gb > 0
        assert result.stages

    assert bottlenecks["iphone"] == "upload/movement"
    assert bottlenecks["oura_ring"] in {"retention storage", "upload/movement"}
    assert bottlenecks["robotaxi"] in {"storage write", "upload/movement", "retention storage"}
    assert bottlenecks["cloud_fleet"] == "preprocess"


def test_movement_strategy_reduces_data_and_preserves_risk_text():
    profile = _profile("robotaxi")
    raw = movement_frontier(profile, strategy_id="full_fleet_upload", dataset_gb=1000, network_gbps=10)
    mined = movement_frontier(profile, strategy_id="local_event_mining", dataset_gb=1000, network_gbps=10)

    assert mined.data_moved_gb < raw.data_moved_gb
    assert mined.egress_cost < raw.egress_cost
    assert mined.quality_retained_pct < raw.quality_retained_pct
    assert mined.residual_risk


def test_pipeline_architecture_summarizes_decision():
    profile = _profile("cloud_fleet")
    pipeline = evaluate_pipeline(profile, sample_multiplier=profile.default_sample_multiplier)
    movement = movement_frontier(profile, strategy_id="regional_cache", dataset_gb=2500, network_gbps=25)
    architecture = pipeline_architecture(
        profile,
        pipeline,
        movement,
        retention_policy=profile.retention_options[0],
    )

    assert architecture.strategy_label == "Regional cache and prefetch"
    assert architecture.bottleneck_stage == pipeline.bottleneck_stage
    assert architecture.accepted_data_risk == movement.residual_risk
    assert architecture.memo_summary
