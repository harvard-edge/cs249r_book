import pytest

from mlsysim import Models, Systems
from mlsysim.core.units import Q_
from mlsysim.engine.v2_05_experiments import (
    ConvergenceFixture,
    compare_straggler_policy,
    get_capacity_dimensions,
    get_parallel_layout_specs,
    get_pipeline_dimensions,
    get_track_scenario,
    get_quality_fixtures,
    memory_plan_record,
    parallel_layout,
    pipeline_schedule,
    scaling_sweep,
    time_to_quality,
    training_memory_plan,
)


@pytest.mark.parametrize("track", ["tinyml", "mobile", "edge", "cloud"])
def test_track_endpoints_are_honest_upstream_training_jobs(track):
    scenario = get_track_scenario(track)

    assert scenario.fleet.total_accelerators >= max(scenario.worker_counts)
    assert scenario.model.parameters.magnitude > 0
    assert "upstream" in scenario.applicability_note
    if track != "cloud":
        assert not scenario.model_parallel_applicable


def test_unknown_track_is_rejected():
    with pytest.raises(ValueError, match="unknown track"):
        get_track_scenario("mcu-mesh")


def test_strong_and_weak_scaling_hold_different_inputs_fixed():
    scenario = get_track_scenario("edge")
    workers = (8, 16)

    strong = scaling_sweep(scenario, workers, mode="strong", global_batch=256)
    weak = scaling_sweep(scenario, workers, mode="weak", local_batch=16)

    assert [point.global_batch for point in strong] == [256, 256]
    assert [point.local_batch for point in weak] == [16, 16]
    assert weak[1].global_batch == 2 * weak[0].global_batch
    assert strong[1].local_batch < strong[0].local_batch


def test_slower_fabric_increases_scaling_communication_time():
    scenario = get_track_scenario("cloud")
    workers = (256,)
    fast_fleet = scenario.fleet.model_copy(update={"fabric": Systems.Fabrics.InfiniBand_NDR})
    slow_fleet = scenario.fleet.model_copy(update={"fabric": Systems.Fabrics.Ethernet_10G})

    fast = scaling_sweep(
        scenario.__class__(**{**scenario.__dict__, "fleet": fast_fleet}),
        workers,
        mode="strong",
    )[0]
    slow = scaling_sweep(
        scenario.__class__(**{**scenario.__dict__, "fleet": slow_fleet}),
        workers,
        mode="strong",
    )[0]

    assert slow.communication_time > fast.communication_time
    assert slow.step_time > fast.step_time


def test_noncausal_deployment_label_does_not_change_scaling_result():
    scenario = get_track_scenario("mobile")
    renamed = scenario.__class__(
        **{**scenario.__dict__, "deployment_target": "renamed deployment audience"}
    )

    original = scaling_sweep(scenario, (8,), mode="strong")[0]
    comparison = scaling_sweep(renamed, (8,), mode="strong")[0]

    assert comparison.step_time == original.step_time
    assert comparison.communication_time == original.communication_time


def test_equal_device_layout_exposes_cross_node_tensor_parallel_cost():
    model = Models.Language.Llama3_8B
    fleet = Systems.Clusters.Lab_64_H100
    common = dict(
        model=model,
        fleet=fleet,
        global_batch=256,
        seq_len=1024,
        microbatch_count=8,
        zero_stage=3,
    )

    within_node = parallel_layout(dp_size=8, tp_size=8, pp_size=1, **common)
    across_nodes = parallel_layout(dp_size=4, tp_size=16, pp_size=1, **common)

    assert within_node.devices == across_nodes.devices == 64
    assert within_node.tensor_parallel_tier == "intra-node"
    assert across_nodes.tensor_parallel_tier == "inter-node"
    assert across_nodes.tp_communication_time > within_node.tp_communication_time


def test_parallel_layout_rejects_unequal_device_budget():
    with pytest.raises(ValueError, match="must equal"):
        parallel_layout(
            Models.Language.Llama3_8B,
            Systems.Clusters.Lab_64_H100,
            global_batch=128,
            seq_len=1024,
            dp_size=4,
            tp_size=8,
            pp_size=1,
            microbatch_count=8,
        )


def test_zero_sharding_can_turn_an_oom_plan_into_a_feasible_plan():
    model = Models.Language.Llama3_70B
    fleet = Systems.Clusters.Lab_64_H100
    common = dict(
        global_batch=64,
        seq_len=1024,
        tp_size=8,
        pp_size=1,
        dp_size=8,
        microbatch_count=8,
        activation_checkpointing="full",
    )

    replicated = training_memory_plan(model, fleet, zero_stage=0, **common)
    sharded = training_memory_plan(model, fleet, zero_stage=3, **common)

    assert not replicated.feasible
    assert sharded.feasible
    assert sharded.weights < replicated.weights
    assert sharded.gradients < replicated.gradients
    assert sharded.optimizer_state < replicated.optimizer_state


def test_pipeline_microbatches_reduce_bubble_but_retain_more_activations():
    model = Models.Language.Llama3_8B
    fleet = Systems.Clusters.Lab_64_H100
    common = dict(
        model=model,
        fleet=fleet,
        microbatch_size=8,
        seq_len=2048,
        pp_size=8,
        dp_size=1,
        tp_size=1,
        activation_checkpointing="none",
    )

    few = pipeline_schedule(microbatch_count=2, **common)
    many = pipeline_schedule(microbatch_count=8, **common)

    assert many.bubble_fraction < few.bubble_fraction
    assert many.retained_microbatches > few.retained_microbatches
    assert many.retained_activation_memory > few.retained_activation_memory


def test_invalid_parallelism_plan_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        training_memory_plan(
            Models.Language.Llama3_8B,
            Systems.Clusters.Lab_64_H100,
            global_batch=64,
            seq_len=1024,
            tp_size=8,
            pp_size=8,
            dp_size=2,
        )


def test_fastest_step_can_lose_at_time_to_common_quality():
    fixture = ConvergenceFixture(
        name="illustrative staleness calibration",
        minimum_steps=1_000,
        critical_batch=256,
    )
    synchronous = time_to_quality(
        name="synchronous",
        step_time=Q_(1.0, "second"),
        global_batch=256,
        fixture=fixture,
    )
    stale = time_to_quality(
        name="drop stragglers",
        step_time=Q_(0.8, "second"),
        global_batch=256,
        fixture=fixture,
        work_multiplier=1.5,
    )

    assert stale.step_time < synchronous.step_time
    assert stale.optimizer_steps > synchronous.optimizer_steps
    assert stale.time_to_quality > synchronous.time_to_quality


def test_larger_batch_has_diminishing_step_reduction():
    fixture = ConvergenceFixture("batch noise fixture", 1_000, 256)
    small = time_to_quality(
        name="small batch",
        step_time=Q_(1, "second"),
        global_batch=128,
        fixture=fixture,
    )
    medium = time_to_quality(
        name="medium batch",
        step_time=Q_(1, "second"),
        global_batch=256,
        fixture=fixture,
    )
    large = time_to_quality(
        name="large batch",
        step_time=Q_(1, "second"),
        global_batch=512,
        fixture=fixture,
    )

    first_gain = small.optimizer_steps - medium.optimizer_steps
    second_gain = medium.optimizer_steps - large.optimizer_steps
    assert first_gain > second_gain > 0


def test_straggler_fixture_is_labeled_and_can_reverse_fastest_step_ranking():
    fixture, policy = get_quality_fixtures("cloud")
    comparison = compare_straggler_policy(
        step_time=Q_(2, "second"),
        global_batch=256,
        fixture=fixture,
        policy=policy,
    )

    assert "illustrative" in fixture.name
    assert comparison.intervention.step_time < comparison.baseline.step_time
    assert comparison.intervention.time_to_quality > comparison.baseline.time_to_quality


def test_memory_serializer_preserves_units_and_exact_inputs():
    inputs = {
        "model": Models.Language.Llama3_8B.name,
        "fleet": Systems.Clusters.Lab_64_H100.name,
        "global_batch": 32,
        "seq_len": 512,
        "tp_size": 8,
        "pp_size": 1,
        "dp_size": 8,
        "zero_stage": 3,
        "microbatch_count": 4,
        "gradient_accumulation_steps": 1,
        "precision": "fp16",
        "activation_checkpointing": "full",
    }
    plan = training_memory_plan(
        Models.Language.Llama3_8B,
        Systems.Clusters.Lab_64_H100,
        global_batch=32,
        seq_len=512,
        tp_size=8,
        pp_size=1,
        dp_size=8,
        zero_stage=3,
        microbatch_count=4,
        activation_checkpointing="full",
    )

    record = memory_plan_record(plan, inputs=inputs)

    assert record["inputs"] == inputs
    assert record["total"]["unit"] == "GB"
    assert Q_(record["total"]["value"], record["total"]["unit"]) == plan.total


def test_layout_specs_scale_to_fleet_and_execute():
    for track in ["edge", "cloud"]:
        scenario = get_track_scenario(track)
        specs = get_parallel_layout_specs(scenario.fleet)
        assert set(specs.keys()) == {"within", "cross", "pipeline"}
        for spec in specs.values():
            assert spec["dp_size"] * spec["tp_size"] * spec["pp_size"] == scenario.fleet.total_accelerators
        within = parallel_layout(
            Models.Language.Llama3_8B,
            scenario.fleet,
            global_batch=256,
            seq_len=1024,
            microbatch_count=8,
            zero_stage=3,
            **specs["within"],
        )
        assert within.devices == scenario.fleet.total_accelerators
        assert within.tensor_parallel_tier == "intra-node"


def test_capacity_and_pipeline_dimensions_match_fleet():
    for track in ["tinyml", "cloud"]:
        scenario = get_track_scenario(track)
        cap_dims = get_capacity_dimensions(scenario.fleet)
        assert cap_dims["tp_size"] * cap_dims["pp_size"] * cap_dims["dp_size"] == scenario.fleet.total_accelerators
        cap_plan = training_memory_plan(
            Models.Language.Llama3_70B,
            scenario.fleet,
            seq_len=1024,
            zero_stage=3,
            microbatch_count=8,
            activation_checkpointing="full",
            **cap_dims,
        )
        assert cap_plan.feasible
        pipe_dims = get_pipeline_dimensions(scenario.fleet)
        assert pipe_dims["tp_size"] * pipe_dims["pp_size"] * pipe_dims["dp_size"] == scenario.fleet.total_accelerators
