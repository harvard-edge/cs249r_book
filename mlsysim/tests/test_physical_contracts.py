"""Independent accounting oracles for principal first-order model domains.

All inputs below are constructed test cases, not device measurements. Expected
values come from explicit operations, stored elements, scheduled transfers, or
conservation of work/energy; they do not call another implementation of the
formula under test. These checks constrain implementation correctness, not
predictive accuracy on hardware.
"""

from itertools import product
import math

import pytest

from mlsysim import Hardware, Models, Q_
from mlsysim.models.types import TransformerWorkload
from mlsysim.physics.communication import calc_hierarchical_allreduce_time, calc_ring_allreduce_time
from mlsysim.physics.memory import calc_activation_memory
from mlsysim.physics.performance import calc_bottleneck, calc_pipeline_bubble, dTime
from mlsysim.physics.reliability import calc_young_daly_interval
from mlsysim.physics.serving import calc_queue_latency_mmc
from mlsysim.physics.transformer import calc_transformer_training_flops
from mlsysim.solvers import RunFootprintModel, ServingModel


@pytest.mark.parametrize("parameters,tokens", [(7, 3), (13, 5)])
@pytest.mark.parametrize("recompute", [False, True])
def test_training_work_counts_forward_backward_and_recomputed_operations(parameters, tokens, recompute):
    # Parameter-dominated dense accounting: multiply + add for each parameter
    # in each token's forward pass and in each of two backward products.
    # Recomputing saved activations adds one forward product, not a backward.
    passes = ["forward", "input gradient", "weight gradient"]
    if recompute:
        passes.append("recomputed forward")
    operations = sum(1 for _ in product(range(tokens), range(parameters), passes, ("multiply", "add")))
    result = calc_transformer_training_flops(parameters, tokens, activation_checkpointing=recompute)
    assert result.m_as("flop") == operations


@pytest.mark.parametrize("device_count,efficiency", [(1, 1.0), (4, 0.25), (7, 0.6)])
def test_training_time_cannot_create_or_destroy_work(device_count, efficiency):
    work, peak = Q_(37, "GFLOP"), Q_(11, "TFLOP/s")
    elapsed = dTime(work, device_count, peak, efficiency)
    executed = elapsed * peak * device_count * efficiency
    assert executed.m_as("flop") == pytest.approx(work.m_as("flop"))
    assert elapsed >= work / (device_count * peak)
    # Changing only the written units must not change time.
    equivalent = dTime(work.to("flop"), device_count, peak.to("GFLOP/s"), efficiency)
    assert equivalent.m_as("second") == pytest.approx(elapsed.m_as("second"))


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("precision,bits", [("fp16", 16), ("int8", 8), ("int4", 4)])
def test_kv_storage_counts_both_tensors_and_grouped_heads(kv_heads, precision, bits):
    # Enumerate actual cache coordinates so a missing K/V factor, query-head
    # substitution, or bits/bytes conversion cannot pass through a shared helper.
    layers, requests, tokens, head_width = 2, 3, 5, 7
    model = TransformerWorkload(
        name="constructed KV storage test", architecture="Transformer", parameters=Q_(1000, "param"),
        layers=layers, hidden_dim=28, heads=4, kv_heads=kv_heads,
    )
    elements = sum(1 for _ in product(
        range(layers), range(requests), range(tokens), ("K", "V"),
        range(kv_heads), range(head_width),
    ))
    assert model.get_kv_cache_size(tokens, requests, precision).m_as("byte") == elements * bits / 8


@pytest.mark.parametrize("precision,bits", [("fp16", 16), ("int8", 8)])
def test_mla_storage_is_one_latent_plus_one_rope_vector_per_token(precision, bits):
    model = TransformerWorkload(
        name="constructed MLA storage test", architecture="Transformer", parameters=Q_(1000, "param"),
        layers=2, hidden_dim=28, heads=4, kv_lora_rank=9, qk_rope_head_dim=3,
    )
    stored = sum(len(range(width)) for _ in product(range(2), range(3), range(5)) for width in (9, 3))
    assert model.get_kv_cache_size(5, 3, precision).m_as("byte") == stored * bits / 8


@pytest.mark.parametrize("layers,sequence,batch,width", [(2, 5, 3, 7), (3, 11, 2, 13)])
def test_full_recomputation_keeps_only_layer_inputs(layers, sequence, batch, width):
    # Full recomputation retains one input activation per layer/token/request.
    elements = sum(1 for _ in product(range(layers), range(sequence), range(batch), range(width)))
    for bytes_per_element in (1, 2, 4):
        result = calc_activation_memory(layers, sequence, batch, width,
                                        precision_bytes=bytes_per_element, strategy="full")
        assert result.m_as("byte") == elements * bytes_per_element


@pytest.mark.parametrize("participants", [1, 2, 3, 8])
def test_ring_time_matches_explicit_reduce_scatter_and_allgather_schedule(participants):
    message, bandwidth, startup = Q_(96, "MiB"), Q_(7, "GB/s"), Q_(3, "us")
    # Each rank transmits one chunk in each round of each phase. Count rounds
    # rather than importing a ring bandwidth/latency factor from the package.
    rounds = [(phase, round_id) for phase in ("reduce-scatter", "allgather")
              for round_id in range(participants - 1)]
    serial_time = Q_(0, "second")
    for _ in rounds:
        serial_time += startup + (message / participants) / bandwidth
    result = calc_ring_allreduce_time(message, participants, bandwidth, startup)
    assert result.m_as("second") == pytest.approx(serial_time.m_as("second"))


@pytest.mark.parametrize("nodes,local_ranks", [(1, 4), (3, 1), (3, 4)])
def test_hierarchical_collective_transfers_only_reduced_shards_between_nodes(nodes, local_ranks):
    message = Q_(96, "MiB")
    local_bw, remote_bw = Q_(13, "GB/s"), Q_(5, "GB/s")
    local_lat, remote_lat = Q_(1, "us"), Q_(7, "us")
    elapsed = Q_(0, "second")
    for _ in ("reduce-scatter", "allgather"):
        for _ in range(local_ranks - 1):
            elapsed += local_lat + message / local_ranks / local_bw
    for _ in ("remote reduce-scatter", "remote allgather"):
        for _ in range(nodes - 1):
            elapsed += remote_lat + message / local_ranks / nodes / remote_bw
    result = calc_hierarchical_allreduce_time(message, nodes, local_ranks, local_bw, remote_bw,
                                             local_lat, remote_lat)
    assert result.m_as("second") == pytest.approx(elapsed.m_as("second"))


@pytest.mark.parametrize("stages,microbatches", [(1, 1), (3, 1), (3, 5), (7, 11)])
def test_pipeline_bubble_matches_explicit_noninterleaved_forward_schedule(stages, microbatches):
    # A stage works on microbatch b at slot b+stage. Count every idle device
    # slot during this ideal uniform-stage forward wave. No interleaved claim.
    busy_slots = {(stage, batch + stage) for stage in range(stages) for batch in range(microbatches)}
    final_slot = max(slot for _, slot in busy_slots)
    all_slots = {(stage, slot) for stage in range(stages) for slot in range(final_slot + 1)}
    expected = len(all_slots - busy_slots) / len(all_slots)
    assert calc_pipeline_bubble(stages, microbatches) == pytest.approx(expected)


@pytest.mark.parametrize("scale", [0.25, 1, 4])
def test_roofline_crossover_respects_independent_resource_budgets(scale):
    # Construct the crossover by assigning equal work/time budgets to both
    # resources, then perturb only compute work on either side of the ridge.
    seconds = 0.013
    peak, bandwidth = Q_(17, "TFLOP/s"), Q_(23, "GB/s")
    work = peak * Q_(seconds * scale, "second")
    traffic = bandwidth * Q_(seconds, "second")
    result = calc_bottleneck(work, traffic, peak, bandwidth)
    assert result["compute_ms"] == pytest.approx(seconds * scale * 1000)
    assert result["memory_ms"] == pytest.approx(seconds * 1000)
    if scale != 1:
        assert result["bottleneck"] == ("Memory" if scale < 1 else "Compute")
    changed_units = calc_bottleneck(work.to("flop"), traffic.to("KiB"), peak.to("GFLOP/s"), bandwidth.to("MiB/s"))
    assert changed_units["intensity"] == pytest.approx(result["intensity"])


@pytest.mark.parametrize("servers,utilization", [(1, 0.2), (1, 0.8), (3, 0.7)])
def test_queue_wait_quantiles_match_birth_death_stationary_probabilities(servers, utilization):
    # Independent birth/death balance: p(n+1)/p(n)=lambda/(min(n+1,c)*mu).
    # Finite normalization uses a tail below floating precision for these inputs.
    service_rate = 13.0
    arrival_rate = utilization * servers * service_rate
    weights = [1.0]
    for population in range(1, 1000):
        weights.append(weights[-1] * arrival_rate / (min(population, servers) * service_rate))
    p_wait = sum(weights[servers:]) / sum(weights)
    rho, median, p99 = calc_queue_latency_mmc(Q_(arrival_rate, "Hz"), Q_(service_rate, "Hz"), servers)
    assert rho == pytest.approx(utilization)
    # P(Wq > t)=P(wait)*exp(-(c*mu-lambda)*t); atom at zero matters.
    for quantile, observed in ((0.5, median), (0.99, p99)):
        expected = max(0, math.log(p_wait / (1 - quantile)) / (servers * service_rate - arrival_rate))
        assert observed.m_as("second") == pytest.approx(expected)


def test_unstable_queue_does_not_report_finite_wait():
    _, median, p99 = calc_queue_latency_mmc(Q_(26, "Hz"), Q_(13, "Hz"), 2)
    assert math.isinf(median.m_as("second"))
    assert math.isinf(p99.m_as("second"))


@pytest.mark.parametrize("pue", [1.0, 1.4, 2.0])
def test_run_footprint_conserves_device_hours_and_applies_facility_overhead_once(pue):
    devices, hours, watts, it_ratio = 3, 7, 400, 1.25
    grid_grams_per_kwh, water_liters_per_kwh = 250, 0.8
    result = RunFootprintModel().solve(
        device_hours=Q_(devices * hours, "hour"), devices=devices,
        device_power=Q_(watts, "W"), it_ratio=it_ratio, pue=pue,
        intensities={"constructed grid": Q_(grid_grams_per_kwh, "g/kWh")},
        embodied_per_device=Q_(100, "kg"), service_life=Q_(1000, "hour"),
        wue={"constructed cooling": Q_(water_liters_per_kwh, "liter/kWh")},
    )
    # Add per-device watt-hours instead of taking the solver's power outputs.
    accelerator_wh = sum(watts for _ in product(range(devices), range(hours)))
    it_wh = accelerator_wh * it_ratio
    facility_wh = it_wh * pue
    assert result.duration.m_as("hour") == hours
    assert result.accelerator_energy.m_as("Wh") == accelerator_wh
    assert result.facility_energy.m_as("Wh") == pytest.approx(facility_wh)
    assert result.carbon["constructed grid"].m_as("g") == pytest.approx(facility_wh / 1000 * grid_grams_per_kwh)
    # WUE uses IT energy, so PUE cannot multiply this water account.
    assert result.water["constructed cooling"].m_as("liter") == pytest.approx(it_wh / 1000 * water_liters_per_kwh)
    assert result.embodied_allocated.m_as("kg") == pytest.approx(devices * hours * 100 / 1000)


def test_checkpoint_interval_balances_write_and_expected_rework_losses():
    write_seconds, failure_interval_seconds = 3.0, 100000.0
    interval = calc_young_daly_interval(Q_(write_seconds, "second"), Q_(failure_interval_seconds, "second"))
    seconds = interval.m_as("second")
    # Uniform failure location loses half an interval on average. The optimum
    # balances checkpoint-frequency cost with that expected rework fraction.
    def loss(candidate):
        return write_seconds / candidate + (candidate / 2) / failure_interval_seconds
    assert loss(seconds) < loss(seconds * 0.9)
    assert loss(seconds) < loss(seconds * 1.1)
    assert write_seconds / seconds == pytest.approx((seconds / 2) / failure_interval_seconds)


def test_serving_capacity_boundary_uses_independently_counted_weights_and_kv():
    model = Models.Language.GPT2_Small
    batch, context = 3, 17
    weight_bits = model.parameters.m_as("count") * 16
    cache_elements = sum(1 for _ in product(
        range(model.layers), range(batch), range(context), ("K", "V"),
        range(model.kv_heads or model.heads), range(model.head_width),
    ))
    bytes_required = (weight_bits + cache_elements * 16) / 8
    base = Hardware.Cloud.H100
    for delta, expected_feasible in ((-1, False), (0, True), (1, True)):
        device = base.model_copy(update={"memory": base.memory.model_copy(update={
            "capacity": Q_(bytes_required + delta, "byte"),
        })})
        result = ServingModel().solve(model, device, context, batch_size=batch)
        assert result.feasible is expected_feasible
        assert result.total_memory_required.m_as("byte") == pytest.approx(bytes_required)
