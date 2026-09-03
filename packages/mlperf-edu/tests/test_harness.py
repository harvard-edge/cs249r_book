import time

import pytest

from mlperf.harness import HarnessSample, ScenarioConfig, plan_batches, run_harness


def test_offline_harness_batches_samples_deterministically():
    seen = []

    def handler(samples: list[HarnessSample]):
        seen.append([sample.id for sample in samples])
        return [{"batch_size": len(samples)} for _ in samples]

    result = run_harness(
        ScenarioConfig(scenario="offline", sample_count=5, batch_size=2), handler
    )

    assert seen == [[0, 1], [2, 3], [4]]
    assert [response.sample_id for response in result.responses] == [0, 1, 2, 3, 4]
    assert result.metrics["scenario"] == "offline"
    assert result.metrics["sample_count"] == 5
    assert result.metrics["ok_count"] == 5
    assert result.metrics["throughput_qps"] > 0
    assert result.responses[0].metadata == {"batch_size": 2}


def test_single_stream_harness_dispatches_one_sample_at_a_time():
    seen = []

    def handler(samples: list[HarnessSample]):
        seen.append([sample.id for sample in samples])
        return [{"token": sample.id} for sample in samples]

    result = run_harness(
        ScenarioConfig(scenario="SingleStream", sample_count=4, batch_size=4), handler
    )

    assert seen == [[0], [1], [2], [3]]
    assert result.config.scenario == "single_stream"
    assert result.metrics["latency_p50_s"] >= 0
    assert result.metrics["latency_p99_s"] >= result.metrics["latency_p50_s"]


def test_server_harness_schedule_is_seeded_and_deterministic():
    config = ScenarioConfig(scenario="server", sample_count=6, target_qps=3.0, seed=7)

    first = plan_batches(config)
    second = plan_batches(config)

    first_schedule = [batch[0].scheduled_time_s for batch in first]
    second_schedule = [batch[0].scheduled_time_s for batch in second]
    assert first_schedule == second_schedule
    assert first_schedule == sorted(first_schedule)
    assert all(value > 0 for value in first_schedule)


def test_server_harness_latency_includes_queueing_delay():
    def handler(samples: list[HarnessSample]):
        time.sleep(0.002)
        return [{"token": sample.id} for sample in samples]

    result = run_harness(
        ScenarioConfig(scenario="server", sample_count=4, target_qps=100000.0, seed=1),
        handler,
    )
    latencies = [response.latency_s for response in result.responses]

    assert latencies == sorted(latencies)
    assert latencies[-1] > latencies[0]
    assert result.metrics["latency_p99_s"] == latencies[-1]


def test_server_harness_requires_positive_qps():
    try:
        plan_batches(ScenarioConfig(scenario="server", sample_count=1, target_qps=0.0))
    except ValueError as exc:
        assert "target_qps" in str(exc)
    else:
        raise AssertionError("server scenario should require positive target_qps")


def test_harness_report_shape_is_json_ready():
    result = run_harness(
        ScenarioConfig(scenario="offline", sample_count=2, batch_size=2),
        lambda samples: {"tokens": len(samples) * 4},
    )

    report = result.to_report()
    assert report["scenario"] == "offline"
    assert report["config"]["sample_count"] == 2
    assert report["metrics"]["sample_count"] == 2
    assert report["responses"][0]["metadata"] == {"tokens": 8}


@pytest.mark.parametrize("output", [None, {}, [], [None]])
def test_harness_rejects_empty_or_missing_work(output):
    with pytest.raises(ValueError, match="response"):
        run_harness(
            ScenarioConfig(scenario="offline", sample_count=1),
            lambda _samples: output,
        )


def test_harness_rejects_partial_batch_output():
    with pytest.raises(ValueError, match="exactly one response per sample"):
        run_harness(
            ScenarioConfig(scenario="offline", sample_count=2, batch_size=2),
            lambda _samples: [{"token": 1}],
        )


def test_harness_uses_workload_response_validator():
    result = run_harness(
        ScenarioConfig(scenario="offline", sample_count=2, batch_size=2),
        lambda samples: [{"token": sample.id} for sample in samples],
        response_validator=lambda _sample, metadata: metadata["token"] > 0,
    )

    assert [response.ok for response in result.responses] == [False, True]
    assert result.metrics["ok_count"] == 1
