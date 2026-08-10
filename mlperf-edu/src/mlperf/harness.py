from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable


SCENARIOS = ("offline", "single_stream", "server")


@dataclass(frozen=True)
class ScenarioConfig:
    scenario: str = "offline"
    sample_count: int = 1
    batch_size: int = 1
    target_qps: float | None = None
    seed: int = 42
    realtime: bool = False

    def normalized(self) -> "ScenarioConfig":
        scenario = self.scenario.lower()
        aliases = {
            "offlinestream": "offline",
            "singlestream": "single_stream",
            "single-stream": "single_stream",
            "offline": "offline",
            "server": "server",
        }
        scenario = aliases.get(scenario, scenario)
        if scenario not in SCENARIOS:
            raise ValueError(f"unsupported harness scenario: {self.scenario}")
        if self.sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if scenario == "server" and (self.target_qps is None or self.target_qps <= 0):
            raise ValueError("server scenario requires target_qps > 0")
        return ScenarioConfig(
            scenario=scenario,
            sample_count=int(self.sample_count),
            batch_size=int(self.batch_size),
            target_qps=self.target_qps,
            seed=int(self.seed),
            realtime=bool(self.realtime),
        )


@dataclass(frozen=True)
class HarnessSample:
    id: int
    index: int
    scheduled_time_s: float


@dataclass(frozen=True)
class HarnessResponse:
    sample_id: int
    index: int
    scheduled_time_s: float
    started_at_s: float
    completed_at_s: float
    latency_s: float
    ok: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessRunResult:
    config: ScenarioConfig
    responses: list[HarnessResponse]
    started_at_s: float
    completed_at_s: float
    metrics: dict[str, Any]

    def to_report(self) -> dict[str, Any]:
        return {
            "scenario": self.config.scenario,
            "config": {
                "sample_count": self.config.sample_count,
                "batch_size": self.config.batch_size,
                "target_qps": self.config.target_qps,
                "seed": self.config.seed,
                "realtime": self.config.realtime,
            },
            "metrics": self.metrics,
            "responses": [
                {
                    "sample_id": response.sample_id,
                    "index": response.index,
                    "scheduled_time_s": response.scheduled_time_s,
                    "latency_s": response.latency_s,
                    "ok": response.ok,
                    "metadata": response.metadata,
                }
                for response in self.responses
            ],
        }


def plan_batches(config: ScenarioConfig) -> list[list[HarnessSample]]:
    config = config.normalized()
    if config.scenario == "offline":
        samples = [
            HarnessSample(id=i, index=i, scheduled_time_s=0.0)
            for i in range(config.sample_count)
        ]
        return chunk_samples(samples, config.batch_size)
    if config.scenario == "single_stream":
        return [
            [HarnessSample(id=i, index=i, scheduled_time_s=0.0)]
            for i in range(config.sample_count)
        ]
    return [[sample] for sample in server_samples(config)]


def server_samples(config: ScenarioConfig) -> list[HarnessSample]:
    config = config.normalized()
    rng = random.Random(config.seed)
    scheduled = 0.0
    samples: list[HarnessSample] = []
    qps = float(config.target_qps or 0.0)
    for i in range(config.sample_count):
        scheduled += rng.expovariate(qps)
        samples.append(HarnessSample(id=i, index=i, scheduled_time_s=scheduled))
    return samples


def chunk_samples(
    samples: list[HarnessSample], batch_size: int
) -> list[list[HarnessSample]]:
    return [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]


def run_harness(
    config: ScenarioConfig,
    query_handler: Callable[[list[HarnessSample]], Any],
    response_validator: Callable[[HarnessSample, dict[str, Any]], bool] | None = None,
) -> HarnessRunResult:
    config = config.normalized()
    batches = plan_batches(config)
    responses: list[HarnessResponse] = []
    started = time.perf_counter()
    virtual_available_s = 0.0

    for batch in batches:
        if config.realtime and batch:
            sleep_until = started + batch[0].scheduled_time_s
            delay = sleep_until - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        batch_started = time.perf_counter()
        output = query_handler(batch)
        batch_completed = time.perf_counter()
        metadata_by_id = normalize_handler_output(batch, output)
        service_time_s = batch_completed - batch_started
        if config.scenario == "server" and not config.realtime:
            batch_started_s = max(virtual_available_s, batch[0].scheduled_time_s)
            batch_completed_s = batch_started_s + service_time_s
            virtual_available_s = batch_completed_s
        else:
            batch_started_s = batch_started - started
            batch_completed_s = batch_completed - started
        for sample in batch:
            if config.scenario == "server":
                latency_s = max(0.0, batch_completed_s - sample.scheduled_time_s)
            else:
                latency_s = service_time_s
            metadata = metadata_by_id[sample.id]
            ok = (
                bool(response_validator(sample, metadata))
                if response_validator is not None
                else bool(metadata)
            )
            responses.append(
                HarnessResponse(
                    sample_id=sample.id,
                    index=sample.index,
                    scheduled_time_s=sample.scheduled_time_s,
                    started_at_s=batch_started_s,
                    completed_at_s=batch_completed_s,
                    latency_s=latency_s,
                    ok=ok,
                    metadata=metadata,
                )
            )

    completed = time.perf_counter()
    duration_s = max(
        (response.completed_at_s for response in responses), default=completed - started
    )
    metrics = harness_metrics(config, responses, duration_s)
    return HarnessRunResult(
        config=config,
        responses=responses,
        started_at_s=started,
        completed_at_s=completed,
        metrics=metrics,
    )


def normalize_handler_output(
    batch: list[HarnessSample], output: Any
) -> dict[int, dict[str, Any]]:
    if output is None:
        raise ValueError("query handler returned no response")
    if isinstance(output, dict):
        if not output:
            raise ValueError("query handler returned an empty response")
        return {sample.id: dict(output) for sample in batch}
    if isinstance(output, list):
        if len(output) != len(batch):
            raise ValueError(
                "query handler must return exactly one response per sample: "
                f"expected {len(batch)}, received {len(output)}"
            )
        rows: dict[int, dict[str, Any]] = {}
        for sample, item in zip(batch, output, strict=True):
            if item is None or (isinstance(item, dict) and not item):
                raise ValueError(
                    f"query handler returned an empty response for sample {sample.id}"
                )
            rows[sample.id] = dict(item) if isinstance(item, dict) else {"value": item}
        return rows
    return {sample.id: {"value": output} for sample in batch}


def harness_metrics(
    config: ScenarioConfig,
    responses: list[HarnessResponse],
    duration_s: float,
) -> dict[str, Any]:
    latencies = [response.latency_s for response in responses]
    ok_count = sum(1 for response in responses if response.ok)
    return {
        "scenario": config.scenario,
        "sample_count": len(responses),
        "ok_count": ok_count,
        "failed_count": len(responses) - ok_count,
        "duration_seconds": float(duration_s),
        "throughput_qps": float(len(responses) / duration_s) if duration_s > 0 else 0.0,
        "latency_mean_s": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p90_s": percentile(latencies, 90),
        "latency_p99_s": percentile(latencies, 99),
    }


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile_value / 100.0) * len(ordered)))
    return float(ordered[min(len(ordered) - 1, rank - 1)])
