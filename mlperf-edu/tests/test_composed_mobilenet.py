import json

import pytest

from mlperf.registry import load_registry
from mlperf.runners.vision import (
    run_mobilenet_composed_max,
    run_mobilenet_composed_min,
)


def test_composed_mobilenet_executes_fp16_and_reports_honest_accounting(tmp_path):
    workload = load_registry()["mobilenet-cifar100-composed-fp16"]

    report = run_mobilenet_composed_min(workload, tmp_path)
    manifest = json.loads(
        (tmp_path / "mobilenet-cifar100-composed-fp16_min.provd.json").read_text()
    )

    assert workload.scenario == "offline"
    assert report["status"] == "passed"
    assert report["backend"] == "pytorch-cpu-fp16"
    assert report["compression"]["execution_precision"] == "fp16"
    assert report["compression"]["quantization"] == "fake-int8"
    assert report["metrics"]["execution_dtype"] == "float16"
    assert report["metrics"]["n_params"] == 2_351_972
    assert report["metrics"]["effective_compression_ratio"] == pytest.approx(
        3.100770983188137
    )
    assert report["quality"]["target_met"] is True
    assert manifest["scenario"] == workload.scenario


def test_composed_mobilenet_rejects_empty_max_measurement(tmp_path, monkeypatch):
    workload = load_registry()["mobilenet-cifar100-composed-fp16"]
    monkeypatch.setenv("MLPERF_EDU_MOBILENET_COMP_REPETITIONS", "0")

    with pytest.raises(ValueError, match="positive batch and repetition"):
        run_mobilenet_composed_max(workload, tmp_path)
