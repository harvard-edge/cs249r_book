from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

from mlperf import edu_cli, fingerprint


def test_execution_device_annotation_distinguishes_request_from_execution(monkeypatch):
    monkeypatch.setenv("MLPERF_EDU_DEVICE", "MPS")
    report = {"backend": "pytorch-mps"}

    edu_cli.annotate_execution_device(report)

    assert report["device_requested"] == "mps"
    assert report["device_executed"] == "mps"


def test_execution_device_annotation_records_auto_cpu_and_preserves_plugins(
    monkeypatch,
):
    monkeypatch.delenv("MLPERF_EDU_DEVICE", raising=False)
    report = {"backend": "pytorch-cpu"}
    edu_cli.annotate_execution_device(report)
    assert report["device_requested"] == "auto"
    assert report["device_executed"] == "cpu"

    plugin_report = {
        "backend": "custom-runtime",
        "device_requested": "accelerator:0",
        "device_executed": "custom-accelerator:0",
    }
    edu_cli.annotate_execution_device(plugin_report)
    assert plugin_report["device_requested"] == "accelerator:0"
    assert plugin_report["device_executed"] == "custom-accelerator:0"


def test_detect_hardware_hash_binds_complete_comparison_record(monkeypatch):
    monkeypatch.setattr(fingerprint, "_detect_machine_model", lambda: "Test Laptop")
    monkeypatch.setattr(fingerprint, "_detect_chip", lambda: "Test CPU")
    monkeypatch.setattr(fingerprint, "_detect_cpu", lambda: "test-arch")
    monkeypatch.setattr(
        fingerprint,
        "_detect_cpu_topology",
        lambda: {"logical_cores": 12, "physical_cores": 6, "sockets": 1},
    )
    monkeypatch.setattr(fingerprint, "_detect_gpu", lambda: "Test Accelerator")
    monkeypatch.setattr(
        fingerprint,
        "_detect_accelerator",
        lambda: {
            "availability_backend": "CUDA",
            "runtime": "CUDA",
            "runtime_version": "13.0",
            "driver_version": "999.1",
        },
    )
    monkeypatch.setattr(fingerprint, "_detect_memory_gb", lambda: 32.0)
    monkeypatch.setattr(fingerprint, "_detect_pytorch_version", lambda: "2.test")
    monkeypatch.setattr(
        fingerprint, "_detect_available_backends", lambda: ["CPU", "CUDA"]
    )
    monkeypatch.setattr(
        fingerprint,
        "_detect_torch_runtime",
        lambda: {
            "available": True,
            "intra_op_threads": 6,
            "inter_op_threads": 2,
            "default_dtype": "float32",
            "float32_matmul_precision": "highest",
        },
    )
    monkeypatch.setattr(
        fingerprint,
        "_detect_performance_environment",
        lambda: {"OMP_NUM_THREADS": "6"},
    )
    monkeypatch.setattr(
        fingerprint,
        "_detect_cache_sizes",
        lambda: {"l1d": 1, "l1i": 2, "l2": 3, "l3": 4},
    )
    monkeypatch.setattr(fingerprint, "_detect_audio_backend", lambda: None)
    monkeypatch.setattr(fingerprint.platform, "system", lambda: "TestOS")
    monkeypatch.setattr(fingerprint.platform, "release", lambda: "1")
    monkeypatch.setattr(fingerprint.platform, "version", lambda: "build-1")
    monkeypatch.setattr(fingerprint.platform, "python_version", lambda: "3.test")

    detected = fingerprint.detect_hardware()

    assert detected["backend"] == "CUDA"
    assert detected["availability_detected_backend"] == "CUDA"
    assert detected["available_backends"] == ["CPU", "CUDA"]
    assert detected["cpu_topology"]["physical_cores"] == 6
    assert detected["fingerprint_hash"] == detected["fingerprint_sha256"][:16]
    assert detected["fingerprint_sha256"] == (
        fingerprint.comparison_fingerprint_sha256(detected)
    )

    changed = copy.deepcopy(detected)
    changed["torch_runtime"]["intra_op_threads"] = 12
    assert (
        fingerprint.comparison_fingerprint_sha256(changed)
        != detected["fingerprint_sha256"]
    )
    changed = copy.deepcopy(detected)
    changed["performance_environment"]["OMP_NUM_THREADS"] = "12"
    assert (
        fingerprint.comparison_fingerprint_sha256(changed)
        != detected["fingerprint_sha256"]
    )


def test_torch_runtime_records_threads_numeric_and_determinism_policies(monkeypatch):
    fake_torch = SimpleNamespace(
        get_num_threads=lambda: 8,
        get_num_interop_threads=lambda: 3,
        get_default_dtype=lambda: "torch.float32",
        get_default_device=lambda: "cpu",
        get_float32_matmul_precision=lambda: "high",
        are_deterministic_algorithms_enabled=lambda: True,
        get_deterministic_debug_mode=lambda: 2,
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(
                allow_tf32=False,
                deterministic=True,
                benchmark=False,
                version=lambda: 99999,
            ),
            quantized=SimpleNamespace(
                engine="qnnpack", supported_engines=["qnnpack", "none"]
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    runtime = fingerprint._detect_torch_runtime()

    assert runtime == {
        "available": True,
        "intra_op_threads": 8,
        "inter_op_threads": 3,
        "default_dtype": "float32",
        "default_device": "cpu",
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": False,
        "cudnn_version": 99999,
        "quantized_engine": "qnnpack",
        "quantized_supported_engines": ["qnnpack", "none"],
        "deterministic_algorithms_enabled": True,
        "deterministic_debug_mode": 2,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def test_cpu_topology_records_logical_physical_and_socket_counts(monkeypatch):
    values = {"hw.logicalcpu": 12, "hw.physicalcpu": 10, "hw.packages": 1}
    monkeypatch.setattr(fingerprint.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(fingerprint, "_sysctl_int", values.get)
    monkeypatch.setattr(fingerprint.os, "cpu_count", lambda: 99)

    assert fingerprint._detect_cpu_topology() == {
        "logical_cores": 12,
        "physical_cores": 10,
        "sockets": 1,
    }


def test_machine_model_fallback_does_not_expose_hostname(monkeypatch):
    monkeypatch.setattr(fingerprint.platform, "system", lambda: "OtherOS")
    monkeypatch.setattr(fingerprint.platform, "machine", lambda: "test-arch")
    monkeypatch.setattr(
        fingerprint.platform,
        "node",
        lambda: (_ for _ in ()).throw(AssertionError("hostname must not be read")),
    )

    assert fingerprint._detect_machine_model() == "test-arch"


def test_cuda_accelerator_records_runtime_driver_and_device(monkeypatch):
    properties = SimpleNamespace(total_memory=24 * 1024**3)
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="13.0", hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 1,
            get_device_properties=lambda index: properties,
            get_device_capability=lambda index: (9, 0),
            device_count=lambda: 2,
            get_device_name=lambda index: f"Test GPU {index}",
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        fingerprint, "_detect_accelerator_driver", lambda runtime: "999.1"
    )

    accelerator = fingerprint._detect_accelerator()

    assert accelerator == {
        "availability_backend": "CUDA",
        "runtime": "CUDA",
        "runtime_version": "13.0",
        "driver_version": "999.1",
        "device_count": 2,
        "inspection_device_index": 1,
        "device_name": "Test GPU 1",
        "total_memory_bytes": 24 * 1024**3,
        "compute_capability": "9.0",
    }


def test_performance_environment_uses_explicit_allowlist(monkeypatch):
    monkeypatch.setattr(
        fingerprint,
        "PERFORMANCE_ENVIRONMENT_ALLOWLIST",
        ("MLPERF_EDU_DEVICE", "OMP_NUM_THREADS"),
    )
    monkeypatch.setenv("MLPERF_EDU_DEVICE", "cpu")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("HF_TOKEN", "must-not-appear")
    monkeypatch.setenv("MLPERF_EDU_DATA_DIR", "/private/course-data")
    monkeypatch.setenv("MLPERF_EDU_PRIVATE_PROMPT", "private prompt")

    environment = fingerprint._detect_performance_environment()

    assert environment == {"MLPERF_EDU_DEVICE": "cpu", "OMP_NUM_THREADS": "4"}
    assert "HF_TOKEN" not in environment
    assert "MLPERF_EDU_DATA_DIR" not in environment
    assert "MLPERF_EDU_PRIVATE_PROMPT" not in environment


def test_performance_environment_has_no_retired_workload_knobs():
    retired_prefixes = (
        "MLPERF_EDU_ANOMALY_",
        "MLPERF_EDU_CODEGEN_",
        "MLPERF_EDU_DDP_",
        "MLPERF_EDU_DLRM_",
        "MLPERF_EDU_MOBILENET_",
        "MLPERF_EDU_RAG_",
        "MLPERF_EDU_REACT_",
        "MLPERF_EDU_RESNET_",
        "MLPERF_EDU_SLM_",
        "MLPERF_EDU_TOOLCALL_",
        "MLPERF_EDU_WAKE_",
    )
    assert not any(
        name.startswith(retired_prefixes)
        for name in fingerprint.PERFORMANCE_ENVIRONMENT_ALLOWLIST
    )


def test_run_fingerprint_labels_selected_backend_and_binds_run_context(monkeypatch):
    hardware = {
        "backend": "MPS",
        "availability_detected_backend": "MPS",
        "available_backends": ["CPU", "MPS"],
        "cpu_topology": {
            "logical_cores": 12,
            "physical_cores": 12,
            "sockets": 1,
        },
        "accelerator": {
            "availability_backend": "MPS",
            "runtime": "Metal Performance Shaders",
        },
        "torch_runtime": {"intra_op_threads": 12},
        "performance_environment": {"OMP_NUM_THREADS": "12"},
        "fingerprint_hash": "0123456789abcdef",
        "fingerprint_sha256": "0" * 64,
    }
    monkeypatch.setattr(edu_cli, "load_report_manifest", lambda report: None)
    monkeypatch.setattr(
        edu_cli,
        "software_fingerprint_summary",
        lambda hw=None: {
            "torch_runtime": hw["torch_runtime"],
            "performance_environment": hw["performance_environment"],
        },
    )
    report = {
        "workload": "test-workload",
        "profile": "max",
        "scenario": "single_stream",
        "seed": 0,
        "backend": "pytorch-cpu",
        "device_requested": "cpu",
        "device_executed": "cpu",
        "dtype": "float32",
        "compilation": {
            "enabled": True,
            "mode": "reduce-overhead",
            "backend": "inductor",
        },
    }

    run_fingerprint = edu_cli.build_run_fingerprint(report, hardware=hardware)

    execution = run_fingerprint["execution"]
    assert run_fingerprint["hardware"]["availability_detected_backend"] == "MPS"
    assert execution["backends"] == ["pytorch-cpu"]
    assert execution["report_selected_backends"] == ["pytorch-cpu"]
    assert execution["report_selected_devices"] == ["cpu"]
    assert execution["report_executed_devices"] == ["cpu"]
    assert execution["scenario"] == "single_stream"
    assert execution["scenarios"] == ["single_stream"]
    assert execution["report_selected_precision"] == [{"dtype": "float32"}]
    assert execution["report_selected_compilation"] == [
        {"backend": "inductor", "enabled": True, "mode": "reduce-overhead"}
    ]
    assert run_fingerprint["comparison_fingerprint_sha256"] == (
        edu_cli.run_comparison_fingerprint_sha256(run_fingerprint)
    )

    changed = copy.deepcopy(run_fingerprint)
    changed["execution"]["report_selected_backends"] = ["pytorch-mps"]
    assert (
        edu_cli.run_comparison_fingerprint_sha256(changed)
        != run_fingerprint["comparison_fingerprint_sha256"]
    )
    relocated = copy.deepcopy(run_fingerprint)
    relocated["software"]["python_executable"] = "/another/venv/bin/python"
    relocated["execution"]["status"] = "failed"
    assert (
        edu_cli.run_comparison_fingerprint_sha256(relocated)
        == run_fingerprint["comparison_fingerprint_sha256"]
    )
    reseeded = copy.deepcopy(run_fingerprint)
    reseeded["execution"]["seed"] = 4
    reseeded["software"]["performance_environment"]["MLPERF_EDU_MAX_SEED"] = "4"
    reseeded["hardware"]["fingerprint_hash"] = "fedcba9876543210"
    reseeded["hardware"]["fingerprint_sha256"] = "f" * 64
    assert (
        edu_cli.run_comparison_fingerprint_sha256(reseeded)
        == run_fingerprint["comparison_fingerprint_sha256"]
    )

    changed_environment = copy.deepcopy(run_fingerprint)
    changed_environment["software"]["performance_environment"]["OMP_NUM_THREADS"] = "8"
    assert (
        edu_cli.run_comparison_fingerprint_sha256(changed_environment)
        != run_fingerprint["comparison_fingerprint_sha256"]
    )


def test_training_comparison_fingerprint_ignores_output_weights_not_inputs():
    run_fingerprint = _fingerprint_with_assets("training")
    baseline = edu_cli.run_comparison_fingerprint_sha256(run_fingerprint)

    changed_checkpoint = copy.deepcopy(run_fingerprint)
    changed_checkpoint["asset_hashes"]["weights"]["path"] = "/tmp/seed-4/model.pt"
    changed_checkpoint["asset_hashes"]["weights"]["sha256"] = "sha256:" + "4" * 64
    assert edu_cli.run_comparison_fingerprint_sha256(changed_checkpoint) == baseline

    relocated_dataset = copy.deepcopy(run_fingerprint)
    relocated_dataset["asset_hashes"]["dataset"]["files"][0]["path"] = (
        "/course/cache/dataset.bin"
    )
    assert edu_cli.run_comparison_fingerprint_sha256(relocated_dataset) == baseline

    changed_dataset = copy.deepcopy(run_fingerprint)
    changed_dataset["asset_hashes"]["dataset"]["files"][0]["sha256"] = (
        "sha256:" + "5" * 64
    )
    assert edu_cli.run_comparison_fingerprint_sha256(changed_dataset) != baseline


def test_inference_comparison_fingerprint_binds_input_weights_but_not_paths():
    run_fingerprint = _fingerprint_with_assets("single_stream")
    baseline = edu_cli.run_comparison_fingerprint_sha256(run_fingerprint)

    relocated = copy.deepcopy(run_fingerprint)
    relocated["asset_hashes"]["weights"]["path"] = "/tmp/model-copy.pt"
    relocated["asset_hashes"]["dataset"]["files"][0]["path"] = "/tmp/dataset-copy.bin"
    assert edu_cli.run_comparison_fingerprint_sha256(relocated) == baseline

    changed_weights = copy.deepcopy(run_fingerprint)
    changed_weights["asset_hashes"]["weights"]["sha256"] = "sha256:" + "6" * 64
    assert edu_cli.run_comparison_fingerprint_sha256(changed_weights) != baseline

    changed_scenario = copy.deepcopy(run_fingerprint)
    changed_scenario["execution"]["scenario"] = "offline"
    changed_scenario["execution"]["scenarios"] = ["offline"]
    assert edu_cli.run_comparison_fingerprint_sha256(changed_scenario) != baseline


def _fingerprint_with_assets(scenario: str) -> dict:
    return {
        "schema": "mlperf-edu-run-fingerprint/0.1",
        "hardware": {
            "backend": "CPU",
            "chip": "Test CPU",
            "fingerprint_sha256": "0" * 64,
        },
        "software": {
            "python": "3.test",
            "python_executable": "/private/venv/bin/python",
        },
        "execution": {
            "workload": "test-workload",
            "profile": "max",
            "scenario": scenario,
            "scenarios": [scenario],
            "seed": 0,
            "status": "passed",
            "backends": ["pytorch-cpu"],
            "report_selected_backends": ["pytorch-cpu"],
            "data_modes": ["real"],
        },
        "asset_hashes": {
            "weights": {
                "path": "/tmp/seed-0/model.pt",
                "sha256": "sha256:" + "1" * 64,
                "n_bytes": 1024,
            },
            "dataset": {
                "name": "test-dataset",
                "files": [
                    {
                        "path": "/Users/vj/cache/dataset.bin",
                        "sha256": "sha256:" + "2" * 64,
                        "n_bytes": 2048,
                    }
                ],
            },
        },
        "comparison_fingerprint_hash_algorithm": "sha256",
        "comparison_fingerprint_hash_scope": "canonical-run-comparison-record",
    }


def test_run_fingerprint_reuses_manifest_bound_hardware(monkeypatch):
    manifest_hardware = {
        "machine_model": "Bound Machine",
        "chip": "Bound Chip",
        "backend": "CPU",
        "availability_detected_backend": "CPU",
        "available_backends": ["CPU"],
        "fingerprint_hash": "0123456789abcdef",
        "fingerprint_sha256": "0" * 64,
    }
    monkeypatch.setattr(
        edu_cli,
        "load_report_manifest",
        lambda _report: {"leaves": {"hardware": {"fingerprint": manifest_hardware}}},
    )
    monkeypatch.setattr(
        edu_cli,
        "detect_hardware",
        lambda: (_ for _ in ()).throw(
            AssertionError("hardware must not be detected a second time")
        ),
    )

    result = edu_cli.build_run_fingerprint(
        {"workload": "toy", "profile": "max", "backend": "pytorch-cpu"}
    )

    assert result["hardware"]["machine_model"] == "Bound Machine"
    assert result["hardware"]["chip"] == "Bound Chip"


def test_execution_fingerprint_uses_exposed_precision_and_compilation_config():
    summary = edu_cli.execution_fingerprint_summary(
        {
            "workload": "compiled-workload",
            "backend": "pytorch-cuda",
            "metrics": {"dtype": "bfloat16"},
            "configuration": {
                "torch_compile": True,
                "compile_mode": "max-autotune",
                "compile_backend": "inductor",
            },
        }
    )

    assert summary["report_selected_precision"] == [{"dtype": "bfloat16"}]
    assert summary["report_selected_compilation"] == [
        {"backend": "inductor", "mode": "max-autotune", "torch_compile": True}
    ]
