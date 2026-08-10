"""
MLPerf EDU: System Fingerprint

Auto-detects hardware and software configuration at runtime.
Every benchmark run stamps this into the JSON artifact.
No manual hardware claims — all evidence is measured.
"""

import hashlib
import json
import os
import platform
import subprocess
from typing import Any


FINGERPRINT_RECORD_SCHEMA = "mlperf-edu-comparison-fingerprint/0.2"
PERFORMANCE_ENVIRONMENT_ALLOWLIST = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_DEVICE_ORDER",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "GLOO_SOCKET_IFNAME",
    "GOMP_CPU_AFFINITY",
    "HIP_VISIBLE_DEVICES",
    "KMP_AFFINITY",
    "KMP_BLOCKTIME",
    "MKL_DYNAMIC",
    "MKL_NUM_THREADS",
    "MLPERF_EDU_DECODE_MAX_BATCH",
    "MLPERF_EDU_DECODE_MAX_PREFILL_CTX",
    "MLPERF_EDU_DECODE_MAX_REPETITIONS",
    "MLPERF_EDU_DECODE_MAX_STEPS",
    "MLPERF_EDU_DECODE_MAX_WARMUPS",
    "MLPERF_EDU_DECODE_MIN_BATCH",
    "MLPERF_EDU_DECODE_MIN_PREFILL_CTX",
    "MLPERF_EDU_DECODE_MIN_STEPS",
    "MLPERF_EDU_DEVICE",
    "MLPERF_EDU_GRAPH_MAX_DROPOUT",
    "MLPERF_EDU_GRAPH_MAX_EPOCHS",
    "MLPERF_EDU_GRAPH_MAX_HIDDEN_CHANNELS",
    "MLPERF_EDU_GRAPH_MAX_LAYERS",
    "MLPERF_EDU_GRAPH_MAX_LR",
    "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE",
    "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_REPETITIONS",
    "MLPERF_EDU_KEYWORD_SPOTTING_MAX_BATCH_SIZE",
    "MLPERF_EDU_KEYWORD_SPOTTING_MAX_REPETITIONS",
    "MLPERF_EDU_KEYWORD_SPOTTING_MAX_WARMUP_REPETITIONS",
    "MLPERF_EDU_MAX_BATCH_SIZE",
    "MLPERF_EDU_MAX_BETA2",
    "MLPERF_EDU_MAX_EVAL_INTERVAL",
    "MLPERF_EDU_MAX_EVAL_ITERS",
    "MLPERF_EDU_MAX_ITERS",
    "MLPERF_EDU_MAX_LR",
    "MLPERF_EDU_MAX_MIN_LR",
    "MLPERF_EDU_MAX_MODEL_SIZE",
    "MLPERF_EDU_MAX_QUALITY_TARGET",
    "MLPERF_EDU_MAX_SEED",
    "MLPERF_EDU_MAX_SEQ_LEN",
    "MLPERF_EDU_MAX_WARMUP_ITERS",
    "MLPERF_EDU_MINIGO_GAMES_PER_GENERATION",
    "MLPERF_EDU_MINIGO_GENERATIONS",
    "MLPERF_EDU_NCF_LEARNING_RATE",
    "MLPERF_EDU_NCF_LR_SCHEDULE",
    "MLPERF_EDU_NCF_MAX_EPOCHS",
    "MLPERF_EDU_PREFILL_MAX_BATCH",
    "MLPERF_EDU_PREFILL_MAX_CONTEXT",
    "MLPERF_EDU_PREFILL_MAX_ITER",
    "MLPERF_EDU_PREFILL_MAX_WARMUP",
    "MLPERF_EDU_PREFILL_MIN_BATCH",
    "MLPERF_EDU_PREFILL_MIN_CONTEXT",
    "MLPERF_EDU_PREFILL_MIN_ITER",
    "MLPERF_EDU_PREFILL_MIN_WARMUP",
    "MLPERF_EDU_PRO_REPETITIONS",
    "MLPERF_EDU_RETRIEVAL_BATCH_SIZE",
    "MLPERF_EDU_SEED",
    "MLPERF_EDU_TEXT_CLASSIFICATION_MAX_BATCH_SIZE",
    "MLPERF_EDU_TEXT_CLASSIFICATION_MAX_LENGTH",
    "MLPERF_EDU_TEXT_CLASSIFICATION_MAX_REPETITIONS",
    "MLPERF_EDU_TIMESERIES_HORIZON",
    "MLPERF_EDU_TIMESERIES_MAX_EPOCHS",
    "MLPERF_EDU_TIMESERIES_MAX_EVAL_BATCHES",
    "MLPERF_EDU_TIMESERIES_MAX_TRAIN_BATCHES",
    "MLPERF_EDU_TIMESERIES_PATIENCE",
    "MLPERF_EDU_TIMESERIES_WORKERS",
    "NCCL_ALGO",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_PROTO",
    "NCCL_SOCKET_IFNAME",
    "NUMEXPR_NUM_THREADS",
    "NVIDIA_TF32_OVERRIDE",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "OMP_SCHEDULE",
    "OPENBLAS_NUM_THREADS",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "PYTORCH_MPS_PREFER_METAL",
    "ROCR_VISIBLE_DEVICES",
    "TOKENIZERS_PARALLELISM",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "TORCH_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_HARDWARE_COMPARISON_KEYS = (
    "machine_model",
    "chip",
    "cpu",
    "cpu_topology",
    "gpu",
    "accelerator",
    "memory_gb",
    "cache_sizes",
    "backend",
    "availability_detected_backend",
    "available_backends",
)
_SOFTWARE_COMPARISON_KEYS = (
    "os",
    "os_version",
    "python_version",
    "pytorch_version",
    "torch_runtime",
    "performance_environment",
    "audio_backend",
)


def detect_hardware() -> dict[str, Any]:
    """Detect the system and runtime configuration stamped into run artifacts.

    This is the single source of truth for hardware claims.
    The paper must not state hardware that this function did not detect.
    """
    available_backends = _detect_available_backends()
    availability_detected_backend = _preferred_available_backend(available_backends)
    info = {
        "machine_model": _detect_machine_model(),
        "chip": _detect_chip(),
        "cpu": _detect_cpu(),
        "cpu_topology": _detect_cpu_topology(),
        "gpu": _detect_gpu(),
        "accelerator": _detect_accelerator(),
        "memory_gb": _detect_memory_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "pytorch_version": _detect_pytorch_version(),
        # ``backend`` is retained for report compatibility. It has always meant
        # the best backend found during availability detection, not necessarily
        # the backend selected by a workload.
        "backend": availability_detected_backend,
        "availability_detected_backend": availability_detected_backend,
        "available_backends": available_backends,
        "torch_runtime": _detect_torch_runtime(),
        "performance_environment": _detect_performance_environment(),
        "cache_sizes": _detect_cache_sizes(),
        "audio_backend": _detect_audio_backend(),
    }

    digest = comparison_fingerprint_sha256(info)
    info["fingerprint_schema"] = FINGERPRINT_RECORD_SCHEMA
    info["fingerprint_hash_algorithm"] = "sha256"
    info["fingerprint_hash_scope"] = "canonical-comparison-record"
    # Keep the historical short key while also retaining the collision-resistant
    # full digest used for evidence comparison.
    info["fingerprint_hash"] = digest[:16]
    info["fingerprint_sha256"] = digest

    return info


def comparison_fingerprint_record(info: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical comparison-relevant record bound by the digest."""
    return {
        "schema": FINGERPRINT_RECORD_SCHEMA,
        "hardware": {key: info.get(key) for key in _HARDWARE_COMPARISON_KEYS},
        "software_runtime": {key: info.get(key) for key in _SOFTWARE_COMPARISON_KEYS},
    }


def comparison_fingerprint_sha256(info: dict[str, Any]) -> str:
    """Hash the complete canonical comparison-relevant fingerprint record."""
    payload = json.dumps(
        comparison_fingerprint_record(info),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _detect_machine_model() -> str:
    """Detect machine model (e.g., 'MacBook Pro')."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.model"], capture_output=True, text=True, timeout=5
            )
            model_id = result.stdout.strip()
            # Also try system_profiler for human-readable name
            result2 = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result2.stdout.split("\n"):
                if "Model Name" in line:
                    return line.split(":")[1].strip()
            return model_id
        except Exception:
            pass
    if platform.system() == "Linux":
        for path in (
            "/sys/devices/virtual/dmi/id/product_name",
            "/sys/class/dmi/id/product_name",
        ):
            try:
                with open(path, encoding="utf-8") as handle:
                    value = handle.read().strip()
            except OSError:
                continue
            if value:
                return value
    # Do not fall back to platform.node(): it is commonly a private hostname,
    # not a hardware model, and should not leak into public evidence.
    return platform.machine() or "Unknown"


def _detect_chip() -> str:
    """Detect chip/CPU brand (e.g., 'Apple M5 Max')."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            pass
    # Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


def _detect_cpu() -> str:
    """Detect CPU architecture."""
    return platform.machine()


def _detect_cpu_topology() -> dict[str, int | None]:
    """Detect logical cores, physical cores, and sockets without guessing."""
    logical_cores = os.cpu_count()
    physical_cores: int | None = None
    sockets: int | None = None

    if platform.system() == "Darwin":
        physical_cores = _sysctl_int("hw.physicalcpu")
        logical_cores = _sysctl_int("hw.logicalcpu") or logical_cores
        sockets = _sysctl_int("hw.packages")
    elif platform.system() == "Linux":
        physical_cores, sockets = _detect_linux_physical_topology()

    return {
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
        "sockets": sockets,
    }


def _sysctl_int(name: str) -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        value = int(result.stdout.strip())
        return value if value > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _detect_linux_physical_topology() -> tuple[int | None, int | None]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            blocks = cpuinfo.read().split("\n\n")
    except OSError:
        return None, None

    core_ids: set[tuple[str, str]] = set()
    socket_ids: set[str] = set()
    cores_per_socket: dict[str, int] = {}
    for block in blocks:
        fields = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
        socket = fields.get("physical id")
        core = fields.get("core id")
        if socket is not None:
            socket_ids.add(socket)
            try:
                cores_per_socket[socket] = int(fields.get("cpu cores", ""))
            except ValueError:
                pass
        if socket is not None and core is not None:
            core_ids.add((socket, core))

    physical_cores = len(core_ids) or None
    if (
        physical_cores is None
        and socket_ids
        and len(cores_per_socket) == len(socket_ids)
    ):
        physical_cores = sum(cores_per_socket.values()) or None
    return physical_cores, len(socket_ids) or None


def _detect_gpu() -> str | None:
    """Detect GPU if available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS (Metal Performance Shaders)"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.get_device_name(0)
    except (AttributeError, ImportError, RuntimeError):
        pass
    return None


def _detect_accelerator() -> dict[str, Any] | None:
    """Detect accelerator identity plus runtime and driver metadata."""
    try:
        import torch
    except (ImportError, RuntimeError):
        return None

    try:
        if torch.cuda.is_available():
            hip_version = getattr(torch.version, "hip", None)
            runtime = "ROCm" if hip_version else "CUDA"
            runtime_version = hip_version or getattr(torch.version, "cuda", None)
            device_index = _safe_call(torch.cuda.current_device, default=0)
            properties = _safe_call(
                torch.cuda.get_device_properties, int(device_index), default=None
            )
            capability = _safe_call(
                torch.cuda.get_device_capability, int(device_index), default=None
            )
            if isinstance(capability, tuple):
                capability = ".".join(str(part) for part in capability)
            return {
                "availability_backend": "CUDA",
                "runtime": runtime,
                "runtime_version": runtime_version,
                "driver_version": _detect_accelerator_driver(runtime),
                "device_count": _safe_call(torch.cuda.device_count, default=None),
                "inspection_device_index": device_index,
                "device_name": _safe_call(
                    torch.cuda.get_device_name, int(device_index), default=None
                ),
                "total_memory_bytes": getattr(properties, "total_memory", None),
                "compute_capability": capability,
            }

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            macos_version = platform.mac_ver()[0] or None
            return {
                "availability_backend": "MPS",
                "runtime": "Metal Performance Shaders",
                "runtime_version": macos_version,
                "driver_version": macos_version,
                "driver_source": "integrated-with-macos",
                "device_count": 1,
                "device_name": "Apple MPS (Metal Performance Shaders)",
            }

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device_index = _safe_call(torch.xpu.current_device, default=0)
            return {
                "availability_backend": "XPU",
                "runtime": "Intel XPU",
                "runtime_version": getattr(torch.version, "xpu", None),
                "driver_version": None,
                "device_count": _safe_call(torch.xpu.device_count, default=None),
                "inspection_device_index": device_index,
                "device_name": _safe_call(
                    torch.xpu.get_device_name, int(device_index), default=None
                ),
            }
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return None


def _detect_accelerator_driver(runtime: str) -> str | None:
    if runtime != "CUDA":
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    versions = sorted(
        {line.strip() for line in result.stdout.splitlines() if line.strip()}
    )
    return ",".join(versions) or None


def _detect_memory_gb() -> float:
    """Detect total system memory in GB."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return round(int(result.stdout.strip()) / (1024**3), 1)
        except Exception:
            pass
    # Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    return round(kb / (1024**2), 1)
    except Exception:
        pass
    return 0.0


def _detect_pytorch_version() -> str:
    """Detect PyTorch version."""
    try:
        import torch

        return torch.__version__
    except (ImportError, RuntimeError):
        return "not installed"


def _detect_backend() -> str:
    """Return the preferred availability-detected backend."""
    return _preferred_available_backend(_detect_available_backends())


def _detect_available_backends() -> list[str]:
    """Return the compute backends visible to PyTorch on this host."""
    backends = ["CPU"]
    try:
        import torch

        if torch.cuda.is_available():
            backends.append("CUDA")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            backends.append("MPS")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            backends.append("XPU")
    except (AttributeError, ImportError, RuntimeError):
        pass
    return backends


def _preferred_available_backend(backends: list[str]) -> str:
    for backend in ("CUDA", "MPS", "XPU", "CPU"):
        if backend in backends:
            return backend
    return "CPU"


def _detect_torch_runtime() -> dict[str, Any]:
    """Capture performance-relevant PyTorch process configuration."""
    try:
        import torch
    except (ImportError, RuntimeError):
        return {"available": False}

    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    cuda_matmul = getattr(cuda_backend, "matmul", None)
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    quantized = getattr(getattr(torch, "backends", None), "quantized", None)
    quantized_supported_engines = getattr(quantized, "supported_engines", None) or []
    default_device = None
    if hasattr(torch, "get_default_device"):
        default_device = _safe_call(torch.get_default_device, default=None)

    return {
        "available": True,
        "intra_op_threads": _safe_call(
            getattr(torch, "get_num_threads", None), default=None
        ),
        "inter_op_threads": _safe_call(
            getattr(torch, "get_num_interop_threads", None), default=None
        ),
        "default_dtype": _torch_value_name(
            _safe_call(getattr(torch, "get_default_dtype", None), default=None)
        ),
        "default_device": _torch_value_name(default_device),
        "float32_matmul_precision": _safe_call(
            getattr(torch, "get_float32_matmul_precision", None), default=None
        ),
        "cuda_matmul_allow_tf32": getattr(cuda_matmul, "allow_tf32", None),
        "cudnn_allow_tf32": getattr(cudnn, "allow_tf32", None),
        "cudnn_version": _safe_call(getattr(cudnn, "version", None), default=None),
        "quantized_engine": getattr(quantized, "engine", None),
        "quantized_supported_engines": list(quantized_supported_engines),
        "deterministic_algorithms_enabled": _safe_call(
            getattr(torch, "are_deterministic_algorithms_enabled", None),
            default=None,
        ),
        "deterministic_debug_mode": _safe_call(
            getattr(torch, "get_deterministic_debug_mode", None), default=None
        ),
        "cudnn_deterministic": getattr(cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(cudnn, "benchmark", None),
    }


def _safe_call(function: Any, *args: Any, default: Any) -> Any:
    if not callable(function):
        return default
    try:
        return function(*args)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return default


def _torch_value_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).removeprefix("torch.")


def _detect_performance_environment() -> dict[str, str]:
    """Record only explicitly allowlisted performance environment settings.

    Credential variables, user text, and machine-local asset paths are excluded.
    """
    return {
        name: os.environ[name]
        for name in PERFORMANCE_ENVIRONMENT_ALLOWLIST
        if name in os.environ
    }


def _detect_cache_sizes() -> dict[str, int | None]:
    """Detect CPU cache sizes in bytes. Returns what is measurable, nothing more."""
    caches = {"l1d": None, "l1i": None, "l2": None, "l3": None}

    if platform.system() == "Darwin":
        mapping = {
            "hw.l1dcachesize": "l1d",
            "hw.l1icachesize": "l1i",
            "hw.l2cachesize": "l2",
            "hw.l3cachesize": "l3",
        }
        for sysctl_key, cache_key in mapping.items():
            try:
                result = subprocess.run(
                    ["sysctl", "-n", sysctl_key],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                val = result.stdout.strip()
                if val:
                    caches[cache_key] = int(val)
            except Exception:
                pass
    elif platform.system() == "Linux":
        # Try /sys/devices/system/cpu/cpu0/cache/
        cache_dir = "/sys/devices/system/cpu/cpu0/cache"
        if os.path.isdir(cache_dir):
            for idx_dir in sorted(os.listdir(cache_dir)):
                idx_path = os.path.join(cache_dir, idx_dir)
                try:
                    with open(os.path.join(idx_path, "level")) as f:
                        level = int(f.read().strip())
                    with open(os.path.join(idx_path, "type")) as f:
                        ctype = f.read().strip()
                    with open(os.path.join(idx_path, "size")) as f:
                        size_str = f.read().strip()
                        # Parse "32K", "256K", "6144K"
                        if size_str.endswith("K"):
                            size_bytes = int(size_str[:-1]) * 1024
                        elif size_str.endswith("M"):
                            size_bytes = int(size_str[:-1]) * 1024 * 1024
                        else:
                            size_bytes = int(size_str)

                    if level == 1 and ctype == "Data":
                        caches["l1d"] = size_bytes
                    elif level == 1 and ctype == "Instruction":
                        caches["l1i"] = size_bytes
                    elif level == 2:
                        caches["l2"] = size_bytes
                    elif level == 3:
                        caches["l3"] = size_bytes
                except Exception:
                    pass

    return caches


def _detect_audio_backend() -> str | None:
    """Detect torchaudio backend availability."""
    try:
        import torchaudio

        return f"torchaudio {torchaudio.__version__}"
    except (ImportError, OSError, RuntimeError):
        return None


def format_fingerprint(hw: dict[str, Any]) -> str:
    """Format hardware fingerprint as a human-readable string."""
    lines = [
        f"Machine:  {hw['machine_model']}",
        f"Chip:     {hw['chip']}",
        f"Memory:   {hw['memory_gb']} GB",
        f"GPU:      {hw.get('gpu') or 'None'}",
        f"OS:       {hw['os']}",
        f"Python:   {hw['python_version']}",
        f"PyTorch:  {hw['pytorch_version']}",
        f"Available: {', '.join(hw.get('available_backends', [hw['backend']]))}",
        f"Detected:  {hw.get('availability_detected_backend', hw['backend'])}",
    ]

    caches = hw.get("cache_sizes", {})
    if any(v is not None for v in caches.values()):
        cache_parts = []
        for level in ["l1d", "l2", "l3"]:
            val = caches.get(level)
            if val:
                if val >= 1024 * 1024:
                    cache_parts.append(f"{level.upper()}={val // (1024 * 1024)}MB")
                else:
                    cache_parts.append(f"{level.upper()}={val // 1024}KB")
        lines.append(f"Caches:   {', '.join(cache_parts)}")

    lines.append(f"ID:       {hw['fingerprint_hash']}")
    return "\n".join(lines)


if __name__ == "__main__":
    hw = detect_hardware()
    print("=== MLPerf EDU Hardware Fingerprint ===")
    print(format_fingerprint(hw))
