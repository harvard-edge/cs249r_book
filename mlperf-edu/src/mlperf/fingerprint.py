"""
MLPerf EDU: System Fingerprint

Auto-detects hardware and software configuration at runtime.
Every benchmark run stamps this into the JSON artifact.
No manual hardware claims — all evidence is measured.
"""

import platform
import subprocess
import sys
import os
import hashlib
from typing import Dict, Any, Optional


def detect_hardware() -> Dict[str, Any]:
    """Detect actual system hardware. Returns a dict stamped into every run artifact.
    
    This is the single source of truth for hardware claims.
    The paper must not state hardware that this function did not detect.
    """
    info = {
        "machine_model": _detect_machine_model(),
        "chip": _detect_chip(),
        "cpu": _detect_cpu(),
        "gpu": _detect_gpu(),
        "memory_gb": _detect_memory_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "pytorch_version": _detect_pytorch_version(),
        "backend": _detect_backend(),
        "cache_sizes": _detect_cache_sizes(),
        "audio_backend": _detect_audio_backend(),
    }
    
    # Compute a deterministic fingerprint hash for cross-run comparison
    fingerprint_str = f"{info['chip']}|{info['memory_gb']}|{info['os']}|{info['pytorch_version']}"
    info["fingerprint_hash"] = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    
    return info


def _detect_machine_model() -> str:
    """Detect machine model (e.g., 'MacBook Pro')."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.model"],
                capture_output=True, text=True, timeout=5
            )
            model_id = result.stdout.strip()
            # Also try system_profiler for human-readable name
            result2 = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True, timeout=10
            )
            for line in result2.stdout.split("\n"):
                if "Model Name" in line:
                    return line.split(":")[1].strip()
            return model_id
        except Exception:
            pass
    return platform.node()


def _detect_chip() -> str:
    """Detect chip/CPU brand (e.g., 'Apple M5 Max')."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
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


def _detect_gpu() -> Optional[str]:
    """Detect GPU if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "Apple MPS (Metal Performance Shaders)"
    except ImportError:
        pass
    return None


def _detect_memory_gb() -> float:
    """Detect total system memory in GB."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
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
    except ImportError:
        return "not installed"


def _detect_backend() -> str:
    """Detect the active compute backend."""
    try:
        import torch
        if torch.cuda.is_available():
            return "CUDA"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "MPS"
        else:
            return "CPU"
    except ImportError:
        return "CPU"


def _detect_cache_sizes() -> Dict[str, Optional[int]]:
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
                    capture_output=True, text=True, timeout=5
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


def _detect_audio_backend() -> Optional[str]:
    """Detect torchaudio backend availability."""
    try:
        import torchaudio
        return f"torchaudio {torchaudio.__version__}"
    except ImportError:
        return None


def format_fingerprint(hw: Dict[str, Any]) -> str:
    """Format hardware fingerprint as a human-readable string."""
    lines = [
        f"Machine:  {hw['machine_model']}",
        f"Chip:     {hw['chip']}",
        f"Memory:   {hw['memory_gb']} GB",
        f"GPU:      {hw.get('gpu') or 'None'}",
        f"OS:       {hw['os']}",
        f"Python:   {hw['python_version']}",
        f"PyTorch:  {hw['pytorch_version']}",
        f"Backend:  {hw['backend']}",
    ]
    
    caches = hw.get("cache_sizes", {})
    if any(v is not None for v in caches.values()):
        cache_parts = []
        for level in ["l1d", "l2", "l3"]:
            val = caches.get(level)
            if val:
                if val >= 1024 * 1024:
                    cache_parts.append(f"{level.upper()}={val // (1024*1024)}MB")
                else:
                    cache_parts.append(f"{level.upper()}={val // 1024}KB")
        lines.append(f"Caches:   {', '.join(cache_parts)}")
    
    lines.append(f"ID:       {hw['fingerprint_hash']}")
    return "\n".join(lines)


def tensor_cache_analysis(tensor_bytes: int, hw: Dict[str, Any]) -> str:
    """Determine which cache level a tensor fits in, based on measured cache sizes.
    
    Returns a factual statement about tensor size vs cache capacity.
    Does NOT guess — only reports what was detected.
    """
    caches = hw.get("cache_sizes", {})
    
    fits_in = []
    for level_name, level_key in [("L1d", "l1d"), ("L2", "l2"), ("L3", "l3")]:
        size = caches.get(level_key)
        if size is not None and tensor_bytes <= size:
            fits_in.append(f"{level_name} ({size // 1024}KB)")
    
    tensor_kb = tensor_bytes / 1024
    if fits_in:
        return f"{tensor_kb:.0f}KB fits in {fits_in[0]}"
    elif any(v is not None for v in caches.values()):
        return f"{tensor_kb:.0f}KB exceeds detected cache sizes"
    else:
        return f"{tensor_kb:.0f}KB (cache sizes not detected on this platform)"


if __name__ == "__main__":
    hw = detect_hardware()
    print("=== MLPerf EDU Hardware Fingerprint ===")
    print(format_fingerprint(hw))
    print()
    
    # DLRM embedding table analysis
    dlrm_bytes = (943 * 32 + 1682 * 32) * 4  # float32
    print(f"DLRM embedding tables: {tensor_cache_analysis(dlrm_bytes, hw)}")
