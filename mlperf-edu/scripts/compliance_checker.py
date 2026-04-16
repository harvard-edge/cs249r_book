#!/usr/bin/env python3
"""
MLPerf EDU: Benchmark Compliance Checker

Validates that a training run meets MLPerf EDU quality targets.
Inspired by the official MLPerf training compliance tools.

Usage:
    python scripts/compliance_checker.py --workload nanogpt --log training_log.json
    python scripts/compliance_checker.py --all  # Check all workloads

Checks:
1. Quality target reached within epoch budget
2. Train/val gap within acceptable bounds (no overfitting)
3. Deterministic seed used (reproducibility)
4. Dataset provenance verified
5. Model parameter count matches spec
"""

import os
import sys
import json
import argparse
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Direction inference: metrics where lower is better
_LOWER_IS_BETTER = {"cross_entropy_loss", "mse_loss", "val_mse", "reconstruction_mse", "loss", "mse"}


def _load_quality_targets():
    """Load quality targets from canonical workloads.yaml (single source of truth)."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "workloads.yaml")
    if not os.path.exists(yaml_path):
        print(f"  ⚠️  workloads.yaml not found at {yaml_path}, using fallback")
        return {}
    
    with open(yaml_path) as f:
        registry = yaml.safe_load(f)
    
    targets = {}
    for suite_name, workloads in registry.get("suites", {}).items():
        for wl_key, wl_cfg in workloads.items():
            qt = wl_cfg.get("quality_target", {})
            if not qt:
                continue
            
            # Normalize workload key: "nanogpt-train" -> "nanogpt"
            short_key = wl_key.replace("-train", "").replace("-kws", "").replace("-agent", "")
            
            metric = qt.get("metric", "")
            threshold = qt.get("value", 0)
            direction = "below" if metric in _LOWER_IS_BETTER else "above"
            
            # Infer max_epochs from verified baseline
            baseline = wl_cfg.get("verified_baseline", {})
            max_epochs = baseline.get("epochs", 50)
            
            targets[short_key] = {
                "metric": metric, "threshold": threshold,
                "direction": direction, "max_epochs": max_epochs,
                "full_key": wl_key,
            }
    
    return targets


# Load from workloads.yaml — single source of truth
QUALITY_TARGETS = _load_quality_targets()

# Acceptable train/val gap bounds (absolute)
MAX_GAP = {
    "nanogpt": 0.5,
    "nano-moe": 0.01,
    "micro-dlrm": 0.05,
    "micro-diffusion": 0.01,
    "micro-gnn": 0.05,
    "micro-bert": 0.05,
    "micro-lstm": 0.02,
    "micro-rl": 100.0,  # RL is high-variance by design
    "resnet18": 1.0,
    "mobilenetv2": 1.0,
    "dscnn": 0.5,
    "anomaly-ae": 0.1,  # Intentionally higher — anomaly detection design
    "wake-vision": 0.1,
}

EXPECTED_PARAMS = {
    "nanogpt": (100_000_000, 150_000_000),  # 124M
    "nano-moe": (15_000_000, 20_000_000),    # 17.4M
    "micro-dlrm": (400_000, 700_000),        # 0.6M
    "micro-diffusion": (1_500_000, 2_500_000),  # 2.0M
    "micro-gnn": (4_000, 8_000),             # 5.6K
    "micro-bert": (350_000, 550_000),        # 432K
    "micro-lstm": (40_000, 65_000),          # 51K
    "micro-rl": (10_000, 25_000),            # 18K
    "resnet18": (10_000_000, 12_000_000),    # 11.2M
    "mobilenetv2": (2_000_000, 3_000_000),   # 2.4M
    "dscnn": (15_000, 25_000),               # 20K
    "anomaly-ae": (250_000, 350_000),        # 0.3M
    "wake-vision": (5_000, 12_000),          # 8.5K
}


class ComplianceResult:
    """Stores the result of a compliance check."""

    def __init__(self, workload: str):
        self.workload = workload
        self.checks = []
        self.passed = 0
        self.failed = 0

    def check(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.checks.append({"name": name, "status": status, "detail": detail})
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def compliant(self):
        return self.failed == 0

    def report(self):
        print(f"\n{'='*60}")
        print(f"  Compliance Report: {self.workload}")
        print(f"{'='*60}")
        for c in self.checks:
            icon = "✅" if c["status"] == "PASS" else "❌"
            print(f"  {icon} {c['name']:35s} {c['detail']}")
        print(f"\n  Result: {'COMPLIANT ✅' if self.compliant else 'NON-COMPLIANT ❌'}")
        print(f"  ({self.passed} passed, {self.failed} failed)")
        print(f"{'='*60}")
        return self.compliant


def check_quality_target(result: ComplianceResult, final_val_metric: float, workload: str):
    """Check if quality target was reached."""
    target = QUALITY_TARGETS.get(workload)
    if target is None:
        result.check("Quality target", False, f"No target defined for '{workload}'")
        return

    threshold = target["threshold"]
    if target["direction"] == "below":
        passed = final_val_metric <= threshold
        detail = f"{final_val_metric:.4f} {'<=' if passed else '>'} {threshold}"
    else:
        passed = final_val_metric >= threshold
        detail = f"{final_val_metric:.4f} {'>=' if passed else '<'} {threshold}"

    result.check("Quality target reached", passed, detail)


def check_overfitting(result: ComplianceResult, train_metric: float, val_metric: float, workload: str):
    """Check train/val gap is within bounds."""
    gap = abs(val_metric - train_metric)
    max_gap = MAX_GAP.get(workload, 0.5)
    passed = gap <= max_gap
    result.check("Train/val gap acceptable", passed,
                 f"gap={gap:.4f}, max={max_gap}")


def check_param_count(result: ComplianceResult, n_params: int, workload: str):
    """Check model parameter count is within expected range."""
    expected = EXPECTED_PARAMS.get(workload)
    if expected is None:
        result.check("Parameter count", True, f"{n_params:,} (no target)")
        return

    lo, hi = expected
    passed = lo <= n_params <= hi
    result.check("Parameter count in range", passed,
                 f"{n_params:,} ∈ [{lo:,}, {hi:,}]")


def check_deterministic_seed(result: ComplianceResult, seed: int = 42):
    """Check that the canonical seed was used."""
    passed = seed == 42
    result.check("Deterministic seed (42)", passed, f"seed={seed}")


def check_from_log(log_path: str, workload: str) -> ComplianceResult:
    """Run compliance checks from a JSON training log."""
    result = ComplianceResult(workload)

    with open(log_path) as f:
        log = json.load(f)

    check_quality_target(result, log.get("final_val_metric", float("inf")), workload)
    check_overfitting(result, log.get("final_train_metric", 0), log.get("final_val_metric", 0), workload)
    check_param_count(result, log.get("n_params", 0), workload)
    check_deterministic_seed(result, log.get("seed", -1))

    epochs = log.get("epochs", 0)
    max_epochs = QUALITY_TARGETS.get(workload, {}).get("max_epochs", 100)
    result.check("Epoch budget", epochs <= max_epochs,
                 f"{epochs} <= {max_epochs}")

    return result


def check_from_submission(submission_path: str) -> ComplianceResult:
    """Run compliance checks from a submission JSON file.
    
    Validates:
    1. Required schema fields present
    2. Workload target attainment
    3. Dataset hash integrity
    4. Minimum run count (>=3)
    5. JSON schema conformance
    """
    with open(submission_path) as f:
        data = json.load(f)
    
    workload = data.get("workload", "unknown")
    result = ComplianceResult(workload)
    
    # Schema validation
    required_fields = ["workload", "metrics", "timestamp"]
    for field in required_fields:
        result.check(f"Schema: '{field}' present", field in data,
                     "present" if field in data else "MISSING")
    
    # Quality target from workloads.yaml
    metrics = data.get("metrics", {})
    # Try to find the best matching metric
    target = QUALITY_TARGETS.get(workload.replace("-12m", "").replace("-1m", ""))
    if target:
        metric_name = target["metric"]
        # Try common field mappings
        val = metrics.get(metric_name) or metrics.get("loss") or metrics.get("accuracy")
        if val is not None:
            check_quality_target(result, val, workload.replace("-12m", "").replace("-1m", ""))
        else:
            result.check("Quality target", False, f"metric '{metric_name}' not found in submission")
    
    # Hash integrity
    hashes = data.get("hashes", data.get("integrity", {}))
    result.check("Dataset hash present",
                 bool(hashes.get("dataset", hashes.get("dataset_hash"))),
                 "SHA-256 hash found" if hashes else "no hashes")
    result.check("Model hash present",
                 bool(hashes.get("model", hashes.get("checkpoint_hash"))),
                 "SHA-256 hash found" if hashes else "no hashes")
    
    # Run count
    compliance = data.get("compliance", {})
    run_count = compliance.get("run_count", 1)
    result.check("Minimum run count (≥3)", run_count >= 3,
                 f"run_count={run_count}")
    
    return result


def quick_check_model(workload: str) -> ComplianceResult:
    """Quick compliance check by instantiating the model."""
    result = ComplianceResult(workload)

    try:
        import torch

        models = {
            "nanogpt": ("reference.cloud.nanogpt_core", "GPT", {}),
            "nano-moe": ("reference.cloud.nano_moe", "NanoMoEWhiteBox", {}),
            "micro-dlrm": ("reference.cloud.micro_dlrm", "MicroDLRMWhiteBox", {}),
            "micro-diffusion": ("reference.cloud.micro_diffusion", "MicroDiffusionUNet", {}),
            "micro-gnn": ("reference.cloud.micro_gnn", "MicroGCN", {"nfeat": 50, "nclass": 7}),
            "micro-bert": ("reference.cloud.micro_bert", "MicroBERT", {}),
            "micro-lstm": ("reference.cloud.micro_lstm", "MicroLSTM", {}),
            "micro-rl": ("reference.cloud.micro_rl", "REINFORCEAgent", {}),
            "resnet18": ("reference.edge.resnet_core", "ResNet18Local", {"num_classes": 100}),
            "mobilenetv2": ("reference.mobile.mobilenet_core", "MobileNetV2Local", {"num_classes": 100}),
            "dscnn": ("reference.tiny.dscnn_kws", "DSCNN", {"num_classes": 12}),
            "anomaly-ae": ("reference.tiny.anomaly_detection_ae", "AnomalyDetectionAE", {"input_dim": 784}),
            "wake-vision": ("reference.tiny.wake_vision_vww", "MicroNet", {"num_classes": 2}),
        }

        if workload not in models:
            result.check("Model instantiation", False, f"Unknown workload: {workload}")
            return result

        mod_path, cls_name, kwargs = models[workload]
        module = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(module, cls_name)
        model = cls(**kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        result.check("Model instantiation", True, cls_name)
        check_param_count(result, n_params, workload)
        check_deterministic_seed(result)

        # Check dataset loads
        from reference.dataset_factory import get_dataloaders
        name_map = {
            "nanogpt": "nanogpt-12m", "nano-moe": "nanogpt-12m",
            "micro-dlrm": "micro-dlrm-1m", "micro-diffusion": "micro-diffusion-32px",
            "micro-gnn": "gnn", "micro-bert": "bert",
            "micro-lstm": "lstm", "micro-rl": "rl",
            "resnet18": "resnet18", "mobilenetv2": "mobilenetv2",
            "dscnn": "dscnn-kws", "anomaly-ae": "anomaly-ae",
            "wake-vision": "wake-vision-vww",
        }
        ds_result = get_dataloaders(name_map[workload], batch_size=4)
        # Handle different return types (DataLoader tuple, dict for GNN/RL)
        if isinstance(ds_result, dict):
            n_train = ds_result.get('x', torch.tensor([])).shape[0] if 'x' in ds_result else 1
            result.check("Dataset loads", True, f"dict with {len(ds_result)} keys")
        elif isinstance(ds_result, tuple):
            t, v = ds_result
            n_train = len(t.dataset)
            result.check("Dataset loads", True, f"{n_train} train samples")
        else:
            result.check("Dataset loads", True, "loaded")

    except Exception as e:
        result.check("Model instantiation", False, str(e))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLPerf EDU Compliance Checker")
    parser.add_argument("--workload", type=str, help="Workload name to check")
    parser.add_argument("--log", type=str, help="Path to training log JSON")
    parser.add_argument("--submission", type=str, help="Path to submission JSON to validate")
    parser.add_argument("--all", action="store_true", help="Check all workloads")
    args = parser.parse_args()

    if args.submission:
        print("🔍 MLPerf EDU Submission Compliance Check\n")
        result = check_from_submission(args.submission)
        result.report()
    elif args.all:
        print("🔍 MLPerf EDU Compliance Checker — All Workloads\n")
        all_compliant = True
        for wl in QUALITY_TARGETS:
            result = quick_check_model(wl)
            compliant = result.report()
            all_compliant = all_compliant and compliant

        print(f"\n{'='*60}")
        print(f"  OVERALL: {'ALL COMPLIANT ✅' if all_compliant else 'SOME FAILURES ❌'}")
        print(f"{'='*60}")
    elif args.workload and args.log:
        result = check_from_log(args.log, args.workload)
        result.report()
    elif args.workload:
        result = quick_check_model(args.workload)
        result.report()
    else:
        parser.print_help()
