#!/usr/bin/env python3
"""
Smoke test for iter-5: combined provenance + roofline emitter.

Two gated checks (analogous to iter-2's 3x lookup gap and iter-3's 5x
intensity ratio):

  Check 1 (GATED): Tamper detection.
    Build a manifest binding a small "weights" file. Verify it: PASS.
    Mutate one byte in the weights file. Verify again: must FAIL on
    weights.sha256. The iter-1 era str(report) self-hash could not
    distinguish these two cases.

  Check 2 (GATED): Roofline self-consistency.
    Run two synthetic workloads through measure_roofline:
      A. cache-resident, bandwidth-bound, dispatch-bound (toy gather)
      B. compute-saturated (big GEMM)
    Acceptance: B.dispatch_utilization / A.dispatch_utilization >= 4x.
    If the emitter cannot tell these apart by 4x, it cannot be trusted
    to populate the iter-4 taxonomy's 12 unmeasured-dispatch cells.

Run: python3 scripts/smoke_roofline_provenance.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from mlperf.manifest import build_provd, verify_provd  # noqa: E402
from mlperf.roofline import measure_roofline  # noqa: E402


def check1_tamper_detection() -> bool:
    """Verifier must catch a single-byte mutation in weights file."""
    print("=== Check 1: Tamper Detection ===")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        weights_path = td / "fake_weights.pt"
        weights_path.write_bytes(b"\x00" * 1024)  # 1 KB of zeros
        report_path = td / "fake_report.json"
        report = {"latency_p50": 0.001, "throughput": 1000}
        report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")))

        manifest = build_provd(
            workload="smoke-tamper",
            scenario="SingleStream",
            division="closed",
            hardware_fingerprint={"machine": "smoke", "system": {"machine": "test"}},
            report=report,
            report_path=report_path,
            weights_path=weights_path,
            weights_n_params=256,
            weights_dtype="float32",
            dataset_name="smoke",
            dataset_files=[],
            rng_seed=42,
            repo_root=REPO_ROOT,
        )
        manifest_path = td / "smoke.provd.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))

        print("  step 1: verify pristine artifact...")
        res = verify_provd(manifest_path, repo_root=REPO_ROOT)
        weight_check = next((ok for n, ok, _ in res.checks if n == "weights.sha256"), None)
        if weight_check is not True:
            print(f"  FAIL: pristine verify did not pass weights.sha256 (got {weight_check})")
            return False
        print(f"  pristine verify: weights.sha256 = OK ({sum(1 for _,ok,_ in res.checks if ok)}/{len(res.checks)} leaves)")

        print("  step 2: mutate one byte in weights file...")
        data = bytearray(weights_path.read_bytes())
        data[0] ^= 0xFF
        weights_path.write_bytes(bytes(data))

        print("  step 3: re-verify; expect weights.sha256 FAIL...")
        res2 = verify_provd(manifest_path, repo_root=REPO_ROOT)
        weight_check2 = next((ok for n, ok, _ in res2.checks if n == "weights.sha256"), None)
        if weight_check2 is True:
            print("  FAIL: tampered file still verified clean")
            return False
        print("  tampered verify: weights.sha256 = FAIL (as expected)")
        print("  PASS: tamper detection works.")
        return True


def check2_roofline_self_consistency() -> bool:
    """Emitter must distinguish dispatch-bound from device-saturated by >= 4x."""
    print("\n=== Check 2: Roofline Self-Consistency ===")
    out_dir = Path(tempfile.mkdtemp())

    # Workload A: tiny gather, dispatch-dominated.
    table_a = torch.nn.EmbeddingBag(2048, 8, mode="sum")
    idx_a = torch.randint(0, 2048, (256,), dtype=torch.long)
    off_a = torch.arange(256, dtype=torch.long)
    n_iter = 200
    with measure_roofline("smoke-dispatch-bound",
                           analytic_flops=lambda: 256 * 8 * n_iter,           # negligible FLOPs
                           analytic_bytes=lambda: 256 * 8 * 4 * n_iter,        # one row per lookup
                           n_iter=n_iter, output_dir=out_dir):
        for _ in range(n_iter):
            table_a(idx_a, off_a)
    sidecar_a_path = Path(os.environ["MLPERF_EDU_LAST_SIDECAR"])
    a = json.loads(sidecar_a_path.read_text())

    # Workload B: large GEMM, compute-dominated.
    M, N, K = 1024, 1024, 1024
    x = torch.randn(M, K)
    w = torch.randn(K, N)
    flops_per_iter = 2 * M * N * K
    bytes_per_iter = (M * K + K * N + M * N) * 4
    n_iter_b = 50
    with measure_roofline("smoke-device-saturated",
                           analytic_flops=lambda: flops_per_iter * n_iter_b,
                           analytic_bytes=lambda: bytes_per_iter * n_iter_b,
                           n_iter=n_iter_b, output_dir=out_dir):
        for _ in range(n_iter_b):
            torch.matmul(x, w)
    sidecar_b_path = Path(os.environ["MLPERF_EDU_LAST_SIDECAR"])
    b = json.loads(sidecar_b_path.read_text())

    util_a = a["measurement"]["dispatch_utilization"]
    util_b = b["measurement"]["dispatch_utilization"]
    intensity_a = a["measurement"]["intensity_FLOPS_per_byte"]
    intensity_b = b["measurement"]["intensity_FLOPS_per_byte"]
    ratio = util_b / util_a if util_a > 0 else float("inf")

    print(f"  A (gather):  intensity={intensity_a:.3f}, util={util_a:.4f}, "
          f"axis_dispatch={a['regime_inference']['axis_dispatch']}")
    print(f"  B (GEMM):    intensity={intensity_b:.3f}, util={util_b:.4f}, "
          f"axis_dispatch={b['regime_inference']['axis_dispatch']}")
    print(f"  utilization ratio B/A: {ratio:.2f}x (gate >= 4x)")

    if ratio < 4.0:
        print("  FAIL: emitter cannot distinguish dispatch_bound from device_saturated by 4x.")
        return False
    print("  PASS: roofline emitter resolves the regime distinction.")
    return True


def main() -> int:
    ok1 = check1_tamper_detection()
    ok2 = check2_roofline_self_consistency()
    print()
    if ok1 and ok2:
        print("ITER-5 SMOKE: PASS (provenance + roofline emitter both work)")
        return 0
    print(f"ITER-5 SMOKE: FAIL (check1={ok1}, check2={ok2})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
