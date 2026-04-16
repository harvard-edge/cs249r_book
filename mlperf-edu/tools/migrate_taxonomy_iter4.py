#!/usr/bin/env python3
"""
One-shot iter-4 migration: add `regime:` blocks to every workload in
workloads.yaml per Emer's iter-4 placement table.

Workloads with empirical evidence (iter-2 DLRM smoke, iter-3 NanoGPT
phase smoke) get fully populated regime blocks. Workloads without
runtime measurements get static-analysis values for axes A and B and
`unmeasured` for axis C — per Emer's directive: "Do not back-fill
Axis C with guesses."

After running this script, `python3 tools/check_taxonomy.py` should
exit 0.

Run once: python3 tools/migrate_taxonomy_iter4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLOADS_YAML = REPO_ROOT / "workloads.yaml"

# Threshold constants (must match check_taxonomy.py).
LLC_BYTES = 12 * 1024 * 1024
RIDGE = 30  # M1 fp32 ridge FLOP/byte
PEAK_BW = 68.25

# Per Emer's placement table in the iter-4 proposal. Values for axes
# without runtime evidence are static-analysis or `unmeasured`.
PLACEMENTS: dict[str, dict] = {
    # === Cloud ===
    "nanogpt-train": {
        "working_set": dict(value="unmeasured", peak_bytes_per_step=64 * 1024 * 1024,
                            note="Per-layer activation peak ~12 MB at B=16 T=64; weights ~46 MB streamed once per epoch. Falls in the 6-48 MB grey band; exact classification depends on batch and depends on caching behavior — measure with cache-miss counters."),
        "arithmetic_intensity": dict(value="compute_bound", flops_per_byte=120,
                                     classification_rule="weights reused across all tokens in a batch"),
        "dispatch": dict(value="unmeasured", note="Training-loop dispatch needs a roofline-emitter run; logged for iter 5."),
    },
    "nano-moe-train": {
        "working_set": dict(value="unmeasured", peak_bytes_per_step=20 * 1024 * 1024,
                            note="17M params, top-2 routing. ~21 MB per-step is in the 6-48 MB grey band; conditional compute pattern complicates cache modeling — measure."),
        "arithmetic_intensity": dict(value="compute_bound", flops_per_byte=80,
                                     classification_rule="active expert MLPs reuse weights across batch"),
        "dispatch": dict(value="unmeasured",
                         note="Routing/gating may push to dispatch_bound at small batch; needs measurement."),
    },
    "micro-dlrm-train": {
        "working_set": dict(value="cache_resident", peak_bytes_per_step=21 * 1024,
                            note="943x8 + 1682x8 + 21x8 ~ 21 KB embedding tables fit in L1 trivially."),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=1.0,
                                     classification_rule="sparse gather is 0 FLOPs; only the small MLP contributes"),
        "dispatch": dict(value="dispatch_bound", utilization=0.05,
                         observation_source="iter-2 smoke_dlrm_dram.py probe; cache variant is dispatch-bound at any realistic batch size"),
    },
    "micro-dlrm-dram-train": {
        "working_set": dict(value="dram_bound", peak_bytes_per_step=64 * 1024 * 1024,
                            note="2M-row x 256-dim virtual table = 2 GB total; per-step working set ~64 MB at B=8192 with random hashing"),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=0.5,
                                     classification_rule="one MAC per loaded float; sparse gather dominates"),
        "dispatch": dict(value="device_saturated", utilization=0.6, achieved_bw_gbps=40.0,
                         observation_source="iter-2 smoke_dlrm_dram.py: m_spa=256 explicitly chosen to clear PyTorch's ~50us dispatch floor"),
    },
    "nanogpt-prefill": {
        "working_set": dict(value="dram_bound", peak_bytes_per_step=96468992,
                            note="Per-layer attention scores tensor (1792x1792 per head, 6 heads, fp32) is 19.3M floats = 77 MB alone — well past the 4*LLC = 48 MB threshold. The cited 96 MB also includes Q/K/V and FFN activations."),
        "arithmetic_intensity": dict(value="compute_bound", flops_per_byte=289,
                                     classification_rule="weights reused across ctx_len=1792 tokens; well past M1 ridge of 30"),
        "dispatch": dict(value="device_saturated", utilization=0.55,
                         observation_source="iter-3 smoke_nanogpt_phases.py; prefill latency 13ms over 1792 tokens"),
    },
    "nanogpt-decode": {
        "working_set": dict(value="unmeasured", peak_bytes_per_step=34603008,
                            note="32 MB KV cache stream per step lands in the 6-48 MB grey band. Empirically the achieved BW (4 GB/s vs 68 GB/s peak) suggests DRAM streaming dominates, but per-step working set technically classifies as ambiguous on this axis. Real bottleneck is on Axis C."),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=0.5,
                                     classification_rule="one new token through all weights + full KV reread; 0.5 FLOP/byte"),
        "dispatch": dict(value="dispatch_bound", utilization=0.059, achieved_bw_gbps=4.01,
                         observation_source="iter-3 smoke_nanogpt_phases.py on M-series MPS; the canonical dispatch_bound case"),
    },
    "micro-diffusion-train": {
        "working_set": dict(value="unmeasured", peak_bytes_per_step=8 * 1024 * 1024,
                            note="2M-param U-Net on CIFAR-10 32x32; ~8 MB per-step in the grey band — measure."),
        "arithmetic_intensity": dict(value="compute_bound", flops_per_byte=70,
                                     classification_rule="dense convs on small images, weights reused across batch"),
        "dispatch": dict(value="unmeasured", note="Conv kernel sizes likely large enough but verify"),
    },
    "micro-gnn-train": {
        "working_set": dict(value="cache_resident", peak_bytes_per_step=200 * 1024,
                            note="Cora has 2708 nodes, 5429 edges; full graph fits in L1"),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=2.0,
                                     classification_rule="sparse adjacency multiply has very low arithmetic intensity"),
        "dispatch": dict(value="unmeasured",
                         note="Per-step kernels are microscopic; very likely dispatch_bound but needs measurement"),
    },
    "micro-bert-train": {
        "working_set": dict(value="cache_resident", peak_bytes_per_step=4 * 1024 * 1024,
                            note="432K params, short SST-2 sequences"),
        "arithmetic_intensity": dict(value="compute_bound", flops_per_byte=80,
                                     classification_rule="bidirectional attention reuses weights across batch"),
        "dispatch": dict(value="unmeasured"),
    },
    "micro-lstm-train": {
        "working_set": dict(value="cache_resident", peak_bytes_per_step=512 * 1024,
                            note="51K params, 96-step horizon"),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=4.0,
                                     classification_rule="sequential per-timestep matmuls have low intensity"),
        "dispatch": dict(value="unmeasured",
                         note="Sequential timesteps invite dispatch overhead; likely dispatch_bound, verify"),
    },
    "micro-rl-train": {
        "working_set": dict(value="cache_resident", peak_bytes_per_step=64 * 1024,
                            note="17K params, tiny actor-critic"),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=2.0,
                                     classification_rule="env step in Python dominates; nn forward is microscopic"),
        "dispatch": dict(value="unmeasured", note="Almost certainly dispatch_bound; verify"),
    },
    # === Edge ===
    "resnet18-train": {
        "working_set": dict(value="unmeasured", peak_bytes_per_step=46 * 1024 * 1024,
                            note="11.2M params; activations + weights ~46 MB at B=64. Just under 4*LLC threshold; classification depends on whether activations stream or stay resident — measure."),
        "arithmetic_intensity": dict(value="compute_bound", flops_per_byte=120,
                                     classification_rule="standard convs, well-studied compute-bound regime"),
        "dispatch": dict(value="unmeasured"),
    },
    "mobilenetv2-train": {
        "working_set": dict(value="unmeasured", peak_bytes_per_step=10 * 1024 * 1024,
                            note="2.4M params, depthwise-separable convs; 10 MB per-step in grey band — measure."),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=12,
                                     classification_rule="depthwise convs have notoriously low arithmetic intensity"),
        "dispatch": dict(value="unmeasured",
                         note="Despite 'edge' label, MobileNetV2 is mostly memory-bound, not compute-bound"),
    },
    # === Tiny ===
    "dscnn-kws-train": {
        "working_set": dict(value="cache_resident", peak_bytes_per_step=2 * 1024 * 1024,
                            note="20K params + 40x101 spectrogram inputs"),
        "arithmetic_intensity": dict(value="compute_bound", flops_per_byte=70,
                                     classification_rule="moderate intensity from spectrogram convs"),
        "dispatch": dict(value="unmeasured"),
    },
    "anomaly-ae-train": {
        "working_set": dict(value="cache_resident", peak_bytes_per_step=2 * 1024 * 1024,
                            note="0.3M params FC autoencoder on MNIST"),
        "arithmetic_intensity": dict(value="compute_bound", flops_per_byte=80,
                                     classification_rule="dense matmuls dominate"),
        "dispatch": dict(value="unmeasured"),
    },
    "wake-vision-vww": {
        "working_set": dict(value="cache_resident", peak_bytes_per_step=512 * 1024,
                            note="8.5K-param micro-CNN"),
        "arithmetic_intensity": dict(value="compute_bound", flops_per_byte=80,
                                     classification_rule="small but compute-dense"),
        "dispatch": dict(value="unmeasured", note="Tiny model may push dispatch overhead per step; verify"),
    },
    # === Agent ===
    "nano-rag-agent": {
        "working_set": dict(value="dram_bound", peak_bytes_per_step=84 * 1024 * 1024,
                            note="Generation phase: 20M-param decode (~80 MB activations + weights re-read) + retrieval index lookup."),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=0.7,
                                     classification_rule="generation is decode (bw-bound); retrieval is index gather"),
        "dispatch": dict(value="dispatch_bound", utilization=0.10,
                         observation_source="composite workload: decode + retrieval, both small-kernel"),
    },
    "nano-codegen-agent": {
        "working_set": dict(value="dram_bound", peak_bytes_per_step=58 * 1024 * 1024,
                            note="13.7M-param iterative regeneration; per-step working set ~55 MB exceeds 4*LLC."),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=0.7,
                                     classification_rule="iterative decode loop"),
        "dispatch": dict(value="dispatch_bound", utilization=0.10),
    },
    "nano-react-agent": {
        "working_set": dict(value="dram_bound", peak_bytes_per_step=58 * 1024 * 1024,
                            note="13.7M-param multi-step reasoning; iter-2 found this still uses non-KV-cache forward — current 58 MB number assumes the (broken) recompute path; will shrink once iter-7 lands KV-cache."),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=0.7,
                                     classification_rule="ReAct decode loop"),
        "dispatch": dict(value="dispatch_bound", utilization=0.10,
                         note="To be patched in iter-7 to use KV-cache path; regime values may shift"),
    },
    "nano-toolcall-agent": {
        "working_set": dict(value="dram_bound", peak_bytes_per_step=58 * 1024 * 1024,
                            note="Bonus workload (not in core 16); structured output generation"),
        "arithmetic_intensity": dict(value="bandwidth_bound", flops_per_byte=0.7,
                                     classification_rule="tool-call decode"),
        "dispatch": dict(value="dispatch_bound", utilization=0.10),
    },
}


def main() -> int:
    doc = yaml.safe_load(WORKLOADS_YAML.read_text())
    suites = doc.get("suites", {})

    n_added = 0
    n_replaced = 0
    n_missing = 0
    for div, workloads in suites.items():
        for name, body in workloads.items():
            if name not in PLACEMENTS:
                print(f"  SKIP (no placement): {div}/{name}")
                n_missing += 1
                continue
            had_regime = "regime" in body
            body["regime"] = PLACEMENTS[name]
            # Strip the now-redundant flat fields from iters 2 and 3.
            for old_field in ("working_set_regime", "compute_regime",
                               "regime_note", "smoke_test_result"):
                body.pop(old_field, None)
            if had_regime:
                n_replaced += 1
            else:
                n_added += 1

    out = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=120)
    WORKLOADS_YAML.write_text(out)
    print(f"\nMigrated workloads.yaml: +{n_added} regime blocks, "
          f"replaced {n_replaced} pre-iter-4 entries, {n_missing} skipped.")
    return 0 if n_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
