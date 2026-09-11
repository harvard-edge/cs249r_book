#!/usr/bin/env python3
"""Apply conservative mechanical hygiene fixes to question YAML files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_DIR = VAULT_DIR / "questions"

CODE_SPAN_RE = re.compile(r"`([^`]+)`")
MATH_SPAN_RE = re.compile(r"\$([^$]+)\$")
HTML_TAG_RE = re.compile(r"<([^>]+)>")

QUESTION_REWRITES = {
    "cloud-2864": "What is the maximum concurrent request count for 8192-token requests with 80 layers, 8 KV heads, and head_dim 128?",
    "cloud-3808": "How does block-sparse GEMM remove capacity padding, and for batch 512, 8 experts, and capacity factor 1.5, what padding remains?",
    "cloud-1121": "Under strict Demographic Parity with the same total daily accepts, how many Group Y candidates are accepted and how many are historically unqualified?",
    "cloud-1607": "Which mitigation best prevents memorization under the compute budget: deduplication, DP-SGD, or RLHF refusal guardrails?",
    "cloud-1914": "What causes high CPU I/O wait and low page-cache hits when PyTorch shuffles individual files, and how would you fix it?",
    "cloud-3544": "How would you compare centralized and distributed edge labeling for 10M images in 3 months while preserving IAA and limiting bias?",
    "cloud-1854": "With 12ms base latency and 1ms per extra batched request, what maximum batch window supports batch size 16?",
    "cloud-4405": "How would you design disaggregated prefill/decode to meet p99 TTFT under 500ms and p99 generation under 4s including KV transfer?",
    "cloud-3285": "How do host-device data movement, DMA, zero-copy, and pinned memory affect latency and throughput in the two architectures?",
    "cloud-4508": "What is speculative decoding latency per accepted token, and below what acceptance rate does it underperform plain decoding?",
    "cloud-2734": "For 2,000 intra-rack 800G 2m links, should you choose active copper, optics, or rack redesign for 3-year TCO?",
    "cloud-4168": "How do NVLink and Infinity Fabric differ for tensor parallelism, and which is better for the given shard and AllReduce sizes?",
    "cloud-0983": "With 4 gradient buckets, perfect CUDA-stream overlap, and no scheduling overhead, what is the new backward-plus-sync time?",
    "cloud-1447": "How many A100 GPU-hours does a 0.5-hour shift-left validation step save per successfully deployed model?",
    "cloud-1241": "At 32MB/ms H2D bandwidth, what latency do 1,000 requests need in naive sync versus pipelined pinned-memory CUDA streams?",
    "cloud-4055": "How does 1GB AllReduce time on the torus compare with 256 H100s on 400Gbps Ethernet, and why does topology matter?",
    "cloud-1207": "Why is 60s nvidia-smi power insufficient for INT8 versus FP16 rollout, and what energy metric should guide deployment?",
    "cloud-2770": "How would you compare classifier filtering, sandboxed tools with human review, and instruction hierarchy for prompt injection defense?",
    "edge-1946": "How would you design hardware-aware NAS for Jetson Orin using SRAM, LPDDR5, INT8 FLOPs, and measured latency constraints?",
    "edge-2295": "How would you design a paged KV manager for Orin and compute concurrent 1024-token sessions in 4GB of KV memory?",
    "edge-1839": "How would you compare Cloud AI 100, Jetson Orin, and Intel Atom for smart-camera efficiency, programmability, and TCO?",
    "edge-1843": "For a perfectly efficient weight-stationary systolic array, what is the theoretical minimum layer time in milliseconds?",
    "edge-2370": "What TOPS and LPDDR5 bandwidth are required, is the system compute- or bandwidth-bound, and which Orin power mode suffices?",
    "edge-2363": "Using M/D/1, what is mean frame response time, how does it compare with M/M/1, and what latency is saved?",
    "edge-0974": "Where should weights be placed to minimize latency, and what is the flash-access time penalty per inference?",
    "edge-1726": "When Edge TPU activations exceed SRAM, how would you choose among downsampling, CPU/TPU splitting, and receptive-field changes?",
    "edge-2409": "Does camera DMA, model weight reads, or activation traffic starve the others under aggregate LPDDR5 utilization?",
    "edge-1993": "How would you route mixed critical and batch requests across Hailo-8 accelerators under failures or thermal throttling?",
    "edge-2284": "What is total broadcast time including bytes, DMA setup, and PCIe arbitration, and how much can pipelining save?",
    "edge-1585": "What BTU/hr thermal dissipation is required for eight 75W Cloud AI 100 cards to avoid throttling?",
    "edge-1376": "With 8 GB/s bandwidth and 32M parameters, what is the maximum inference rate for packed 4-bit weights?",
    "edge-2432": "During the 5-second shutdown window, which subsystems must checkpoint and in what canonical order?",
    "mobile-1605": "How do the two attention mechanisms trade off memory bandwidth, compute efficiency, and latency on this hardware?",
    "mobile-1614": "How would hardware-aware NAS use TOPS, memory limits, and MCUNet-style constraints to find a mobile architecture?",
    "mobile-2028": "What GFLOPs/s are required for nominal INT8 and mixed-precision fallback, and does fallback still hit 120 FPS?",
    "mobile-2157": "Why does depthwise convolution lose L1 reuse and fall into the memory-bound region of the Roofline model?",
    "mobile-1377": "How would you combine per-example clipping, secure noise, and INT8 quantization on a Hexagon NPU without weakening DP?",
    "mobile-1751": "What TOPS, power, and memory budget support on-device demographic-parity monitoring without hurting user experience?",
    "mobile-0903": "How would you design data curation, synchronization, and privacy-preserving collection for the target device?",
    "mobile-1903": "From the 30.3 FPS baseline, what steady-state FPS does double-buffering unlock, and which stage remains binding?",
    "mobile-1982": "From the 300ms naive estimate, what is the realistic worst-case queue drain time after cold-cache and arrival effects?",
    "mobile-1917": "How would you pipeline UNet activations to avoid system-RAM spills when tensors peak at 120MB and NPU SRAM is 32MB?",
    "mobile-1881": "What stage rate binds the Cloud-to-NPU load pipeline under a 64MB ring, and is the 3-minute SLA feasible?",
    "mobile-1932": "What memory-savings factor comes from FP16-to-INT4 weights after accounting for the static KV cache?",
    "mobile-2031": "How do FP16 and INT8 per-token decode latency compare on a contended 50GB/s LPDDR5 bus?",
    "mobile-1891": "Can checkpointing fit in the 1.5s grace window after serialization, UFS contention, and CPU/I/O overlap?",
    "tinyml-1389": "How would you estimate parameter count, INT8 memory footprint, and Cortex-M7 plus Ethos-U55 inference latency?",
    "tinyml-1634": "Does a 5- or 15-minute checkpoint interval minimize 1-hour expected energy, and by what factor?",
    "tinyml-1661": "Can the 2mF capacitor finish a 16KB checkpoint before brownout, and what capacitance would succeed?",
    "cloud-1249": "Using Young's formula, what are the optimal checkpoint interval and daily checkpointing overhead?",
    "edge-1119": "For a decode queue with lambda=8/s and mu=9.7/s, what is the correct wait time and why does caching matter?",
    "edge-2280": "What maximum processor utilization rho keeps average wait time at or below 40ms?",
    "edge-1549": "Which strategy yields higher sustained 24-hour throughput in a 40C sealed enclosure, and why?",
    "global-0085": "Why are GPUDirect RDMA latency and bandwidth limited even when the GPU and NIC share a PCIe switch?",
    "mobile-0136": "Which standard mobile SoC processor type is most energy-efficient for these operations?",
}


def clean_question_text(text: str) -> str:
    cleaned = CODE_SPAN_RE.sub(r"\1", text)
    cleaned = MATH_SPAN_RE.sub(r"\1", cleaned)
    cleaned = HTML_TAG_RE.sub(r"\1", cleaned)
    cleaned = cleaned.replace("\\times", "x")
    cleaned = cleaned.replace("\\cdot", "x")
    cleaned = cleaned.replace("\\approx", "approximately")
    cleaned = cleaned.replace("\\lambda", "lambda")
    cleaned = cleaned.replace("\\sigma", "sigma")
    cleaned = cleaned.replace("\\alpha", "alpha")
    cleaned = cleaned.replace("\\pm", "+/-")
    cleaned = cleaned.replace("\\ge", ">=")
    cleaned = cleaned.replace("\\le", "<=")
    cleaned = cleaned.replace("\\_", "_")
    cleaned = " ".join(cleaned.split())
    return cleaned


def has_ordered_markers(text: str, markers: tuple[str, ...]) -> bool:
    cursor = 0
    for marker in markers:
        idx = text.find(marker, cursor)
        if idx < 0:
            return False
        cursor = idx + len(marker)
    return True


def normalize_common_mistake(text: str) -> str:
    stripped = text.strip()
    if not stripped or has_ordered_markers(
        stripped,
        ("**The Pitfall:**", "**The Rationale:**", "**The Consequence:**"),
    ):
        return text
    return (
        f"**The Pitfall:** {stripped}\n"
        "**The Rationale:** This mistake focuses on the surface symptom instead of the governing ML-systems constraint.\n"
        "**The Consequence:** The resulting design, estimate, or diagnosis can miss the real bottleneck and lead to incorrect deployment decisions.\n"
    )


def normalize_napkin_math(text: str) -> str:
    stripped = text.strip()
    if not stripped or has_ordered_markers(
        stripped,
        ("**Assumptions", "**Calculations:**", "**Conclusion"),
    ):
        return text
    return (
        "**Assumptions & Constraints:**\n"
        "- Use the quantities and constraints stated in the scenario.\n\n"
        "**Calculations:**\n"
        f"{stripped}\n\n"
        "**Conclusion & Interpretation:**\n"
        "- The calculation identifies the limiting systems constraint for this question.\n"
    )


def fix_record(data: dict[str, Any]) -> bool:
    changed = False

    if data.get("status") != "deleted" and "deletion_reason" in data:
        data.pop("deletion_reason", None)
        changed = True

    question = data.get("question")
    if isinstance(question, str):
        qid = data.get("id")
        cleaned = QUESTION_REWRITES.get(qid, clean_question_text(question))
        if data.get("status") == "published" and "?" not in cleaned:
            cleaned = cleaned.rstrip(".:;! ") + "?"
        if cleaned != question:
            data["question"] = cleaned
            changed = True

    details = data.get("details")
    if isinstance(details, dict):
        common_mistake = details.get("common_mistake")
        if isinstance(common_mistake, str):
            normalized = normalize_common_mistake(common_mistake)
            if normalized != common_mistake:
                details["common_mistake"] = normalized
                changed = True
        napkin_math = details.get("napkin_math")
        if isinstance(napkin_math, str):
            normalized = normalize_napkin_math(napkin_math)
            if normalized != napkin_math:
                details["napkin_math"] = normalized
                changed = True

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    args = parser.parse_args()

    changed_paths: list[Path] = []
    for path in sorted(args.questions_dir.glob("*/*/*.yaml")):
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            continue
        if fix_record(data):
            path.write_text(
                yaml.safe_dump(
                    data,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                    width=1000,
                )
            )
            changed_paths.append(path)

    print(f"Applied hygiene fixes to {len(changed_paths)} file(s).")
    for path in changed_paths[:50]:
        print(f"- {path.relative_to(VAULT_DIR.parents[1])}")
    if len(changed_paths) > 50:
        print(f"... and {len(changed_paths) - 50} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
