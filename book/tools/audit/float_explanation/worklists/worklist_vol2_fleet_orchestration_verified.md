# Verified findings — fleet_orchestration.qmd (vol2)
Prior findings: 1 | Survived: 0 | Refuted: 1

## SURVIVING findings

*(none)*

## REFUTED findings

- `tbl-fleet-orchestration-slurm-partitions` — REFUTED: explanation in caption (L614) and next ¶ (L618).

  The first pass correctly noted that the ref sentence at L616 misrepresents the table ("GPU allocation strategies significantly impact utilization" is not what the table shows). However, the refutation bar is whether any neighborhood element tells the reader what the float shows and why it matters — not whether the ref sentence alone is adequate. The caption at L614 is the richest neighborhood element: it names the organizing logic (logical pools matched to workload characteristics), states the NVLink/PCIe distinction and its consequence for parallelism strategy (NVLink supports tensor parallelism, PCIe serves data-parallel workloads), and explains the inference/debug separation. The setup sentence at L605 primes the reader by framing the table as a configuration example organized by accelerator type and interconnect. The paragraph at L618 provides quantitative follow-through: `--gpus=16 --gpus-per-node=8` guarantees two complete NVLink nodes; the same request without the per-node constraint may spread GPUs across three or four partial nodes, degrading intra-node communication. That paragraph makes the NVLink partition row's significance concrete with numbers. The ref sentence is a mismatch with the table's content, but the caption and next paragraph together give the reader both what the table shows and why partition design determines parallelism strategy availability before any scheduling decision is made. The neighborhood clears the refutation bar.
