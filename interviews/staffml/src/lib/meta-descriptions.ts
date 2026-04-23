/**
 * Human-readable definitions for the jargon a new StaffML user encounters.
 *
 * "L4 Analyze" means nothing to a first-time visitor. Neither does "tinyml"
 * or "operations" or "data". This module is the single source of truth for
 * tooltip text on every level / track / competency / zone badge in the app.
 *
 * Pure data, zero deps. Edit here to update tooltips everywhere.
 */

import { getLevelDef } from "./levels";

// ─── Level descriptions ─────────────────────────────────────
// Levels already have name + role + verb + example in lib/levels.ts.
// This helper composes them into a single tooltip string.
export function levelTooltip(id: string): { title: string; body: string } {
  const def = getLevelDef(id);
  return {
    title: `${def.id} — ${def.name} (${def.role})`,
    body: `${def.verb}\nExample: ${def.example}`,
  };
}

// ─── Track descriptions ─────────────────────────────────────
// Tracks are the deployment context. They aren't in a typed lib because
// they're just strings in the corpus, so this is the only place they're
// defined for the UI.
const TRACK_DESCRIPTIONS: Record<string, { title: string; body: string }> = {
  cloud: {
    title: "Cloud track",
    body: "Datacenter ML systems: 8-GPU servers, multi-node training, racks of accelerators, hundreds of GB of HBM. Latency budgets in milliseconds, throughput in thousands of QPS.",
  },
  edge: {
    title: "Edge track",
    body: "On-prem and industrial ML: single-box deployments, embedded GPUs, intermittent connectivity. Power and thermal constraints matter as much as throughput.",
  },
  mobile: {
    title: "Mobile track",
    body: "On-device ML on phones and tablets: NPUs, NEON/AMX, model compression, battery life. Models must be small, latency must hide behind 60fps frames.",
  },
  tinyml: {
    title: "TinyML track",
    body: "Microcontroller ML: kilobytes of RAM, milliwatt power budgets, no operating system. Quantization and pruning aren't optimizations — they're requirements.",
  },
  global: {
    title: "Cross-track",
    body: "Concepts that apply across deployment targets — fundamentals you'll see whether you serve at scale or on a microcontroller.",
  },
};

export function trackTooltip(track: string): { title: string; body: string } {
  return (
    TRACK_DESCRIPTIONS[track] ?? {
      title: track,
      body: "Deployment track for this question.",
    }
  );
}

// ─── Competency area descriptions ───────────────────────────
// The 11 ikigai zones from the StaffML taxonomy. These are the high-level
// "what does a senior ML systems engineer need to know" categories.
const COMPETENCY_DESCRIPTIONS: Record<string, { title: string; body: string }> = {
  compute: {
    title: "Compute",
    body: "Accelerator architecture, FLOPs, arithmetic intensity, kernel design — the silicon side of inference and training.",
  },
  memory: {
    title: "Memory",
    body: "HBM, KV cache, model state, gradient checkpointing, paging — anything where size or bandwidth bites.",
  },
  fluency: {
    title: "Fluency",
    body: "The vocabulary and mental model of ML systems. Knowing what an NPU is, what FlashAttention does, what 'prefill vs decode' means.",
  },
  architecture: {
    title: "Architecture",
    body: "How model designs (attention, MoE, retrieval, hybrid) map onto real hardware constraints.",
  },
  latency: {
    title: "Latency",
    body: "TTFT, TPOT, p50/p99 budgets, scheduling, request shaping. The user-facing time domain.",
  },
  "cross-cutting": {
    title: "Cross-cutting",
    body: "System-level skills that span every layer: profiling, debugging, capacity planning, incident response.",
  },
  data: {
    title: "Data",
    body: "Data engineering, validation, versioning, drift detection — the part of ML systems that isn't model code.",
  },
  networking: {
    title: "Networking",
    body: "NVLink, InfiniBand, AllReduce, network bottlenecks, multi-node communication patterns.",
  },
  power: {
    title: "Power",
    body: "Energy per inference, thermal budgets, datacenter PUE, sustainability — the constraint that ends up dominating at scale.",
  },
  optimization: {
    title: "Optimization",
    body: "Quantization, sparsity, distillation, pruning, compilation — making models faster, smaller, cheaper without breaking them.",
  },
  precision: {
    title: "Precision",
    body: "FP32 / FP16 / BF16 / FP8 / INT8 numerics, accumulator widths, calibration. Where bits meet accuracy.",
  },
  reliability: {
    title: "Reliability",
    body: "Failure modes, fault tolerance, checkpointing, replay, redundancy. The 'what happens when something breaks' axis.",
  },
  parallelism: {
    title: "Parallelism",
    body: "Data, tensor, pipeline, expert parallelism. How big models actually run on multiple devices.",
  },
};

export function competencyTooltip(area: string): { title: string; body: string } {
  const key = area.toLowerCase();
  return (
    COMPETENCY_DESCRIPTIONS[key] ?? {
      title: area,
      body: "Competency area in the StaffML taxonomy.",
    }
  );
}
