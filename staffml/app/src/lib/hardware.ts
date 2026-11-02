// Hardware, interconnect, latency, and model numbers for the reference card,
// napkin calculator, roofline, and simulator. Every value comes from the
// MLSysIM registries through src/data/mlsysim.json, which
// scripts/export_mlsysim_data.py regenerates before every CI build. Do not
// type numbers here; change MLSysIM and rerun the export.
import mlsysim from '../data/mlsysim.json';

export interface HardwareSpec {
  name: string;
  tier: 'cloud' | 'edge' | 'mobile' | 'tinyml';
  compute_tflops: number;
  compute_unit: string; // "FP16" or "INT8" etc
  bandwidth_tbs: number; // TB/s
  memory_gb: number;
  memory_type: string;
  tdp_w: number;
  year: number;
  mlsysim: string; // registry entry this row was exported from
}

export const HARDWARE_SPECS = mlsysim.hardware as HardwareSpec[];

export interface InterconnectSpec {
  name: string;
  bandwidth_gbs: number; // GB/s, as MLSysIM records the link
  latency_us: number;
  mlsysim: string;
}

export const INTERCONNECTS: InterconnectSpec[] = mlsysim.interconnects;

export interface LatencyRef {
  operation: string;
  latency_ns: number;
  human_scale: string;
}

/** Scale a latency so 1 ns reads as 1 second, the classic way to feel the gaps. */
export function humanScale(latencyNs: number): string {
  const seconds = latencyNs;
  const units: [string, number][] = [['year', 365 * 86400], ['day', 86400], ['hour', 3600], ['minute', 60], ['second', 1]];
  const [unit, size] = units.find(([, s]) => seconds >= s) ?? units[units.length - 1];
  const value = Number((seconds / size).toPrecision(2));
  return `${value} ${unit}${value === 1 ? '' : 's'}`;
}

// MLSysIM has the speed of light in fiber but no coast-to-coast route length,
// so this one row is not from the registry. Add a route distance to MLSysIM
// to bring it in.
const CROSS_US_FIBER_NS = 40_000_000;

export const LATENCY_HIERARCHY: LatencyRef[] = [
  ...mlsysim.latency.map(({ operation, latency_ns }) => ({ operation, latency_ns })),
  { operation: 'Cross-US Fiber', latency_ns: CROSS_US_FIBER_NS },
].map(l => ({ ...l, human_scale: humanScale(l.latency_ns) }));

// Key formulas for napkin math
export const FORMULAS = {
  ridge_point: (compute_tflops: number, bandwidth_tbs: number) =>
    compute_tflops / bandwidth_tbs, // Ops/Byte

  model_memory_gb: (params_b: number, bytes_per_param: number) =>
    (params_b * 1e9 * bytes_per_param) / 1e9,

  training_flops: (params_b: number, tokens_b: number) =>
    6 * params_b * 1e9 * tokens_b * 1e9, // 6PD rule

  training_time_days: (flops: number, gpu_tflops: number, num_gpus: number, mfu: number) =>
    flops / (gpu_tflops * 1e12 * num_gpus * mfu * 86400),

  kv_cache_mb: (layers: number, heads: number, head_dim: number, seq_len: number, batch: number, bytes: number) =>
    (2 * layers * heads * head_dim * seq_len * batch * bytes) / 1e6,

  allreduce_time_ms: (message_gb: number, bandwidth_gbs: number, num_gpus: number) =>
    (2 * (num_gpus - 1) / num_gpus) * (message_gb / bandwidth_gbs) * 1000,

  // Distributed training simulation
  pipeline_bubble_pct: (num_stages: number, num_microbatches: number) =>
    (num_stages - 1) / (num_microbatches + num_stages - 1) * 100,

  checkpoint_size_gb: (params_b: number) =>
    params_b * 1e9 * 14 / 1e9, // 14 bytes/param for mixed-precision Adam (fp16 param + fp32 master + fp32 momentum + fp32 variance)

  compute_time_ms: (flops_per_iter: number, gpu_tflops: number, num_gpus: number, mfu: number) =>
    flops_per_iter / (gpu_tflops * 1e12 * num_gpus * mfu) * 1000,

  cluster_mtbf_hours: (num_gpus: number, gpu_mtbf_hours: number) =>
    gpu_mtbf_hours / num_gpus,

  young_daly_checkpoint_interval_min: (checkpoint_time_min: number, mtbf_min: number) =>
    Math.sqrt(2 * checkpoint_time_min * mtbf_min),
};

// Well-known model configs for the simulator
export interface ModelConfig {
  name: string;
  params_b: number;
  layers: number;
  hidden_dim: number;
  heads: number;
  flops_per_token: number; // training FLOPs per token, 6 x parameters (forward + backward)
  mlsysim: string;
}

export const MODEL_CONFIGS: ModelConfig[] = mlsysim.models;
