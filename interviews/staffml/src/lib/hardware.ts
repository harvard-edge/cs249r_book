// Hardware constants from mlsysim/core/constants.py — single source of truth
// Used as a reference card during drill/gauntlet napkin math

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
}

export const HARDWARE_SPECS: HardwareSpec[] = [
  // Cloud GPUs
  { name: 'NVIDIA H100 SXM', tier: 'cloud', compute_tflops: 989, compute_unit: 'FP16', bandwidth_tbs: 3.35, memory_gb: 80, memory_type: 'HBM3', tdp_w: 700, year: 2022 },
  { name: 'NVIDIA H200', tier: 'cloud', compute_tflops: 989, compute_unit: 'FP16', bandwidth_tbs: 4.8, memory_gb: 141, memory_type: 'HBM3e', tdp_w: 700, year: 2023 },
  { name: 'NVIDIA B200', tier: 'cloud', compute_tflops: 2250, compute_unit: 'FP16', bandwidth_tbs: 8.0, memory_gb: 192, memory_type: 'HBM3e', tdp_w: 1000, year: 2024 },
  { name: 'NVIDIA A100 80GB', tier: 'cloud', compute_tflops: 312, compute_unit: 'FP16', bandwidth_tbs: 2.039, memory_gb: 80, memory_type: 'HBM2e', tdp_w: 400, year: 2020 },
  { name: 'AMD MI300X', tier: 'cloud', compute_tflops: 1307, compute_unit: 'FP16', bandwidth_tbs: 5.3, memory_gb: 192, memory_type: 'HBM3', tdp_w: 750, year: 2023 },
  { name: 'Google TPU v5p', tier: 'cloud', compute_tflops: 459, compute_unit: 'BF16', bandwidth_tbs: 2.76, memory_gb: 95, memory_type: 'HBM', tdp_w: 300, year: 2024 },
  // Edge
  { name: 'Jetson Orin NX', tier: 'edge', compute_tflops: 25, compute_unit: 'INT8', bandwidth_tbs: 0.102, memory_gb: 16, memory_type: 'LPDDR5', tdp_w: 25, year: 2022 },
  { name: 'Google Coral Edge TPU', tier: 'edge', compute_tflops: 4, compute_unit: 'INT8', bandwidth_tbs: 0.008, memory_gb: 1, memory_type: 'DRAM', tdp_w: 2, year: 2019 },
  // Mobile
  { name: 'Apple A17 Pro', tier: 'mobile', compute_tflops: 35, compute_unit: 'FP16', bandwidth_tbs: 0.1, memory_gb: 8, memory_type: 'LPDDR5', tdp_w: 5, year: 2023 },
  { name: 'Snapdragon 8 Gen 3', tier: 'mobile', compute_tflops: 45, compute_unit: 'INT8', bandwidth_tbs: 0.077, memory_gb: 12, memory_type: 'LPDDR5X', tdp_w: 5, year: 2023 },
  // TinyML
  { name: 'ESP32-S3', tier: 'tinyml', compute_tflops: 0.0005, compute_unit: 'INT8', bandwidth_tbs: 0.0002, memory_gb: 0.000512, memory_type: 'SRAM', tdp_w: 1.2, year: 2021 },
];

export interface InterconnectSpec {
  name: string;
  bandwidth_gbs: number; // GB/s (unidirectional)
  latency_us: number;
}

export const INTERCONNECTS: InterconnectSpec[] = [
  { name: 'NVLink H100', bandwidth_gbs: 900, latency_us: 0.5 },
  { name: 'NVLink B200', bandwidth_gbs: 1800, latency_us: 0.3 },
  { name: 'PCIe Gen5 x16', bandwidth_gbs: 64, latency_us: 1.0 },
  { name: 'InfiniBand NDR', bandwidth_gbs: 50, latency_us: 5.0 },
  { name: 'InfiniBand XDR', bandwidth_gbs: 100, latency_us: 3.0 },
  { name: 'Ethernet 100GbE', bandwidth_gbs: 12.5, latency_us: 10.0 },
];

export interface LatencyRef {
  operation: string;
  latency_ns: number;
  human_scale: string;
}

export const LATENCY_HIERARCHY: LatencyRef[] = [
  { operation: 'L1 Cache / Register', latency_ns: 1, human_scale: '1 second' },
  { operation: 'L2 Cache', latency_ns: 4, human_scale: '4 seconds' },
  { operation: 'HBM3 Access', latency_ns: 300, human_scale: '5 minutes' },
  { operation: 'NVLink Transfer', latency_ns: 500, human_scale: '8 minutes' },
  { operation: 'PCIe Gen5', latency_ns: 1000, human_scale: '16 minutes' },
  { operation: 'InfiniBand NDR', latency_ns: 5000, human_scale: '1.4 hours' },
  { operation: 'NVMe SSD Read', latency_ns: 100000, human_scale: '1.1 days' },
  { operation: 'Cross-US Fiber', latency_ns: 40000000, human_scale: '1.2 years' },
];

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
};
