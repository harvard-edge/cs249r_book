import {
  FORMULAS, ModelConfig, HardwareSpec, InterconnectSpec
} from "./hardware";

export interface SimConfig {
  model: ModelConfig;
  hardware: HardwareSpec;
  numGpus: number;
  gpusPerNode: number;
  intraConnect: InterconnectSpec; // within node (NVLink)
  interConnect: InterconnectSpec; // between nodes (InfiniBand)
  batchSize: number;
  seqLen: number;
  precision: number; // bytes per param: 2=FP16, 4=FP32
  mfu: number; // model FLOP utilization 0-1
  tpDegree: number; // tensor parallelism
  ppDegree: number; // pipeline parallelism
}

export interface SimResult {
  // Memory
  modelMemoryGb: number;
  memoryPerGpu: number; // total: params + optimizer + activations
  fitsInMemory: boolean;
  tpCrossesNodes: boolean;
  // Compute
  flopsPerIter: number;
  computeTimeMs: number;
  // Communication
  dpDegree: number;
  gradientSizeGb: number;
  allreduceTimeMs: number;
  commOverheadPct: number;
  // Pipeline
  pipelineBubblePct: number;
  // Throughput
  iterTimeMs: number;
  tokensPerSec: number;
  mfuEffective: number;
  // Reliability
  clusterMtbfHours: number;
  checkpointSizeGb: number;
  checkpointIntervalMin: number;
  // Time to train
  trainingTimeDays: number;
}

/**
 * Distributed training simulation based on MLSysIM principles.
 * Canonical equations for compute, communication, and memory tradeoffs.
 */
export function simulate(config: SimConfig): SimResult {
  const { model, hardware, numGpus, gpusPerNode, batchSize, seqLen, precision, mfu, tpDegree, ppDegree, intraConnect, interConnect } = config;

  // Constraint validation
  const dpDegree = Math.max(1, Math.floor(numGpus / (tpDegree * ppDegree)));
  const numNodes = Math.ceil(numGpus / gpusPerNode);
  const tpCrossesNodes = tpDegree > gpusPerNode;

  // Memory: model params + optimizer states (14 bytes/param for mixed-precision Adam)
  // + activation memory (approximate: selective recompute formula 10 * L * B * S * H)
  const modelMemoryGb = FORMULAS.model_memory_gb(model.params_b, precision);
  const optimizerMemoryGb = FORMULAS.model_memory_gb(model.params_b, 12); // 12 bytes for Adam states (fp32 master + momentum + variance)
  const activationMemoryGb = (10 * model.layers * (batchSize / dpDegree) * seqLen * model.hidden_dim) / (tpDegree * 1e9);
  const totalMemoryPerGpu = (modelMemoryGb + optimizerMemoryGb) / (tpDegree * ppDegree) + activationMemoryGb;
  const memoryPerGpu = totalMemoryPerGpu;
  const fitsInMemory = memoryPerGpu < hardware.memory_gb * 0.8; // 80% usable

  // Compute (forward + backward)
  const tokensPerIter = batchSize * seqLen;
  const flopsPerIter = model.flops_per_token * tokensPerIter; 
  const computeTimeMs = FORMULAS.compute_time_ms(flopsPerIter, hardware.compute_tflops, numGpus, mfu);

  // Communication — AllReduce for data parallelism
  const gradientSizeGb = FORMULAS.model_memory_gb(model.params_b, precision) / tpDegree;
  // Hierarchical AllReduce: bottleneck is inter-node link, but intra-node reduce
  // shrinks the message by gpusPerNode before crossing the network
  const effectiveBandwidth = numNodes > 1
    ? interConnect.bandwidth_gbs * Math.min(gpusPerNode, tpDegree > 1 ? 1 : gpusPerNode)
    : intraConnect.bandwidth_gbs;
  // If TP crosses nodes, add TP communication overhead via inter-node link
  const tpCommPenalty = tpCrossesNodes ? 1.5 : 1.0;
  const allreduceTimeMs = dpDegree > 1
    ? FORMULAS.allreduce_time_ms(gradientSizeGb, effectiveBandwidth, dpDegree) * tpCommPenalty
    : 0;

  // Pipeline bubble — microbatches = global_batch / (DP * micro_batch_size)
  // Approximate micro_batch_size as batch / (DP * PP) to fill pipeline
  const microBatchSize = Math.max(1, Math.floor(batchSize / (dpDegree * ppDegree)));
  const numMicrobatches = Math.max(1, Math.floor(batchSize / (dpDegree * microBatchSize)));
  const pipelineBubblePct = ppDegree > 1
    ? FORMULAS.pipeline_bubble_pct(ppDegree, numMicrobatches)
    : 0;

  // Total iteration time (with partial compute-comm overlap: ~30% overlap for well-optimized systems)
  const overlapFactor = 0.7; // 30% of AllReduce overlaps with backward compute
  const bubbleOverhead = computeTimeMs * (pipelineBubblePct / 100);
  const iterTimeMs = computeTimeMs + allreduceTimeMs * overlapFactor + bubbleOverhead;
  const commOverheadPct = iterTimeMs > 0 ? (allreduceTimeMs / iterTimeMs) * 100 : 0;

  // Throughput
  const tokensPerSec = tokensPerIter / (iterTimeMs / 1000);
  const peakFlops = hardware.compute_tflops * 1e12 * numGpus;
  const actualFlops = flopsPerIter / (iterTimeMs / 1000);
  const mfuEffective = (actualFlops / peakFlops) * 100;

  // Reliability
  const gpuMtbfHours = 10000; // typical GPU MTBF
  const clusterMtbfHours = FORMULAS.cluster_mtbf_hours(numGpus, gpuMtbfHours);
  const checkpointSizeGb = FORMULAS.checkpoint_size_gb(model.params_b);
  const checkpointTimeMins = (checkpointSizeGb / (effectiveBandwidth * 0.5)) / 60; // assume 50% bandwidth
  const checkpointIntervalMin = FORMULAS.young_daly_checkpoint_interval_min(
    Math.max(checkpointTimeMins, 0.5), clusterMtbfHours * 60
  );

  // Training time (assume 1T tokens)
  const totalFlops = FORMULAS.training_flops(model.params_b, 1);
  const trainingTimeDays = totalFlops / (actualFlops * 86400);

  return {
    modelMemoryGb, memoryPerGpu, fitsInMemory, tpCrossesNodes,
    flopsPerIter, computeTimeMs,
    dpDegree, gradientSizeGb, allreduceTimeMs, commOverheadPct,
    pipelineBubblePct,
    iterTimeMs, tokensPerSec, mfuEffective,
    clusterMtbfHours, checkpointSizeGb, checkpointIntervalMin,
    trainingTimeDays,
  };
}
