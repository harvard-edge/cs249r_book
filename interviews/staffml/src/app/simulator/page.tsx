"use client";

import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { Server, Terminal, AlertTriangle, Zap, Clock, Cpu } from "lucide-react";
import clsx from "clsx";
import {
  HARDWARE_SPECS, INTERCONNECTS, FORMULAS, MODEL_CONFIGS,
  HardwareSpec, InterconnectSpec, ModelConfig
} from "@/lib/hardware";

interface SimConfig {
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

interface SimResult {
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

function simulate(config: SimConfig): SimResult {
  const { model, hardware, numGpus, gpusPerNode, batchSize, seqLen, precision, mfu, tpDegree, ppDegree, intraConnect, interConnect } = config;

  // Constraint validation
  const dpDegree = Math.max(1, Math.floor(numGpus / (tpDegree * ppDegree)));
  const numNodes = Math.ceil(numGpus / gpusPerNode);
  const tpCrossesNodes = tpDegree > gpusPerNode;

  // Memory: model params + optimizer states (14 bytes/param for mixed-precision Adam)
  // + activation memory (approximate: 2 * layers * hidden^2 * batch * seq * bytes / TP)
  const modelMemoryGb = FORMULAS.model_memory_gb(model.params_b, precision);
  const optimizerMemoryGb = FORMULAS.model_memory_gb(model.params_b, 12); // 12 bytes for Adam states (fp32 master + momentum + variance)
  const activationMemoryGb = (2 * model.layers * model.hidden_dim * model.hidden_dim * (batchSize / dpDegree) * precision) / (tpDegree * 1e9);
  const totalMemoryPerGpu = (modelMemoryGb + optimizerMemoryGb) / (tpDegree * ppDegree) + activationMemoryGb;
  const memoryPerGpu = totalMemoryPerGpu;
  const fitsInMemory = memoryPerGpu < hardware.memory_gb * 0.8; // 80% usable

  // Compute (forward + backward = 3x forward)
  const tokensPerIter = batchSize * seqLen;
  const flopsPerIter = model.flops_per_token * tokensPerIter * 3; // 3x for fwd+bwd
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

export default function SimulatorPage() {
  const [mounted, setMounted] = useState(false);
  const cloudGpus = HARDWARE_SPECS.filter(h => h.tier === 'cloud');

  const [config, setConfig] = useState<SimConfig>({
    model: MODEL_CONFIGS[2], // Llama-70B
    hardware: cloudGpus.find(h => h.name.includes('H100')) || cloudGpus[0],
    numGpus: 64,
    gpusPerNode: 8,
    intraConnect: INTERCONNECTS.find(i => i.name.includes('NVLink H100'))!,
    interConnect: INTERCONNECTS.find(i => i.name.includes('InfiniBand NDR'))!,
    batchSize: 1024,
    seqLen: 2048,
    precision: 2,
    mfu: 0.4,
    tpDegree: 8,
    ppDegree: 1,
  });

  useEffect(() => { setMounted(true); }, []);

  const result = useMemo(() => simulate(config), [config]);
  const numNodes = Math.ceil(config.numGpus / config.gpusPerNode);

  const update = (patch: Partial<SimConfig>) => setConfig(prev => ({ ...prev, ...patch }));

  if (!mounted) return <div className="flex-1 flex items-center justify-center"><Terminal className="w-6 h-6 text-textTertiary animate-pulse" /></div>;

  return (
    <div className="flex-1 flex flex-col px-6 py-8 overflow-y-auto">
      <div className="max-w-6xl mx-auto w-full">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <Server className="w-8 h-8 text-accentBlue" />
          <div>
            <h1 className="text-2xl font-extrabold text-white tracking-tight">Distributed Training Simulator</h1>
            <p className="text-sm text-textSecondary">Configure a cluster, see compute/communication/bubble breakdown</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Configuration */}
          <div className="lg:col-span-1 space-y-5">
            {/* Model */}
            <div>
              <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Model</label>
              <select
                value={config.model.name}
                onChange={(e) => update({ model: MODEL_CONFIGS.find(m => m.name === e.target.value)! })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-textPrimary font-mono focus:outline-none focus:border-accentBlue/50"
              >
                {MODEL_CONFIGS.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
              </select>
            </div>

            {/* Hardware */}
            <div>
              <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Accelerator</label>
              <select
                value={config.hardware.name}
                onChange={(e) => {
                  const hw = cloudGpus.find(h => h.name === e.target.value)!;
                  // Auto-select matching NVLink
                  const nvlink = INTERCONNECTS.find(i => i.name.includes('NVLink') && i.name.includes(hw.name.split(' ')[1])) || config.intraConnect;
                  update({ hardware: hw, intraConnect: nvlink });
                }}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-textPrimary font-mono focus:outline-none focus:border-accentBlue/50"
              >
                {cloudGpus.map(h => <option key={h.name} value={h.name}>{h.name}</option>)}
              </select>
            </div>

            {/* Cluster size */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[10px] font-mono text-textTertiary uppercase block mb-2">Total GPUs</label>
                <select value={config.numGpus} onChange={(e) => update({ numGpus: Number(e.target.value) })}
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50">
                  {[8, 16, 32, 64, 128, 256, 512, 1024, 2048].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] font-mono text-textTertiary uppercase block mb-2">GPUs/Node</label>
                <select value={config.gpusPerNode} onChange={(e) => update({ gpusPerNode: Number(e.target.value) })}
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50">
                  {[1, 2, 4, 8].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </div>

            {/* Parallelism */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[10px] font-mono text-textTertiary uppercase block mb-2">Tensor Parallel</label>
                <select value={config.tpDegree} onChange={(e) => update({ tpDegree: Number(e.target.value) })}
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50">
                  {[1, 2, 4, 8].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] font-mono text-textTertiary uppercase block mb-2">Pipeline Parallel</label>
                <select value={config.ppDegree} onChange={(e) => update({ ppDegree: Number(e.target.value) })}
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50">
                  {[1, 2, 4, 8, 16].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </div>

            {/* Batch + Seq */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[10px] font-mono text-textTertiary uppercase block mb-2">Global Batch</label>
                <select value={config.batchSize} onChange={(e) => update({ batchSize: Number(e.target.value) })}
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50">
                  {[64, 128, 256, 512, 1024, 2048, 4096].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] font-mono text-textTertiary uppercase block mb-2">Seq Length</label>
                <select value={config.seqLen} onChange={(e) => update({ seqLen: Number(e.target.value) })}
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50">
                  {[512, 1024, 2048, 4096, 8192, 32768].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </div>

            {/* MFU slider */}
            <div>
              <label className="text-[10px] font-mono text-textTertiary uppercase block mb-2">
                Target MFU: {(config.mfu * 100).toFixed(0)}%
              </label>
              <input
                type="range" min="10" max="70" value={config.mfu * 100}
                onChange={(e) => update({ mfu: Number(e.target.value) / 100 })}
                className="w-full accent-accentBlue"
              />
            </div>

            {/* Cluster summary */}
            <div className="p-3 rounded-lg border border-border bg-surface/50 text-[10px] font-mono text-textTertiary space-y-1">
              <div>{numNodes} nodes × {config.gpusPerNode} GPUs = {config.numGpus} total</div>
              <div>DP={result.dpDegree} × TP={config.tpDegree} × PP={config.ppDegree} = {result.dpDegree * config.tpDegree * config.ppDegree}</div>
              <div>Intra: {config.intraConnect.name} | Inter: {config.interConnect.name}</div>
            </div>
          </div>

          {/* Right: Results */}
          <div className="lg:col-span-2 space-y-4">
            {/* Warnings */}
            {!result.fitsInMemory && (
              <div className="p-3 rounded-lg border border-accentRed/30 bg-accentRed/5 flex items-center gap-2 text-sm text-accentRed">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                OOM: {result.memoryPerGpu.toFixed(1)} GB/GPU exceeds {(config.hardware.memory_gb * 0.7).toFixed(0)} GB usable
              </div>
            )}

            {/* Time breakdown bar */}
            <div className="p-5 rounded-xl border border-border bg-surface/80">
              <h3 className="text-xs font-mono text-textTertiary uppercase mb-4">Iteration Time Breakdown</h3>
              <div className="flex h-10 rounded-lg overflow-hidden mb-3">
                <div
                  className="bg-accentBlue/60 flex items-center justify-center text-[10px] font-mono text-white"
                  style={{ width: `${Math.max(5, (result.computeTimeMs / result.iterTimeMs) * 100)}%` }}
                  title={`Compute: ${result.computeTimeMs.toFixed(1)}ms`}
                >
                  Compute
                </div>
                {result.allreduceTimeMs > 0 && (
                  <div
                    className="bg-accentAmber/60 flex items-center justify-center text-[10px] font-mono text-white"
                    style={{ width: `${Math.max(5, result.commOverheadPct)}%` }}
                    title={`AllReduce: ${result.allreduceTimeMs.toFixed(1)}ms`}
                  >
                    AllReduce
                  </div>
                )}
                {result.pipelineBubblePct > 0 && (
                  <div
                    className="bg-accentRed/40 flex items-center justify-center text-[10px] font-mono text-white"
                    style={{ width: `${Math.max(5, result.pipelineBubblePct * result.computeTimeMs / result.iterTimeMs * 100)}%` }}
                    title={`Pipeline bubble: ${result.pipelineBubblePct.toFixed(1)}%`}
                  >
                    Bubble
                  </div>
                )}
              </div>
              <div className="text-xs font-mono text-textTertiary">
                Total: {result.iterTimeMs.toFixed(1)} ms/iter | Compute: {result.computeTimeMs.toFixed(1)} ms | Comm: {result.allreduceTimeMs.toFixed(1)} ms
                {result.pipelineBubblePct > 0 && ` | Bubble: ${result.pipelineBubblePct.toFixed(1)}%`}
              </div>
            </div>

            {/* Key metrics grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                {
                  label: 'Throughput',
                  value: result.tokensPerSec > 1e6
                    ? `${(result.tokensPerSec / 1e6).toFixed(1)}M`
                    : result.tokensPerSec > 1e3
                    ? `${(result.tokensPerSec / 1e3).toFixed(0)}K`
                    : result.tokensPerSec.toFixed(0),
                  unit: 'tok/s',
                  color: 'text-accentGreen',
                },
                {
                  label: 'Effective MFU',
                  value: `${result.mfuEffective.toFixed(1)}`,
                  unit: '%',
                  color: result.mfuEffective > 40 ? 'text-accentGreen' : result.mfuEffective > 20 ? 'text-accentAmber' : 'text-accentRed',
                },
                {
                  label: 'Comm Overhead',
                  value: `${result.commOverheadPct.toFixed(0)}`,
                  unit: '%',
                  color: result.commOverheadPct < 20 ? 'text-accentGreen' : result.commOverheadPct < 40 ? 'text-accentAmber' : 'text-accentRed',
                },
                {
                  label: 'Memory/GPU',
                  value: `${result.memoryPerGpu.toFixed(1)}`,
                  unit: `/ ${config.hardware.memory_gb} GB`,
                  color: result.fitsInMemory ? 'text-accentGreen' : 'text-accentRed',
                },
              ].map(m => (
                <div key={m.label} className="p-4 rounded-xl border border-border bg-surface/50 text-center">
                  <div className={clsx("text-2xl font-bold font-mono", m.color)}>{m.value}</div>
                  <div className="text-[10px] text-textTertiary">{m.unit}</div>
                  <div className="text-[10px] text-textTertiary uppercase mt-1">{m.label}</div>
                </div>
              ))}
            </div>

            {/* Reliability + Training time */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="p-4 rounded-xl border border-border bg-surface/50">
                <div className="text-[10px] font-mono text-textTertiary uppercase mb-2">Cluster MTBF</div>
                <div className={clsx("text-lg font-bold font-mono",
                  result.clusterMtbfHours > 24 ? "text-accentGreen" : result.clusterMtbfHours > 4 ? "text-accentAmber" : "text-accentRed"
                )}>
                  {result.clusterMtbfHours.toFixed(1)} hrs
                </div>
                <div className="text-[10px] text-textTertiary mt-1">
                  Checkpoint every {result.checkpointIntervalMin.toFixed(0)} min ({result.checkpointSizeGb.toFixed(0)} GB)
                </div>
              </div>
              <div className="p-4 rounded-xl border border-border bg-surface/50">
                <div className="text-[10px] font-mono text-textTertiary uppercase mb-2">Gradient AllReduce</div>
                <div className="text-lg font-bold font-mono text-white">
                  {result.gradientSizeGb.toFixed(1)} GB
                </div>
                <div className="text-[10px] text-textTertiary mt-1">
                  via {numNodes > 1 ? config.interConnect.name : config.intraConnect.name} ({result.allreduceTimeMs.toFixed(1)} ms)
                </div>
              </div>
              <div className="p-4 rounded-xl border border-border bg-surface/50">
                <div className="text-[10px] font-mono text-textTertiary uppercase mb-2">Training Time (1T tokens)</div>
                <div className="text-lg font-bold font-mono text-white">
                  {result.trainingTimeDays > 365
                    ? `${(result.trainingTimeDays / 365).toFixed(1)} yrs`
                    : `${result.trainingTimeDays.toFixed(0)} days`}
                </div>
                <div className="text-[10px] text-textTertiary mt-1">
                  {result.tokensPerSec.toFixed(0)} tok/s sustained
                </div>
              </div>
            </div>

            {/* What-if insights */}
            <div className="p-4 rounded-xl border border-border bg-surface/50">
              <h3 className="text-xs font-mono text-textTertiary uppercase mb-3">Diagnosis</h3>
              <div className="space-y-2 text-sm text-textSecondary">
                {/* Constraint violations */}
                {result.tpCrossesNodes && (
                  <p className="flex items-start gap-2">
                    <span className="text-accentRed font-bold shrink-0">!!</span>
                    TP={config.tpDegree} exceeds GPUs/node={config.gpusPerNode}. Tensor parallelism across nodes
                    requires inter-node AllReduce for every matmul — expect severe performance degradation. Keep TP within one node.
                  </p>
                )}
                {config.numGpus % (config.tpDegree * config.ppDegree) !== 0 && (
                  <p className="flex items-start gap-2">
                    <span className="text-accentRed font-bold shrink-0">!!</span>
                    Invalid config: {config.numGpus} GPUs is not divisible by TP={config.tpDegree} × PP={config.ppDegree} = {config.tpDegree * config.ppDegree}.
                    DP degree would be fractional.
                  </p>
                )}
                {result.commOverheadPct > 30 && (
                  <p className="flex items-start gap-2">
                    <span className="text-accentRed font-bold shrink-0">!</span>
                    Communication-bound: AllReduce takes {result.commOverheadPct.toFixed(0)}% of iteration time.
                    Try increasing TP degree (reduces DP degree and gradient size) or use gradient compression.
                  </p>
                )}
                {result.pipelineBubblePct > 15 && (
                  <p className="flex items-start gap-2">
                    <span className="text-accentAmber font-bold shrink-0">!</span>
                    Pipeline bubble at {result.pipelineBubblePct.toFixed(0)}%. Increase microbatch count (raise global batch) or reduce PP stages.
                  </p>
                )}
                {!result.fitsInMemory && (
                  <p className="flex items-start gap-2">
                    <span className="text-accentRed font-bold shrink-0">!</span>
                    Model does not fit. Increase TP degree, enable activation recomputation, or use ZeRO-3.
                  </p>
                )}
                {result.mfuEffective < 30 && result.commOverheadPct < 20 && (
                  <p className="flex items-start gap-2">
                    <span className="text-accentAmber font-bold shrink-0">!</span>
                    Low MFU ({result.mfuEffective.toFixed(0)}%) despite low comm overhead. Check batch size, kernel efficiency, or data loading pipeline.
                  </p>
                )}
                {result.commOverheadPct <= 30 && result.pipelineBubblePct <= 15 && result.fitsInMemory && result.mfuEffective >= 30 && (
                  <p className="flex items-start gap-2">
                    <span className="text-accentGreen font-bold shrink-0">✓</span>
                    Configuration looks healthy. MFU: {result.mfuEffective.toFixed(1)}%, Comm: {result.commOverheadPct.toFixed(0)}%, Memory: {result.memoryPerGpu.toFixed(1)}/{config.hardware.memory_gb} GB.
                  </p>
                )}
                {result.clusterMtbfHours < 4 && (
                  <p className="flex items-start gap-2">
                    <span className="text-accentAmber font-bold shrink-0">!</span>
                    MTBF is only {result.clusterMtbfHours.toFixed(1)} hours. Expect ~{Math.ceil(24 / result.clusterMtbfHours)} failures/day. Checkpoint aggressively (every {result.checkpointIntervalMin.toFixed(0)} min).
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
