import { describe, it, expect } from "vitest";
import { MODEL_CONFIGS, HARDWARE_SPECS, INTERCONNECTS, FORMULAS } from "../lib/hardware";

function simulate(config: any) {
  const { model, hardware, numGpus, gpusPerNode, batchSize, seqLen, precision, mfu, tpDegree, ppDegree } = config;

  const dpDegree = Math.max(1, Math.floor(numGpus / (tpDegree * ppDegree)));

  const modelMemoryGb = FORMULAS.model_memory_gb(model.params_b, precision);
  const optimizerMemoryGb = FORMULAS.model_memory_gb(model.params_b, 12);
  const activationMemoryGb = (10 * model.layers * (batchSize / dpDegree) * seqLen * model.hidden_dim) / (tpDegree * 1e9);
  const totalMemoryPerGpu = (modelMemoryGb + optimizerMemoryGb) / (tpDegree * ppDegree) + activationMemoryGb;

  const tokensPerIter = batchSize * seqLen;
  const flopsPerIter = model.flops_per_token * tokensPerIter; 
  const computeTimeMs = FORMULAS.compute_time_ms(flopsPerIter, hardware.compute_tflops, numGpus, mfu);

  return {
    model_memory_gb: modelMemoryGb,
    total_memory_per_gpu: totalMemoryPerGpu,
    flops_per_iter: flopsPerIter,
    compute_time_ms: computeTimeMs,
    dp_degree: dpDegree
  };
}

describe("Simulator Logic", () => {
  const llama70b = MODEL_CONFIGS.find(m => m.name.includes("70B"))!;
  const h100 = HARDWARE_SPECS.find(h => h.name.includes("H100"))!;
  
  it("calculates training flops correctly (6PD per token)", () => {
    // Llama-70B: 70B params. 6PD per token = 420G FLOPs.
    expect(llama70b.flops_per_token).toBe(420e9);
    
    const config = {
      model: llama70b,
      hardware: h100,
      numGpus: 64,
      gpusPerNode: 8,
      batchSize: 1024,
      seqLen: 2048,
      precision: 2,
      mfu: 0.4,
      tpDegree: 8,
      ppDegree: 1,
    };
    
    const result = simulate(config);
    const tokensPerIter = config.batchSize * config.seqLen;
    expect(result.flops_per_iter).toBe(llama70b.flops_per_token * tokensPerIter);
    expect(result.flops_per_iter).toBe(420e9 * 1024 * 2048);
  });

  it("calculates activation memory with seqLen", () => {
    const config = {
      model: llama70b,
      hardware: h100,
      numGpus: 64,
      gpusPerNode: 8,
      batchSize: 1024,
      seqLen: 2048,
      precision: 2,
      mfu: 0.4,
      tpDegree: 8,
      ppDegree: 1,
    };
    
    const result1 = simulate(config);
    const result2 = simulate({ ...config, seqLen: 4096 });
    
    const constantPart = (result1.model_memory_gb + (llama70b.params_b * 12)) / (config.tpDegree * config.ppDegree);
    const act1 = result1.total_memory_per_gpu - constantPart;
    const act2 = result2.total_memory_per_gpu - constantPart;
    
    expect(act2).toBeCloseTo(act1 * 2, 5);
  });
});
