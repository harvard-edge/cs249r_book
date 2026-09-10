import { describe, it, expect } from "vitest";
import { MODEL_CONFIGS, HARDWARE_SPECS, INTERCONNECTS } from "../lib/hardware";
import { simulate, SimConfig } from "../lib/simulator";

describe("Simulator Logic", () => {
  const llama70b = MODEL_CONFIGS.find(m => m.name.includes("70B"))!;
  const h100 = HARDWARE_SPECS.find(h => h.name.includes("H100"))!;
  const nvlink = INTERCONNECTS.find(i => i.name.includes("NVLink H100"))!;
  const ib = INTERCONNECTS.find(i => i.name.includes("InfiniBand NDR"))!;
  
  const baseConfig: SimConfig = {
    model: llama70b,
    hardware: h100,
    numGpus: 64,
    gpusPerNode: 8,
    intraConnect: nvlink,
    interConnect: ib,
    batchSize: 1024,
    seqLen: 2048,
    precision: 2,
    mfu: 0.4,
    tpDegree: 8,
    ppDegree: 1,
  };

  it("calculates training flops correctly (6PD per token)", () => {
    // Llama-70B: 70B params. 6PD per token = 420G FLOPs.
    expect(llama70b.flops_per_token).toBe(420e9);
    
    const result = simulate(baseConfig);
    const tokensPerIter = baseConfig.batchSize * baseConfig.seqLen;
    expect(result.flopsPerIter).toBe(llama70b.flops_per_token * tokensPerIter);
    expect(result.flopsPerIter).toBe(420e9 * 1024 * 2048);
  });

  it("calculates activation memory with seqLen", () => {
    const result1 = simulate(baseConfig);
    const result2 = simulate({ ...baseConfig, seqLen: 4096 });
    
    // In lib/simulator.ts we use camelCase: modelMemoryGb, memoryPerGpu
    const constantPart = (result1.modelMemoryGb + (llama70b.params_b * 12)) / (baseConfig.tpDegree * baseConfig.ppDegree);
    const act1 = result1.memoryPerGpu - constantPart;
    const act2 = result2.memoryPerGpu - constantPart;
    
    expect(act2).toBeCloseTo(act1 * 2, 5);
  });

  it("identifies OOM correctly", () => {
    // Force OOM by using huge batch size
    const result = simulate({ ...baseConfig, batchSize: 1000000 });
    expect(result.fitsInMemory).toBe(false);
  });
});
