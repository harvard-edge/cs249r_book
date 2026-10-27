/**
 * The hardware reference, napkin calculator, roofline, and simulator read
 * their numbers from src/data/mlsysim.json, exported from the MLSysIM
 * registries by scripts/export_mlsysim_data.py.
 *
 * 2026-10-27: the hand-copied table this replaced cited a deleted MLSysIM file
 * and had drifted on five devices. CI runs the export with --check, which
 * catches stale values; these tests catch rows that lose their source.
 */
import { describe, it, expect } from 'vitest';
import { HARDWARE_SPECS, INTERCONNECTS, LATENCY_HIERARCHY, MODEL_CONFIGS, humanScale } from '@/lib/hardware';

describe('MLSysIM-backed reference data', () => {
  it('traces every hardware, interconnect, and model row to a registry entry', () => {
    for (const row of [...HARDWARE_SPECS, ...INTERCONNECTS, ...MODEL_CONFIGS]) {
      expect(row.mlsysim, row.name).toMatch(/^(Hardware|Systems|Models)\./);
    }
  });

  it('keeps the rows the simulator and napkin calculator look up by name', () => {
    expect(HARDWARE_SPECS.find(h => h.name.includes('H100'))).toBeDefined();
    expect(INTERCONNECTS.find(i => i.name === 'NVLink H100')).toBeDefined();
    expect(INTERCONNECTS.find(i => i.name === 'InfiniBand NDR')).toBeDefined();
  });

  it('labels each latency on the 1 ns = 1 second scale', () => {
    expect(humanScale(1)).toBe('1 second');
    expect(humanScale(300)).toBe('5 minutes');
    expect(humanScale(40_000_000)).toBe('1.3 years');
    for (const l of LATENCY_HIERARCHY) expect(l.human_scale).toBe(humanScale(l.latency_ns));
  });
});
