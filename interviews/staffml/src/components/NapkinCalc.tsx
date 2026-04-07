"use client";

import { useState } from "react";
import { Calculator, ChevronDown, ChevronRight } from "lucide-react";
import clsx from "clsx";
import { HARDWARE_SPECS, FORMULAS, HardwareSpec } from "@/lib/hardware";

type CalcMode = 'model_memory' | 'training_time' | 'allreduce' | 'ridge_point' | 'kv_cache';

const CALC_MODES: { id: CalcMode; label: string; desc: string }[] = [
  { id: 'model_memory', label: 'Model Memory', desc: 'How much memory does the model need?' },
  { id: 'training_time', label: 'Training Time', desc: 'How long to train on N GPUs?' },
  { id: 'allreduce', label: 'AllReduce Time', desc: 'Communication overhead for gradients' },
  { id: 'ridge_point', label: 'Ridge Point', desc: 'Where compute meets bandwidth' },
  { id: 'kv_cache', label: 'KV Cache Size', desc: 'Inference memory for attention' },
];

export default function NapkinCalc({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const [mode, setMode] = useState<CalcMode>('model_memory');

  // Inputs
  const [paramsB, setParamsB] = useState('70');
  const [bytesPerParam, setBytesPerParam] = useState('2');
  const [tokensB, setTokensB] = useState('1');
  const [numGpus, setNumGpus] = useState('64');
  const [mfu, setMfu] = useState('0.4');
  const [hwIdx, setHwIdx] = useState(0);
  const [layers, setLayers] = useState('80');
  const [heads, setHeads] = useState('64');
  const [headDim, setHeadDim] = useState('128');
  const [seqLen, setSeqLen] = useState('2048');
  const [batch, setBatch] = useState('1');

  const hw = HARDWARE_SPECS.filter(h => h.tier === 'cloud')[hwIdx] || HARDWARE_SPECS[0];
  const cloudGpus = HARDWARE_SPECS.filter(h => h.tier === 'cloud');

  const compute = () => {
    const p = parseFloat(paramsB) || 0;
    const bpp = parseFloat(bytesPerParam) || 2;
    const t = parseFloat(tokensB) || 1;
    const g = parseInt(numGpus) || 1;
    const m = parseFloat(mfu) || 0.4;
    const l = parseInt(layers) || 80;
    const h = parseInt(heads) || 64;
    const hd = parseInt(headDim) || 128;
    const s = parseInt(seqLen) || 2048;
    const b = parseInt(batch) || 1;

    switch (mode) {
      case 'model_memory': {
        const mem = FORMULAS.model_memory_gb(p, bpp);
        return { result: `${mem.toFixed(1)} GB`, detail: `${p}B params × ${bpp} bytes = ${mem.toFixed(1)} GB` };
      }
      case 'training_time': {
        const flops = FORMULAS.training_flops(p, t);
        const days = FORMULAS.training_time_days(flops, hw.compute_tflops, g, m);
        return {
          result: days > 365 ? `${(days / 365).toFixed(1)} years` : `${days.toFixed(1)} days`,
          detail: `6 × ${p}B × ${t}T = ${(flops / 1e21).toFixed(1)}e21 FLOPS | ${g}× ${hw.name.split(' ')[1]} @ ${(m * 100).toFixed(0)}% MFU`,
        };
      }
      case 'allreduce': {
        const gradSize = FORMULAS.model_memory_gb(p, bpp);
        const time = FORMULAS.allreduce_time_ms(gradSize, hw.bandwidth_tbs * 1000, g);
        return {
          result: `${time.toFixed(1)} ms`,
          detail: `${gradSize.toFixed(1)} GB gradients across ${g} GPUs via ring AllReduce`,
        };
      }
      case 'ridge_point': {
        const ridge = FORMULAS.ridge_point(hw.compute_tflops, hw.bandwidth_tbs);
        return {
          result: `${ridge.toFixed(0)} Ops/Byte`,
          detail: `${hw.compute_tflops} TFLOPS / ${hw.bandwidth_tbs} TB/s on ${hw.name}`,
        };
      }
      case 'kv_cache': {
        const cache = FORMULAS.kv_cache_mb(l, h, hd, s, b, bpp);
        return {
          result: `${cache.toFixed(0)} MB`,
          detail: `2 × ${l} layers × ${h} heads × ${hd}d × ${s} seq × ${b} batch × ${bpp}B`,
        };
      }
    }
  };

  const output = compute();

  return (
    <div className="border-t border-border">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2 text-[10px] font-mono text-textTertiary uppercase tracking-widest hover:text-textSecondary transition-colors"
      >
        <span className="flex items-center gap-1.5">
          <Calculator className="w-3 h-3" /> Napkin Calculator
        </span>
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>

      {open && (
        <div className="px-4 pb-4">
          {/* Mode tabs */}
          <div className="flex flex-wrap gap-1 mb-3">
            {CALC_MODES.map(c => (
              <button
                key={c.id}
                onClick={() => setMode(c.id)}
                className={clsx(
                  "px-2 py-1 rounded text-[9px] font-mono transition-all",
                  mode === c.id ? "bg-accentBlue/10 text-accentBlue" : "text-textTertiary hover:text-textSecondary"
                )}
              >
                {c.label}
              </button>
            ))}
          </div>

          {/* Inputs */}
          <div className="space-y-2 mb-3">
            {(mode === 'model_memory' || mode === 'training_time' || mode === 'allreduce') && (
              <div className="flex items-center gap-2">
                <label className="text-[9px] text-textTertiary w-16">Params (B)</label>
                <input value={paramsB} onChange={e => setParamsB(e.target.value)}
                  className="flex-1 bg-background border border-border rounded px-2 py-1 text-xs font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50" />
              </div>
            )}
            {(mode === 'training_time' || mode === 'allreduce' || mode === 'ridge_point') && (
              <div className="flex items-center gap-2">
                <label className="text-[9px] text-textTertiary w-16">Hardware</label>
                <select value={hwIdx} onChange={e => setHwIdx(Number(e.target.value))}
                  className="flex-1 bg-background border border-border rounded px-2 py-1 text-xs font-mono text-textPrimary focus:outline-none">
                  {cloudGpus.map((h, i) => <option key={h.name} value={i}>{h.name.replace('NVIDIA ', '')}</option>)}
                </select>
              </div>
            )}
            {(mode === 'training_time' || mode === 'allreduce') && (
              <div className="flex items-center gap-2">
                <label className="text-[9px] text-textTertiary w-16">GPUs</label>
                <input value={numGpus} onChange={e => setNumGpus(e.target.value)}
                  className="flex-1 bg-background border border-border rounded px-2 py-1 text-xs font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50" />
              </div>
            )}
            {mode === 'training_time' && (
              <>
                <div className="flex items-center gap-2">
                  <label className="text-[9px] text-textTertiary w-16">Tokens (T)</label>
                  <input value={tokensB} onChange={e => setTokensB(e.target.value)}
                    className="flex-1 bg-background border border-border rounded px-2 py-1 text-xs font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50" />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-[9px] text-textTertiary w-16">MFU</label>
                  <input value={mfu} onChange={e => setMfu(e.target.value)}
                    className="flex-1 bg-background border border-border rounded px-2 py-1 text-xs font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50" />
                </div>
              </>
            )}
            {mode === 'kv_cache' && (
              <>
                <div className="flex items-center gap-2">
                  <label className="text-[9px] text-textTertiary w-16">Layers</label>
                  <input value={layers} onChange={e => setLayers(e.target.value)}
                    className="flex-1 bg-background border border-border rounded px-2 py-1 text-xs font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50" />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-[9px] text-textTertiary w-16">Heads</label>
                  <input value={heads} onChange={e => setHeads(e.target.value)}
                    className="flex-1 bg-background border border-border rounded px-2 py-1 text-xs font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50" />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-[9px] text-textTertiary w-16">Head dim</label>
                  <input value={headDim} onChange={e => setHeadDim(e.target.value)}
                    className="flex-1 bg-background border border-border rounded px-2 py-1 text-xs font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50" />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-[9px] text-textTertiary w-16">Seq len</label>
                  <input value={seqLen} onChange={e => setSeqLen(e.target.value)}
                    className="flex-1 bg-background border border-border rounded px-2 py-1 text-xs font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50" />
                </div>
              </>
            )}
          </div>

          {/* Result */}
          <div className="p-3 rounded-lg bg-background border border-accentBlue/20">
            <div className="text-lg font-bold font-mono text-accentBlue">{output.result}</div>
            <div className="text-[10px] font-mono text-textTertiary mt-1">{output.detail}</div>
          </div>

          <div className="mt-2 text-[9px] text-textTertiary/50 italic">
            Formulas from mlsysim/core/formulas.py
          </div>
        </div>
      )}
    </div>
  );
}
