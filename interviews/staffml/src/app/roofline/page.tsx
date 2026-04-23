"use client";

import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { Terminal, Cpu } from "lucide-react";
import clsx from "clsx";
import { HARDWARE_SPECS, FORMULAS, HardwareSpec } from "@/lib/hardware";

// Well-known model workloads for plotting.
//
// Note on colors: these are a categorical data palette, not theme
// chrome. Each workload needs a distinct hue so the viewer can match
// the dot on the chart to the label in the legend. Unlike the
// roofline curve and ridge point (which follow the theme's accent
// blue), the workload palette stays constant across dark/light
// modes — same as you'd do with any Matplotlib tab10-style chart.
const WORKLOADS = [
  { name: 'BERT-Base (FP16)', params_b: 0.11, ops_per_byte: 50, color: '#3b82f6' },
  { name: 'ResNet-50 (FP16)', params_b: 0.025, ops_per_byte: 120, color: '#10b981' },
  { name: 'GPT-2 (FP16)', params_b: 1.5, ops_per_byte: 30, color: '#f5a623' },
  { name: 'Llama-70B (FP16)', params_b: 70, ops_per_byte: 15, color: '#ef4444' },
  { name: 'Llama-70B (INT8)', params_b: 70, ops_per_byte: 30, color: '#cc4444' },
  { name: 'MobileNetV2 (INT8)', params_b: 0.003, ops_per_byte: 200, color: '#8b5cf6' },
];

// SVG dimensions
const W = 680;
const H = 420;
const PAD = { top: 50, right: 40, bottom: 60, left: 70 };
const PW = W - PAD.left - PAD.right;
const PH = H - PAD.top - PAD.bottom;

// Log scale helpers
function logScale(val: number, min: number, max: number, size: number): number {
  const logMin = Math.log10(Math.max(min, 0.01));
  const logMax = Math.log10(max);
  const logVal = Math.log10(Math.max(val, min));
  return ((logVal - logMin) / (logMax - logMin)) * size;
}

export default function RooflinePage() {
  const [mounted, setMounted] = useState(false);
  const [selectedHw, setSelectedHw] = useState<HardwareSpec>(
    HARDWARE_SPECS.find(h => h.name.includes('H100')) || HARDWARE_SPECS[0]
  );
  const [customOI, setCustomOI] = useState<string>('');

  useEffect(() => { setMounted(true); }, []);

  // Compute roofline
  const ridge = useMemo(() => FORMULAS.ridge_point(selectedHw.compute_tflops, selectedHw.bandwidth_tbs), [selectedHw]);

  // Axis ranges (log scale)
  const xMin = 0.1, xMax = 1000; // Operational Intensity (Ops/Byte)
  const yMin = 0.1, yMax = selectedHw.compute_tflops * 2; // TFLOPS

  // Roofline segments
  const rooflinePoints = useMemo(() => {
    const points: Array<{ x: number; y: number }> = [];
    // Bandwidth-bound region (slope)
    for (let oi = xMin; oi <= ridge; oi *= 1.2) {
      points.push({ x: oi, y: oi * selectedHw.bandwidth_tbs });
    }
    // Ridge point
    points.push({ x: ridge, y: selectedHw.compute_tflops });
    // Compute-bound region (flat)
    for (let oi = ridge * 1.2; oi <= xMax; oi *= 1.2) {
      points.push({ x: oi, y: selectedHw.compute_tflops });
    }
    return points;
  }, [selectedHw, ridge]);

  const toSvg = (x: number, y: number) => ({
    sx: PAD.left + logScale(x, xMin, xMax, PW),
    sy: PAD.top + PH - logScale(y, yMin, yMax, PH),
  });

  const rooflinePath = useMemo(() => {
    return rooflinePoints.map((p, i) => {
      const { sx, sy } = toSvg(p.x, p.y);
      return `${i === 0 ? 'M' : 'L'}${sx.toFixed(1)},${sy.toFixed(1)}`;
    }).join(' ');
  }, [rooflinePoints]);

  // Custom OI point
  const customOINum = parseFloat(customOI);
  const customPoint = !isNaN(customOINum) && customOINum > 0 ? {
    oi: customOINum,
    perf: Math.min(customOINum * selectedHw.bandwidth_tbs, selectedHw.compute_tflops),
    bound: customOINum < ridge ? 'Memory-bound' : 'Compute-bound',
  } : null;

  if (!mounted) {
    return <div className="flex-1 flex items-center justify-center"><Terminal className="w-6 h-6 text-textTertiary animate-pulse" /></div>;
  }

  return (
    <div className="flex-1 flex flex-col px-6 py-10">
      <div className="max-w-5xl mx-auto w-full">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <svg viewBox="0 0 32 32" className="w-8 h-8">
            <path d="M5,25 L16,9 L27,9" stroke="var(--accent-blue)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none" />
            <circle cx="16" cy="9" r="2.5" fill="var(--accent-blue)" />
            <circle cx="16" cy="9" r="1" fill="currentColor" />
          </svg>
          <div>
            <h1 className="text-3xl font-extrabold text-textPrimary tracking-tight">Interactive Roofline</h1>
            <p className="text-sm text-textSecondary">Visualize compute vs. bandwidth bottlenecks on real hardware</p>
          </div>
        </div>

        <p className="text-xs text-textTertiary mb-6 max-w-2xl leading-relaxed">
          The Roofline Model shows the maximum achievable performance for a given workload.
          Below the ridge point, your workload is <span className="text-accentRed">memory-bandwidth limited</span> — moving data is the bottleneck.
          Above it, you are <span className="text-accentGreen">compute limited</span> — the ALU is the bottleneck.
          Select hardware below and see where common models fall.
        </p>

        {/* Hardware selector */}
        <div className="flex flex-wrap items-center gap-2 mb-6">
          {HARDWARE_SPECS.filter(h => h.tier === 'cloud').map(hw => (
            <button
              key={hw.name}
              onClick={() => setSelectedHw(hw)}
              className={clsx(
                "px-3 py-1.5 rounded-md text-xs font-mono transition-all border",
                selectedHw.name === hw.name
                  ? "border-accentBlue bg-accentBlue/10 text-textPrimary"
                  : "border-border text-textTertiary hover:border-borderHighlight hover:text-textSecondary"
              )}
            >
              {hw.name.replace('NVIDIA ', '').replace(' SXM', '')}
            </button>
          ))}
        </div>

        {/* Roofline SVG */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-surface border border-border rounded-xl p-4 mb-6">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: '420px' }}>
            {/* Background */}
            <rect width={W} height={H} fill="var(--surface)" rx="8" />

            {/* Grid lines */}
            {[0.1, 1, 10, 100, 1000].map(v => {
              const { sx } = toSvg(v, yMin);
              return <line key={`xg-${v}`} x1={sx} y1={PAD.top} x2={sx} y2={PAD.top + PH} stroke="var(--border)" strokeWidth="0.5" />;
            })}
            {[0.1, 1, 10, 100, 1000, 10000].filter(v => v <= yMax).map(v => {
              const { sy } = toSvg(xMin, v);
              return <line key={`yg-${v}`} x1={PAD.left} y1={sy} x2={PAD.left + PW} y2={sy} stroke="var(--border)" strokeWidth="0.5" />;
            })}

            {/* Roofline curve — themed, follows the app's accent blue so
                it reads correctly in both dark and light mode. */}
            <path d={rooflinePath} stroke="var(--accent-blue)" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />

            {/* Ridge point — also themed. */}
            {(() => {
              const { sx, sy } = toSvg(ridge, selectedHw.compute_tflops);
              return (
                <g>
                  <circle cx={sx} cy={sy} r="5" fill="var(--accent-blue)" />
                  <circle cx={sx} cy={sy} r="2" fill="var(--surface)" />
                  <text x={sx + 8} y={sy - 8} fill="var(--accent-blue)" fontSize="9" fontFamily="JetBrains Mono, monospace">
                    Ridge: {ridge.toFixed(0)} Ops/B
                  </text>
                </g>
              );
            })()}

            {/* Workload dots */}
            {WORKLOADS.map((wl, wi) => {
              const perf = Math.min(wl.ops_per_byte * selectedHw.bandwidth_tbs, selectedHw.compute_tflops);
              const { sx, sy } = toSvg(wl.ops_per_byte, perf);
              // Stagger labels vertically to avoid overlap
              const yOffset = (wi % 2 === 0) ? -10 : 12;
              return (
                <g key={wl.name}>
                  <circle cx={sx} cy={sy} r="5" fill={wl.color} opacity="0.9" />
                  <rect x={sx + 7} y={sy + yOffset - 9} width={wl.name.length * 6.5 + 8} height="14" rx="3"
                    fill="var(--background)" opacity="0.85" />
                  <text x={sx + 10} y={sy + yOffset} fill={wl.color} fontSize="10" fontWeight="600" fontFamily="JetBrains Mono, monospace">
                    {wl.name}
                  </text>
                </g>
              );
            })}

            {/* Custom point */}
            {customPoint && (() => {
              const { sx, sy } = toSvg(customPoint.oi, customPoint.perf);
              return (
                <g>
                  <circle cx={sx} cy={sy} r="6" fill="none" stroke="var(--text-primary)" strokeWidth="2" />
                  <circle cx={sx} cy={sy} r="2" fill="var(--text-primary)" />
                  <text x={sx + 10} y={sy + 3} fill="var(--text-primary)" fontSize="9" fontFamily="JetBrains Mono, monospace" fontWeight="bold">
                    {customPoint.perf.toFixed(1)} TFLOPS ({customPoint.bound})
                  </text>
                </g>
              );
            })()}

            {/* Axis labels */}
            <text x={W / 2} y={H - 10} textAnchor="middle" fill="var(--text-tertiary)" fontSize="10" fontFamily="JetBrains Mono, monospace">
              Operational Intensity (Ops/Byte)
            </text>
            <text x="15" y={H / 2} textAnchor="middle" fill="var(--text-tertiary)" fontSize="10" fontFamily="JetBrains Mono, monospace"
              transform={`rotate(-90, 15, ${H / 2})`}>
              Performance (TFLOPS)
            </text>

            {/* X axis tick labels */}
            {[0.1, 1, 10, 100, 1000].map(v => {
              const { sx } = toSvg(v, yMin);
              return <text key={`xl-${v}`} x={sx} y={PAD.top + PH + 20} textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="JetBrains Mono, monospace">{v}</text>;
            })}
            {/* Y axis tick labels */}
            {[0.1, 1, 10, 100, 1000, 10000].filter(v => v <= yMax).map(v => {
              const { sy } = toSvg(xMin, v);
              return <text key={`yl-${v}`} x={PAD.left - 8} y={sy + 3} textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="JetBrains Mono, monospace">{v >= 1000 ? `${v / 1000}K` : v}</text>;
            })}

            {/* Title */}
            <text x={W / 2} y="25" textAnchor="middle" fill="var(--text-primary)" fontSize="12" fontWeight="bold" fontFamily="JetBrains Mono, monospace">
              {selectedHw.name} — {selectedHw.compute_tflops} TFLOPS, {selectedHw.bandwidth_tbs} TB/s
            </text>

            {/* Region labels */}
            {(() => {
              const midBW = toSvg(Math.sqrt(xMin * ridge), selectedHw.compute_tflops * 0.3);
              const midComp = toSvg(Math.sqrt(ridge * xMax), selectedHw.compute_tflops * 0.7);
              return (
                <>
                  <text x={midBW.sx} y={midBW.sy} textAnchor="middle" fill="var(--text-muted)" fontSize="10" fontFamily="JetBrains Mono, monospace" fontStyle="italic">
                    Memory-bound
                  </text>
                  <text x={midComp.sx} y={midComp.sy} textAnchor="middle" fill="var(--text-muted)" fontSize="10" fontFamily="JetBrains Mono, monospace" fontStyle="italic">
                    Compute-bound
                  </text>
                </>
              );
            })()}
          </svg>
        </motion.div>

        {/* Custom OI input */}
        <div className="flex items-center gap-4 mb-6">
          <label className="text-xs text-textTertiary font-mono">Your workload's Operational Intensity (Ops/Byte):</label>
          <input
            type="text"
            value={customOI}
            onChange={(e) => setCustomOI(e.target.value)}
            placeholder="e.g. 45"
            className="w-32 bg-background border border-border rounded px-3 py-1.5 text-sm font-mono text-textPrimary focus:outline-none focus:border-accentBlue/50"
          />
          {customPoint && (
            <span className={clsx(
              "text-xs font-mono px-2 py-1 rounded",
              customPoint.bound === 'Memory-bound' ? "text-accentRed bg-accentRed/10" : "text-accentGreen bg-accentGreen/10"
            )}>
              {customPoint.perf.toFixed(1)} TFLOPS — {customPoint.bound}
            </span>
          )}
        </div>

        {/* Hardware comparison cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {HARDWARE_SPECS.filter(h => h.tier === 'cloud').map(hw => {
            const r = FORMULAS.ridge_point(hw.compute_tflops, hw.bandwidth_tbs);
            return (
              <button
                key={hw.name}
                onClick={() => setSelectedHw(hw)}
                className={clsx(
                  "p-3 rounded-lg border text-left transition-all",
                  selectedHw.name === hw.name
                    ? "border-accentBlue bg-accentBlue/5"
                    : "border-border bg-surface/50 hover:border-borderHighlight"
                )}
              >
                <div className="text-xs font-medium text-textPrimary mb-1">{hw.name.replace('NVIDIA ', '')}</div>
                <div className="text-[10px] font-mono text-textTertiary space-y-0.5">
                  <div>{hw.compute_tflops} TFLOPS {hw.compute_unit}</div>
                  <div>{hw.bandwidth_tbs} TB/s {hw.memory_type}</div>
                  <div>Ridge: {r.toFixed(0)} Ops/B</div>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
