"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Cpu } from "lucide-react";
import clsx from "clsx";
import { HARDWARE_SPECS, INTERCONNECTS, LATENCY_HIERARCHY, HardwareSpec } from "@/lib/hardware";

export default function HardwareRef() {
  const [open, setOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'specs' | 'latency' | 'interconnects'>('specs');

  return (
    <div className="border-t border-border">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2 text-[10px] font-mono text-textTertiary uppercase tracking-widest hover:text-textSecondary transition-colors"
      >
        <span className="flex items-center gap-1.5">
          <Cpu className="w-3 h-3" /> Hardware Reference
        </span>
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>

      {open && (
        <div className="px-4 pb-4">
          {/* Tabs */}
          <div className="flex items-center gap-1 mb-3">
            {(['specs', 'latency', 'interconnects'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={clsx(
                  "px-2 py-1 rounded text-[10px] font-mono capitalize transition-all",
                  activeTab === tab
                    ? "bg-accentBlue/10 text-accentBlue"
                    : "text-textTertiary hover:text-textSecondary"
                )}
              >
                {tab}
              </button>
            ))}
          </div>

          {activeTab === 'specs' && (
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {HARDWARE_SPECS.map(hw => (
                <div key={hw.name} className="flex items-center justify-between text-[10px] font-mono py-1 border-b border-border/50 last:border-0">
                  <span className="text-textSecondary truncate mr-2">{hw.name}</span>
                  <div className="flex items-center gap-3 text-textTertiary shrink-0">
                    <span>{hw.compute_tflops} T</span>
                    <span>{formatBandwidth(hw.bandwidth_tbs)}</span>
                    <span>{formatMemory(hw.memory_gb)}</span>
                    <span>{hw.tdp_w}W</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'latency' && (
            <div className="space-y-1">
              {LATENCY_HIERARCHY.map(l => (
                <div key={l.operation} className="flex items-center justify-between text-[10px] font-mono py-1 border-b border-border/50 last:border-0">
                  <span className="text-textSecondary">{l.operation}</span>
                  <div className="flex items-center gap-3 text-textTertiary">
                    <span>{formatLatency(l.latency_ns)}</span>
                    <span className="text-textTertiary/50 w-16 text-right">{l.human_scale}</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'interconnects' && (
            <div className="space-y-1">
              {INTERCONNECTS.map(ic => (
                <div key={ic.name} className="flex items-center justify-between text-[10px] font-mono py-1 border-b border-border/50 last:border-0">
                  <span className="text-textSecondary">{ic.name}</span>
                  <div className="flex items-center gap-3 text-textTertiary">
                    <span>{ic.bandwidth_gbs} GB/s</span>
                    <span>{ic.latency_us} µs</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="mt-2 text-[9px] text-textTertiary/50 italic">
            Source: mlsysim/core/constants.py
          </div>
        </div>
      )}
    </div>
  );
}

function formatBandwidth(tbs: number): string {
  if (tbs >= 1) return `${tbs} TB/s`;
  return `${Math.round(tbs * 1000)} GB/s`;
}

function formatMemory(gb: number): string {
  if (gb >= 1) return `${gb} GB`;
  if (gb >= 0.001) return `${Math.round(gb * 1000)} MB`;
  return `${Math.round(gb * 1e6)} KB`;
}

function formatLatency(ns: number): string {
  if (ns >= 1e6) return `${(ns / 1e6).toFixed(0)} ms`;
  if (ns >= 1e3) return `${(ns / 1e3).toFixed(0)} µs`;
  return `${ns} ns`;
}
