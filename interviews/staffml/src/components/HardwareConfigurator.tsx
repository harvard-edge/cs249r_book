"use client";

import { useState } from 'react';
import { Sliders, Cpu, Network, MemoryStick, HardDrive } from 'lucide-react';
import clsx from 'clsx';

interface HardwareState {
  compute: string;
  network: string;
  memory: string;
}

interface ConfiguratorProps {
  onStateChange: (state: HardwareState) => void;
  initialState?: HardwareState;
}

export default function HardwareConfigurator({ onStateChange, initialState }: ConfiguratorProps) {
  const [state, setState] = useState<HardwareState>(initialState || {
    compute: 'H100',
    network: '10GbE',
    memory: '80GB'
  });

  const handleChange = (key: keyof HardwareState, value: string) => {
    const newState = { ...state, [key]: value };
    setState(newState);
    onStateChange(newState);
  };

  return (
    <div className="bg-[#0f1115] border border-border rounded-xl p-5 mb-6 shadow-inner relative overflow-hidden group">
      <div className="absolute inset-0 bg-gradient-to-r from-accentBlue/0 via-accentBlue/5 to-accentBlue/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 ease-in-out pointer-events-none"></div>
      
      <div className="flex items-center gap-2 mb-4 border-b border-border/50 pb-3">
        <Sliders className="w-4 h-4 text-accentBlue" />
        <h3 className="text-xs font-mono font-semibold text-textPrimary uppercase tracking-widest">Hardware Control Plane</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Compute Dial */}
        <div className="flex flex-col gap-2">
          <label className="text-[10px] font-mono text-textTertiary uppercase flex items-center gap-1.5">
            <Cpu className="w-3 h-3" /> Compute Node
          </label>
          <select 
            value={state.compute}
            onChange={(e) => handleChange('compute', e.target.value)}
            className="bg-surface border border-border text-textPrimary text-xs font-mono rounded-lg px-3 py-2.5 focus:outline-none focus:border-accentBlue focus:ring-1 focus:ring-accentBlue/50 transition-all appearance-none cursor-pointer hover:bg-surfaceHover"
          >
            <option value="T4">NVIDIA T4 (16GB)</option>
            <option value="A100">NVIDIA A100 (40GB)</option>
            <option value="H100">NVIDIA H100 (80GB)</option>
            <option value="B200">NVIDIA B200 (192GB)</option>
          </select>
        </div>

        {/* Network Dial */}
        <div className="flex flex-col gap-2">
          <label className="text-[10px] font-mono text-textTertiary uppercase flex items-center gap-1.5">
            <Network className="w-3 h-3" /> Interconnect
          </label>
          <select 
            value={state.network}
            onChange={(e) => handleChange('network', e.target.value)}
            className="bg-surface border border-border text-textPrimary text-xs font-mono rounded-lg px-3 py-2.5 focus:outline-none focus:border-accentBlue focus:ring-1 focus:ring-accentBlue/50 transition-all appearance-none cursor-pointer hover:bg-surfaceHover"
          >
            <option value="1GbE">1 GbE Ethernet</option>
            <option value="10GbE">10 GbE Ethernet</option>
            <option value="PCIe">PCIe Gen5 Switch</option>
            <option value="InfiniBand">NDR InfiniBand (400G)</option>
          </select>
        </div>

        {/* Storage / Precision Dial */}
        <div className="flex flex-col gap-2">
          <label className="text-[10px] font-mono text-textTertiary uppercase flex items-center gap-1.5">
            <HardDrive className="w-3 h-3" /> Storage Tier
          </label>
          <select 
            value={state.memory}
            onChange={(e) => handleChange('memory', e.target.value)}
            className="bg-surface border border-border text-textPrimary text-xs font-mono rounded-lg px-3 py-2.5 focus:outline-none focus:border-accentBlue focus:ring-1 focus:ring-accentBlue/50 transition-all appearance-none cursor-pointer hover:bg-surfaceHover"
          >
            <option value="HDD">Network HDD (100 MB/s)</option>
            <option value="SSD">SATA SSD (500 MB/s)</option>
            <option value="NVMe">Local NVMe (7 GB/s)</option>
            <option value="Distributed">Distributed Flash (20 GB/s)</option>
          </select>
        </div>
      </div>
    </div>
  );
}