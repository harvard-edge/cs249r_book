"use client";

import { useCallback, useState, useEffect } from 'react';
import ReactFlow, { Background, Controls, Edge, Node, useNodesState, useEdgesState, MarkerType } from 'reactflow';
import 'reactflow/dist/style.css';
import { Network, Server, Zap } from 'lucide-react';
import clsx from 'clsx';

// Custom Node for GPUs
const CustomNode = ({ data, selected }: any) => {
  return (
    <div className={clsx(
      "px-4 py-2 shadow-xl rounded-md border-2 bg-surface",
      selected ? "border-accentBlue" : "border-borderHighlight",
      data.isBottleneck && "border-accentRed shadow-[0_0_15px_rgba(239,68,68,0.3)]"
    )}>
      <div className="flex items-center gap-2">
        <Server className={clsx("w-4 h-4", data.isBottleneck ? "text-accentRed" : "text-accentBlue")} />
        <div className="font-mono text-xs font-bold text-textPrimary">{data.label}</div>
      </div>
      <div className="text-[9px] font-mono text-textTertiary mt-1">{data.sublabel}</div>
    </div>
  );
};

const nodeTypes = {
  custom: CustomNode,
};

interface AnimatedFlowProps {
  interconnectType: '10GbE' | 'InfiniBand';
}

export default function AnimatedFlow({ interconnectType }: AnimatedFlowProps) {
  const isBottleneck = interconnectType === '10GbE';

  const initialNodes: Node[] = [
    {
      id: 'rack1',
      type: 'custom',
      position: { x: 50, y: 50 },
      data: { label: 'Rack 1 (64 GPUs)', sublabel: 'Data Parallel Gradients' },
    },
    {
      id: 'rack2',
      type: 'custom',
      position: { x: 50, y: 250 },
      data: { label: 'Rack 2 (64 GPUs)', sublabel: 'Data Parallel Gradients' },
    },
    {
      id: 'switch',
      type: 'custom',
      position: { x: 350, y: 150 },
      data: { 
        label: 'Core Switch', 
        sublabel: isBottleneck ? 'BUFFER FULL' : 'NOMINAL',
        isBottleneck: isBottleneck
      },
    },
  ];

  const initialEdges: Edge[] = [
    {
      id: 'e1-switch',
      source: 'rack1',
      target: 'switch',
      animated: true,
      style: { stroke: isBottleneck ? '#ef4444' : '#10b981', strokeWidth: isBottleneck ? 4 : 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: isBottleneck ? '#ef4444' : '#10b981' },
      label: isBottleneck ? '1.25 GB/s (CHOKED)' : '50 GB/s (NDR)',
      labelStyle: { fill: isBottleneck ? '#ef4444' : '#10b981', fontWeight: 700, fontFamily: 'monospace', fontSize: 10 },
      labelBgStyle: { fill: '#111216', fillOpacity: 0.8 },
    },
    {
      id: 'e2-switch',
      source: 'rack2',
      target: 'switch',
      animated: true,
      style: { stroke: isBottleneck ? '#ef4444' : '#10b981', strokeWidth: isBottleneck ? 4 : 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: isBottleneck ? '#ef4444' : '#10b981' },
      label: isBottleneck ? '1.25 GB/s (CHOKED)' : '50 GB/s (NDR)',
      labelStyle: { fill: isBottleneck ? '#ef4444' : '#10b981', fontWeight: 700, fontFamily: 'monospace', fontSize: 10 },
      labelBgStyle: { fill: '#111216', fillOpacity: 0.8 },
    },
  ];

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes and edges when interconnect changes
  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [interconnectType]);

  return (
    <div className="w-full h-[400px] bg-[#000000] border border-border rounded-xl overflow-hidden relative">
      <div className="absolute top-4 left-4 z-10 bg-surface/80 backdrop-blur border border-border p-3 rounded-lg">
         <div className="text-[10px] font-mono text-textTertiary uppercase mb-1">Live Simulation</div>
         <div className={clsx("text-sm font-mono font-bold flex items-center gap-2", isBottleneck ? "text-accentRed" : "text-accentGreen")}>
           <Zap className="w-4 h-4" />
           {isBottleneck ? "TRANSFER: 112s / STEP" : "TRANSFER: 2.8s / STEP"}
         </div>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        className="dark"
      >
        <Background color="#333333" gap={16} />
      </ReactFlow>
    </div>
  );
}
