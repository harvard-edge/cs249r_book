"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Search, ZoomIn, ZoomOut, Maximize } from "lucide-react";
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import { type Concept, getConcepts, getEdges, getChapters, formatChapter } from "@/lib/taxonomy";

// Chapter color palette — distinct hues
const PALETTE = [
  "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#a855f7",
  "#06b6d4", "#f97316", "#ec4899", "#3b82f6", "#14b8a6",
  "#8b5cf6", "#10b981", "#e11d48", "#0ea5e9", "#d946ef",
  "#84cc16", "#f43f5e", "#2563eb", "#eab308", "#059669",
  "#7c3aed", "#0891b2", "#dc2626", "#4f46e5", "#16a34a",
  "#ea580c", "#9333ea",
];

interface Props {
  onSelect: (concept: Concept) => void;
  selectedId: string | null;
}

export default function TaxonomyGraph({ onSelect, selectedId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const graphRef = useRef<Graph | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // Build graph once
  useEffect(() => {
    if (!containerRef.current) return;

    const concepts = getConcepts();
    const edges = getEdges();
    const chapters = getChapters();
    const chColor: Record<string, string> = {};
    chapters.forEach((ch, i) => {
      chColor[ch] = PALETTE[i % PALETTE.length];
    });

    const conceptMap = new Map(concepts.map(c => [c.id, c]));
    const graph = new Graph();

    // Add nodes
    concepts.forEach((c) => {
      const ch = c.source_chapters[0] || "unknown";
      const size = Math.max(3, Math.min(15, 3 + Math.sqrt(c.question_count) * 1.2));
      graph.addNode(c.id, {
        label: c.name,
        size,
        color: chColor[ch] || "#999",
        x: Math.random() * 100,
        y: Math.random() * 100,
        // Custom attributes
        chapter: ch,
        qcount: c.question_count,
        role: c.role,
      });
    });

    // Add edges
    edges.forEach((e) => {
      if (graph.hasNode(e.source) && graph.hasNode(e.target)) {
        try {
          graph.addEdge(e.source, e.target, {
            size: 0.5,
            color: "rgba(100,100,100,0.15)",
          });
        } catch {
          // Skip duplicate edges
        }
      }
    });

    // Run ForceAtlas2 layout
    forceAtlas2.assign(graph, {
      iterations: 200,
      settings: {
        gravity: 1,
        scalingRatio: 10,
        barnesHutOptimize: true,
        strongGravityMode: true,
        slowDown: 5,
      },
    });

    graphRef.current = graph;

    // Initialize Sigma
    const sigma = new Sigma(graph, containerRef.current, {
      renderLabels: true,
      labelSize: 10,
      labelColor: { color: "#c9d1d9" },
      labelRenderedSizeThreshold: 8,
      defaultEdgeColor: "rgba(100,100,100,0.15)",
      defaultNodeColor: "#999",
      minCameraRatio: 0.1,
      maxCameraRatio: 10,
    });

    // Node reducers for highlighting
    sigma.setSetting("nodeReducer", (node, data) => {
      const res = { ...data };

      // Highlight untested with red ring
      const attrs = graph.getNodeAttributes(node);
      if (attrs.qcount === 0) {
        res.borderColor = "#f85149";
        res.borderSize = 2;
      }

      // Search highlighting
      if (searchQuery) {
        const label = (data.label || "").toLowerCase();
        const nodeId = node.toLowerCase();
        const q = searchQuery.toLowerCase();
        if (!label.includes(q) && !nodeId.includes(q)) {
          res.color = "rgba(100,100,100,0.2)";
          res.label = "";
        }
      }

      // Selected node
      if (node === selectedId) {
        res.highlighted = true;
        res.zIndex = 1;
      }

      // Hovered node — highlight neighbors
      if (hoveredNode) {
        if (node === hoveredNode) {
          res.highlighted = true;
          res.zIndex = 1;
        } else if (graph.hasEdge(hoveredNode, node) || graph.hasEdge(node, hoveredNode)) {
          res.highlighted = true;
        } else {
          res.color = "rgba(100,100,100,0.2)";
          res.label = "";
        }
      }

      return res;
    });

    sigma.setSetting("edgeReducer", (edge, data) => {
      const res = { ...data };
      if (hoveredNode) {
        const src = graph.source(edge);
        const tgt = graph.target(edge);
        if (src !== hoveredNode && tgt !== hoveredNode) {
          res.hidden = true;
        } else {
          res.size = 1.5;
          res.color = "rgba(200,200,200,0.5)";
        }
      }
      return res;
    });

    // Events
    sigma.on("clickNode", ({ node }) => {
      const concept = conceptMap.get(node);
      if (concept) onSelect(concept);
    });

    sigma.on("enterNode", ({ node }) => {
      setHoveredNode(node);
      containerRef.current!.style.cursor = "pointer";
    });

    sigma.on("leaveNode", () => {
      setHoveredNode(null);
      containerRef.current!.style.cursor = "default";
    });

    sigmaRef.current = sigma;

    return () => {
      sigma.kill();
      sigmaRef.current = null;
      graphRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Refresh on search/selection/hover changes
  useEffect(() => {
    sigmaRef.current?.refresh();
  }, [searchQuery, selectedId, hoveredNode]);

  const handleZoom = useCallback((direction: "in" | "out" | "reset") => {
    const sigma = sigmaRef.current;
    if (!sigma) return;
    const camera = sigma.getCamera();
    if (direction === "reset") {
      camera.animatedReset({ duration: 300 });
    } else {
      const ratio = direction === "in" ? camera.ratio / 1.5 : camera.ratio * 1.5;
      camera.animate({ ratio }, { duration: 200 });
    }
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* Controls */}
      <div className="flex items-center gap-2 mb-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-textTertiary" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Highlight concepts..."
            className="w-full pl-9 pr-4 py-2 bg-surface border border-border rounded-lg text-sm text-white placeholder:text-textTertiary focus:outline-none focus:border-accentBlue/50"
          />
        </div>
        <button
          onClick={() => handleZoom("in")}
          className="p-2 bg-surface border border-border rounded-lg text-textTertiary hover:text-white transition-colors"
          title="Zoom in"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => handleZoom("out")}
          className="p-2 bg-surface border border-border rounded-lg text-textTertiary hover:text-white transition-colors"
          title="Zoom out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={() => handleZoom("reset")}
          className="p-2 bg-surface border border-border rounded-lg text-textTertiary hover:text-white transition-colors"
          title="Reset view"
        >
          <Maximize className="w-4 h-4" />
        </button>
      </div>

      {/* Graph container */}
      <div
        ref={containerRef}
        className="flex-1 rounded-lg border border-border bg-[#0a0e14] overflow-hidden"
        style={{ minHeight: 400 }}
      />

      {/* Legend */}
      <div className="mt-2 flex items-center gap-4 text-[10px] text-textTertiary">
        <span>Node size = question count</span>
        <span>Color = chapter</span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full border-2 border-[#f85149] bg-transparent" />
          Untested
        </span>
        <span>Click node for details</span>
      </div>
    </div>
  );
}
