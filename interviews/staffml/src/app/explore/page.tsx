"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import clsx from "clsx";
import {
  ArrowLeft,
  ArrowRight,
  Layers,
  LocateFixed,
  Network,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { getQuestions, getTracks, type Question } from "@/lib/corpus";
import { getAreas, getAreaStyle } from "@/lib/taxonomy";
import { LEVELS } from "@/lib/levels";

const WORLD_WIDTH = 1800;
const WORLD_HEIGHT = 1120;
const CENTER_X = WORLD_WIDTH / 2;
const CENTER_Y = WORLD_HEIGHT / 2;
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 4;
const HIT_RADIUS = 10;

type Point = { x: number; y: number };

interface AtlasNode {
  question: Question;
  x: number;
  y: number;
  lon: number;
  lat: number;
  r: number;
  color: string;
  areaName: string;
  levelIndex: number;
  chainCount: number;
}

interface AreaAnchor {
  id: string;
  name: string;
  x: number;
  y: number;
  lon: number;
  lat: number;
  color: string;
}

interface GlobeProjection {
  x: number;
  y: number;
  scale: number;
  depth: number;
}

function hashString(input: string): number {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hashUnit(input: string, salt: string): number {
  return (hashString(`${salt}:${input}`) % 10000) / 10000;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function screenToWorld(point: Point, pan: Point, zoom: number): Point {
  return {
    x: (point.x - pan.x) / zoom,
    y: (point.y - pan.y) / zoom,
  };
}

function projectGlobe(lon: number, lat: number, rotation: number, radius: number): GlobeProjection {
  const rotatedLon = lon + rotation;
  const cosLat = Math.cos(lat);
  const x3 = cosLat * Math.cos(rotatedLon);
  const z3 = cosLat * Math.sin(rotatedLon);
  const y3 = Math.sin(lat);
  const front = (z3 + 1) / 2;
  return {
    x: CENTER_X + x3 * radius,
    y: CENTER_Y - y3 * radius * 0.9,
    scale: 0.62 + front * 0.58,
    depth: z3,
  };
}

function questionMatches(q: Question, query: string): boolean {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return true;
  const haystack = [
    q.title,
    q.question,
    q.topic,
    q.zone,
    q.competency_area,
    q.track,
    q.level,
  ].join(" ").toLowerCase();
  return normalized.split(/\s+/).every((term) => haystack.includes(term));
}

function formatTrack(track: string) {
  return track === "tinyml" ? "TinyML" : track.charAt(0).toUpperCase() + track.slice(1);
}

export default function ExplorePage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const asideRef = useRef<HTMLElement>(null);
  const dragRef = useRef<{ active: boolean; moved: boolean; last: Point }>({
    active: false,
    moved: false,
    last: { x: 0, y: 0 },
  });

  const [canvasSize, setCanvasSize] = useState({ width: 960, height: 620 });
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(0.55);
  const [rotation, setRotation] = useState(0);
  const [selectedTrack, setSelectedTrack] = useState<string>("all");
  const [selectedLevel, setSelectedLevel] = useState<string>("all");
  const [selectedArea, setSelectedArea] = useState<string>("all");
  const [query, setQuery] = useState("");
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const questions = useMemo(() => getQuestions(), []);
  const tracks = useMemo(() => getTracks(), []);
  const levels = LEVELS.map((level) => level.id);
  const areas = useMemo(() => getAreas(), []);
  const areaById = useMemo(() => new Map(areas.map((area) => [area.id, area])), [areas]);

  const areaAnchors = useMemo<AreaAnchor[]>(() => {
    return areas.map((area, index) => {
      const angle = (index / areas.length) * Math.PI * 2 - Math.PI / 2;
      const style = getAreaStyle(area.id);
      const lat = (hashUnit(area.id, "area-lat") - 0.5) * 1.05;
      const projection = projectGlobe(angle, lat, 0, 390);
      return {
        id: area.id,
        name: area.name,
        x: projection.x,
        y: projection.y,
        lon: angle,
        lat,
        color: style.primary,
      };
    });
  }, [areas]);

  const anchorByArea = useMemo(
    () => new Map(areaAnchors.map((anchor) => [anchor.id, anchor])),
    [areaAnchors],
  );

  const nodes = useMemo<AtlasNode[]>(() => {
    return questions.map((question) => {
      const anchor = anchorByArea.get(question.competency_area) ?? {
        x: CENTER_X,
        y: CENTER_Y,
        color: "#6b7280",
        name: question.competency_area,
        lon: 0,
        lat: 0,
      };
      const levelIndex = Math.max(0, levels.indexOf(question.level));
      const topicAngle = hashUnit(question.topic, "topic-angle") * Math.PI * 2;
      const zoneAngle = hashUnit(question.zone, "zone-angle") * Math.PI * 2;
      const topicRadius = 58 + hashUnit(question.topic, "topic-radius") * 118;
      const levelOffset = (levelIndex - 2.5) * 16;
      const jitter = 28 + hashUnit(question.id, "jitter-radius") * 38;
      const jitterAngle = hashUnit(question.id, "jitter-angle") * Math.PI * 2;
      const chainCount = question.chain_ids?.length ?? 0;
      const lon = anchor.lon + (hashUnit(question.topic, "topic-lon") - 0.5) * 0.7 + (hashUnit(question.id, "node-lon") - 0.5) * 0.12;
      const lat = clamp(
        anchor.lat + (hashUnit(question.zone, "zone-lat") - 0.5) * 0.55 + (levelIndex - 2.5) * 0.045 + (hashUnit(question.id, "node-lat") - 0.5) * 0.1,
        -1.08,
        1.08,
      );

      return {
        question,
        x:
          anchor.x +
          Math.cos(topicAngle) * topicRadius +
          Math.cos(zoneAngle) * levelOffset +
          Math.cos(jitterAngle) * jitter,
        y:
          anchor.y +
          Math.sin(topicAngle) * topicRadius * 0.7 +
          Math.sin(zoneAngle) * levelOffset +
          Math.sin(jitterAngle) * jitter,
        lon,
        lat,
        r: chainCount > 0 ? 3.4 : 2.6,
        color: anchor.color,
        areaName: areaById.get(question.competency_area)?.name ?? question.competency_area,
        levelIndex,
        chainCount,
      };
    });
  }, [anchorByArea, areaById, levels, questions]);

  const filteredNodes = useMemo(() => {
    return nodes.filter(({ question }) => {
      if (selectedTrack !== "all" && question.track !== selectedTrack) return false;
      if (selectedLevel !== "all" && question.level !== selectedLevel) return false;
      if (selectedArea !== "all" && question.competency_area !== selectedArea) return false;
      return questionMatches(question, query);
    });
  }, [nodes, query, selectedArea, selectedLevel, selectedTrack]);

  const nodeById = useMemo(
    () => new Map(nodes.map((node) => [node.question.id, node])),
    [nodes],
  );
  const hoveredNode = hoveredId ? nodeById.get(hoveredId) ?? null : null;
  const selectedNode = selectedId ? nodeById.get(selectedId) ?? null : null;
  const hasActiveFilters =
    query.trim().length > 0 ||
    selectedTrack !== "all" ||
    selectedLevel !== "all" ||
    selectedArea !== "all";

  const relatedNodes = useMemo(() => {
    if (!selectedNode) return [];
    const selected = selectedNode.question;
    return nodes
      .filter(({ question }) => {
        if (question.id === selected.id) return false;
        const sharesChain = selected.chain_ids?.some((id) => question.chain_ids?.includes(id));
        return sharesChain || question.topic === selected.topic;
      })
      .slice(0, 12);
  }, [nodes, selectedNode]);

  const panelMatches = useMemo(() => {
    return [...filteredNodes]
      .sort((a, b) => {
        if (!hasActiveFilters) {
          const levelDelta = b.levelIndex - a.levelIndex;
          if (levelDelta !== 0) return levelDelta;
        }
        const chainDelta = b.chainCount - a.chainCount;
        if (chainDelta !== 0) return chainDelta;
        const areaDelta = a.areaName.localeCompare(b.areaName);
        if (areaDelta !== 0) return areaDelta;
        return a.question.title.localeCompare(b.question.title);
      })
      .slice(0, 10);
  }, [filteredNodes, hasActiveFilters]);

  const globeRadius = Math.min(WORLD_WIDTH, WORLD_HEIGHT) * 0.47;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;

    const fit = () => {
      const rect = el.getBoundingClientRect();
      const width = Math.max(320, Math.floor(rect.width));
      const height = Math.max(360, Math.floor(rect.height));
      const nextZoom = clamp(Math.min(width / WORLD_WIDTH, height / WORLD_HEIGHT) * 0.94, MIN_ZOOM, 1.1);
      setCanvasSize({ width, height });
      setZoom(nextZoom);
      setPan({
        x: (width - WORLD_WIDTH * nextZoom) / 2,
        y: (height - WORLD_HEIGHT * nextZoom) / 2,
      });
    };

    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let frame = 0;
    let lastTick = 0;
    const tick = (time: number) => {
      if (time - lastTick > 70) {
        setRotation((value) => (value + 0.01) % (Math.PI * 2));
        lastTick = time;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvasSize.width * dpr);
    canvas.height = Math.floor(canvasSize.height * dpr);
    canvas.style.width = `${canvasSize.width}px`;
    canvas.style.height = `${canvasSize.height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    const globeGradient = ctx.createRadialGradient(
      CENTER_X - globeRadius * 0.25,
      CENTER_Y - globeRadius * 0.22,
      globeRadius * 0.08,
      CENTER_X,
      CENTER_Y,
      globeRadius,
    );
    globeGradient.addColorStop(0, "rgba(59,130,246,0.09)");
    globeGradient.addColorStop(0.62, "rgba(15,23,42,0.015)");
    globeGradient.addColorStop(1, "rgba(15,23,42,0.055)");

    ctx.beginPath();
    ctx.ellipse(CENTER_X, CENTER_Y, globeRadius, globeRadius * 0.9, 0, 0, Math.PI * 2);
    ctx.fillStyle = globeGradient;
    ctx.fill();
    ctx.strokeStyle = "rgba(100,116,139,0.18)";
    ctx.lineWidth = 1.2 / zoom;
    ctx.stroke();

    // Longitudes/latitudes make the map read as a planet instead of a Venn diagram.
    ctx.strokeStyle = "rgba(100,116,139,0.10)";
    ctx.lineWidth = 0.8 / zoom;
    for (const lat of [-0.9, -0.45, 0, 0.45, 0.9]) {
      ctx.beginPath();
      ctx.ellipse(
        CENTER_X,
        CENTER_Y - Math.sin(lat) * globeRadius * 0.9,
        Math.cos(lat) * globeRadius,
        Math.cos(lat) * globeRadius * 0.13,
        0,
        0,
        Math.PI * 2,
      );
      ctx.stroke();
    }
    for (let i = 0; i < 8; i++) {
      const phase = (i / 8) * Math.PI;
      ctx.beginPath();
      ctx.ellipse(CENTER_X, CENTER_Y, Math.abs(Math.cos(phase + rotation)) * globeRadius, globeRadius * 0.9, 0, 0, Math.PI * 2);
      ctx.stroke();
    }

    const projectedAnchors = areaAnchors.map((anchor) => ({
      anchor,
      projection: projectGlobe(anchor.lon, anchor.lat, rotation, globeRadius),
    }));

    ctx.lineWidth = 0.8 / zoom;
    for (let i = 0; i < projectedAnchors.length; i++) {
      const a = projectedAnchors[i];
      const b = projectedAnchors[(i + 1) % projectedAnchors.length];
      const c = projectedAnchors[(i + 4) % projectedAnchors.length];
      for (const target of [b, c]) {
        const alpha = Math.max(0.05, Math.min(0.18, (a.projection.depth + target.projection.depth + 2) / 18));
        ctx.beginPath();
        ctx.moveTo(a.projection.x, a.projection.y);
        ctx.lineTo(target.projection.x, target.projection.y);
        ctx.strokeStyle = `rgba(100,116,139,${alpha})`;
        ctx.stroke();
      }
    }

    const drawFlightRoute = (
      source: AtlasNode,
      target: AtlasNode,
      {
        alpha = 0.18,
        color = "59,130,246",
        width = 1,
        dashed = false,
      }: { alpha?: number; color?: string; width?: number; dashed?: boolean } = {},
    ) => {
      const a = projectGlobe(source.lon, source.lat, rotation, globeRadius);
      const b = projectGlobe(target.lon, target.lat, rotation, globeRadius);
      if (a.depth < -0.78 && b.depth < -0.78) return;

      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const lift = clamp(distance * 0.24, 18, 96);
      const side = Math.sign(Math.sin(target.lon - source.lon)) || 1;
      const cx = (a.x + b.x) / 2 + (-dy / distance) * lift * side;
      const cy = (a.y + b.y) / 2 + (dx / distance) * lift * side;
      const visibility = Math.max(0.2, (a.scale + b.scale) / 2);

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.quadraticCurveTo(cx, cy, b.x, b.y);
      ctx.strokeStyle = `rgba(${color},${alpha * visibility})`;
      ctx.lineWidth = width / zoom;
      if (dashed) ctx.setLineDash([5 / zoom, 5 / zoom]);
      ctx.stroke();
      ctx.restore();
    };

    const routeNodes = filteredNodes.length <= 220 ? filteredNodes : panelMatches;
    const drawChainRoutes = (candidates: AtlasNode[]) => {
      const byChain = new Map<string, AtlasNode[]>();
      for (const node of candidates) {
        for (const chainId of node.question.chain_ids ?? []) {
          const group = byChain.get(chainId) ?? [];
          group.push(node);
          byChain.set(chainId, group);
        }
      }

      for (const [chainId, group] of Array.from(byChain.entries())) {
        const ordered = [...group]
          .sort((a, b) =>
            (a.question.chain_positions?.[chainId] ?? 0) -
            (b.question.chain_positions?.[chainId] ?? 0),
          )
          .slice(0, 8);
        for (let i = 1; i < ordered.length; i++) {
          drawFlightRoute(ordered[i - 1], ordered[i], {
            alpha: 0.58,
            color: "245,158,11",
            width: 2,
          });
        }
      }
    };

    if (selectedNode) {
      for (const node of relatedNodes) {
        const sharesChain = selectedNode.question.chain_ids?.some((id) => node.question.chain_ids?.includes(id));
        drawFlightRoute(selectedNode, node, {
          alpha: sharesChain ? 0.46 : 0.20,
          color: sharesChain ? "245,158,11" : "59,130,246",
          width: sharesChain ? 2 : 1.3,
          dashed: !sharesChain,
        });
      }
    } else if (filteredNodes.length <= 160) {
      drawChainRoutes(routeNodes);
      const ordered = [...filteredNodes].sort((a, b) =>
        a.question.topic.localeCompare(b.question.topic) || a.lon - b.lon || a.lat - b.lat,
      );
      for (let i = 1; i < ordered.length; i++) {
        drawFlightRoute(ordered[i - 1], ordered[i], {
          alpha: 0.22,
          color: "59,130,246",
          width: 1.2,
          dashed: true,
        });
      }

      const byTopic = new Map<string, AtlasNode[]>();
      for (const node of filteredNodes) {
        const group = byTopic.get(node.question.topic) ?? [];
        if (group.length < 6) group.push(node);
        byTopic.set(node.question.topic, group);
      }
      for (const group of Array.from(byTopic.values())) {
        for (let i = 1; i < group.length; i++) {
          drawFlightRoute(group[i - 1], group[i], {
            alpha: 0.30,
            color: "59,130,246",
            width: 1.25,
            dashed: true,
          });
        }
      }
    } else {
      drawChainRoutes(routeNodes);
      for (let i = 1; i < panelMatches.length; i++) {
        drawFlightRoute(panelMatches[i - 1], panelMatches[i], {
          alpha: 0.18,
          color: "59,130,246",
          width: 1,
          dashed: true,
        });
      }
    }

    for (const { anchor, projection } of projectedAnchors) {
      const isActive = selectedArea === "all" || selectedArea === anchor.id;
      if (projection.depth < -0.7) continue;
      ctx.beginPath();
      ctx.arc(projection.x, projection.y, 4.5 * projection.scale, 0, Math.PI * 2);
      ctx.fillStyle = isActive ? anchor.color : "rgba(148,163,184,0.45)";
      ctx.globalAlpha = isActive ? 0.7 : 0.35;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    for (const node of filteredNodes) {
      const projection = projectGlobe(node.lon, node.lat, rotation, globeRadius);
      const screenX = projection.x * zoom + pan.x;
      const screenY = projection.y * zoom + pan.y;
      if (
        screenX < -20 ||
        screenY < -20 ||
        screenX > canvasSize.width + 20 ||
        screenY > canvasSize.height + 20
      ) {
        continue;
      }
      const isSelected = node.question.id === selectedId;
      const isHovered = node.question.id === hoveredId;
      const dim = selectedNode && !isSelected && !relatedNodes.some((r) => r.question.id === node.question.id);
      const radius = (isSelected ? 7.5 : isHovered ? 6.2 : node.r + node.levelIndex * 0.18) * projection.scale;
      const depthAlpha = projection.depth < -0.45 ? 0.22 : projection.depth < 0 ? 0.45 : 0.9;

      ctx.beginPath();
      ctx.arc(projection.x, projection.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = dim ? `${node.color}45` : node.color;
      ctx.globalAlpha = dim ? Math.min(0.35, depthAlpha) : depthAlpha;
      ctx.fill();
      ctx.globalAlpha = 1;

      if (node.chainCount > 0 || isHovered || isSelected) {
        ctx.strokeStyle = isSelected ? "#ffffff" : "rgba(255,255,255,0.65)";
        ctx.lineWidth = isSelected ? 2.5 / zoom : 1.25 / zoom;
        ctx.stroke();
      }
    }

    ctx.restore();
  }, [
    areaAnchors,
    canvasSize,
    filteredNodes,
    globeRadius,
    hoveredId,
    pan,
    panelMatches,
    relatedNodes,
    rotation,
    selectedArea,
    selectedId,
    selectedNode,
    zoom,
  ]);

  const resetView = () => {
    const nextZoom = clamp(
      Math.min(canvasSize.width / WORLD_WIDTH, canvasSize.height / WORLD_HEIGHT) * 0.94,
      MIN_ZOOM,
      1.1,
    );
    setZoom(nextZoom);
    setPan({
      x: (canvasSize.width - WORLD_WIDTH * nextZoom) / 2,
      y: (canvasSize.height - WORLD_HEIGHT * nextZoom) / 2,
    });
  };

  const findNodeAt = (point: Point) => {
    const world = screenToWorld(point, pan, zoom);
    let best: AtlasNode | null = null;
    let bestDistance = Infinity;

    for (const node of filteredNodes) {
      const projection = projectGlobe(node.lon, node.lat, rotation, globeRadius);
      const dx = projection.x - world.x;
      const dy = projection.y - world.y;
      const distance = dx * dx + dy * dy;
      const relaxedHitRadius = filteredNodes.length <= 80 ? 26 : HIT_RADIUS;
      const threshold = Math.pow((relaxedHitRadius + node.r * projection.scale) / zoom, 2);
      if (distance < threshold && distance < bestDistance) {
        best = node;
        bestDistance = distance;
      }
    }

    return best;
  };

  const zoomAt = (point: Point, deltaY: number) => {
    const worldBefore = screenToWorld(point, pan, zoom);
    const nextZoom = clamp(zoom * (deltaY > 0 ? 0.88 : 1.14), MIN_ZOOM, MAX_ZOOM);
    setZoom(nextZoom);
    setPan({
      x: point.x - worldBefore.x * nextZoom,
      y: point.y - worldBefore.y * nextZoom,
    });
  };

  const eventPoint = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      zoomAt({ x: event.clientX - rect.left, y: event.clientY - rect.top }, event.deltaY);
    };
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [pan, zoom]);

  useEffect(() => {
    asideRef.current?.scrollTo({ top: 0 });
  }, [selectedId]);

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = eventPoint(event);
    dragRef.current = { active: true, moved: false, last: point };
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const point = eventPoint(event);
    const drag = dragRef.current;

    if (drag.active) {
      const dx = point.x - drag.last.x;
      const dy = point.y - drag.last.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
      if (drag.moved) {
        setPan((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
      }
      drag.last = point;
      return;
    }

    setHoveredId(findNodeAt(point)?.question.id ?? null);
  };

  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    const point = eventPoint(event);
    const hit = findNodeAt(point);
    if (!drag.moved && hit) {
      setSelectedId(hit.question.id);
      setHoveredId(hit.question.id);
    }
    dragRef.current = { active: false, moved: false, last: point };
  };

  return (
    <div className="bg-background overflow-auto lg:overflow-hidden lg:h-[calc(100dvh-14.5rem)] lg:min-h-[620px]">
      <div className="h-full flex flex-col">
        <header className="border-b border-border bg-background/95 px-4 sm:px-6 py-3">
          <div className="max-w-7xl mx-auto">
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-sm text-textTertiary hover:text-textSecondary transition-colors mb-2"
            >
              <ArrowLeft className="w-3.5 h-3.5" /> Back to Vault
            </Link>
            <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
              <div>
                <div className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-[0.16em] font-semibold text-accentBlue px-2.5 py-1 rounded-full border border-accentBlue/30 bg-accentBlue/5 mb-2">
                  <Sparkles className="w-3 h-3" />
                  Question planet preview
                </div>
                <h1 className="text-2xl sm:text-3xl font-extrabold text-textPrimary tracking-tight">
                  Explore the StaffML Vault
                </h1>
                <p className="text-sm text-textSecondary mt-1 max-w-2xl leading-relaxed">
                  Use this as a discovery planet: search for a concept, narrow by track or level,
                  then follow connected question routes into practice. Orange flight paths show
                  question chains; blue routes show nearby topic neighborhoods.
                </p>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center shrink-0">
                <Metric label="showing" value={filteredNodes.length.toLocaleString()} />
                <Metric label="total" value={questions.length.toLocaleString()} />
                <Metric label="chains" value={questions.filter((q) => q.chain_ids?.length).length.toLocaleString()} />
              </div>
            </div>
          </div>
        </header>

        <section className="border-b border-borderSubtle bg-surface/40 px-4 sm:px-6 py-2.5">
          <div className="max-w-7xl mx-auto flex flex-col xl:flex-row gap-3 xl:items-center">
            <div className="relative flex-1 min-w-[220px]">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-textMuted" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              placeholder="Search the planet: kv cache, quantization, drift, tensor parallel..."
                className="w-full pl-9 pr-9 py-2 rounded-lg border border-border bg-background text-sm text-textPrimary placeholder:text-textTertiary focus:outline-none focus:border-borderHighlight"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  aria-label="Clear search"
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-textTertiary hover:text-textPrimary"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            <FilterSelect label="Track" value={selectedTrack} onChange={setSelectedTrack}>
              <option value="all">All tracks</option>
              {tracks.map((track) => (
                <option key={track} value={track}>{formatTrack(track)}</option>
              ))}
            </FilterSelect>

            <FilterSelect label="Level" value={selectedLevel} onChange={setSelectedLevel}>
              <option value="all">All levels</option>
              {levels.map((level) => (
                <option key={level} value={level}>{level}</option>
              ))}
            </FilterSelect>

            <FilterSelect label="Area" value={selectedArea} onChange={setSelectedArea}>
              <option value="all">All areas</option>
              {areas.map((area) => (
                <option key={area.id} value={area.id}>{area.name}</option>
              ))}
            </FilterSelect>

            <button
              type="button"
              onClick={resetView}
              className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-border bg-background text-sm text-textSecondary hover:text-textPrimary hover:bg-surface transition-colors"
            >
              <LocateFixed className="w-4 h-4" />
              Reset
            </button>
          </div>
        </section>

        <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-hidden">
          <div ref={wrapRef} className="relative flex-1 min-h-[420px] lg:min-h-0 overflow-hidden">
            <canvas
              ref={canvasRef}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerLeave={() => {
                dragRef.current.active = false;
                setHoveredId(null);
              }}
              onPointerUp={onPointerUp}
              className={clsx(
                "block w-full h-full touch-none cursor-grab bg-[radial-gradient(circle_at_center,var(--surface)_0,transparent_64%)]",
                dragRef.current.active && "cursor-grabbing",
              )}
              aria-label="Interactive question planet of StaffML Vault questions"
            />
            <div className="pointer-events-none absolute left-4 bottom-4 rounded-lg border border-borderSubtle bg-background/85 backdrop-blur px-3 py-2 text-[11px] text-textTertiary">
              Drag to pan - scroll to zoom - click a dot to inspect a question
            </div>
            {hoveredNode && (
              <div className="pointer-events-none absolute right-4 top-4 max-w-xs rounded-xl border border-border bg-background/95 shadow-xl p-3">
                <div className="flex items-center gap-1.5 mb-1">
                  <span className="text-[10px] font-mono text-textTertiary uppercase">{hoveredNode.question.level}</span>
                  <span className="text-[10px] font-mono text-textTertiary uppercase">{hoveredNode.question.track}</span>
                </div>
                <p className="text-sm font-semibold text-textPrimary leading-snug">{hoveredNode.question.title}</p>
                <p className="text-[11px] text-textTertiary mt-1">{hoveredNode.areaName} - {hoveredNode.question.topic}</p>
              </div>
            )}
          </div>

          <aside
            ref={asideRef}
            className="lg:w-[380px] min-h-0 border-t lg:border-t-0 lg:border-l border-border bg-background overflow-auto"
          >
            {selectedNode ? (
              <QuestionPanel
                node={selectedNode}
                related={relatedNodes}
                onSelect={setSelectedId}
                onClose={() => setSelectedId(null)}
              />
            ) : (
              <EmptyPanel
                areaAnchors={areaAnchors}
                matches={panelMatches}
                hasActiveFilters={hasActiveFilters}
                onSelect={setSelectedId}
                onReset={() => {
                  setQuery("");
                  setSelectedTrack("all");
                  setSelectedLevel("all");
                  setSelectedArea("all");
                }}
              />
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-borderSubtle bg-surface/60 px-3 py-2 min-w-[86px]">
      <div className="text-lg font-bold font-mono text-textPrimary leading-none">{value}</div>
      <div className="text-[10px] uppercase tracking-wide text-textTertiary mt-1">{label}</div>
    </div>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  children,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
}) {
  return (
    <label className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-wide text-textTertiary">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="normal-case tracking-normal font-sans min-w-[128px] rounded-lg border border-border bg-background px-3 py-2 text-sm text-textSecondary focus:outline-none focus:border-borderHighlight"
      >
        {children}
      </select>
    </label>
  );
}

function EmptyPanel({
  areaAnchors,
  matches,
  hasActiveFilters,
  onSelect,
  onReset,
}: {
  areaAnchors: AreaAnchor[];
  matches: AtlasNode[];
  hasActiveFilters: boolean;
  onSelect: (id: string) => void;
  onReset: () => void;
}) {
  return (
    <div className="p-5">
      <div className="w-10 h-10 rounded-xl border border-accentBlue/30 bg-accentBlue/10 flex items-center justify-center mb-4">
        <Network className="w-5 h-5 text-accentBlue" />
      </div>
      <h2 className="text-lg font-bold text-textPrimary mb-2">Question planet</h2>
      <p className="text-sm text-textSecondary leading-relaxed mb-5">
        Each dot is a Vault question. Orange flight paths connect chain steps;
        blue routes connect local topic neighborhoods. Use the planet to choose
        a path, then use the cards for the exact question title.
      </p>

      {!hasActiveFilters && (
        <section className="mb-6 rounded-xl border border-accentBlue/20 bg-accentBlue/5 p-4">
          <h3 className="text-sm font-bold text-textPrimary mb-3">How to use this</h3>
          <div className="space-y-3">
            <HowToStep
              step="1"
              title="Search for a concept"
              text="Try kv cache, quantization, drift, batching, memory, or tensor parallel."
            />
            <HowToStep
              step="2"
              title="Narrow the interview context"
              text="Use track and level filters to turn the full Vault into a small candidate set."
            />
            <HowToStep
              step="3"
              title="Pick a path"
              text="Click a card or dot, practice it, then use connected questions to move easier, harder, or sideways."
            />
          </div>
        </section>
      )}

      {matches.length > 0 ? (
        <section className="mb-6">
          <h3 className="text-[11px] font-mono uppercase tracking-wide text-textTertiary mb-2">
            {hasActiveFilters ? "Matching questions" : "Suggested starting points"}
          </h3>
          <div className="space-y-2">
            {matches.map((node) => (
              <button
                key={node.question.id}
                type="button"
                onClick={() => onSelect(node.question.id)}
                className="w-full text-left p-3 rounded-lg border border-borderSubtle bg-surface/50 hover:bg-surface hover:border-borderHighlight transition-colors"
              >
                <div className="flex items-center gap-1.5 mb-1 text-[10px] font-mono uppercase text-textTertiary">
                  <span>{node.question.level}</span>
                  <span>{node.question.track}</span>
                  <span>{node.areaName}</span>
                </div>
                <p className="text-sm font-semibold text-textPrimary leading-snug">{node.question.title}</p>
              </button>
            ))}
          </div>
        </section>
      ) : (
        <section className="mb-6 rounded-xl border border-borderSubtle bg-surface/60 p-4">
          <h3 className="text-sm font-bold text-textPrimary mb-1">No matching questions</h3>
          <p className="text-sm text-textSecondary leading-relaxed mb-3">
            Try a broader phrase, clear a track or level filter, or return to the full planet.
          </p>
          <button
            type="button"
            onClick={onReset}
            className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-border bg-background text-sm text-textSecondary hover:text-textPrimary hover:bg-surface transition-colors"
          >
            <LocateFixed className="w-4 h-4" />
            Reset filters
          </button>
        </section>
      )}

      <section>
        <h3 className="text-[11px] font-mono uppercase tracking-wide text-textTertiary mb-2">
          Area legend
        </h3>
        <div className="space-y-2">
          {areaAnchors.map((anchor) => (
            <div key={anchor.id} className="flex items-center gap-2 text-sm text-textSecondary">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: anchor.color }} />
              {anchor.name}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function HowToStep({ step, title, text }: { step: string; title: string; text: string }) {
  return (
    <div className="flex gap-3">
      <div className="w-6 h-6 rounded-full bg-accentBlue text-white flex items-center justify-center text-[11px] font-bold shrink-0">
        {step}
      </div>
      <div>
        <div className="text-sm font-semibold text-textPrimary">{title}</div>
        <p className="text-[12px] text-textSecondary leading-relaxed">{text}</p>
      </div>
    </div>
  );
}

function QuestionPanel({
  node,
  related,
  onSelect,
  onClose,
}: {
  node: AtlasNode;
  related: AtlasNode[];
  onSelect: (id: string) => void;
  onClose: () => void;
}) {
  const q = node.question;
  return (
    <div className="p-5">
      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="inline-flex items-center gap-2 text-[10px] font-mono uppercase tracking-wide text-textTertiary">
          <span className="px-2 py-1 rounded border border-border bg-surface">{q.level}</span>
          <span className="px-2 py-1 rounded border border-border bg-surface">{q.track}</span>
          {node.chainCount > 0 && (
            <span className="px-2 py-1 rounded border border-accentGreen/30 bg-accentGreen/10 text-accentGreen">
              chain
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close question panel"
          className="text-textTertiary hover:text-textPrimary"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <h2 className="text-xl font-extrabold text-textPrimary tracking-tight leading-tight mb-3">
        {q.title}
      </h2>
      <p className="text-sm text-textSecondary leading-relaxed mb-4">
        {q.question || "Open the practice view to see the full scenario and solution."}
      </p>

      <div className="grid grid-cols-2 gap-2 mb-5">
        <InfoTile icon={<Layers className="w-3.5 h-3.5" />} label="Area" value={node.areaName} />
        <InfoTile icon={<Network className="w-3.5 h-3.5" />} label="Topic" value={q.topic} />
        <InfoTile icon={<Sparkles className="w-3.5 h-3.5" />} label="Zone" value={q.zone} />
        <InfoTile icon={<LocateFixed className="w-3.5 h-3.5" />} label="Phase" value={q.phase ?? "mixed"} />
      </div>

      <Link
        href={`/practice?q=${q.id}`}
        className="inline-flex items-center justify-center gap-1.5 w-full px-4 py-2.5 rounded-lg bg-accentBlue text-white text-sm font-bold hover:opacity-90 transition-opacity mb-6"
      >
        Practice this question
        <ArrowRight className="w-4 h-4" />
      </Link>

      <section>
        <h3 className="text-[11px] font-mono uppercase tracking-wide text-textTertiary mb-2">
          Nearby questions
        </h3>
        {related.length > 0 ? (
          <div className="space-y-2">
            {related.map((item) => (
              <button
                key={item.question.id}
                type="button"
                onClick={() => onSelect(item.question.id)}
                className="w-full text-left p-3 rounded-lg border border-borderSubtle bg-surface/50 hover:bg-surface hover:border-borderHighlight transition-colors"
              >
                <div className="flex items-center gap-1.5 mb-1 text-[10px] font-mono uppercase text-textTertiary">
                  <span>{item.question.level}</span>
                  <span>{item.question.track}</span>
                </div>
                <p className="text-sm font-semibold text-textPrimary leading-snug">{item.question.title}</p>
              </button>
            ))}
          </div>
        ) : (
          <p className="text-sm text-textTertiary">
            No topic or chain neighbors are visible for this question yet.
          </p>
        )}
      </section>
    </div>
  );
}

function InfoTile({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-lg border border-borderSubtle bg-surface/50 p-3">
      <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wide text-textTertiary mb-1">
        {icon}
        {label}
      </div>
      <div className="text-sm font-semibold text-textPrimary capitalize leading-snug">{value}</div>
    </div>
  );
}
