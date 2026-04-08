"use client";

import { useMemo, useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { ArrowLeft, Search, X } from "lucide-react";
import {
  blocks,
  rowLabels,
  elements,
  elMap,
  compounds,
  type Block,
  type BlockKey,
  type Element,
  type Compound,
  type FormulaToken,
} from "@/data/periodicTable";
import styles from "./PeriodicTable.module.css";

// ── Helpers ──────────────────────────────────────────────────────────────
const blockEntries = Object.entries(blocks) as Array<[BlockKey, Block]>;

function compoundSyms(compound: Compound): Set<string> {
  const set = new Set<string>();
  compound.formula.forEach((t) => {
    if (t.kind === "sym" && elMap[t.sym]) set.add(t.sym);
  });
  return set;
}

// Pre-compute compound ↔ symbol maps once.
const compoundSymCache: Map<Compound, Set<string>> = new Map();
const symToCompounds: Record<string, Compound[]> = {};
let TOTAL_COMPOUNDS = 0;
compounds.forEach((section) =>
  section.items.forEach((c) => {
    TOTAL_COMPOUNDS += 1;
    const syms = compoundSyms(c);
    compoundSymCache.set(c, syms);
    syms.forEach((sym) => {
      if (!symToCompounds[sym]) symToCompounds[sym] = [];
      symToCompounds[sym].push(c);
    });
  }),
);

// ── Page ─────────────────────────────────────────────────────────────────
export default function FrameworkPage() {
  const [activeBlock, setActiveBlock] = useState<BlockKey | null>(null);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Element | null>(null);
  // Cross-ref state: clicking element highlights its compounds; clicking
  // a compound highlights its elements.
  const [highlightedCompounds, setHighlightedCompounds] = useState<Set<Compound> | null>(null);
  const [highlightedSyms, setHighlightedSyms] = useState<Set<string> | null>(null);

  const dimmedSet = useMemo(() => {
    const q = query.toLowerCase().trim();
    const dimmed = new Set<string>();
    elements.forEach((e) => {
      const blockMatch = !activeBlock || e.block === activeBlock;
      const searchMatch =
        !q || e.name.toLowerCase().includes(q) || e.sym.toLowerCase().includes(q);
      if (!(blockMatch && searchMatch)) dimmed.add(e.sym);
    });
    return dimmed;
  }, [activeBlock, query]);

  const clearXref = useCallback(() => {
    setHighlightedCompounds(null);
    setHighlightedSyms(null);
  }, []);

  const handleElementClick = useCallback((e: Element) => {
    // Match the original: highlight matching compound cards AND scroll the
    // first match into view (scroll happens behind the modal so the page is
    // already positioned correctly when the user closes the panel).
    const cards = symToCompounds[e.sym] || [];
    if (cards.length > 0) {
      setHighlightedCompounds(new Set(cards));
      setHighlightedSyms(null);
      const firstName = cards[0].name;
      requestAnimationFrame(() => {
        try {
          document
            .querySelector<HTMLElement>(`[data-pt-compound="${CSS.escape(firstName)}"]`)
            ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
        } catch {
          /* scroll is best-effort */
        }
      });
    }
    setSelected(e);
  }, []);

  const handleCompoundClick = useCallback(
    (c: Compound) => {
      const syms = compoundSymCache.get(c);
      if (!syms || syms.size === 0) return;
      setHighlightedCompounds(new Set([c]));
      setHighlightedSyms(syms);
      requestAnimationFrame(() => {
        document.getElementById("table-anchor")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      });
    },
    [],
  );

  // Esc closes overlay / clears xref.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSelected(null);
        clearXref();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [clearXref]);

  // Click anywhere outside an element / compound card / panel clears the
  // cross-reference highlight, matching the original page's behavior.
  const xrefIsActive = highlightedCompounds !== null || highlightedSyms !== null;
  useEffect(() => {
    if (!xrefIsActive) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      if (target.closest("[data-pt-element]")) return;
      if (target.closest("[data-pt-compound]")) return;
      if (target.closest("[data-pt-panel]")) return;
      clearXref();
    };
    document.addEventListener("click", handler);
    return () => document.removeEventListener("click", handler);
  }, [xrefIsActive, clearXref]);

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10">
        {/* Back link */}
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-textTertiary hover:text-textSecondary transition-colors mb-8"
        >
          <ArrowLeft className="w-3.5 h-3.5" /> Back to Vault
        </Link>

        {/* Hero */}
        <div className="text-center mb-10">
          <span className="inline-block text-[10px] uppercase tracking-[0.14em] font-semibold text-accentBlue px-2.5 py-1 rounded-full border border-accentBlue/30 bg-accentBlue/5 mb-4">
            v0.1 · {elements.length} Elements · {TOTAL_COMPOUNDS} Compounds
          </span>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-textPrimary tracking-tight mb-3">
            The <span className="text-accentBlue italic">Periodic Table</span> of Machine Learning Systems
          </h1>
          <p className="text-[15px] text-textSecondary leading-relaxed max-w-2xl mx-auto">
            Two fundamental axes — abstraction layer and information-processing role — organize ML
            concepts the way electron shells and valence organize chemistry.
          </p>
        </div>

        {/* Two axes explainer */}
        <section className="mb-6">
          <h2 className="text-base font-bold text-textPrimary mb-3">The Two Axes</h2>
          <div className="grid sm:grid-cols-2 gap-3">
            <div className="rounded-xl border border-borderSubtle bg-surface/60 p-4">
              <h3 className="text-sm font-bold text-textPrimary mb-1.5 flex items-center gap-2">
                <span className="text-accentBlue">↓</span> Rows: Abstraction Layer
              </h3>
              <p className="text-[12.5px] text-textSecondary mb-2 leading-relaxed">
                Like electron shells, each layer <strong className="text-textPrimary">builds on and contains</strong> the
                ones above. You can&apos;t have optimization without a model, or deployment without hardware.
              </p>
              <ol className="text-[11.5px] text-textTertiary leading-relaxed space-y-0.5">
                {rowLabels.map((label, i) => (
                  <li key={label}>
                    <span className="text-textPrimary font-semibold">
                      {i + 1}. {label}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
            <div className="rounded-xl border border-borderSubtle bg-surface/60 p-4">
              <h3 className="text-sm font-bold text-textPrimary mb-1.5 flex items-center gap-2">
                <span className="text-accentBlue">→</span> Columns: Information-Processing Role
              </h3>
              <p className="text-[12.5px] text-textSecondary mb-2 leading-relaxed">
                From computer architecture and systems theory — five roles that exist in{" "}
                <strong className="text-textPrimary">any</strong> information-processing system.
              </p>
              <ul className="text-[11.5px] text-textTertiary leading-relaxed space-y-0.5">
                {blockEntries.map(([key, b]) => (
                  <li key={key}>
                    <span className="font-semibold" style={{ color: b.color }}>
                      {b.name}
                    </span>
                    <span className="text-textTertiary"> — {b.sub}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* "Same column test" proof box */}
        <section className="mb-10 rounded-xl border-l-[3px] border-l-accentBlue border border-borderSubtle bg-surface/60 px-4 py-3.5">
          <p className="text-[12.5px] text-textSecondary mb-2 leading-relaxed">
            <strong className="text-textPrimary">The same-column test:</strong> Concepts in the
            same column genuinely behave alike.
          </p>
          {([
            ["Represent", blocks.R.color, "Tensor, Probability, Parameter, Embedding, Topology, Hidden State, Optimizer State, Caching, Checkpointing, SRAM, DRAM, Artifact Store — all hold and structure information across all 8 layers."],
            ["Compute", blocks.C.color, "Operator, Activation, Dense Dot, Convolution, Pooling, Attention, Routing, Quantization, Fusion, Tiling, Compilation — all transform inputs to outputs."],
            ["Communicate", blocks.X.color, "Chain Rule, Backprop, Tokenization, Skip/Res, Distillation, Weight Averaging, Pipelining, Sync, Prefetching, Interconnect, RPC Protocol — all move information between components without deciding what to do with it."],
            ["Control", blocks.K.color, "Objective, Constraint, Grad Descent, Search, Initialization, Masking, Scheduling, Regularization, Allocation, Arbiter, Load Balancer, Telemetry — all make decisions that govern system behavior."],
            ["Measure", blocks.M.color, "Entropy, Loss Function, Receptive Field, Info Density, Throughput, Energy, Latency — all observe without changing. Noble gases."],
          ] as const).map(([label, color, text]) => (
            <p key={label} className="text-[12px] text-textSecondary mb-1 last:mb-0 leading-relaxed">
              <strong style={{ color }}>{label}: </strong>
              <span className="text-textTertiary">{text}</span>
            </p>
          ))}
        </section>

        {/* Controls */}
        <div id="table-anchor" className="flex flex-col items-center gap-3 mb-4">
          <div className="relative w-full max-w-sm">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-textTertiary pointer-events-none" />
            <input
              type="text"
              placeholder="Search elements…"
              value={query}
              onChange={(ev) => setQuery(ev.target.value)}
              className="w-full pl-8 pr-3 py-2 text-[13px] bg-surface border border-border rounded-lg text-textPrimary placeholder:text-textTertiary focus:outline-none focus:border-accentBlue transition-colors"
            />
          </div>

          <div className="flex flex-wrap justify-center gap-1.5">
            {blockEntries.map(([key, b]) => {
              const isActive = activeBlock === key;
              return (
                <button
                  key={key}
                  onClick={() => setActiveBlock(isActive ? null : key)}
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium transition-colors border ${
                    isActive
                      ? "border-borderHighlight bg-surfaceHover text-textPrimary"
                      : "border-borderSubtle bg-surface/60 text-textTertiary hover:text-textSecondary"
                  }`}
                >
                  <span
                    className="w-2.5 h-2.5 rounded-sm shrink-0"
                    style={{ background: b.color }}
                  />
                  <span style={{ color: isActive ? b.color : undefined }}>{b.name}</span>
                  <span className="hidden sm:inline text-textMuted text-[10px]">— {b.sub}</span>
                </button>
              );
            })}
            {(activeBlock || query) && (
              <button
                onClick={() => {
                  setActiveBlock(null);
                  setQuery("");
                }}
                className="flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium text-textTertiary hover:text-textPrimary transition-colors"
              >
                <X className="w-3 h-3" /> Clear
              </button>
            )}
          </div>
        </div>

        {/* Periodic grid — framed in a card so the empty cells feel intentional */}
        <div className="rounded-2xl border border-borderSubtle bg-surface/40 px-2 sm:px-4 py-5 mb-2 overflow-x-auto">
          <PeriodicGrid
            dimmed={dimmedSet}
            highlightedSyms={highlightedSyms}
            onElementClick={handleElementClick}
          />
        </div>
        <p className="text-center text-[11px] text-textTertiary mt-2">
          Click any element for details · click a compound below to highlight its primitives
        </p>

        {/* Compounds */}
        <section id="compounds-anchor" className="mt-12">
          <h2 className="text-base font-bold text-textPrimary mb-2">Molecular ML (Compounds)</h2>
          <p className="text-[13px] text-textSecondary mb-4 leading-relaxed">
            Just as H₂O = hydrogen + oxygen, every ML system decomposes into primitives from the
            table above. Read each formula left to right — symbols are element codes, operators
            show how they bond.
          </p>

          <CompoundLegend />

          {compounds.map((section) => (
            <div key={section.title} className="mt-6">
              <h3 className="text-[11px] uppercase tracking-[0.08em] font-bold text-textTertiary border-b border-borderSubtle pb-1.5 mb-3">
                {section.title}
                {section.hint && (
                  <span className="ml-2 text-accentBlue normal-case font-normal tracking-normal text-[11px]">
                    {section.hint}
                  </span>
                )}
              </h3>
              <div className="grid gap-2.5 grid-cols-[repeat(auto-fill,minmax(280px,1fr))]">
                {section.items.map((c) => {
                  const isHighlighted = highlightedCompounds?.has(c);
                  const isDimmed = highlightedCompounds && !isHighlighted;
                  return (
                    <button
                      key={c.name}
                      data-pt-compound={c.name}
                      onClick={() => handleCompoundClick(c)}
                      className={`text-left rounded-xl border bg-surface/60 p-3 transition-all ${
                        isHighlighted
                          ? "border-accentBlue shadow-[0_0_14px_color-mix(in_srgb,var(--accent-blue)_25%,transparent)]"
                          : "border-borderSubtle hover:border-border"
                      } ${isDimmed ? "opacity-20" : ""}`}
                    >
                      <div className="text-[13px] font-bold text-accentBlue mb-1.5">{c.name}</div>
                      <FormulaRender tokens={c.formula} />
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </section>

        <footer className="text-center mt-16 pt-6 border-t border-borderSubtle text-[11px] text-textTertiary">
          A project of{" "}
          <a href="https://mlsysbook.ai" className="text-accentBlue hover:underline">
            Machine Learning Systems
          </a>{" "}
          — Harvard CS249r · Vijay Janapa Reddi
        </footer>
      </div>

      {/* Detail overlay */}
      {selected && (
        <ElementDetail element={selected} onClose={() => setSelected(null)} onBondClick={(sym) => {
          const target = elMap[sym];
          if (target) setSelected(target);
        }} />
      )}
    </div>
  );
}

// ── Periodic grid ────────────────────────────────────────────────────────
function PeriodicGrid({
  dimmed,
  highlightedSyms,
  onElementClick,
}: {
  dimmed: Set<string>;
  highlightedSyms: Set<string> | null;
  onElementClick: (e: Element) => void;
}) {
  return (
    <div className={styles.tableWrap}>
      <div className={styles.tableOuter}>
        {/* Y-axis labels */}
        <div className={styles.yLabels}>
          {rowLabels.map((label, i) => (
            <div key={label} className={styles.yLbl}>
              {i + 1}. {label}
            </div>
          ))}
        </div>

        <div>
          {/* Block headers */}
          <div className={styles.blockHeaders}>
            {blockEntries.map(([key, b]) => {
              const widthPx = b.cols.length * 58 + (b.cols.length - 1) * 3;
              return (
                <div
                  key={key}
                  className={styles.bh}
                  style={{
                    width: `${widthPx}px`,
                    background: `color-mix(in srgb, ${b.color} 18%, transparent)`,
                    color: b.color,
                  }}
                >
                  {b.name}
                </div>
              );
            })}
          </div>

          {/* Grid */}
          <div className={styles.grid}>
            {elements.map((e) => {
              const block = blocks[e.block];
              const isDimmed = dimmed.has(e.sym);
              const isXref = highlightedSyms?.has(e.sym);
              return (
                <button
                  key={`${e.row}-${e.col}-${e.sym}`}
                  type="button"
                  data-pt-element=""
                  className={`${styles.el} ${isDimmed ? styles.dimmed : ""} ${
                    isXref ? styles.xref : ""
                  }`}
                  style={
                    {
                      "--el-c": block.color,
                      gridRow: e.row,
                      gridColumn: e.col,
                    } as React.CSSProperties
                  }
                  onClick={() => onElementClick(e)}
                  title={`${e.name} — ${block.name}`}
                >
                  <span className={styles.elNum}>{e.num}</span>
                  <span className={styles.elSym}>{e.sym}</span>
                  <span className={styles.elName}>{e.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Compound legend ──────────────────────────────────────────────────────
function CompoundLegend() {
  const items = [
    ["→", "Sequential"],
    ["∥", "Parallel"],
    ["?", "Conditional"],
    ["⇌", "Adversarial"],
    ["↺", "Feedback Loop"],
    ["[ ]ᴺ", "Repeated Block"],
  ];
  return (
    <div className="flex flex-wrap justify-center gap-x-5 gap-y-1.5 px-3 py-2 rounded-lg border border-borderSubtle bg-surface/40">
      {items.map(([sym, label]) => (
        <div key={label} className="text-[11px] text-textTertiary flex items-center gap-1.5">
          <span className="font-mono font-bold text-accentBlue text-[13px]">{sym}</span>
          {label}
        </div>
      ))}
    </div>
  );
}

// ── Formula renderer ─────────────────────────────────────────────────────
function FormulaRender({ tokens }: { tokens: FormulaToken[] }) {
  return (
    <div className="text-[12px] font-mono leading-relaxed bg-background/60 px-2.5 py-2 rounded-md text-textTertiary break-words">
      {tokens.map((t, i) => {
        if (t.kind === "op") return <span key={i}>{t.text}</span>;
        const el = elMap[t.sym];
        const color = el ? blocks[el.block].color : undefined;
        return (
          <span key={i} className={`relative inline-block group/sym ${el ? "cursor-help" : ""}`}>
            <span
              className="font-semibold"
              style={{
                color: color || "var(--text-primary)",
                borderBottom: el
                  ? `1px dotted color-mix(in srgb, ${color} 50%, transparent)`
                  : undefined,
              }}
            >
              {t.sym}
              {t.sub && <sub className="text-[9px] opacity-80">{t.sub}</sub>}
            </span>
            {el && (
              <span className="pointer-events-none absolute left-1/2 -translate-x-1/2 bottom-full mb-1.5 whitespace-nowrap rounded-md border border-border bg-surfaceElevated px-2 py-1 text-[10px] font-sans font-medium text-textPrimary shadow-lg opacity-0 group-hover/sym:opacity-100 transition-opacity z-20">
                {el.name}
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}

// ── Element detail overlay ───────────────────────────────────────────────
// Pixel-port of the original .panel CSS from periodic-table/index.html.
// Sizes use arbitrary Tailwind values to match the original's rems/px
// exactly rather than rounding to nearest preset.
function ElementDetail({
  element,
  onClose,
  onBondClick,
}: {
  element: Element;
  onClose: () => void;
  onBondClick: (sym: string) => void;
}) {
  const block = blocks[element.block];
  const color = block.color;
  const layerName = rowLabels[element.row - 1];

  return (
    <div
      className="fixed inset-0 z-[100] bg-black/60 flex items-center justify-center p-4"
      onClick={onClose}
    >
      {/*
        Two-layer modal panel:
        - Outer wrapper (relative, non-scrolling) carries the absolute close
          button so it stays anchored in the visual top-right corner.
        - Inner panel (scrollable, max-h-[100dvh-2rem]) wraps the actual
          content so long detail panels never bleed off the viewport on
          landscape phones / short screens.
        Without this split, the close button and bottom rows became
        unreachable on small/landscape viewports.
      */}
      <div
        className="relative w-full max-w-[500px]"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-[0.7rem] right-[1rem] z-10 w-10 h-10 flex items-center justify-center bg-transparent border-0 text-textTertiary hover:text-textPrimary text-[1.4rem] leading-none cursor-pointer"
          aria-label="Close"
        >
          ×
        </button>
        <div
          data-pt-panel=""
          className="bg-surface border border-border rounded-[14px] p-[1.8rem] max-h-[calc(100dvh-2rem)] overflow-y-auto"
          style={{ animation: "ptPop 0.2s ease" }}
        >

        {/* Header */}
        <div className="flex items-end gap-[0.9rem] mb-4 pb-4 border-b border-border">
          <div
            className="w-[70px] h-[70px] rounded-[9px] flex flex-col items-center justify-center shrink-0 bg-background border-2"
            style={{ borderColor: color }}
          >
            <span className="text-[0.6rem] text-textTertiary">#{element.num}</span>
            <span className="text-[1.7rem] font-bold leading-none" style={{ color }}>
              {element.sym}
            </span>
          </div>
          <div className="min-w-0">
            <div className="text-[1.15rem] font-bold text-textPrimary leading-tight">
              {element.name}
            </div>
            <div
              className="text-[0.65rem] uppercase tracking-[0.08em] font-semibold mt-1"
              style={{ color }}
            >
              {block.name} · {layerName}
            </div>
          </div>
        </div>

        {/* Meta */}
        <div className="grid grid-cols-3 gap-[0.5rem] mb-[0.9rem]">
          <MetaCell label="Introduced" value={element.year} />
          <MetaCell label="Role" value={block.name} valueColor={color} />
          <MetaCell label="Layer" value={layerName} />
        </div>

        {/* Description */}
        <p className="text-[0.8rem] leading-[1.6] text-textTertiary mb-[0.9rem]">
          {element.desc}
        </p>

        {/* Why here */}
        <div className="bg-background rounded-[7px] px-[0.8rem] py-[0.6rem] text-[0.73rem] leading-[1.5] text-textTertiary mb-[0.9rem]">
          <strong className="text-textPrimary font-bold">Why this position: </strong>
          {element.whyHere}
        </div>

        {/* Bonds */}
        {element.bonds.length > 0 && (
          <div>
            <h4 className="text-[0.6rem] uppercase tracking-[0.07em] font-semibold text-textMuted mb-[0.3rem]">
              Bonds With
            </h4>
            <div className="flex flex-wrap gap-[0.3rem]">
              {element.bonds.map((sym) => {
                const target = elMap[sym];
                return (
                  <button
                    key={sym}
                    onClick={() => target && onBondClick(sym)}
                    disabled={!target}
                    className="bg-background border border-border rounded-[4px] px-[0.4rem] py-[0.12rem] text-[0.67rem] text-textSecondary hover:border-accentBlue hover:text-accentBlue transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {sym}
                    {target && ` — ${target.name}`}
                  </button>
                );
              })}
            </div>
          </div>
        )}
        </div>
      </div>
    </div>
  );
}

function MetaCell({
  label,
  value,
  valueColor,
}: {
  label: string;
  value: string;
  valueColor?: string;
}) {
  return (
    <div className="bg-background rounded-[6px] px-[0.6rem] py-[0.45rem]">
      <div className="text-[0.5rem] uppercase tracking-[0.07em] text-textMuted">{label}</div>
      <div
        className="text-[0.78rem] font-semibold mt-[2px]"
        style={valueColor ? { color: valueColor } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
