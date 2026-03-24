"use client";

import { useState, useCallback, lazy, Suspense } from "react";
import { Network, List, GitBranch, BarChart3, Terminal } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";
import ConceptBrowser from "@/components/ConceptBrowser";
import ConceptDetail from "@/components/ConceptDetail";
import TaxonomyCoverage from "@/components/TaxonomyCoverage";
import { type Concept, getConceptById, getTaxonomyStats } from "@/lib/taxonomy";

// Lazy-load the graph (heavy: sigma + graphology + WebGL)
const TaxonomyGraph = lazy(() => import("@/components/TaxonomyGraph"));

type Tab = "browse" | "graph" | "coverage";

const tabs: { key: Tab; label: string; icon: typeof List }[] = [
  { key: "browse", label: "Browse", icon: List },
  { key: "graph", label: "Graph", icon: GitBranch },
  { key: "coverage", label: "Coverage", icon: BarChart3 },
];

export default function TaxonomyPage() {
  const [activeTab, setActiveTab] = useState<Tab>("browse");
  const [selectedConcept, setSelectedConcept] = useState<Concept | null>(null);
  const stats = getTaxonomyStats();

  const handleSelect = useCallback((concept: Concept) => {
    setSelectedConcept(concept);
  }, []);

  const handleNavigate = useCallback((id: string) => {
    const concept = getConceptById(id);
    if (concept) setSelectedConcept(concept);
  }, []);

  const handleClose = useCallback(() => {
    setSelectedConcept(null);
  }, []);

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-56px)]">
      {/* Header */}
      <div className="px-6 py-5 border-b border-border">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Network className="w-7 h-7 text-accentBlue" />
              <div>
                <h1 className="text-2xl font-extrabold text-white tracking-tight">
                  Taxonomy Explorer
                </h1>
                <p className="text-xs text-textSecondary">
                  {stats.totalConcepts} concepts &middot; {stats.totalEdges} edges &middot;{" "}
                  {stats.totalQuestions} questions &middot; v{stats.version}
                </p>
              </div>
            </div>

            {/* Quick stats */}
            <div className="hidden md:flex items-center gap-6">
              <StatPill label="Concepts" value={stats.totalConcepts} />
              <StatPill label="Edges" value={stats.totalEdges} />
              <StatPill label="Questions" value={stats.totalQuestions} />
            </div>
          </div>

          {/* Tab bar */}
          <div className="flex items-center gap-1">
            {tabs.map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={clsx(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
                  activeTab === key
                    ? "bg-surface text-white border border-border"
                    : "text-textTertiary hover:text-textSecondary hover:bg-surface/50"
                )}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden max-w-7xl mx-auto w-full">
        {/* Tab content */}
        <div
          className={clsx(
            "flex-1 overflow-auto px-6 py-4",
            activeTab === "browse" && selectedConcept ? "hidden lg:block" : ""
          )}
        >
          {activeTab === "browse" && (
            <ConceptBrowser
              onSelect={handleSelect}
              selected={selectedConcept?.id ?? null}
            />
          )}
          {activeTab === "graph" && (
            <Suspense
              fallback={
                <div className="flex-1 flex items-center justify-center h-full">
                  <Terminal className="w-6 h-6 text-textTertiary animate-pulse" />
                </div>
              }
            >
              <TaxonomyGraph
                onSelect={handleSelect}
                selectedId={selectedConcept?.id ?? null}
              />
            </Suspense>
          )}
          {activeTab === "coverage" && <TaxonomyCoverage />}
        </div>

        {/* Detail panel (shows for browse + graph tabs) */}
        {(activeTab === "browse" || activeTab === "graph") && (
          <AnimatePresence>
            {selectedConcept && (
              <motion.div
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 380, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="shrink-0 overflow-hidden"
              >
                <div className="w-[380px] h-full">
                  <ConceptDetail
                    concept={selectedConcept}
                    onClose={handleClose}
                    onNavigate={handleNavigate}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}

function StatPill({ label, value }: { label: string; value: number }) {
  return (
    <div className="text-center">
      <div className="text-lg font-bold font-mono text-white">
        {value.toLocaleString()}
      </div>
      <div className="text-[10px] text-textTertiary uppercase tracking-wider">
        {label}
      </div>
    </div>
  );
}
