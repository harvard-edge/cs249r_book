"use client";

import { useState } from "react";
import { LEVELS } from "@/lib/levels";
import { X, HelpCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

/** Inline level explainer — shown to first-time users, collapsible to a "?" button */
export default function LevelExplainer({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <>
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="flex items-center gap-1.5 text-[12px] text-textTertiary hover:text-textSecondary transition-colors"
          aria-label="Show level definitions"
        >
          <HelpCircle className="w-3.5 h-3.5" />
          <span>What do levels mean?</span>
        </button>
      )}

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="rounded-xl border border-border bg-surface p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-[13px] font-bold text-textPrimary">
                  Difficulty Levels (Bloom's Taxonomy)
                </h3>
                <button
                  onClick={() => setOpen(false)}
                  className="p-1 text-textTertiary hover:text-textPrimary transition-colors"
                  aria-label="Close level explainer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-2">
                {LEVELS.map((level) => (
                  <div key={level.id} className="flex items-start gap-3">
                    <span
                      className="text-[11px] font-mono font-bold px-1.5 py-0.5 rounded shrink-0 mt-0.5"
                      style={{
                        color: level.color,
                        backgroundColor: `${level.color}12`,
                      }}
                    >
                      {level.id}
                    </span>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-[13px] font-semibold text-textPrimary">
                          {level.name}
                        </span>
                        <span className="text-[11px] text-textTertiary">
                          {level.role}
                        </span>
                      </div>
                      <p className="text-[12px] text-textSecondary leading-snug mt-0.5">
                        {level.verb}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <p className="text-[11px] text-textMuted mt-3">
                Based on Bloom's Taxonomy applied to ML systems engineering.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
