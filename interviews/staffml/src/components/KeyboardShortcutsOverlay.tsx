"use client";

/**
 * Global keyboard-shortcuts cheat sheet.
 *
 * Triggered by `?` from anywhere (except inside text inputs). Mounted once
 * in app/layout.tsx alongside the command palette. Uses the same a11y
 * primitives: role="dialog", aria-modal, Escape to close, focus management.
 *
 * Adding a shortcut: append to SECTIONS below. The overlay is the single
 * source of truth that the rest of the app should reference when adding
 * new keybindings.
 *
 * Why an overlay and not a /help page:
 *   - Lower friction (no navigation, no route change)
 *   - Matches the Linear / GitHub / Notion `?` convention
 *   - Persists current page context (you can compare what you're doing
 *     against the relevant shortcuts without losing your place)
 */

import { useEffect, useRef, useState } from "react";
import { X, Keyboard } from "lucide-react";

interface Shortcut {
  keys: string[];
  description: string;
}

interface Section {
  label: string;
  shortcuts: Shortcut[];
}

const SECTIONS: Section[] = [
  {
    label: "Global",
    shortcuts: [
      { keys: ["⌘", "K"],     description: "Open command palette (search anything)" },
      { keys: ["?"],          description: "Show this keyboard shortcuts overlay" },
      { keys: ["Esc"],        description: "Close any open overlay or drawer" },
    ],
  },
  {
    label: "Practice",
    shortcuts: [
      { keys: ["⌘", "↵"],     description: "Reveal answer (when typing in the answer box)" },
      { keys: ["1"],          description: "Self-rate Wrong" },
      { keys: ["2"],          description: "Self-rate Partial" },
      { keys: ["3"],          description: "Self-rate Nailed it" },
      { keys: ["N"],          description: "Next question" },
    ],
  },
  {
    label: "Mock Interview",
    shortcuts: [
      { keys: ["⌘", "↵"],     description: "Reveal answer (when typing in the answer box)" },
      { keys: ["1"],          description: "Self-rate Skip" },
      { keys: ["2"],          description: "Self-rate Wrong" },
      { keys: ["3"],          description: "Self-rate Partial" },
      { keys: ["4"],          description: "Self-rate Nailed it" },
    ],
  },
  {
    label: "Vault",
    shortcuts: [
      { keys: ["Esc"],        description: "Close the topic detail drawer" },
    ],
  },
];

function KeyChip({ k }: { k: string }) {
  return (
    <kbd className="inline-flex items-center justify-center min-w-[22px] h-[22px] px-1.5 bg-surface border border-border rounded font-mono text-[11px] text-textPrimary">
      {k}
    </kbd>
  );
}

export default function KeyboardShortcutsOverlay() {
  const [open, setOpen] = useState(false);
  const previouslyFocused = useRef<HTMLElement | null>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const surfaceRef = useRef<HTMLDivElement>(null);

  // Global `?` listener — guarded against text inputs and selects
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Allow Escape to close even when overlay holds focus
      if (e.key === "Escape" && open) {
        e.preventDefault();
        setOpen(false);
        return;
      }
      // Don't open when the user is typing or has a select focused
      const t = e.target as HTMLElement | null;
      if (
        t && (
          t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.tagName === "SELECT" ||
          (t as HTMLElement).isContentEditable
        )
      ) {
        return;
      }
      if (e.key === "?" && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        setOpen(o => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Focus trap: cycle Tab/Shift+Tab within the surface while open.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab" || !surfaceRef.current) return;
      const focusables = surfaceRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Focus management on open/close
  useEffect(() => {
    if (open) {
      previouslyFocused.current = document.activeElement as HTMLElement | null;
      setTimeout(() => closeBtnRef.current?.focus(), 0);
    } else {
      previouslyFocused.current?.focus?.();
    }
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[12vh] px-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="keyboard-shortcuts-title"
    >
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={() => setOpen(false)}
        aria-hidden="true"
      />

      <div ref={surfaceRef} className="relative w-full max-w-2xl bg-background border border-border rounded-xl shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-border">
          <div className="flex items-center gap-2">
            <Keyboard className="w-4 h-4 text-accentBlue" />
            <h2 id="keyboard-shortcuts-title" className="text-[14px] font-bold text-textPrimary">
              Keyboard shortcuts
            </h2>
          </div>
          <button
            ref={closeBtnRef}
            onClick={() => setOpen(false)}
            aria-label="Close keyboard shortcuts"
            className="p-1.5 text-textTertiary hover:text-textPrimary rounded transition-colors focus:outline-none focus:ring-2 focus:ring-accentBlue/50"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 max-h-[70vh] overflow-y-auto">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-6">
            {SECTIONS.map((section) => (
              <div key={section.label}>
                <div className="text-[10px] font-mono text-textTertiary uppercase tracking-widest mb-2">
                  {section.label}
                </div>
                <ul className="space-y-1.5">
                  {section.shortcuts.map((s, i) => (
                    <li key={i} className="flex items-center justify-between gap-3">
                      <span className="text-[12px] text-textSecondary">{s.description}</span>
                      <span className="flex items-center gap-1 shrink-0">
                        {s.keys.map((k, j) => (
                          <span key={j} className="flex items-center gap-1">
                            {j > 0 && <span className="text-[10px] text-textTertiary">+</span>}
                            <KeyChip k={k} />
                          </span>
                        ))}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <div className="px-5 py-2 border-t border-border bg-surface/50 text-[10px] font-mono text-textTertiary">
          Press <KeyChip k="?" /> from anywhere to toggle this overlay.
        </div>
      </div>
    </div>
  );
}
