"use client";

/**
 * Cmd+K / Ctrl+K command palette.
 *
 * Single global instance mounted in app/layout.tsx. Three sections:
 *  1. Pages — fixed list of routes (Vault, Practice, Mock Interview, etc.)
 *  2. Topics — fuzzy search across the taxonomy
 *  3. Questions — fuzzy search across all 9,200 question titles
 *
 * Keyboard:
 *   Cmd/Ctrl+K   open/close
 *   Esc          close
 *   ↑↓           navigate
 *   Enter        commit selected result
 *
 * No external library — `cmdk` adds 30kB and we already need only this
 * one component. Roughly 250 lines, zero deps beyond what staffml has.
 *
 * a11y notes:
 *   role="dialog" + aria-modal on the surface
 *   role="listbox" on the result list, role="option" on items
 *   aria-selected on the active row
 *   focus on the input on open, restore on close
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Search, X, Target, Crosshair, BookOpen, BarChart3, Info,
  FileText, Cpu, ArrowRight,
} from "lucide-react";
import clsx from "clsx";
import { searchTopics } from "@/lib/taxonomy";
import { searchQuestions } from "@/lib/corpus";

type Result =
  | { kind: "page"; title: string; subtitle: string; href: string; icon: React.ComponentType<{ className?: string }> }
  | { kind: "topic"; title: string; subtitle: string; topicId: string }
  | { kind: "question"; title: string; subtitle: string; questionId: string };

const PAGES: Result[] = [
  { kind: "page", title: "Vault",          subtitle: "Browse all topics",        href: "/",          icon: BookOpen },
  { kind: "page", title: "Practice",       subtitle: "Untimed, with helpers",    href: "/practice",  icon: Target },
  { kind: "page", title: "Mock Interview", subtitle: "Timed, the gauntlet",      href: "/gauntlet",  icon: Crosshair },
  { kind: "page", title: "Progress",       subtitle: "Stats and weak spots",     href: "/progress",  icon: BarChart3 },
  { kind: "page", title: "Plans",          subtitle: "Curated study tracks",     href: "/plans",     icon: FileText },
  { kind: "page", title: "Roofline",       subtitle: "Compute vs bandwidth",     href: "/roofline",  icon: Cpu },
  { kind: "page", title: "Simulator",      subtitle: "Hardware sandbox",         href: "/simulator", icon: Cpu },
  { kind: "page", title: "About",          subtitle: "What is StaffML",          href: "/about",     icon: Info },
];

function pageMatches(p: Result, q: string): boolean {
  if (p.kind !== "page") return false;
  const needle = q.toLowerCase().trim();
  if (!needle) return true;
  return (p.title + " " + p.subtitle).toLowerCase().includes(needle);
}

// Debounce window for the corpus search. The 9k-question search is fast in
// absolute terms but on every keystroke it competes with input rendering for
// the main thread; debouncing keeps the input responsive.
const SEARCH_DEBOUNCE_MS = 120;

export default function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  // searchQuery lags `query` by SEARCH_DEBOUNCE_MS — used for the expensive
  // searchQuestions / searchTopics calls so we don't re-scan 9k questions
  // on every keystroke.
  const [searchQuery, setSearchQuery] = useState("");
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const surfaceRef = useRef<HTMLDivElement>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  // ─── Global keyboard: open with Cmd+K / Ctrl+K ───────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen(o => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ─── Open: focus input, capture previously-focused element ──
  useEffect(() => {
    if (open) {
      previouslyFocused.current = document.activeElement as HTMLElement | null;
      // small timeout so the input exists in the DOM
      setTimeout(() => inputRef.current?.focus(), 0);
      setActiveIdx(0);
      setQuery("");
      setSearchQuery("");
    } else {
      previouslyFocused.current?.focus?.();
    }
  }, [open]);

  // ─── Debounce: update searchQuery after the user stops typing ──
  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => setSearchQuery(query), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [query, open]);

  // ─── Compute results ─────────────────────────────────
  // Pages filter on the live `query` (cheap, in-memory list filter).
  // Topics + questions filter on the debounced `searchQuery` because they
  // walk the full corpus.
  const results = useMemo<Result[]>(() => {
    const out: Result[] = [];

    const pages = PAGES.filter(p => pageMatches(p, query));
    out.push(...pages);

    if (searchQuery.trim().length >= 2) {
      try {
        const topics = searchTopics(searchQuery).slice(0, 8);
        for (const t of topics) {
          out.push({
            kind: "topic",
            title: t.name,
            subtitle: `${t.questionCount} question${t.questionCount === 1 ? "" : "s"}`,
            topicId: t.id,
          });
        }
      } catch {
        /* searchTopics may throw on cold start; swallow */
      }

      try {
        const questions = searchQuestions(searchQuery, 12);
        for (const q of questions) {
          out.push({
            kind: "question",
            title: q.title,
            subtitle: `${q.level} · ${q.track} · ${q.competency_area}`,
            questionId: q.id,
          });
        }
      } catch {
        /* same */
      }
    }

    return out;
  }, [query, searchQuery]);

  // Reset active index when results change
  useEffect(() => {
    setActiveIdx(0);
  }, [query, searchQuery]);

  // ─── Commit a result ─────────────────────────────────
  const commit = (r: Result) => {
    setOpen(false);
    if (r.kind === "page") router.push(r.href);
    else if (r.kind === "topic") router.push(`/?topic=${r.topicId}`);
    else if (r.kind === "question") router.push(`/practice?q=${r.questionId}`);
  };

  // ─── Local keyboard inside the palette ───────────────
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();   // don't bubble to global ? overlay listener
      setOpen(false);
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx(i => Math.min(i + 1, results.length - 1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx(i => Math.max(i - 1, 0));
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      const r = results[activeIdx];
      if (r) commit(r);
      return;
    }
    // Focus trap: cycle Tab/Shift+Tab within the surface so keyboard users
    // can't escape into the dimmed background. Without this they'd Tab into
    // arbitrary background controls.
    if (e.key === "Tab" && surfaceRef.current) {
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
    }
  };

  if (!open) return null;

  // Group result indexes by kind for section headers
  const sections: { label: string; items: { result: Result; idx: number }[] }[] = [];
  let cursor = 0;
  const pages = results.filter(r => r.kind === "page");
  if (pages.length) {
    sections.push({
      label: "Pages",
      items: pages.map((r, i) => ({ result: r, idx: cursor + i })),
    });
    cursor += pages.length;
  }
  const topics = results.filter(r => r.kind === "topic");
  if (topics.length) {
    sections.push({
      label: "Topics",
      items: topics.map((r, i) => ({ result: r, idx: cursor + i })),
    });
    cursor += topics.length;
  }
  const questions = results.filter(r => r.kind === "question");
  if (questions.length) {
    sections.push({
      label: "Questions",
      items: questions.map((r, i) => ({ result: r, idx: cursor + i })),
    });
    cursor += questions.length;
  }

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] px-4"
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={() => setOpen(false)}
        aria-hidden="true"
      />

      {/* Surface */}
      <div
        ref={surfaceRef}
        className="relative w-full max-w-2xl bg-background border border-border rounded-xl shadow-2xl overflow-hidden"
        onKeyDown={onKeyDown}
      >
        {/* Input row */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          <Search className="w-4 h-4 text-textTertiary shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search pages, topics, or questions…"
            className="flex-1 bg-transparent text-[14px] text-textPrimary placeholder:text-textTertiary focus:outline-none"
            aria-label="Command palette query"
            aria-controls="command-palette-results"
            aria-activedescendant={results[activeIdx] ? `cmdk-row-${activeIdx}` : undefined}
            spellCheck={false}
            autoComplete="off"
          />
          <kbd className="text-[10px] font-mono text-textTertiary border border-border rounded px-1.5 py-0.5 bg-surface">
            ESC
          </kbd>
          <button
            onClick={() => setOpen(false)}
            aria-label="Close command palette"
            className="p-1 text-textTertiary hover:text-textPrimary rounded transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results */}
        <ul
          id="command-palette-results"
          role="listbox"
          aria-label="Command palette results"
          className="max-h-[60vh] overflow-y-auto"
        >
          {sections.length === 0 && (
            <li className="px-4 py-8 text-center text-[13px] text-textTertiary">
              No results. Try a topic name like &ldquo;flash attention&rdquo; or a question keyword.
            </li>
          )}
          {sections.map((section) => (
            <li key={section.label}>
              <div className="px-4 pt-3 pb-1 text-[10px] font-mono text-textTertiary uppercase tracking-widest">
                {section.label}
              </div>
              <ul>
                {section.items.map(({ result, idx }) => {
                  const isActive = idx === activeIdx;
                  const Icon =
                    result.kind === "page"
                      ? result.icon
                      : result.kind === "topic"
                      ? Cpu
                      : Target;
                  return (
                    <li
                      key={`${result.kind}-${idx}`}
                      id={`cmdk-row-${idx}`}
                      role="option"
                      aria-selected={isActive}
                      onMouseEnter={() => setActiveIdx(idx)}
                      onClick={() => commit(result)}
                      className={clsx(
                        "flex items-center gap-3 px-4 py-2.5 cursor-pointer transition-colors",
                        isActive
                          ? "bg-accentBlue/15 text-textPrimary"
                          : "text-textSecondary hover:bg-surfaceHover"
                      )}
                    >
                      <Icon className="w-4 h-4 shrink-0 text-accentBlue/80" />
                      <div className="flex-1 min-w-0">
                        <div className="text-[13px] font-medium truncate">{result.title}</div>
                        <div className="text-[11px] text-textTertiary truncate">{result.subtitle}</div>
                      </div>
                      {isActive && (
                        <ArrowRight className="w-3.5 h-3.5 shrink-0 text-accentBlue/70" />
                      )}
                    </li>
                  );
                })}
              </ul>
            </li>
          ))}
        </ul>

        {/* Footer hint */}
        <div className="px-4 py-2 border-t border-border bg-surface/50 flex items-center gap-3 text-[10px] font-mono text-textTertiary">
          <span><kbd className="border border-border rounded px-1 bg-background">↑</kbd><kbd className="border border-border rounded px-1 bg-background ml-0.5">↓</kbd> navigate</span>
          <span><kbd className="border border-border rounded px-1 bg-background">↵</kbd> open</span>
          <span><kbd className="border border-border rounded px-1 bg-background">esc</kbd> close</span>
          <span className="ml-auto">Cmd+K from anywhere</span>
        </div>
      </div>
    </div>
  );
}
