"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ECOSYSTEM_BASE } from "../lib/env";
import {
  Library, Target, Crosshair, BarChart3, BookOpen, Github,
  Menu, X, Sun, Moon, Map, Cpu, Server, ChevronDown, Info,
  Star, Bug, Send, Atom, MoreHorizontal, Search,
} from "lucide-react";
import clsx from "clsx";
import StreakBadge from "@/components/StreakBadge";
import { buildSiteIssueUrl } from "@/lib/issue-url";
import { getDueCount } from "@/lib/progress";
import { useTheme } from "@/components/ThemeProvider";

// IA reorganization (per dev-tools UX review):
//   - About moved out of primary into the "More" user menu
//   - Plans promoted from Tools dropdown to primary nav
//   - "Tools" renamed to "Lab" (now contains only the calculator-style sandboxes)
//   - Contribute, Dashboard, About moved to "More" user menu (top-right)
const primaryLinks = [
  { href: "/", label: "Vault", icon: Library },
  { href: "/practice", label: "Practice", icon: Target },
  { href: "/gauntlet", label: "Mock Interview", icon: Crosshair },
  { href: "/plans", label: "Plans", icon: Map },
  { href: "/progress", label: "Progress", icon: BarChart3 },
];

// "Lab" is the renamed Tools dropdown — only sandbox calculators belong here.
// /framework was added on dev in parallel and belongs alongside the other
// sandbox tools.
const labLinks = [
  { href: "/framework", label: "Framework", icon: Atom },
  { href: "/roofline", label: "Roofline", icon: Cpu },
  { href: "/simulator", label: "Simulator", icon: Server },
];

// "More" is the new top-right user menu — secondary destinations
const moreLinks = [
  { href: "/about", label: "About", icon: Info, kind: "internal" as const },
  { href: "/contribute", label: "Contribute", icon: Send, kind: "internal" as const },
  { href: "/dashboard", label: "Dashboard", icon: BarChart3, kind: "internal" as const },
];

export default function Nav() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [labOpen, setLabOpen] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);
  const [dueCount, setDueCount] = useState(0);
  const { theme, toggleTheme } = useTheme();
  const labRef = useRef<HTMLDivElement>(null);
  const moreRef = useRef<HTMLDivElement>(null);

  // Detect Mac vs other for ⌘K hint
  // Detect Mac vs other for ⌘K hint. navigator.platform is deprecated;
  // navigator.userAgent is the recommended replacement and is reliable
  // enough for the binary "show ⌘K vs Ctrl K" decision.
  const [isMac, setIsMac] = useState(false);
  useEffect(() => {
    setIsMac(/Mac|iPhone|iPad|iPod/i.test(navigator.userAgent));
  }, []);

  // Check for due SR cards periodically
  useEffect(() => {
    try { setDueCount(getDueCount()); } catch {}
    const interval = setInterval(() => {
      try { setDueCount(getDueCount()); } catch {}
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  // Close dropdowns on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (labRef.current && !labRef.current.contains(e.target as Node)) {
        setLabOpen(false);
      }
      if (moreRef.current && !moreRef.current.contains(e.target as Node)) {
        setMoreOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <nav className="border-b border-border bg-background/80 backdrop-blur-md sticky top-[52px] z-50">
      <div className="h-14 flex items-center px-4 lg:px-6 justify-between">
        <div className="flex items-center gap-4 lg:gap-6">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 group select-none shrink-0" onClick={() => setMobileOpen(false)}>
            <svg viewBox="0 0 32 32" className="w-6 h-6 drop-shadow-[0_0_8px_rgba(59,130,246,0.4)]">
              <path d="M5,25 L16,9 L27,9" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none" />
              <circle cx="16" cy="9" r="2.5" fill="#3b82f6" />
              <circle cx="16" cy="9" r="1" fill="currentColor" />
            </svg>
            <span className="text-base tracking-tight">
              <span className="text-textPrimary font-extrabold">Staff</span><span className="text-accentBlue font-bold ml-[2px]">ML</span>
            </span>
          </Link>

          {/* Desktop primary nav */}
          <div className="hidden md:flex items-center gap-0.5">
            {primaryLinks.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                className={clsx(
                  "flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[13px] font-medium transition-colors",
                  isActive(href)
                    ? "bg-surface text-textPrimary border border-border"
                    : "text-textTertiary hover:text-textSecondary hover:bg-surface/50"
                )}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
                {href === "/practice" && dueCount > 0 && (
                  <span className="ml-0.5 px-1.5 py-0.5 text-[9px] font-bold bg-accentAmber/20 text-accentAmber rounded-full leading-none">
                    {dueCount}
                  </span>
                )}
              </Link>
            ))}

            {/* Lab dropdown — sandbox calculators only */}
            <div className="w-px h-4 bg-border mx-1.5" />
            <div ref={labRef} className="relative">
              <button
                onClick={() => setLabOpen(!labOpen)}
                aria-haspopup="menu"
                aria-expanded={labOpen}
                className={clsx(
                  "flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[13px] font-medium transition-colors",
                  labLinks.some(l => isActive(l.href))
                    ? "bg-surface text-textPrimary border border-border"
                    : "text-textTertiary hover:text-textSecondary hover:bg-surface/50"
                )}
              >
                Lab
                <ChevronDown className={clsx("w-3 h-3 transition-transform", labOpen && "rotate-180")} />
              </button>
              {labOpen && (
                <div role="menu" className="absolute top-full left-0 mt-1 w-48 bg-background border border-border rounded-lg shadow-lg py-1 z-50">
                  {labLinks.map(({ href, label, icon: Icon }) => (
                    <Link
                      key={href}
                      href={href}
                      role="menuitem"
                      onClick={() => setLabOpen(false)}
                      className={clsx(
                        "flex items-center gap-2.5 px-3 py-2 text-[13px] font-medium transition-colors",
                        isActive(href)
                          ? "text-textPrimary bg-surface"
                          : "text-textSecondary hover:text-textPrimary hover:bg-surface/50"
                      )}
                    >
                      <Icon className="w-4 h-4" />
                      {label}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <StreakBadge />

          {/* ⌘K command palette hint — clicks dispatch the same keyboard event */}
          <button
            onClick={() => {
              window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true, ctrlKey: !isMac }));
            }}
            className="hidden md:inline-flex items-center gap-2 px-2.5 py-1 text-[11px] font-medium text-textTertiary hover:text-textSecondary border border-border rounded-md bg-surface/50 hover:bg-surface transition-colors"
            aria-label="Open command palette"
            title="Search anything (Cmd+K)"
          >
            <Search className="w-3 h-3" />
            <span>Search</span>
            <kbd className="ml-1 font-mono text-[9px] text-textTertiary">{isMac ? "⌘K" : "Ctrl K"}</kbd>
          </button>

          <button
            onClick={toggleTheme}
            className="p-2 text-textTertiary hover:text-textSecondary transition-colors"
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>

          {/* More menu — About / Contribute / Dashboard / Star / Report */}
          <div ref={moreRef} className="relative hidden md:block">
            <button
              onClick={() => setMoreOpen(!moreOpen)}
              aria-haspopup="menu"
              aria-expanded={moreOpen}
              aria-label="More menu"
              className="p-2 text-textTertiary hover:text-textSecondary transition-colors"
            >
              <MoreHorizontal className="w-4 h-4" />
            </button>
            {moreOpen && (
              <div role="menu" className="absolute top-full right-0 mt-1 w-52 bg-background border border-border rounded-lg shadow-lg py-1 z-50">
                {moreLinks.map(({ href, label, icon: Icon }) => (
                  <Link
                    key={href}
                    href={href}
                    role="menuitem"
                    onClick={() => setMoreOpen(false)}
                    className={clsx(
                      "flex items-center gap-2.5 px-3 py-2 text-[13px] font-medium transition-colors",
                      isActive(href)
                        ? "text-textPrimary bg-surface"
                        : "text-textSecondary hover:text-textPrimary hover:bg-surface/50"
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    {label}
                  </Link>
                ))}
                <div className="my-1 border-t border-border" />
                <a
                  href="https://github.com/harvard-edge/cs249r_book"
                  target="_blank"
                  rel="noopener noreferrer"
                  role="menuitem"
                  onClick={() => setMoreOpen(false)}
                  className="flex items-center gap-2.5 px-3 py-2 text-[13px] font-medium text-textSecondary hover:text-textPrimary hover:bg-surface/50 transition-colors"
                >
                  <Star className="w-4 h-4" />
                  Star on GitHub
                </a>
                <a
                  href={buildSiteIssueUrl()}
                  target="_blank"
                  rel="noopener noreferrer"
                  role="menuitem"
                  onClick={() => setMoreOpen(false)}
                  className="flex items-center gap-2.5 px-3 py-2 text-[13px] font-medium text-textSecondary hover:text-textPrimary hover:bg-surface/50 transition-colors"
                >
                  <Bug className="w-4 h-4" />
                  Report Issue
                </a>
              </div>
            )}
          </div>
          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 text-textTertiary hover:text-textPrimary transition-colors"
            aria-label="Toggle navigation menu"
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="md:hidden border-t border-border bg-surface px-4 py-3 space-y-1">
          {primaryLinks.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              onClick={() => setMobileOpen(false)}
              className={clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors",
                isActive(href)
                  ? "bg-accentBlue/10 text-accentBlue"
                  : "text-textSecondary hover:text-textPrimary hover:bg-surfaceHover"
              )}
            >
              <Icon className="w-4 h-4" />
              {label}
            </Link>
          ))}
          <div className="pt-2 mt-2 border-t border-border">
            <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest px-3 block mb-2">Lab</span>
            {labLinks.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                onClick={() => setMobileOpen(false)}
                className={clsx(
                  "flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors",
                  isActive(href)
                    ? "bg-accentBlue/10 text-accentBlue"
                    : "text-textSecondary hover:text-textPrimary hover:bg-surfaceHover"
                )}
              >
                <Icon className="w-4 h-4" />
                {label}
              </Link>
            ))}
          </div>
          <div className="pt-2 mt-2 border-t border-border">
            <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest px-3 block mb-2">More</span>
            {moreLinks.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                onClick={() => setMobileOpen(false)}
                className={clsx(
                  "flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors",
                  isActive(href)
                    ? "bg-accentBlue/10 text-accentBlue"
                    : "text-textSecondary hover:text-textPrimary hover:bg-surfaceHover"
                )}
              >
                <Icon className="w-4 h-4" />
                {label}
              </Link>
            ))}
          </div>
          <div className="border-t border-border pt-2 mt-2 flex flex-wrap items-center gap-4">
            <a href={ECOSYSTEM_BASE} target="_blank" rel="noopener noreferrer" className="text-textTertiary hover:text-textSecondary text-xs flex items-center gap-1.5">
              <BookOpen className="w-3.5 h-3.5" /> MLSysBook.ai
            </a>
            <a href="https://github.com/harvard-edge/cs249r_book" target="_blank" rel="noopener noreferrer" className="text-textTertiary hover:text-textSecondary text-xs flex items-center gap-1.5">
              <Star className="w-3.5 h-3.5" /> Star on GitHub
            </a>
            <a href={buildSiteIssueUrl()} target="_blank" rel="noopener noreferrer" className="text-textTertiary hover:text-textSecondary text-xs flex items-center gap-1.5">
              <Bug className="w-3.5 h-3.5" /> Report Issue
            </a>
          </div>
        </div>
      )}
    </nav>
  );
}
