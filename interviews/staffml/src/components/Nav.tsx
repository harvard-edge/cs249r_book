"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ECOSYSTEM_BASE } from "../lib/env";
import {
  Library, Target, Crosshair, BarChart3, BookOpen, Github,
  Menu, X, Sun, Moon, Map, Cpu, Server, ChevronDown, Info,
  Star, Bug, Send, Atom,
} from "lucide-react";
import clsx from "clsx";
import StreakBadge from "@/components/StreakBadge";
import { buildSiteIssueUrl } from "@/lib/issue-url";
import { getDueCount } from "@/lib/progress";
import { useTheme } from "@/components/ThemeProvider";

const primaryLinks = [
  { href: "/", label: "Vault", icon: Library },
  { href: "/practice", label: "Practice", icon: Target },
  { href: "/gauntlet", label: "Mock Interview", icon: Crosshair },
  { href: "/progress", label: "Progress", icon: BarChart3 },
  { href: "/about", label: "About", icon: Info },
];

const toolLinks = [
  { href: "/plans", label: "Study Plans", icon: Map },
  { href: "/framework", label: "Framework", icon: Atom },
  { href: "/contribute", label: "Contribute", icon: Send },
  { href: "/roofline", label: "Roofline", icon: Cpu },
  { href: "/simulator", label: "Simulator", icon: Server },
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
];

export default function Nav() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [dueCount, setDueCount] = useState(0);
  const { theme, toggleTheme } = useTheme();
  const toolsRef = useRef<HTMLDivElement>(null);

  // Check for due SR cards periodically
  useEffect(() => {
    try { setDueCount(getDueCount()); } catch {}
    const interval = setInterval(() => {
      try { setDueCount(getDueCount()); } catch {}
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  // Close tools dropdown on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (toolsRef.current && !toolsRef.current.contains(e.target as Node)) {
        setToolsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" || pathname.startsWith("/explore") : pathname.startsWith(href);

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

            {/* Tools dropdown */}
            <div className="w-px h-4 bg-border mx-1.5" />
            <div ref={toolsRef} className="relative">
              <button
                onClick={() => setToolsOpen(!toolsOpen)}
                className={clsx(
                  "flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[13px] font-medium transition-colors",
                  toolLinks.some(l => isActive(l.href))
                    ? "bg-surface text-textPrimary border border-border"
                    : "text-textTertiary hover:text-textSecondary hover:bg-surface/50"
                )}
              >
                Tools
                <ChevronDown className={clsx("w-3 h-3 transition-transform", toolsOpen && "rotate-180")} />
              </button>
              {toolsOpen && (
                <div className="absolute top-full left-0 mt-1 w-48 bg-background border border-border rounded-lg shadow-lg py-1 z-50">
                  {toolLinks.map(({ href, label, icon: Icon }) => (
                    <Link
                      key={href}
                      href={href}
                      onClick={() => setToolsOpen(false)}
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

        <div className="flex items-center gap-3">
          <StreakBadge />
          {/* Dark mode toggle and Star moved to EcosystemBar for consistency */}
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
            <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest px-3 block mb-2">Tools</span>
            {toolLinks.map(({ href, label, icon: Icon }) => (
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
