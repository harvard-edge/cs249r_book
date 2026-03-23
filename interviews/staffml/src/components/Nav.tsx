"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Terminal, Crosshair, BarChart3, Target, BookOpen, Github, Menu, X, Calendar, Map, Cpu } from "lucide-react";
import clsx from "clsx";
import StreakBadge from "@/components/StreakBadge";

const links = [
  { href: "/", label: "Home", icon: Terminal },
  { href: "/daily", label: "Daily", icon: Calendar },
  { href: "/gauntlet", label: "Gauntlet", icon: Crosshair },
  { href: "/heatmap", label: "Heat Map", icon: BarChart3 },
  { href: "/drill", label: "Drill", icon: Target },
  { href: "/plans", label: "Plans", icon: Map },
  { href: "/roofline", label: "Roofline", icon: Cpu },
];

export default function Nav() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="border-b border-border bg-background/80 backdrop-blur-md sticky top-0 z-50">
      <div className="h-14 flex items-center px-4 sm:px-6 justify-between">
        <div className="flex items-center gap-4 sm:gap-8">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 group select-none" onClick={() => setMobileOpen(false)}>
            {/* Roofline logo mark */}
            <svg viewBox="0 0 32 32" className="w-7 h-7 drop-shadow-[0_0_8px_rgba(59,130,246,0.4)]">
              <path d="M5,25 L16,9 L27,9" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" fill="none" />
              <circle cx="16" cy="9" r="2.5" fill="#3b82f6" />
              <circle cx="16" cy="9" r="1" fill="black" />
            </svg>
            <span className="text-white text-lg font-extrabold tracking-tight">Staff</span>
            <span className="text-white/70 text-lg font-medium tracking-tight">ML</span>
          </Link>

          {/* Desktop nav links */}
          <div className="hidden sm:flex items-center gap-1">
            {links.map(({ href, label, icon: Icon }) => {
              const isActive = pathname === href || (href !== "/" && pathname.startsWith(href));
              return (
                <Link
                  key={href}
                  href={href}
                  className={clsx(
                    "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                    isActive
                      ? "bg-surface text-white border border-border"
                      : "text-textTertiary hover:text-textSecondary hover:bg-surface/50"
                  )}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </Link>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-4">
          <StreakBadge />
          <a
            href="https://mlsysbook.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex text-textTertiary hover:text-textSecondary items-center gap-1.5 text-xs transition-colors"
          >
            <BookOpen className="w-3.5 h-3.5" />
            Textbook
          </a>
          <a
            href="https://github.com/harvard-edge/cs249r_book"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex text-textTertiary hover:text-textSecondary items-center gap-1.5 text-xs transition-colors"
          >
            <Github className="w-3.5 h-3.5" />
            Star
          </a>
          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="sm:hidden p-1.5 text-textTertiary hover:text-white transition-colors"
            aria-label="Toggle navigation menu"
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="sm:hidden border-t border-border bg-surface px-4 py-3 space-y-1">
          {links.map(({ href, label, icon: Icon }) => {
            const isActive = pathname === href || (href !== "/" && pathname.startsWith(href));
            return (
              <Link
                key={href}
                href={href}
                onClick={() => setMobileOpen(false)}
                className={clsx(
                  "flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors",
                  isActive
                    ? "bg-accentBlue/10 text-accentBlue"
                    : "text-textSecondary hover:text-white hover:bg-surfaceHover"
                )}
              >
                <Icon className="w-4 h-4" />
                {label}
              </Link>
            );
          })}
          <div className="border-t border-border pt-2 mt-2 flex items-center gap-4">
            <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer" className="text-textTertiary hover:text-textSecondary text-xs flex items-center gap-1.5">
              <BookOpen className="w-3.5 h-3.5" /> Textbook
            </a>
            <a href="https://github.com/harvard-edge/cs249r_book" target="_blank" rel="noopener noreferrer" className="text-textTertiary hover:text-textSecondary text-xs flex items-center gap-1.5">
              <Github className="w-3.5 h-3.5" /> Star
            </a>
          </div>
        </div>
      )}
    </nav>
  );
}
