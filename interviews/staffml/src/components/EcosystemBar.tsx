"use client";

import { useState, useRef, useEffect } from "react";

/**
 * MLSysBook ecosystem navbar — mirrors the shared Quarto navbar-common.yml
 * so StaffML feels like part of the same site. Adapts the 6-dropdown structure
 * (Read, Build, Teach, Prepare, Connect, About) for React/Next.js.
 */

const BASE = "https://mlsysbook.ai";

interface MenuItem {
  label: string;
  href: string;
  external?: boolean;
  active?: boolean;
  separator?: boolean;
}

interface MenuGroup {
  label: string;
  items: MenuItem[];
}

const MENUS: MenuGroup[] = [
  {
    label: "Read",
    items: [
      { label: "Volume I: Foundations", href: `${BASE}/vol1/` },
      { label: "Volume II: At Scale", href: `${BASE}/vol2/` },
    ],
  },
  {
    label: "Build",
    items: [
      { label: "Labs", href: `${BASE}/labs/` },
      { label: "TinyTorch", href: `${BASE}/tinytorch/` },
      { label: "Hardware Kits", href: `${BASE}/kits/` },
    ],
  },
  {
    label: "Prepare",
    items: [
      { label: "StaffML", href: `${BASE}/staffml/`, active: true },
      { label: "Study Plans", href: "/plans" },
      { label: "Gauntlet", href: "/gauntlet" },
    ],
  },
  {
    label: "Connect",
    items: [
      { label: "Newsletter", href: `${BASE}/newsletter/` },
      { label: "Community", href: `${BASE}/community/` },
    ],
  },
  {
    label: "About",
    items: [
      { label: "Our Story", href: `${BASE}/about/` },
      { label: "People", href: `${BASE}/about/people.html` },
    ],
  },
];

export default function EcosystemBar() {
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const barRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (barRef.current && !barRef.current.contains(e.target as Node)) {
        setOpenMenu(null);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div ref={barRef} className="bg-[#1a1a2e] text-white/80 text-[11px] relative z-[60]">
      <div className="flex items-center justify-between px-4 h-8">
        {/* Left: logo + menus */}
        <div className="flex items-center gap-1">
          <a
            href={BASE}
            className="font-bold text-white hover:text-white/90 transition-colors flex items-center gap-1.5 shrink-0 mr-3 pr-3 border-r border-white/10"
          >
            <svg viewBox="0 0 20 20" className="w-3.5 h-3.5">
              <rect width="20" height="20" rx="3" fill="#a31f34" />
              <text x="10" y="14.5" textAnchor="middle" fill="white" fontSize="11" fontWeight="700" fontFamily="system-ui">M</text>
            </svg>
            MLSysBook
          </a>
          <div className="hidden sm:flex items-center gap-0.5">
            {MENUS.map((menu) => (
              <div key={menu.label} className="relative">
                <button
                  onClick={() => setOpenMenu(openMenu === menu.label ? null : menu.label)}
                  className={`px-2 py-1 rounded hover:bg-white/10 transition-colors ${
                    openMenu === menu.label ? "bg-white/10 text-white" : ""
                  }`}
                >
                  {menu.label}
                </button>
                {openMenu === menu.label && (
                  <div className="absolute top-full left-0 mt-0.5 py-1 bg-[#1a1a2e] border border-white/10 rounded-md shadow-xl min-w-[180px] z-[70]">
                    {menu.items.map((item) => (
                      <a
                        key={item.label}
                        href={item.href}
                        onClick={() => setOpenMenu(null)}
                        className={`block px-3 py-1.5 hover:bg-white/10 transition-colors ${
                          item.active ? "text-white font-semibold" : ""
                        }`}
                      >
                        {item.label}
                        {item.active && <span className="ml-1.5 text-[9px] text-accentBlue">current</span>}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Right: GitHub */}
        <div className="flex items-center gap-3">
          <a
            href="https://github.com/harvard-edge/cs249r_book"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-white transition-colors hidden md:inline"
          >
            GitHub
          </a>
        </div>
      </div>
    </div>
  );
}
