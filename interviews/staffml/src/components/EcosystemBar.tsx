"use client";

import { useState, useRef, useEffect } from "react";

/**
 * MLSysBook ecosystem navbar — mirrors navbar-common.yml from the Quarto sites.
 * Light background, gray links, accent on hover. Matches Vol I, Vol II, Labs, etc.
 */

const BASE = "https://mlsysbook.ai";

interface MenuItem {
  label: string;
  href: string;
  active?: boolean;
}

interface MenuGroup {
  label: string;
  items: MenuItem[];
}

const LEFT_MENUS: MenuGroup[] = [
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
    label: "Teach",
    items: [
      { label: "Course Map", href: `${BASE}/instructors/course-map.html` },
      { label: "Lecture Slides", href: `${BASE}/slides/` },
      { label: "Instructor Hub", href: `${BASE}/instructors/` },
    ],
  },
  {
    label: "Prepare",
    items: [
      { label: "StaffML", href: `${BASE}/staffml/`, active: true },
      { label: "Study Plans", href: "/plans" },
      { label: "Gauntlet Mode", href: "/gauntlet" },
    ],
  },
  {
    label: "Connect",
    items: [
      { label: "Newsletter", href: `${BASE}/newsletter/` },
      { label: "Global Network", href: `${BASE}/community/` },
      { label: "Workshops & Events", href: `${BASE}/community/events.html` },
    ],
  },
  {
    label: "About",
    items: [
      { label: "Our Story", href: `${BASE}/about/` },
      { label: "People", href: `${BASE}/about/people.html` },
      { label: "Contributors", href: `${BASE}/about/contributors.html` },
    ],
  },
];

const RIGHT_LINKS = [
  { label: "Support", href: "https://opencollective.com/mlsysbook", external: true },
  { label: "Star", href: "https://github.com/harvard-edge/cs249r_book", external: true },
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
    <div ref={barRef} className="border-b border-[#dee2e6] bg-white relative z-[60]">
      <div className="flex items-center justify-between px-4 h-9">
        {/* Left: logo + menus */}
        <div className="flex items-center gap-0.5">
          <a
            href={BASE}
            className="font-bold text-[#333] hover:text-[#a31f34] transition-colors flex items-center gap-1.5 shrink-0 mr-3 pr-3 border-r border-[#dee2e6] text-[12px]"
          >
            <svg viewBox="0 0 20 20" className="w-3.5 h-3.5">
              <rect width="20" height="20" rx="3" fill="#a31f34" />
              <text x="10" y="14.5" textAnchor="middle" fill="white" fontSize="11" fontWeight="700" fontFamily="system-ui">M</text>
            </svg>
            MLSysBook
          </a>
          <div className="hidden lg:flex items-center gap-0.5">
            {LEFT_MENUS.map((menu) => (
              <div key={menu.label} className="relative">
                <button
                  onClick={() => setOpenMenu(openMenu === menu.label ? null : menu.label)}
                  className={`px-2 py-1 rounded text-[12px] font-normal transition-colors ${
                    openMenu === menu.label
                      ? "text-[#a31f34]"
                      : "text-[#6c757d] hover:text-[#a31f34]"
                  }`}
                >
                  {menu.label}
                </button>
                {openMenu === menu.label && (
                  <div className="absolute top-full left-0 mt-0.5 py-1.5 bg-white border border-[#dee2e6] rounded-md shadow-lg min-w-[200px] z-[70]">
                    {menu.items.map((item) => (
                      <a
                        key={item.label}
                        href={item.href}
                        onClick={() => setOpenMenu(null)}
                        className={`block px-3 py-1.5 text-[12px] transition-colors ${
                          item.active
                            ? "text-[#a31f34] font-medium"
                            : "text-[#6c757d] hover:text-[#a31f34] hover:bg-[#f8f9fa]"
                        }`}
                      >
                        {item.label}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Right */}
        <div className="hidden md:flex items-center gap-3">
          {RIGHT_LINKS.map(({ label, href }) => (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[12px] text-[#6c757d] hover:text-[#a31f34] transition-colors"
            >
              {label}
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
