"use client";

import { useState, useRef, useEffect } from "react";

/**
 * MLSysBook ecosystem navbar — replicates the Quarto Bootstrap navbar
 * from navbar-common.yml so StaffML looks like part of the same site.
 *
 * Visual spec (from Quarto + _navbar.scss):
 *   - White background, bottom border #dee2e6
 *   - Logo: SEAS shield + "Machine Learning Systems"
 *   - Nav links: #6c757d, hover: accent (#a31f34)
 *   - Dropdowns: white bg, border, shadow-sm
 *   - Font: Inter, ~14px
 *   - Height: ~56px
 *   - Collapses below xl (1200px)
 */

const BASE = "https://mlsysbook.ai";

interface MenuItem {
  icon?: string;
  label: string;
  href: string;
  external?: boolean;
  active?: boolean;
  divider?: boolean;
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
      { label: "StaffML", href: "/", active: true },
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
  { label: "Support", href: "https://opencollective.com/mlsysbook", icon: "♥" },
  { label: "Star", href: "https://github.com/harvard-edge/cs249r_book", icon: "★" },
];

export default function EcosystemBar() {
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
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
    <nav
      ref={barRef}
      className="bg-white border-b border-[#dee2e6] relative z-[60]"
      style={{ fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}
    >
      <div className="flex items-center justify-between px-4 lg:px-6" style={{ height: 50 }}>
        {/* Brand */}
        <div className="flex items-center gap-5">
          <a
            href={BASE}
            className="flex items-center gap-2 shrink-0 no-underline"
          >
            <img
              src="https://mlsysbook.ai/vol1/assets/images/icons/favicon.png"
              alt=""
              className="h-7 w-auto"
            />
            <span className="text-[14px] font-semibold text-[#333] hidden sm:inline">
              Machine Learning Systems
            </span>
          </a>

          {/* Desktop nav links */}
          <div className="hidden xl:flex items-center gap-0.5">
            {LEFT_MENUS.map((menu) => {
              const isActive = menu.items.some(i => i.active);
              return (
                <div key={menu.label} className="relative">
                  <button
                    onClick={() => setOpenMenu(openMenu === menu.label ? null : menu.label)}
                    className="px-2.5 py-1.5 text-[14px] transition-colors rounded"
                    style={{
                      color: openMenu === menu.label ? '#a31f34' : '#6c757d',
                      fontWeight: 400,
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#a31f34')}
                    onMouseLeave={(e) => {
                      if (openMenu !== menu.label) e.currentTarget.style.color = '#6c757d';
                    }}
                  >
                    {menu.label}
                  </button>
                  {openMenu === menu.label && (
                    <div
                      className="absolute top-full left-0 mt-1 py-1 bg-white rounded-md min-w-[220px]"
                      style={{
                        border: '1px solid rgba(0,0,0,0.15)',
                        boxShadow: '0 6px 12px rgba(0,0,0,0.08)',
                      }}
                    >
                      {menu.items.map((item) => (
                        <a
                          key={item.label}
                          href={item.href}
                          onClick={() => setOpenMenu(null)}
                          className="block px-4 py-1.5 text-[13px] no-underline transition-colors"
                          style={{
                            color: item.active ? '#a31f34' : '#6c757d',
                            fontWeight: item.active ? 500 : 400,
                            backgroundColor: 'transparent',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.color = '#a31f34';
                            e.currentTarget.style.backgroundColor = '#f8f9fa';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.color = item.active ? '#a31f34' : '#6c757d';
                            e.currentTarget.style.backgroundColor = 'transparent';
                          }}
                        >
                          {item.label}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-3">
            {RIGHT_LINKS.map(({ label, href, icon }) => (
              <a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[13px] no-underline transition-colors flex items-center gap-1"
                style={{ color: '#6c757d' }}
                onMouseEnter={(e) => (e.currentTarget.style.color = '#a31f34')}
                onMouseLeave={(e) => (e.currentTarget.style.color = '#6c757d')}
              >
                <span className="text-[11px]">{icon}</span> {label}
              </a>
            ))}
          </div>

          {/* Mobile hamburger (xl:hidden) */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="xl:hidden p-1.5 text-[#6c757d]"
            aria-label="Toggle ecosystem navigation"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
              {mobileOpen ? (
                <path d="M5 5L15 15M15 5L5 15" />
              ) : (
                <path d="M3 5h14M3 10h14M3 15h14" />
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile dropdown */}
      {mobileOpen && (
        <div className="xl:hidden border-t border-[#dee2e6] bg-white px-4 py-3 space-y-3">
          {LEFT_MENUS.map((menu) => (
            <div key={menu.label}>
              <span className="text-[10px] font-semibold text-[#adb5bd] uppercase tracking-wider block mb-1">
                {menu.label}
              </span>
              {menu.items.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className="block py-1 text-[13px] no-underline"
                  style={{
                    color: item.active ? '#a31f34' : '#6c757d',
                    fontWeight: item.active ? 500 : 400,
                  }}
                >
                  {item.label}
                </a>
              ))}
            </div>
          ))}
          <div className="border-t border-[#dee2e6] pt-2 flex gap-4">
            {RIGHT_LINKS.map(({ label, href, icon }) => (
              <a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[13px] no-underline flex items-center gap-1"
                style={{ color: '#6c757d' }}
              >
                <span className="text-[11px]">{icon}</span> {label}
              </a>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
}
