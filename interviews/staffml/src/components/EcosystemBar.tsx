"use client";

import { useState, useRef, useEffect } from "react";

/**
 * MLSysBook ecosystem navbar — identical structure and appearance to the
 * Quarto Bootstrap navbar, but rendered with inline styles to avoid
 * loading Bootstrap CSS (which would conflict with Tailwind).
 *
 * Matches the exact HTML from harvard-edge.github.io/cs249r_book_dev/
 * including Bootstrap Icons via CDN.
 */

const BASE = "https://mlsysbook.ai";

interface DropdownItem {
  icon?: string;
  label?: string;
  href?: string;
  external?: boolean;
  active?: boolean;
  divider?: boolean;
}

interface MenuGroup {
  id: string;
  label: string;
  icon?: string;
  items: DropdownItem[];
  alignEnd?: boolean;
}

const LEFT_MENUS: MenuGroup[] = [
  {
    id: "read", label: "Read", items: [
      { icon: "bi-journal", label: "Volume I: Foundations", href: `${BASE}/vol1/` },
      { icon: "bi-journal", label: "Volume II: At Scale", href: `${BASE}/vol2/` },
      { divider: true },
      { icon: "bi-file-pdf", label: "Volume I PDF", href: `${BASE}/vol1/assets/downloads/Machine-Learning-Systems-Vol1.pdf`, external: true },
      { icon: "bi-journal-text", label: "Volume I EPUB", href: `${BASE}/vol1/assets/downloads/Machine-Learning-Systems-Vol1.epub`, external: true },
      { divider: true },
      { icon: "bi-file-pdf", label: "Volume II PDF", href: `${BASE}/vol2/assets/downloads/Machine-Learning-Systems-Vol2.pdf`, external: true },
      { icon: "bi-journal-text", label: "Volume II EPUB", href: `${BASE}/vol2/assets/downloads/Machine-Learning-Systems-Vol2.epub`, external: true },
    ]
  },
  {
    id: "build", label: "Build", items: [
      { icon: "bi-sliders", label: "Labs", href: `${BASE}/labs/` },
      { icon: "bi-fire", label: "TinyTorch", href: `${BASE}/tinytorch/` },
      { icon: "bi-cpu", label: "Hardware Kits", href: `${BASE}/kits/` },
      { icon: "bi-calculator", label: "MLSys·IM", href: `${BASE}/mlsysim/` },
    ]
  },
  {
    id: "teach", label: "Teach", items: [
      { icon: "bi-signpost-split", label: "Course Map", href: `${BASE}/instructors/course-map.html` },
      { icon: "bi-easel", label: "Lecture Slides", href: `${BASE}/slides/` },
      { icon: "bi-person-video3", label: "Instructor Hub", href: `${BASE}/instructors/` },
    ]
  },
  {
    id: "prepare", label: "Prepare", items: [
      { icon: "bi-mortarboard", label: "StaffML", href: "/", active: true },
      { icon: "bi-map", label: "Study Plans", href: "/plans" },
      { icon: "bi-lightning", label: "Gauntlet Mode", href: "/gauntlet" },
    ]
  },
  {
    id: "connect", label: "Connect", items: [
      { icon: "bi-envelope", label: "Newsletter", href: `${BASE}/newsletter/` },
      { icon: "bi-globe", label: "Global Network", href: `${BASE}/community/` },
      { icon: "bi-calendar-event", label: "Workshops & Events", href: `${BASE}/community/events.html` },
      { icon: "bi-people", label: "Partners & Sponsors", href: `${BASE}/community/partners.html` },
    ]
  },
  {
    id: "about", label: "About", items: [
      { icon: "bi-book", label: "Our Story", href: `${BASE}/about/#story` },
      { icon: "bi-bullseye", label: "Mission", href: `${BASE}/about/#mission` },
      { icon: "bi-clock-history", label: "Milestones", href: `${BASE}/about/#milestones` },
      { divider: true },
      { icon: "bi-person-lines-fill", label: "People", href: `${BASE}/about/people.html` },
      { icon: "bi-people", label: "Contributors", href: `${BASE}/about/contributors.html` },
    ]
  },
];

const RIGHT_MENUS: MenuGroup[] = [
  {
    id: "github", label: "GitHub", icon: "bi-github", alignEnd: true, items: [
      { icon: "bi-chat", label: "Discussions", href: "https://github.com/harvard-edge/cs249r_book/discussions", external: true },
      { icon: "bi-pencil", label: "Edit this page", href: "https://github.com/harvard-edge/cs249r_book", external: true },
      { icon: "bi-bug", label: "Report an issue", href: "https://github.com/harvard-edge/cs249r_book/issues/new", external: true },
      { icon: "bi-code", label: "View source", href: "https://github.com/harvard-edge/cs249r_book", external: true },
    ]
  },
];

// ─── Styles matching Bootstrap 5 navbar computed values ────

const S = {
  nav: {
    backgroundColor: '#fff',
    borderBottom: '1px solid #dee2e6',
    padding: '0 16px',
    position: 'sticky' as const,
    top: 0,
    zIndex: 60,
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    fontSize: 16,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: 62,
    flexWrap: 'wrap' as const,
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    textDecoration: 'none',
    color: '#333',
    fontSize: 20,
    fontWeight: 600,
    whiteSpace: 'nowrap' as const,
  },
  navLink: {
    color: '#6c757d',
    textDecoration: 'none',
    padding: '8px 12px',
    fontSize: 16,
    fontWeight: 400,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    border: 'none',
    background: 'none',
    whiteSpace: 'nowrap' as const,
  },
  dropdown: {
    position: 'absolute' as const,
    top: '100%',
    left: 0,
    zIndex: 1000,
    minWidth: 240,
    padding: '4px 0',
    backgroundColor: '#fff',
    border: '1px solid rgba(0,0,0,0.15)',
    borderRadius: 4,
    boxShadow: '0 6px 12px rgba(0,0,0,0.08)',
  },
  dropdownEnd: {
    left: 'auto',
    right: 0,
  },
  dropdownItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '6px 16px',
    color: '#212529',
    textDecoration: 'none',
    fontSize: 16,
    whiteSpace: 'nowrap' as const,
  },
  divider: {
    margin: '4px 0',
    borderTop: '1px solid #dee2e6',
  },
};

export default function EcosystemBar() {
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [hoveredLink, setHoveredLink] = useState<string | null>(null);
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);
  const barRef = useRef<HTMLDivElement>(null);

  // Load Bootstrap Icons font
  useEffect(() => {
    if (document.getElementById("bi-css")) return;
    const link = document.createElement("link");
    link.id = "bi-css";
    link.rel = "stylesheet";
    link.href = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css";
    document.head.appendChild(link);
  }, []);

  // Close on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (barRef.current && !barRef.current.contains(e.target as Node)) {
        setOpenMenu(null);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const linkStyle = (id: string) => ({
    ...S.navLink,
    color: openMenu === id || hoveredLink === id ? '#a31f34' : '#6c757d',
  });

  const itemStyle = (key: string, active?: boolean) => ({
    ...S.dropdownItem,
    color: active ? '#a31f34' : hoveredItem === key ? '#a31f34' : '#212529',
    fontWeight: active ? 500 : 400,
    backgroundColor: hoveredItem === key ? '#f8f9fa' : 'transparent',
  });

  const renderDropdown = (menu: MenuGroup) => (
    <div key={menu.id} style={{ position: 'relative' }}>
      <button
        style={linkStyle(menu.id)}
        onClick={() => setOpenMenu(openMenu === menu.id ? null : menu.id)}
        onMouseEnter={() => setHoveredLink(menu.id)}
        onMouseLeave={() => { if (openMenu !== menu.id) setHoveredLink(null); }}
      >
        {menu.icon && <i className={`bi ${menu.icon}`} />}
        <span>{menu.label}</span>
        <svg width="10" height="10" viewBox="0 0 10 10" style={{ marginLeft: 2, opacity: 0.5 }}>
          <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.2" fill="none" />
        </svg>
      </button>
      {openMenu === menu.id && (
        <div style={{ ...S.dropdown, ...(menu.alignEnd ? S.dropdownEnd : {}) }}>
          {menu.items.map((item, i) =>
            item.divider ? (
              <div key={i} style={S.divider} />
            ) : (
              <a
                key={i}
                href={item.href}
                onClick={() => setOpenMenu(null)}
                style={itemStyle(`${menu.id}-${i}`, item.active)}
                onMouseEnter={() => setHoveredItem(`${menu.id}-${i}`)}
                onMouseLeave={() => setHoveredItem(null)}
                {...(item.external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
              >
                {item.icon && <i className={`bi ${item.icon}`} style={{ fontSize: 15, width: 18, opacity: 0.7 }} />}
                <span>{item.label}</span>
              </a>
            )
          )}
        </div>
      )}
    </div>
  );

  return (
    <div ref={barRef}>
      <div style={S.nav}>
        {/* Brand */}
        <a href={BASE} style={S.brand}>
          <img
            src="/logo-seas-shield.png"
            alt=""
            style={{ height: 28, width: 'auto' }}
          />
          <span className="hidden sm:inline">Machine Learning Systems</span>
        </a>

        {/* Desktop nav */}
        <div className="hidden xl:flex" style={{ alignItems: 'center', gap: 0, flex: 1, marginLeft: 16 }}>
          {/* Left menus */}
          <div style={{ display: 'flex', alignItems: 'center' }}>
            {LEFT_MENUS.map(renderDropdown)}
          </div>

          {/* Spacer */}
          <div style={{ flex: 1 }} />

          {/* Right links + menus */}
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <a
              href="https://opencollective.com/mlsysbook"
              target="_blank"
              rel="noopener noreferrer"
              style={linkStyle('support')}
              onMouseEnter={() => setHoveredLink('support')}
              onMouseLeave={() => setHoveredLink(null)}
            >
              <i className="bi bi-heart" /> <span>Support</span>
            </a>
            <a
              href="https://github.com/harvard-edge/cs249r_book"
              target="_blank"
              rel="noopener noreferrer"
              style={linkStyle('star')}
              onMouseEnter={() => setHoveredLink('star')}
              onMouseLeave={() => setHoveredLink(null)}
            >
              <i className="bi bi-star" /> <span>Star</span>
            </a>
            <a
              href="#subscribe"
              style={linkStyle('subscribe')}
              onMouseEnter={() => setHoveredLink('subscribe')}
              onMouseLeave={() => setHoveredLink(null)}
            >
              <i className="bi bi-envelope" /> <span>Subscribe</span>
            </a>
            {RIGHT_MENUS.map(renderDropdown)}
          </div>
        </div>

        {/* Mobile hamburger */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="xl:hidden"
          style={{ ...S.navLink, padding: 8 }}
          aria-label="Toggle navigation"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6c757d" strokeWidth="2">
            {mobileOpen
              ? <path d="M6 6L18 18M18 6L6 18" />
              : <path d="M3 6h18M3 12h18M3 18h18" />
            }
          </svg>
        </button>
      </div>

      {/* Mobile expanded */}
      {mobileOpen && (
        <div className="xl:hidden" style={{ borderTop: '1px solid #dee2e6', backgroundColor: '#fff', padding: '12px 16px' }}>
          {LEFT_MENUS.map((menu) => (
            <div key={menu.id} style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: '#adb5bd', textTransform: 'uppercase' as const, letterSpacing: 1, marginBottom: 4 }}>
                {menu.label}
              </div>
              {menu.items.filter(i => !i.divider).map((item, i) => (
                <a
                  key={i}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '4px 0', fontSize: 15, textDecoration: 'none',
                    color: item.active ? '#a31f34' : '#6c757d',
                    fontWeight: item.active ? 500 : 400,
                  }}
                  {...(item.external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
                >
                  {item.icon && <i className={`bi ${item.icon}`} style={{ fontSize: 14, width: 18, opacity: 0.6 }} />}
                  {item.label}
                </a>
              ))}
            </div>
          ))}
          <div style={{ borderTop: '1px solid #dee2e6', paddingTop: 8, display: 'flex', gap: 16 }}>
            <a href="https://opencollective.com/mlsysbook" target="_blank" rel="noopener noreferrer"
              style={{ fontSize: 15, color: '#6c757d', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
              <i className="bi bi-heart" style={{ fontSize: 14 }} /> Support
            </a>
            <a href="https://github.com/harvard-edge/cs249r_book" target="_blank" rel="noopener noreferrer"
              style={{ fontSize: 15, color: '#6c757d', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
              <i className="bi bi-star" style={{ fontSize: 14 }} /> Star
            </a>
            <a href="#subscribe"
              style={{ fontSize: 15, color: '#6c757d', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
              <i className="bi bi-envelope" style={{ fontSize: 14 }} /> Subscribe
            </a>
            <a href="https://github.com/harvard-edge/cs249r_book" target="_blank" rel="noopener noreferrer"
              style={{ fontSize: 15, color: '#6c757d', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
              <i className="bi bi-github" style={{ fontSize: 14 }} /> GitHub
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
