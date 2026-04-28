"use client";

import { useState, useRef, useEffect } from "react";
import { ECOSYSTEM_BASE as BASE } from "../lib/env";
import { useTheme } from "@/components/ThemeProvider";

const ASSET_PREFIX = process.env.NEXT_PUBLIC_BASE_PATH || "";

/**
 * MLSysBook ecosystem navbar — identical structure and appearance to the
 * Quarto Bootstrap navbar, but rendered with inline styles to avoid
 * loading Bootstrap CSS (which would conflict with Tailwind).
 *
 * Matches the exact HTML from harvard-edge.github.io/cs249r_book_dev/
 * including Bootstrap Icons via CDN.
 */

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
  // Site-specific dropdown first (matches ecosystem convention)
  {
    id: "staffml", label: "StaffML", items: [
      { icon: "bi-mortarboard", label: "StaffML Home", href: "/" },
      { icon: "bi-collection", label: "Question Vault", href: "/" },
      { divider: true },
      { icon: "bi-lightning", label: "Practice", href: "/practice" },
      { icon: "bi-crosshair", label: "Mock Interview", href: "/gauntlet" },
      { icon: "bi-bar-chart", label: "Progress", href: "/progress" },
      { divider: true },
      { icon: "bi-map", label: "Study Plans", href: "/plans" },
      { icon: "bi-diagram-3", label: "Framework", href: "/framework" },
    ]
  },
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

// Colors extracted from Quarto's compiled Bootstrap CSS
const ACCENT = '#a51c30';       // --bs-navbar-active-color (crimson)
const ACCENT_HOVER = 'rgba(165, 28, 48, 0.8)'; // --bs-navbar-hover-color
const NAV_COLOR = '#6c757d';    // Matches shared/_navbar.scss nav-link color
const BRAND_COLOR = '#333';     // Matches shared/_navbar.scss .navbar-brand color

const S = {
  nav: {
    // Bootstrap navbar default: 0.5rem top/bottom padding plus brand line
    // height → ~60 px total at desktop widths. Matches the measured height
    // of Quarto sites (60.5 px) so StaffML sits flush with ecosystem nav.
    padding: '8.5px 17px',  // matches Quarto measured `.navbar` padding
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: 17,  // --bs-body-font-size: 1.0625rem (Quarto measured 17 px)
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    // Do NOT wrap: wrapping produces a stacked/offset layout at intermediate
    // widths that looks broken. The hamburger path handles narrow screens.
    flexWrap: 'nowrap' as const,
    minHeight: 60,
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    textDecoration: 'none',
    color: BRAND_COLOR,  // Neutral gray, matches shared/_navbar.scss
    fontSize: 21,  // --bs-navbar-brand-font-size: 1.25rem on 17 px root ≈ 21.25 px
    fontWeight: 500,  // Matches shared/_navbar.scss .navbar-brand
    whiteSpace: 'nowrap' as const,
    marginRight: 16,  // --bs-navbar-brand-margin-end: 1rem
    minWidth: 0,     // Allow title to truncate rather than clipping the shield
    overflow: 'hidden' as const,
  },
  navLink: {
    color: NAV_COLOR,
    textDecoration: 'none',
    padding: '8.5px 12px',  // Quarto measured 8.5 × 12.75 px; round for integer pixels
    fontSize: 17,
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
    minWidth: 160,        // Bootstrap `.dropdown-menu` default (10rem), was over-wide at 240
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
    padding: '4px 16px',  // Bootstrap `.dropdown-item` padding is 0.25rem 1rem
    color: '#212529',
    textDecoration: 'none',
    fontSize: 14,         // Quarto dropdown items measured ~14 px; was 16 → visibly over-sized
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
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";

  // Theme-aware colors (matches shared/_navbar.scss light/dark).
  // Light bg is Bootstrap's `--bs-tertiary-bg` (#f8f9fa) — Quarto navbars
  // render against this off-white, with no bottom border. Using pure #fff
  // here previously made the StaffML bar look flatter/brighter than the
  // rest of the ecosystem.
  const bgColor = isDark ? '#212529' : '#f8f9fa';
  const borderColor = isDark ? '#454d55' : '#dee2e6';
  const brandColor = isDark ? '#e6e6e6' : '#333';
  const navColor = isDark ? '#adb5bd' : '#6c757d';
  const dropdownBg = isDark ? '#2d2d2d' : '#fff';
  const dropdownBorder = isDark ? '#454d55' : 'rgba(0,0,0,0.15)';
  const dropdownItemColor = isDark ? '#e6e6e6' : '#212529';
  const dropdownHoverBg = isDark ? '#3a3a3a' : '#f8f9fa';
  const dividerColor = isDark ? '#454d55' : '#dee2e6';

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
    color: openMenu === id || hoveredLink === id ? ACCENT : navColor,
  });

  const itemStyle = (key: string, active?: boolean) => ({
    ...S.dropdownItem,
    color: active ? ACCENT : hoveredItem === key ? ACCENT : dropdownItemColor,
    fontWeight: active ? 500 : 400,
    backgroundColor: hoveredItem === key ? dropdownHoverBg : 'transparent',
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
        <span className={menu.alignEnd ? "hidden nav-xl:inline" : ""}>{menu.label}</span>
        <span style={{
          display: 'inline-block', marginLeft: 4,
          borderTop: '4px solid currentColor', borderRight: '4px solid transparent', borderLeft: '4px solid transparent',
          opacity: 0.6, verticalAlign: '1px',
        }} />
      </button>
      {openMenu === menu.id && (
        <div style={{ ...S.dropdown, ...(menu.alignEnd ? S.dropdownEnd : {}), backgroundColor: dropdownBg, border: `1px solid ${dropdownBorder}` }}>
          {menu.items.map((item, i) =>
            item.divider ? (
              <div key={i} style={{ ...S.divider, borderTopColor: dividerColor }} />
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
    <div ref={barRef} style={{
      position: 'sticky' as const, top: 0, zIndex: 60,
      backgroundColor: bgColor,
      // Quarto navbars have no bottom border in light mode — the seam is
      // provided by the bg-color contrast against the white page body.
      // In dark mode keep a hairline so the nav doesn't bleed into the
      // dark page background.
      borderBottom: isDark ? `1px solid ${borderColor}` : 'none',
      transition: 'background-color 0.2s, border-color 0.2s',
      // Why `overflow-x: clip` (not `hidden`):
      // At iPad landscape (1024 px) and other narrow desktop widths, the 7
      // left dropdowns + 6 right-side icons + brand exceed the viewport.
      // Without containment, that overflow propagates to body.scrollWidth
      // and the whole page scrolls horizontally. Quarto sites avoid the
      // same scroll because their navbar lives inside `.fixed-top`
      // (position:fixed), which is removed from normal flow — body width
      // never sees the navbar's true content width. We achieve the same
      // visual result without changing positioning by clipping the X axis
      // on this wrapper.
      // `clip` (not `hidden`) is critical: per CSS Overflow Module 3, when
      // `overflow-x: clip` is paired with `overflow-y: visible`, the Y axis
      // stays visible. `overflow-x: hidden` would force `overflow-y: auto`
      // and clip the dropdown menus that extend below the bar. Browser
      // support: Safari 16+, Chrome 90+, Firefox 81+; older browsers fall
      // back to `visible` (no regression vs current behavior).
      overflowX: 'clip' as const,
    }}>
      <div style={{ ...S.nav, position: 'relative' as const, borderBottom: 'none' }}>
        {/* Brand */}
        <a href={BASE} style={{ ...S.brand, color: brandColor }}>
          <img
            src={`${ASSET_PREFIX}/logo-seas-shield.png`}
            alt=""
            style={{ height: 28, width: 'auto', flexShrink: 0 }}
          />
          {/* Title always shown — Quarto mobile bars still show "ML Systems"
              (truncated), so hiding the span below `nav-sm` was worse than
              the ecosystem. Parent `S.brand` has `overflow: 'hidden'` +
              `whiteSpace: 'nowrap'`, so narrow viewports get an ellipsis
              rather than a wrap. */}
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', minWidth: 0 }}>
            Machine Learning Systems
          </span>
        </a>

        {/* Desktop nav — appears at Bootstrap's lg breakpoint (992px) to match
            the Quarto navbar (collapse-below: lg in navbar-common.yml). */}
        <div className="hidden nav-lg:flex" style={{ alignItems: 'center', gap: 0, flex: 1, marginLeft: 16, minWidth: 0 }}>
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
              <i className="bi bi-heart" /> <span className="hidden nav-xl:inline">Support</span>
            </a>
            <a
              href="https://github.com/harvard-edge/cs249r_book"
              target="_blank"
              rel="noopener noreferrer"
              style={linkStyle('star')}
              onMouseEnter={() => setHoveredLink('star')}
              onMouseLeave={() => setHoveredLink(null)}
            >
              <i className="bi bi-star" /> <span className="hidden nav-xl:inline">Star</span>
            </a>
            <a
              href="#subscribe"
              style={linkStyle('subscribe')}
              onMouseEnter={() => setHoveredLink('subscribe')}
              onMouseLeave={() => setHoveredLink(null)}
            >
              <i className="bi bi-envelope" /> <span className="hidden nav-xl:inline">Subscribe</span>
            </a>
            {/* Dark mode toggle — matches Quarto navbar position */}
            <button
              onClick={toggleTheme}
              style={{ ...S.navLink, padding: '8px 10px', cursor: 'pointer' }}
              aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
            >
              <i className={`bi ${isDark ? 'bi-sun' : 'bi-moon-stars-fill'}`} style={{ fontSize: 14 }} />
            </button>
            {RIGHT_MENUS.map(renderDropdown)}
            {/* Search — opens the existing CommandPalette (Cmd+K).
                Matches the search icon on Quarto navbars. */}
            <button
              onClick={() => window.dispatchEvent(new CustomEvent('staffml:open-palette'))}
              style={{ ...S.navLink, padding: '8px 10px', cursor: 'pointer' }}
              aria-label="Search (Cmd+K)"
              title="Search (⌘K)"
            >
              <i className="bi bi-search" style={{ fontSize: 14 }} />
            </button>
          </div>
        </div>

        {/* Mobile hamburger — display is controlled entirely by Tailwind
            responsive classes; do NOT set `display` in the inline style,
            because an inline `display: flex` would beat the `nav-lg:hidden`
            media-query rule and leave the hamburger visible at 992 px+. */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="flex nav-lg:hidden items-center"
          style={{
            color: NAV_COLOR,
            padding: 8,
            cursor: 'pointer',
            border: 'none',
            background: 'none',
          }}
          aria-label="Toggle navigation"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={NAV_COLOR} strokeWidth="2">
            {mobileOpen
              ? <path d="M6 6L18 18M18 6L6 18" />
              : <path d="M3 6h18M3 12h18M3 18h18" />
            }
          </svg>
        </button>
      </div>

      {/* Mobile expanded */}
      {mobileOpen && (
        <div className="nav-lg:hidden" style={{ borderTop: `1px solid ${borderColor}`, backgroundColor: bgColor, padding: '12px 16px' }}>
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
                    color: item.active ? '#a51c30' : '#6c757d',
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
              style={{ fontSize: 15, color: NAV_COLOR, textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
              <i className="bi bi-heart" style={{ fontSize: 14 }} /> Support
            </a>
            <a href="https://github.com/harvard-edge/cs249r_book" target="_blank" rel="noopener noreferrer"
              style={{ fontSize: 15, color: NAV_COLOR, textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
              <i className="bi bi-star" style={{ fontSize: 14 }} /> Star
            </a>
            <a href="#subscribe"
              style={{ fontSize: 15, color: NAV_COLOR, textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
              <i className="bi bi-envelope" style={{ fontSize: 14 }} /> Subscribe
            </a>
            <a href="https://github.com/harvard-edge/cs249r_book" target="_blank" rel="noopener noreferrer"
              style={{ fontSize: 15, color: NAV_COLOR, textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
              <i className="bi bi-github" style={{ fontSize: 14 }} /> GitHub
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
