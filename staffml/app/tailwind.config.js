/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // Breakpoints for the ecosystem navbar (EcosystemBar.tsx).
      // Defaults (sm/md/lg/xl/2xl) remain untouched for app layouts.
      //
      // Aligned with Bootstrap (which the rest of the Quarto ecosystem
      // uses): `nav-lg` = 992 matches Bootstrap's `lg` breakpoint and
      // Quarto's `collapse-below: lg` setting, so StaffML collapses to
      // hamburger at the same viewport as every other ecosystem subsite.
      //
      // The corollary at iPad landscape (1024–1194 px): the full desktop
      // nav renders but the right-side icons get visually clipped past
      // the viewport edge. That's the same trade-off Quarto makes — see
      // EcosystemBar.tsx's `overflowX: 'clip'` comment for how we keep
      // the page from horizontally scrolling despite the overflow.
      //
      // Note on `nav-xl`: Quarto's _navbar.scss hides right-side text
      // labels in the range `1200px <= width < 1400px` (the "icon-only"
      // band where dropdowns crowd text labels). Setting `nav-xl` to
      // 1400 px keeps that label-reveal behavior: labels appear only
      // once the viewport is wide enough to fit them alongside the full
      // dropdown row.
      screens: {
        'nav-sm': '576px',   // brand title visible
        'nav-lg': '992px',   // desktop nav appears, hamburger hides (Bootstrap `lg`)
        'nav-xl': '1400px',  // right-side label-reveal threshold (post-icon band)
      },
      colors: {
        background: "var(--background)",
        surface: "var(--surface)",
        surfaceHover: "var(--surface-hover)",
        surfaceElevated: "var(--surface-elevated)",
        border: "var(--border)",
        borderSubtle: "var(--border-subtle)",
        borderHighlight: "var(--border-highlight)",
        textPrimary: "var(--text-primary)",
        textSecondary: "var(--text-secondary)",
        textTertiary: "var(--text-tertiary)",
        textMuted: "var(--text-muted)",
        accentBlue: "var(--accent-blue)",
        accentRed: "var(--accent-red)",
        accentAmber: "var(--accent-amber)",
        accentGreen: "var(--accent-green)",
        accentPurple: "var(--accent-purple)",
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"IBM Plex Mono"', 'monospace'],
        sans: ['"Inter"', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
