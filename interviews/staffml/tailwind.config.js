/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // Bootstrap-aligned breakpoints for the ecosystem navbar so
      // StaffML's collapse points match the Quarto sites exactly.
      // Defaults (sm/md/lg/xl/2xl) remain untouched for app layouts.
      //
      // Note on `nav-xl`: Quarto's _navbar.scss hides right-side text
      // labels in the range `1200px <= width < 1400px` (the "icon-only"
      // band where 6 dropdowns crowd 4 text labels). Setting `nav-xl` to
      // 1400 px — not 1200 — makes StaffML match that exact behavior:
      // labels appear only once the viewport is wide enough to fit them
      // alongside the full dropdown row.
      screens: {
        'nav-sm': '576px',   // bs sm — brand title visible
        'nav-lg': '992px',   // bs lg — desktop nav appears, hamburger hides
        'nav-xl': '1400px',  // Quarto's label-reveal threshold (post-icon band)
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
