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
      // Quarto's `collapse-below: lg` setting for the app's own Nav.
      //
      // The ecosystem row has more menus than the app's own Nav. Keep its
      // hamburger until the compact row fits; reveal the full brand and
      // right-side labels only when the expanded row also fits.
      screens: {
        'nav-sm': '576px',   // brand title visible
        'nav-lg': '992px',   // desktop nav appears, hamburger hides (Bootstrap `lg`)
        'nav-ecosystem': '1152px', // compact ecosystem row fits with icons
        'nav-xl': '1536px',  // full brand and right-side labels fit
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
