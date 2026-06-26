"use client";

import { createContext, useContext, useEffect, useState, useCallback } from "react";

type Theme = "light" | "dark";

interface ThemeContextValue {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "light",
  toggleTheme: () => {},
});

export function useTheme() {
  return useContext(ThemeContext);
}

// Cross-site interop with the Quarto-built ecosystem: those subsites store
// the user's choice under `quarto-color-scheme`. We mirror in both
// directions so a user toggling on the book lands in dark mode here, and
// vice versa. Keep the key string in sync with shared/scripts/theme-persist.js
// and shared/config/site-head.html.
const QUARTO_KEY = "quarto-color-scheme";

// localStorage access can throw — not just on the method call, but on the
// property access itself — in strict-storage browsers (Safari "Block all
// cookies") and sandboxed iframes without allow-same-origin. Funnel every
// access through these guards so a future caller can't reintroduce an
// unguarded read/write that aborts a mount effect.
function safeGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch (_) {
    return null;
  }
}

function safeSet(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch (_) {
    /* storage unavailable; in-memory state still updates. */
  }
}

function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "light";
  const stored = safeGet("staffml_theme");
  if (stored === "light" || stored === "dark") return stored;
  // Secondary source: pick up the choice last set on a sibling subsite.
  const fromQuarto = safeGet(QUARTO_KEY);
  if (fromQuarto === "light" || fromQuarto === "dark") return fromQuarto;
  // Light is the ecosystem-wide default (matches the book, labs, kits,
  // slides, etc.). OS preference is intentionally NOT honored here — users
  // opt into dark via the theme toggle; the choice persists in
  // localStorage and mirrors to Quarto sites via quarto-color-scheme.
  return "light";
}

export default function ThemeProvider({ children }: { children: React.ReactNode }) {
  // SSR/first-paint defaults to light to match the inline script in layout.tsx.
  // The useEffect below reconciles with the DOM value the inline script set,
  // which may be "dark" if the user has a stored choice.
  const [theme, setTheme] = useState<Theme>("light");

  // Sync with actual DOM on mount (inline script already set data-theme)
  useEffect(() => {
    setTheme(getInitialTheme());
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      safeSet("staffml_theme", next);
      // Mirror to the ecosystem-shared key so Quarto subsites (book,
      // labs, kits, slides, ...) inherit the choice on next nav.
      safeSet(QUARTO_KEY, next);
      return next;
    });
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}
