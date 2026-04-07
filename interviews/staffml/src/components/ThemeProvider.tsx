"use client";

import { createContext, useContext, useEffect, useState, useCallback } from "react";

type Theme = "light" | "dark";

interface ThemeContextValue {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "dark",
  toggleTheme: () => {},
});

export function useTheme() {
  return useContext(ThemeContext);
}

function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "dark";
  const stored = localStorage.getItem("staffml_theme") as Theme | null;
  if (stored === "light" || stored === "dark") return stored;
  // Dark is the site-wide default. OS preference is intentionally NOT
  // honored — users can opt into light via the theme toggle.
  return "dark";
}

export default function ThemeProvider({ children }: { children: React.ReactNode }) {
  // SSR/first-paint defaults to dark to match the inline script in layout.tsx.
  // The useEffect below reconciles with the DOM value the inline script set,
  // which may be "light" if the user has an OS preference or stored choice.
  const [theme, setTheme] = useState<Theme>("dark");

  // Sync with actual DOM on mount (inline script already set data-theme)
  useEffect(() => {
    setTheme(getInitialTheme());
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("staffml_theme", next);
      return next;
    });
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}
