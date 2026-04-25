"use client";

import { usePathname } from "next/navigation";

import Footer from "./Footer";

/**
 * Conditionally renders the global Footer.
 *
 * The footer is intentionally hidden on workspace tools (Practice,
 * Gauntlet, Simulator, Roofline) where the page is a self-contained
 * full-viewport surface with its own sticky bottom chrome. Showing the
 * site footer below those pages reads as orphaned chrome and creates a
 * visible empty band between the workspace and the footer attribution
 * row (the footer's own `mt-12` margin shows through).
 *
 * Content-style routes (/about, /contribute, etc.) keep the footer.
 *
 * Add a route here if it becomes a workspace tool. Removing one is fine
 * too — falling back to the footer is harmless.
 */
const ROUTES_WITHOUT_FOOTER = new Set<string>([
  "/practice",
  "/gauntlet",
  "/explore",
  "/simulator",
  "/roofline",
]);

export default function MaybeFooter() {
  const pathname = usePathname();
  // Normalize: static exports served as flat HTML files give a pathname
  // like "/practice.html" instead of "/practice", and some hosts add a
  // trailing slash. Strip both before lookup. Default to rendering the
  // footer if pathname is unavailable — a momentary footer flash on a
  // tool page is far better than a missing footer on a content page.
  const route = (pathname ?? "")
    .replace(/\.html$/, "")
    .replace(/\/+$/, "") || "/";
  if (ROUTES_WITHOUT_FOOTER.has(route)) return null;
  return <Footer />;
}
