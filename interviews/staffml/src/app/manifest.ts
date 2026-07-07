import type { MetadataRoute } from "next";
import { QUESTION_COUNT_DISPLAY } from "@/lib/corpus";

/**
 * Web app manifest — makes StaffML installable as a PWA.
 *
 * Next.js emits this as `/manifest.webmanifest` at build time and injects
 * the `<link rel="manifest">` into every page, so it works with the static
 * export. Icon/start_url paths must carry NEXT_PUBLIC_BASE_PATH manually
 * (same convention as the og-image URLs in layout.tsx) — Next only
 * base-paths the manifest route itself, not the URLs inside it.
 *
 * Icons are rasterized from public/favicon.svg (the roofline mark). The
 * `maskable` variants use a full-bleed background with the mark inside the
 * safe zone so Android's circle/squircle masks never clip it;
 * apple-touch-icon (linked in layout.tsx) is full-bleed too because iOS
 * applies its own corner radius.
 */
// Required by `output: export` — metadata routes must opt into static
// rendering explicitly or `next build` fails collecting page data.
export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  const base = process.env.NEXT_PUBLIC_BASE_PATH || "";
  return {
    name: "StaffML — ML Systems Interview Prep",
    short_name: "StaffML",
    description: `Physics-grounded system design prep for ML Engineers. ${QUESTION_COUNT_DISPLAY} questions across cloud, edge, mobile, and TinyML. 100% client-side.`,
    id: `${base}/`,
    start_url: `${base}/`,
    scope: `${base}/`,
    display: "standalone",
    // Default theme is light (see public/theme-bootstrap.js) — match it so
    // the installed app's window chrome doesn't flash a mismatched color.
    background_color: "#ffffff",
    theme_color: "#ffffff",
    icons: [
      { src: `${base}/icons/icon-192.png`, sizes: "192x192", type: "image/png", purpose: "any" },
      { src: `${base}/icons/icon-512.png`, sizes: "512x512", type: "image/png", purpose: "any" },
      { src: `${base}/icons/icon-maskable-192.png`, sizes: "192x192", type: "image/png", purpose: "maskable" },
      { src: `${base}/icons/icon-maskable-512.png`, sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
