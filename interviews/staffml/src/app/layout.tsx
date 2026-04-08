import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";
import EcosystemBar from "@/components/EcosystemBar";
import Providers from "@/components/Providers";
import CommandPalette from "@/components/CommandPalette";
import KeyboardShortcutsOverlay from "@/components/KeyboardShortcutsOverlay";
import {
  SOCIAL_DESCRIPTION,
  SOCIAL_DESCRIPTION_LONG,
  SOCIAL_IMAGE_ALT,
  QUESTION_COUNT_FORMATTED,
} from "@/lib/stats";

// All user-visible question/topic counts come from src/lib/stats.ts, which
// derives them from src/data/vault-manifest.json at build time. Do not
// hand-type numbers into metadata — they go stale the next time the corpus
// regenerates and embarrass us on social cards.
export const metadata: Metadata = {
  metadataBase: new URL("https://staffml.ai"),
  title: {
    default: "StaffML — ML Systems Interview Prep",
    template: "%s | StaffML",
  },
  description: SOCIAL_DESCRIPTION,
  icons: {
    icon: "/favicon.svg",
  },
  openGraph: {
    title: "StaffML — ML Systems Interview Prep",
    description: SOCIAL_DESCRIPTION_LONG,
    type: "website",
    siteName: "StaffML",
    images: [{ url: "/og-image.svg", width: 1200, height: 630, alt: SOCIAL_IMAGE_ALT }],
  },
  twitter: {
    card: "summary_large_image",
    title: "StaffML — ML Systems Interview Prep",
    description: `${QUESTION_COUNT_FORMATTED} physics-grounded ML systems questions with napkin math. Free, open source, no accounts.`,
    images: ["/og-image.svg"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/*
          Dev-mode needs 'unsafe-eval' because Next.js's React Refresh / HMR
          runtime evaluates strings as JavaScript. Production locks this down.
        */}
        <meta
          httpEquiv="Content-Security-Policy"
          content={`default-src 'self'; script-src 'self' 'unsafe-inline'${
            process.env.NODE_ENV === "development" ? " 'unsafe-eval'" : ""
          }; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; connect-src 'self' https://api.github.com https://mlsysbook.ai https://harvard-edge.github.io; img-src 'self' data: https://mlsysbook.ai https://harvard-edge.github.io;`}
        />
        <script dangerouslySetInnerHTML={{ __html: `
          (function() {
            // Dark is the default site-wide; users can opt into light via the
            // theme toggle, which persists in localStorage.
            var t = localStorage.getItem('staffml_theme');
            if (t !== 'light' && t !== 'dark') t = 'dark';
            document.documentElement.dataset.theme = t;
          })();
        `}} />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
      </head>
      <body className="min-h-screen flex flex-col bg-background selection:bg-accentBlue/30 selection:text-textPrimary">
        <Providers>
          <EcosystemBar />
          <Nav />
          <main className="flex-1 flex flex-col">{children}</main>
          <CommandPalette />
          <KeyboardShortcutsOverlay />
        </Providers>
      </body>
    </html>
  );
}
