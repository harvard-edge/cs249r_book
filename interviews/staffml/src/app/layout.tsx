import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";
import EcosystemBar from "@/components/EcosystemBar";
import AnnouncementBar from "@/components/AnnouncementBar";
import MaybeFooter from "@/components/MaybeFooter";
import Providers from "@/components/Providers";
import { QUESTION_COUNT_DISPLAY } from "@/lib/corpus";

export const metadata: Metadata = {
  metadataBase: new URL("https://staffml.ai"),
  title: {
    default: "StaffML — ML Systems Interview Prep",
    template: "%s | StaffML",
  },
  description: `Physics-grounded system design prep for ML Engineers. ${QUESTION_COUNT_DISPLAY} questions across cloud, edge, mobile, and TinyML. 100% client-side.`,
  icons: {
    icon: "/favicon.svg",
  },
  openGraph: {
    title: "StaffML — ML Systems Interview Prep",
    description: `${QUESTION_COUNT_DISPLAY} physics-grounded ML systems questions with napkin math verification. Free, open source, no accounts, runs entirely in your browser.`,
    type: "website",
    siteName: "StaffML",
    images: [{ url: "/og-image.svg", width: 1200, height: 630, alt: `StaffML — ${QUESTION_COUNT_DISPLAY} ML systems interview questions` }],
  },
  twitter: {
    card: "summary_large_image",
    title: "StaffML — ML Systems Interview Prep",
    description: `${QUESTION_COUNT_DISPLAY} physics-grounded ML systems questions with napkin math. Free, open source, no accounts.`,
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
          Content Security Policy.

          NOTE: 'unsafe-inline' in script-src is REQUIRED for Next.js App
          Router. Next.js 15 emits multiple inline <script> tags for React
          Server Components streaming (`self.__next_f.push(...)`) that
          carry the RSC payload — without 'unsafe-inline' the browser blocks
          them and React hydration silently fails in production. A proper
          fix would be a Next.js Middleware that injects per-request nonces,
          but this app is a STATIC export (no server runtime) so middleware
          nonces are not available. We accept the residual XSS risk from
          script-src 'unsafe-inline' and mitigate it with the OTHER controls
          on this list (no DOM-injection sinks in app code, escaped React
          rendering everywhere, and the safeHref helper for corpus URLs).

          We DID externalize our own theme bootstrap to
          public/theme-bootstrap.js — defense in depth, one less inline
          script of our own that could ever drift, even though we can't
          fully drop 'unsafe-inline' because of Next.js itself.

          'unsafe-inline' for style-src is needed by Tailwind + framer-motion.

          frame-ancestors and X-Frame-Options are intentionally NOT here:
          per CSP Level 2 §6.2, frame-ancestors is IGNORED when delivered
          via <meta http-equiv>. X-Frame-Options can only ship via HTTP
          response headers, not meta. Static exports on GitHub Pages cannot
          set HTTP headers, so clickjacking defense requires either a
          reverse proxy (Cloudflare Pages, Netlify) or moving to a hosting
          target that supports custom headers. Tracked as known limitation.

          base-uri and form-action ARE honored in meta CSP and stay in.
        */}
        <meta
          httpEquiv="Content-Security-Policy"
          content={`default-src 'self'; script-src 'self' 'unsafe-inline' https://static.cloudflareinsights.com${
            process.env.NODE_ENV === "development" ? " 'unsafe-eval'" : ""
          }; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; connect-src 'self' https://api.github.com https://mlsysbook.ai https://harvard-edge.github.io https://staffml-vault.mlsysbook-ai-account.workers.dev https://staffml-vault.mlsysbook.ai https://cloudflareinsights.com; img-src 'self' data: https://mlsysbook.ai https://harvard-edge.github.io; base-uri 'self'; form-action 'self';`}
        />
        {/*
          Theme bootstrap — render-blocking external script (no async/defer)
          so it runs before the body parses and avoids FOUC. Source lives at
          public/theme-bootstrap.js — keep it small and side-effect-only.
        */}
        <script src={`${process.env.NEXT_PUBLIC_BASE_PATH || ""}/theme-bootstrap.js`} />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
      </head>
      <body className="min-h-dvh flex flex-col bg-background selection:bg-accentBlue/30 selection:text-textPrimary">
        <Providers>
          <EcosystemBar />
          <Nav />
          <AnnouncementBar />
          <main className="flex-1 flex flex-col">{children}</main>
          <MaybeFooter />
        </Providers>
      </body>
    </html>
  );
}
