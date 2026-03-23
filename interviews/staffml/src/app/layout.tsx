import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";
import Providers from "@/components/Providers";

export const metadata: Metadata = {
  title: "StaffML — ML Systems Interview Prep",
  description: "Physics-grounded system design prep for Staff ML Engineers. 3,180+ questions across cloud, edge, mobile, and TinyML.",
  icons: {
    icon: "/favicon.svg",
  },
  openGraph: {
    title: "StaffML — ML Systems Interview Prep",
    description: "3,180+ physics-grounded system design questions with napkin math verification. Free, open source, backed by real hardware constants.",
    type: "website",
    siteName: "StaffML",
  },
  twitter: {
    card: "summary_large_image",
    title: "StaffML — ML Systems Interview Prep",
    description: "3,180+ physics-grounded questions with napkin math. Prep for Staff ML Systems interviews.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
      </head>
      <body className="min-h-screen flex flex-col bg-background selection:bg-accentBlue/30 selection:text-white">
        <Providers>
          <Nav />
          <main className="flex-1 flex flex-col">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
