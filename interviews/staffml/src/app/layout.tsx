import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StaffML | ML Systems Design",
  description: "Physics-grounded system design for Staff ML Engineers.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      {/* Adding JetBrains Mono for that premium IDE look */}
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
      </head>
      <body className="antialiased min-h-screen flex flex-col bg-black selection:bg-[#333] selection:text-white">
        {children}
      </body>
    </html>
  );
}
