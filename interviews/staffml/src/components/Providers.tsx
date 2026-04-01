"use client";

import { useEffect } from "react";
import { ToastProvider } from "@/components/Toast";
import ThemeProvider from "@/components/ThemeProvider";
import { flushAnalytics } from "@/lib/analytics";

export default function Providers({ children }: { children: React.ReactNode }) {
  // Flush pending analytics on page unload
  useEffect(() => {
    const handleUnload = () => flushAnalytics();
    window.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') flushAnalytics();
    });
    window.addEventListener('pagehide', handleUnload);
    return () => window.removeEventListener('pagehide', handleUnload);
  }, []);

  return (
    <ThemeProvider>
      <ToastProvider>
        {children}
      </ToastProvider>
    </ThemeProvider>
  );
}
