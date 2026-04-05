"use client";

import { useEffect } from "react";
import { ToastProvider } from "@/components/Toast";
import ThemeProvider from "@/components/ThemeProvider";
import { flushAnalytics, track } from "@/lib/analytics";

export default function Providers({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Session start — fire once per tab
    const hasAttempts = !!window.localStorage.getItem('staffml_progress');
    track({
      type: 'session_start',
      isReturning: hasAttempts,
      screenWidth: window.innerWidth,
    });

    // Flush pending analytics on page unload
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
