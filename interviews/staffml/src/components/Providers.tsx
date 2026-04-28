"use client";

import { useEffect } from "react";
import { ToastProvider } from "@/components/Toast";
import ThemeProvider from "@/components/ThemeProvider";
import { CorpusProvider } from "@/lib/corpus-provider";
import { flushAnalytics, track } from "@/lib/analytics";
import { installErrorReporter } from "@/lib/error-reporter";

export default function Providers({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Capture runtime errors + unhandled rejections and forward them to
    // the analytics-worker as `client_error` events. Surfaces prod bugs
    // (like the hydration shape mismatch fixed in PR #1440) without
    // depending on users reporting them.
    installErrorReporter();

    // Session start — fire once per tab
    const hasAttempts = !!window.localStorage.getItem('staffml_progress');
    track({
      type: 'session_start',
      isReturning: hasAttempts,
      screenWidth: window.innerWidth,
    });

    // Flush pending analytics on page unload
    const handleUnload = () => flushAnalytics();
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden') flushAnalytics();
    };
    window.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('pagehide', handleUnload);
    return () => {
      window.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('pagehide', handleUnload);
    };
  }, []);

  return (
    <ThemeProvider>
      <CorpusProvider>
        <ToastProvider>
          {children}
        </ToastProvider>
      </CorpusProvider>
    </ThemeProvider>
  );
}
