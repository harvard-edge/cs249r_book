"use client";

import { useState, useEffect, useCallback, createContext, useContext } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Trophy, Flame, CheckCircle2 } from "lucide-react";
import clsx from "clsx";

interface ToastMessage {
  id: number;
  type: 'success' | 'badge' | 'info';
  title: string;
  description?: string;
}

interface ToastContextValue {
  show: (msg: Omit<ToastMessage, 'id'>) => void;
}

const ToastContext = createContext<ToastContextValue>({ show: () => {} });

export function useToast() {
  return useContext(ToastContext);
}

let nextId = 0;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const show = useCallback((msg: Omit<ToastMessage, 'id'>) => {
    const id = nextId++;
    setToasts(prev => [...prev, { ...msg, id }]);
    // Auto-dismiss after 4s
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 4000);
  }, []);

  const dismiss = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ show }}>
      {children}
      {/* Toast container */}
      <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 max-w-sm">
        <AnimatePresence>
          {toasts.map(toast => (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              className={clsx(
                "flex items-start gap-3 p-4 rounded-xl border shadow-lg backdrop-blur-md",
                toast.type === 'badge'
                  ? "bg-accentAmber/10 border-accentAmber/30"
                  : toast.type === 'success'
                  ? "bg-accentGreen/10 border-accentGreen/30"
                  : "bg-surface border-border"
              )}
            >
              {toast.type === 'badge' && <Trophy className="w-5 h-5 text-accentAmber shrink-0 mt-0.5" />}
              {toast.type === 'success' && <CheckCircle2 className="w-5 h-5 text-accentGreen shrink-0 mt-0.5" />}
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-white">{toast.title}</div>
                {toast.description && (
                  <div className="text-xs text-textSecondary mt-0.5">{toast.description}</div>
                )}
              </div>
              <button
                onClick={() => dismiss(toast.id)}
                className="text-textTertiary hover:text-white transition-colors shrink-0"
              >
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}
