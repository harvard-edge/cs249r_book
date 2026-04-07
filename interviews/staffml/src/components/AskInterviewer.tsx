"use client";

/**
 * Ask Interviewer panel — clarification ritual for Mock Interview mode.
 *
 * Three behaviors that auto-detect at runtime:
 *
 *   1. JOURNAL  — no NEXT_PUBLIC_INTERVIEWER_ENDPOINT configured.
 *                 User types clarifications, they're logged for post-mortem,
 *                 no AI call happens. Works for everyone, no infra needed.
 *
 *   2. HOSTED   — NEXT_PUBLIC_INTERVIEWER_ENDPOINT is set. Each clarification
 *                 is POSTed to the Worker, which forwards to whichever LLM
 *                 provider is configured (default: Cloudflare Workers AI
 *                 Llama 3.1 8B). Inline attribution shows the actual model
 *                 that answered.
 *
 *   3. FALLBACK — Hosted mode was configured but failed (rate-limited, 503,
 *                 network error). Shows the friendly error inline AND
 *                 surfaces the Copy-as-prompt button prominently. The user's
 *                 question is still logged for post-mortem.
 *
 * Copy-as-prompt is ALWAYS available as a button regardless of mode. It's
 * the universal safety net — works with any LLM the user has access to,
 * including local Ollama, ChatGPT, Claude, etc.
 *
 * The Socratic system prompt is enforced server-side in the Worker (so it
 * can't be bypassed) but is also embedded in the Copy-as-prompt output so
 * users pasting into other LLMs get the same constraint.
 */

import { useState, useRef, useEffect } from "react";
import {
  ChevronDown, ChevronRight, MessageCircle, Send, Info,
  ClipboardCopy, Check, AlertTriangle,
} from "lucide-react";
import clsx from "clsx";

const INTERVIEWER_ENDPOINT =
  process.env.NEXT_PUBLIC_INTERVIEWER_ENDPOINT?.replace(/\/+$/, "") || "";

// The same Socratic constraint enforced in the Worker. Embedded here so the
// Copy-as-prompt output carries it into whatever LLM the user pastes into.
const SOCRATIC_PROMPT_FOR_COPY = `You are a senior ML systems interviewer running a clarification round. Your only job is to answer the candidate's clarifying questions about constraints, scale, latency budgets, SLOs, traffic patterns, hardware availability, team size, and timeline.

You must NOT solve the problem. You must NOT propose architectures, algorithms, frameworks, or implementations. If the candidate asks "how should I do X" or "what's the right approach," redirect with: "That's the part I want to see you reason through. What constraint do you need from me first?"

Keep answers under 60 words. Be specific and concrete — give numbers when reasonable. Use a senior interviewer's tone: direct, no fluff, no apologies.`;

interface Message {
  /** Stable id for React keys + aria-live announcements. Generated client-side
   *  per message; never rely on array index because we reset and append. */
  id: string;
  role: "user" | "interviewer";
  text: string;
  // For interviewer messages: provenance metadata returned by the Worker
  vendorLabel?: string;
  modelLabel?: string;
  privacyNote?: string;
}

let _msgIdCounter = 0;
function nextMessageId(): string {
  _msgIdCounter += 1;
  return `m${_msgIdCounter}`;
}

interface AskInterviewerProps {
  /** The current question's scenario text. Sent as context to the LLM. */
  questionContext: string;
  /** Open by default (e.g. when realism = "open"). */
  defaultOpen?: boolean;
  /** Notified every time the user submits a clarification. Used by the
   *  gauntlet results phase to surface "you asked N clarifications." */
  onAsk?: (question: string) => void;
}

const HOSTED_AVAILABLE = INTERVIEWER_ENDPOINT.length > 0;

export default function AskInterviewer({ questionContext, defaultOpen = false, onAsk }: AskInterviewerProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const transcriptRef = useRef<HTMLDivElement>(null);
  // AbortController for the in-flight fetch. Cleared on questionContext change
  // so a stale response can never inject into the next question's transcript.
  const inFlightRef = useRef<AbortController | null>(null);

  // Reset transcript when the question changes. CRITICAL: also clear `busy`
  // and abort any in-flight fetch — otherwise navigating mid-fetch leaves
  // the input permanently disabled until the user reopens the panel.
  useEffect(() => {
    setMessages([]);
    setDraft("");
    setError(null);
    setBusy(false);
    if (inFlightRef.current) {
      inFlightRef.current.abort();
      inFlightRef.current = null;
    }
  }, [questionContext]);

  // Cleanup any in-flight fetch on unmount as well
  useEffect(() => {
    return () => {
      inFlightRef.current?.abort();
      inFlightRef.current = null;
    };
  }, []);

  // Auto-scroll the transcript on new messages — but only if the user is
  // already near the bottom. Don't yank them away from older messages they're
  // actively reading.
  useEffect(() => {
    const el = transcriptRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom < 64) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  const submit = async () => {
    const question = draft.trim();
    if (!question || busy) return;

    // Always log the user's clarification for post-mortem aggregation
    setMessages((prev) => [...prev, { id: nextMessageId(), role: "user", text: question }]);
    onAsk?.(question);
    setDraft("");
    setError(null);

    if (!HOSTED_AVAILABLE) {
      // Journal mode — no AI call. The clarification is the educational act.
      // Show a one-time inline note so users understand why there's no answer.
      setMessages((prev) => [
        ...prev,
        {
          id: nextMessageId(),
          role: "interviewer",
          text:
            "(journal mode) Your clarification is logged for the post-mortem. " +
            "Use the Copy-as-prompt button below to ask in your own LLM.",
        },
      ]);
      return;
    }

    // Tear down any older in-flight request before starting a new one.
    inFlightRef.current?.abort();
    const controller = new AbortController();
    inFlightRef.current = controller;
    setBusy(true);
    try {
      const res = await fetch(`${INTERVIEWER_ENDPOINT}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          question,
          context: questionContext,
          history: messages.map((m) => ({
            role: m.role,
            content: m.text,
          })),
        }),
      });

      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { message?: string; error?: string };
        const msg = data.message || data.error || `interviewer service returned ${res.status}`;
        setMessages((prev) => [
          ...prev,
          { id: nextMessageId(), role: "interviewer", text: `⚠ ${msg}` },
        ]);
        setError(msg);
        return;
      }

      const data = (await res.json()) as {
        answer: string;
        provider: string;
        vendorLabel: string;
        modelLabel: string;
        privacyNote: string;
      };
      setMessages((prev) => [
        ...prev,
        {
          id: nextMessageId(),
          role: "interviewer",
          text: data.answer,
          vendorLabel: data.vendorLabel,
          modelLabel: data.modelLabel,
          privacyNote: data.privacyNote,
        },
      ]);
    } catch (e) {
      // AbortError from a stale request is expected — swallow silently.
      if (e instanceof DOMException && e.name === "AbortError") return;
      const msg = e instanceof Error ? e.message : String(e);
      setMessages((prev) => [
        ...prev,
        { id: nextMessageId(), role: "interviewer", text: `⚠ Failed to reach interviewer service: ${msg}` },
      ]);
      setError(msg);
    } finally {
      // Only clear busy/inFlightRef if THIS controller is still the active one.
      // If a newer submit started (or questionContext changed), don't unset.
      if (inFlightRef.current === controller) {
        setBusy(false);
        inFlightRef.current = null;
      }
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      submit();
    }
  };

  // ─── Copy-as-prompt: build a self-contained prompt for any LLM ─────
  const copyAsPrompt = async () => {
    const userTurns = messages.filter((m) => m.role === "user");
    const numbered = userTurns.length > 0
      ? userTurns.map((m, i) => `${i + 1}. ${m.text}`).join("\n")
      : draft.trim()
      ? `1. ${draft.trim()}`
      : "(no clarifying questions yet — write yours below this prompt)";

    const text = `${SOCRATIC_PROMPT_FOR_COPY}

---

Scenario:
${questionContext}

My clarifying questions so far:
${numbered}

Please answer each as the interviewer.`;

    // navigator.clipboard is undefined on insecure-context (http on non-localhost)
    // and may throw on permission denial. Handle both cleanly.
    if (typeof navigator === "undefined" || !navigator.clipboard) {
      setError("Clipboard API not available — try using a secure (https) context.");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "permission denied";
      setError(`Could not write to clipboard: ${msg}`);
    }
  };

  const userClarificationCount = messages.filter((m) => m.role === "user").length;

  return (
    <div className="border-t border-border">
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls="ask-interviewer-body"
        className="w-full flex items-center justify-between px-4 py-2 text-[10px] font-mono text-textTertiary uppercase tracking-widest hover:text-textSecondary transition-colors"
      >
        <span className="flex items-center gap-1.5">
          <MessageCircle className="w-3 h-3" /> Ask Interviewer
          {userClarificationCount > 0 && (
            <span className="ml-1 px-1 text-[9px] font-bold bg-accentBlue/20 text-accentBlue rounded">
              {userClarificationCount}
            </span>
          )}
        </span>
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>

      {open && (
        <div id="ask-interviewer-body" className="px-4 pb-4">
          {/* Mode banner (journal vs hosted) — shown once at the top */}
          {messages.length === 0 && (
            HOSTED_AVAILABLE ? (
              <div className="flex items-start gap-2 mb-3 p-2.5 rounded-md bg-accentBlue/5 border border-accentBlue/20">
                <Info className="w-3.5 h-3.5 text-accentBlue shrink-0 mt-0.5" />
                <p className="text-[11px] text-textSecondary leading-relaxed">
                  Practice the clarification ritual real interviews reward. Your questions go to a
                  small AI interviewer with a Socratic constraint — it can answer constraints, never
                  solve the problem. <span className="font-semibold">AI may be wrong — verify against the model answer.</span>
                </p>
              </div>
            ) : (
              <div className="flex items-start gap-2 mb-3 p-2.5 rounded-md bg-accentAmber/10 border border-accentAmber/30">
                <Info className="w-3.5 h-3.5 text-accentAmber shrink-0 mt-0.5" />
                <p className="text-[11px] text-textSecondary leading-relaxed">
                  Journal mode. Your clarifying questions are logged for the post-mortem but no
                  AI is wired. Use the <span className="font-semibold">Copy as prompt</span> button
                  below to ask in your own LLM.
                </p>
              </div>
            )
          )}

          {/* Transcript */}
          {messages.length > 0 && (
            <div
              ref={transcriptRef}
              role="log"
              aria-live="polite"
              aria-label="Clarification transcript"
              className="mb-3 max-h-56 overflow-y-auto space-y-2.5 p-2.5 rounded-md border border-border bg-background"
            >
              {messages.map((m) => (
                <div key={m.id} className="text-[11px] leading-relaxed">
                  <span
                    className={clsx(
                      "font-mono text-[9px] uppercase mr-2",
                      m.role === "user" ? "text-accentBlue" : "text-accentGreen",
                    )}
                  >
                    {m.role === "user" ? "you" : "interviewer"}
                  </span>
                  <span className={m.role === "user" ? "text-textPrimary" : "text-textSecondary"}>
                    {m.text}
                  </span>
                  {m.role === "interviewer" && m.modelLabel && (
                    <div className="mt-1 ml-1 text-[9px] font-mono text-textTertiary/70 italic">
                      {m.modelLabel} via {m.vendorLabel} · AI may be wrong · check the model answer
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Composer */}
          <label className="sr-only" htmlFor="ask-interviewer-input">
            Ask a clarifying question
          </label>
          <textarea
            id="ask-interviewer-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a clarifying question — e.g. 'what's the latency budget?' or 'how many concurrent users?'"
            rows={2}
            className="w-full bg-background border border-border rounded-md p-2 font-mono text-[11px] text-textPrimary resize-none focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/60 leading-relaxed"
            spellCheck={false}
            disabled={busy}
          />

          <div className="flex items-center justify-between gap-2 mt-2">
            <span className="text-[10px] font-mono text-textTertiary">⌘↵ to send</span>
            <div className="flex items-center gap-2">
              <button
                onClick={copyAsPrompt}
                disabled={busy}
                title="Copy a self-contained prompt for use in any LLM (ChatGPT, Claude, Ollama, etc.)"
                className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[10px] font-medium text-textSecondary border border-border rounded hover:bg-surfaceHover transition-all disabled:opacity-40"
              >
                {copied ? <Check className="w-3 h-3 text-accentGreen" /> : <ClipboardCopy className="w-3 h-3" />}
                {copied ? "Copied" : "Copy as prompt"}
              </button>
              {HOSTED_AVAILABLE && (
                <button
                  onClick={submit}
                  disabled={busy || !draft.trim()}
                  className="inline-flex items-center gap-1.5 px-3 py-1 text-[11px] font-bold bg-accentBlue text-white rounded transition-all hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Send className="w-3 h-3" />
                  {busy ? "Asking…" : "Ask"}
                </button>
              )}
            </div>
          </div>

          {/* Privacy footer — shows the active provider's privacy note when present,
              otherwise the journal-mode default. Always visible so the social
              contract is in front of the user at the point of action. */}
          <div className="mt-3 pt-2 border-t border-borderSubtle flex items-start gap-1.5 text-[9px] text-textTertiary leading-relaxed">
            <AlertTriangle className="w-2.5 h-2.5 shrink-0 mt-0.5" />
            <span>
              {HOSTED_AVAILABLE ? (
                <>
                  Questions you type are sent to an external LLM service via StaffML's relay.
                  The relay does not log requests. Don't paste anything sensitive to your employer.
                  Your model answer (below) is the source of truth.
                </>
              ) : (
                <>
                  Journal mode — nothing leaves your browser. Use Copy-as-prompt and paste into the
                  LLM of your choice if you want answers.
                </>
              )}
            </span>
          </div>

          {error && !messages.find((m) => m.text.includes(error)) && (
            <p className="mt-2 text-[10px] text-accentRed" role="alert">
              {error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
