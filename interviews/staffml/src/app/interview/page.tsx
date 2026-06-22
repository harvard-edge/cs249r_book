"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic2, Play, Clock, CheckCircle2, BarChart3,
  Send, Square, Loader2, RotateCcw, ChevronDown, ChevronUp,
} from "lucide-react";
import clsx from "clsx";
import Link from "next/link";
import {
  getTracks, getLevels, getCompetencyAreas,
  getQuestionFullDetail, type Question,
} from "@/lib/corpus";
import { getLevelDef } from "@/lib/levels";
import { track as trackAnalytics } from "@/lib/analytics";
import MarkdownText from "@/components/MarkdownText";
import NapkinCalc from "@/components/NapkinCalc";
import HardwareRef from "@/components/HardwareRef";
import FirstRunExplainer from "@/components/FirstRunExplainer";
import { useToast } from "@/components/Toast";
import {
  createSession,
  advanceInChain,
  retreatInChain,
  selectNextChain,
  buildWorkerPayload,
  applyResponse,
  generateReport,
} from "@/lib/interview-conductor";
import { sendInterviewTurn } from "@/lib/interview-api";
import type {
  InterviewConfig,
  InterviewSession,
  InterviewReport,
  TranscriptEntry,
} from "@/lib/interview-types";

type Phase = "setup" | "active" | "feedback";

const TRACK_LABELS: Record<string, string> = {
  cloud: "Cloud",
  edge: "Edge",
  global: "Global",
  mobile: "Mobile",
  tinyml: "TinyML",
};

const DURATIONS = [
  { label: "Quick", minutes: 15 },
  { label: "Standard", minutes: 30 },
  { label: "Full", minutes: 45 },
];

const INTERVIEW_STORAGE_KEY = "staffml_interview_active";
const INTERVIEW_RESULTS_KEY = "staffml_interviews";

export default function InterviewPage() {
  const { show: showToast } = useToast();
  const [phase, setPhase] = useState<Phase>("setup");
  const [mounted, setMounted] = useState(false);

  // Setup
  const [selectedTrack, setSelectedTrack] = useState("cloud");
  const [selectedLevel, setSelectedLevel] = useState("L4");
  const [selectedDuration, setSelectedDuration] = useState(1);
  const [focusAreas, setFocusAreas] = useState<string[]>([]);

  // Active
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [userInput, setUserInput] = useState("");
  const [sending, setSending] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [toolsOpen, setToolsOpen] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Feedback
  const [report, setReport] = useState<InterviewReport | null>(null);

  // Resume check
  const [showResume, setShowResume] = useState(false);
  const [savedSession, setSavedSession] = useState<InterviewSession | null>(null);

  useEffect(() => { setMounted(true); }, []);

  useEffect(() => {
    if (!mounted) return;
    try {
      const raw = localStorage.getItem(INTERVIEW_STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw) as InterviewSession;
        if (saved.status === "active" && saved.transcript.length > 0) {
          setSavedSession(saved);
          setShowResume(true);
        }
      }
    } catch { /* ignore */ }
  }, [mounted]);

  // Timer
  useEffect(() => {
    if (phase !== "active" || !session) return;
    timerRef.current = setInterval(() => {
      setElapsed(e => e + 1);
    }, 1000);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [phase, session]);

  // Auto-scroll
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [session?.transcript.length]);

  // Save session to localStorage on changes
  useEffect(() => {
    if (session && phase === "active") {
      try {
        localStorage.setItem(INTERVIEW_STORAGE_KEY, JSON.stringify({ ...session, elapsedSeconds: elapsed }));
      } catch { /* ignore */ }
    }
  }, [session, elapsed, phase]);

  // ── Handlers ──────────────────────────────────────────────────

  const handleBegin = useCallback(async () => {
    const config: InterviewConfig = {
      track: selectedTrack,
      level: selectedLevel,
      duration: DURATIONS[selectedDuration].minutes,
      competencyFocus: focusAreas.length > 0 ? focusAreas : undefined,
    };

    const newSession = createSession(config);
    setSession(newSession);
    setElapsed(0);
    setPhase("active");

    trackAnalytics({
      type: "interview_started" as never,
      track: config.track,
      level: config.level,
      duration: config.duration,
    } as never);

    // Hydrate first question and get AI's opening
    let q: Question | null = null;
    if (newSession.currentQuestionId) {
      try {
        q = (await getQuestionFullDetail(newSession.currentQuestionId)) ?? null;
      } catch {
        // Vault worker may be unavailable; proceed without full question details
      }
    }
    setCurrentQuestion(q);
    await sendTurn(newSession, "", q, false);
  }, [selectedTrack, selectedLevel, selectedDuration, focusAreas]);

  const handleResume = useCallback(() => {
    if (savedSession) {
      setSession(savedSession);
      setElapsed(savedSession.elapsedSeconds || 0);
      setPhase("active");
      setShowResume(false);
      if (savedSession.currentQuestionId) {
        getQuestionFullDetail(savedSession.currentQuestionId).then(q => setCurrentQuestion(q ?? null));
      }
    }
  }, [savedSession]);

  const handleNewSession = useCallback(() => {
    setShowResume(false);
    setSavedSession(null);
    try { localStorage.removeItem(INTERVIEW_STORAGE_KEY); } catch { /* ignore */ }
  }, []);

  const sendTurn = useCallback(async (
    sess: InterviewSession,
    message: string,
    question: Question | null,
    timeExpired: boolean,
  ) => {
    setSending(true);
    try {
      const payload = await buildWorkerPayload(sess, message, question, timeExpired);
      const response = await sendInterviewTurn(payload);
      const updated = applyResponse(sess, response, message);

      // Handle chain navigation
      let nextSession = updated;
      let nextQuestion = question;

      const tryHydrate = async (id: string): Promise<Question | null> => {
        try { return (await getQuestionFullDetail(id)) ?? null; } catch { return null; }
      };

      if (response.meta.nextAction === "advance_chain") {
        const advanced = advanceInChain(updated);
        if (advanced) {
          nextSession = advanced;
          nextQuestion = await tryHydrate(advanced.currentQuestionId!);
        } else {
          const next = selectNextChain(updated);
          if (next) {
            nextSession = { ...updated, currentChainId: next.chainId, currentQuestionId: next.questionId, currentChainPosition: next.position };
            nextQuestion = await tryHydrate(next.questionId);
          }
        }
      } else if (response.meta.nextAction === "retreat_chain") {
        const retreated = retreatInChain(updated);
        if (retreated) {
          nextSession = retreated;
          nextQuestion = await tryHydrate(retreated.currentQuestionId!);
        }
      } else if (response.meta.nextAction === "switch_area") {
        const next = selectNextChain(updated);
        if (next) {
          nextSession = { ...updated, currentChainId: next.chainId, currentQuestionId: next.questionId, currentChainPosition: next.position };
          nextQuestion = await tryHydrate(next.questionId);
        }
      } else if (response.meta.nextAction === "conclude") {
        nextSession = { ...updated, status: "completed" };
      }

      setSession(nextSession);
      setCurrentQuestion(nextQuestion);

      if (nextSession.status === "completed" || response.meta.intent === "closing") {
        handleEnd(nextSession);
      }
    } catch (err) {
      showToast({ type: "info", title: "Interview error", description: err instanceof Error ? err.message : "Unknown error" });
    } finally {
      setSending(false);
    }
  }, [showToast]);

  const handleSend = useCallback(async () => {
    if (!session || !userInput.trim() || sending) return;
    const message = userInput.trim();
    setUserInput("");
    const timeExpired = elapsed >= session.config.duration * 60;
    await sendTurn(session, message, currentQuestion, timeExpired);
  }, [session, userInput, sending, elapsed, currentQuestion, sendTurn]);

  const handleEnd = useCallback((sess?: InterviewSession) => {
    const s = sess ?? session;
    if (!s) return;
    if (timerRef.current) clearInterval(timerRef.current);

    const finalSession = { ...s, status: "completed" as const, elapsedSeconds: elapsed };
    const rep = generateReport(finalSession);
    setReport(rep);
    setSession(finalSession);
    setPhase("feedback");

    try { localStorage.removeItem(INTERVIEW_STORAGE_KEY); } catch { /* ignore */ }

    // Save result
    try {
      const results = JSON.parse(localStorage.getItem(INTERVIEW_RESULTS_KEY) ?? "[]");
      results.push({
        id: finalSession.id,
        track: finalSession.config.track,
        level: finalSession.config.level,
        duration: Math.round(elapsed / 60),
        overallScore: rep.overallScore,
        areasAssessed: rep.areaBreakdown.length,
        completedAt: Date.now(),
      });
      if (results.length > 100) results.splice(0, results.length - 100);
      localStorage.setItem(INTERVIEW_RESULTS_KEY, JSON.stringify(results));
    } catch { /* ignore */ }

    trackAnalytics({
      type: "interview_completed" as never,
      track: finalSession.config.track,
      level: finalSession.config.level,
      overallScore: rep.overallScore,
      areasAssessed: rep.areaBreakdown.length,
    } as never);
  }, [session, elapsed]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  // ── Render ────────────────────────────────────────────────────

  if (!mounted) return null;

  // Resume modal
  if (showResume) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-8">
        <div className="max-w-md w-full bg-surface border border-border rounded-xl p-8 text-center">
          <Mic2 className="w-10 h-10 text-accentBlue mx-auto mb-4" />
          <h2 className="text-xl font-bold text-textPrimary mb-2">Interview in Progress</h2>
          <p className="text-textSecondary text-sm mb-6">
            You have an active interview with {savedSession?.transcript.length} exchanges.
            Resume where you left off or start fresh.
          </p>
          <div className="flex gap-3 justify-center">
            <button onClick={handleResume} className="px-6 py-2.5 rounded-lg bg-accentBlue text-white font-medium">
              Resume
            </button>
            <button onClick={handleNewSession} className="px-6 py-2.5 rounded-lg border border-border text-textSecondary hover:text-textPrimary">
              Start New
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <main className={clsx("bg-background", phase === "active" ? "h-screen overflow-hidden" : "min-h-screen")}>
      <div className={clsx("max-w-4xl mx-auto px-4 sm:px-6 lg:px-8", phase === "active" ? "pt-2 pb-0 h-full" : "py-8")}>
        <AnimatePresence mode="wait">
          {phase === "setup" && (
            <motion.div key="setup" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <SetupPhase
                selectedTrack={selectedTrack}
                setSelectedTrack={setSelectedTrack}
                selectedLevel={selectedLevel}
                setSelectedLevel={setSelectedLevel}
                selectedDuration={selectedDuration}
                setSelectedDuration={setSelectedDuration}
                focusAreas={focusAreas}
                setFocusAreas={setFocusAreas}
                onBegin={handleBegin}
              />
            </motion.div>
          )}

          {phase === "active" && session && (
            <motion.div key="active" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="flex flex-col h-full min-h-0">
              {/* Status bar */}
              <div className="flex items-center justify-between py-3 border-b border-border mb-4 shrink-0">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1.5 text-sm font-mono text-textSecondary">
                    <Clock className="w-3.5 h-3.5" />
                    {formatTime(elapsed)} / {session.config.duration} min
                  </div>
                  <div className="flex gap-1.5 flex-wrap">
                    {Object.entries(session.areaAssessments).slice(0, 8).map(([area, assessment]) => (
                      <span key={area} className={clsx(
                        "text-[10px] px-1.5 py-0.5 rounded font-mono",
                        assessment.questionsAsked > 0 ? "bg-accentGreen/20 text-accentGreen" : "bg-surface text-textMuted",
                      )}>
                        {area.slice(0, 4)}
                      </span>
                    ))}
                  </div>
                </div>
                <button
                  onClick={() => handleEnd()}
                  className="text-xs px-3 py-1.5 rounded border border-border text-textSecondary hover:text-accentRed hover:border-accentRed/50 transition-colors"
                >
                  <Square className="w-3 h-3 inline mr-1" />
                  End
                </button>
              </div>

              {/* Transcript */}
              <div className="flex-1 overflow-y-auto space-y-4 pb-4">
                {session.transcript.map((entry) => (
                  <div key={entry.id} className={clsx(
                    "flex",
                    entry.role === "candidate" ? "justify-end" : "justify-start",
                  )}>
                    <div className={clsx(
                      "max-w-[85%] rounded-xl px-4 py-3 text-sm leading-relaxed",
                      entry.role === "candidate"
                        ? "bg-accentBlue/10 text-textPrimary border border-accentBlue/20"
                        : "bg-surface border border-border text-textSecondary",
                    )}>
                      {entry.role === "interviewer" ? (
                        <MarkdownText text={entry.text} />
                      ) : (
                        <span>{entry.text}</span>
                      )}
                    </div>
                  </div>
                ))}

                {sending && (
                  <div className="flex justify-start">
                    <div className="bg-surface border border-border rounded-xl px-4 py-3">
                      <Loader2 className="w-4 h-4 animate-spin text-textMuted" />
                    </div>
                  </div>
                )}
                <div ref={transcriptEndRef} />
              </div>

              {/* Input area */}
              <div className="shrink-0 border-t border-border pt-4 space-y-3">
                {/* Tools drawer */}
                <div>
                  <button
                    onClick={() => setToolsOpen(o => !o)}
                    className="text-[11px] font-mono text-textMuted flex items-center gap-1 hover:text-textSecondary"
                  >
                    {toolsOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                    Tools
                  </button>
                  {toolsOpen && (
                    <div className="mt-2 grid grid-cols-2 gap-3">
                      <NapkinCalc />
                      <HardwareRef />
                    </div>
                  )}
                </div>

                <div className="flex gap-2">
                  <textarea
                    ref={textareaRef}
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your response..."
                    disabled={sending}
                    rows={3}
                    className="flex-1 resize-none rounded-lg border border-border bg-background px-4 py-3 text-sm text-textPrimary placeholder:text-textMuted focus:outline-none focus:border-accentBlue disabled:opacity-50"
                  />
                  <button
                    onClick={handleSend}
                    disabled={sending || !userInput.trim()}
                    className="self-end px-4 py-3 rounded-lg bg-accentBlue text-white font-medium disabled:opacity-30 hover:bg-accentBlue/90 transition-colors"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-[10px] text-textMuted text-center">
                  {"⌘"}+Enter to send
                </p>
              </div>
            </motion.div>
          )}

          {phase === "feedback" && report && session && (
            <motion.div key="feedback" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <FeedbackPhase
                report={report}
                session={session}
                onNewInterview={() => { setPhase("setup"); setSession(null); setReport(null); }}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}

// ─── Setup Phase ────────────────────────────────────────────────────────────

function SetupPhase({
  selectedTrack, setSelectedTrack,
  selectedLevel, setSelectedLevel,
  selectedDuration, setSelectedDuration,
  focusAreas, setFocusAreas,
  onBegin,
}: {
  selectedTrack: string; setSelectedTrack: (t: string) => void;
  selectedLevel: string; setSelectedLevel: (l: string) => void;
  selectedDuration: number; setSelectedDuration: (d: number) => void;
  focusAreas: string[]; setFocusAreas: (a: string[]) => void;
  onBegin: () => void;
}) {
  const tracks = getTracks();
  const levels = getLevels().filter(l => ["L4", "L5", "L6+"].includes(l));
  const areas = getCompetencyAreas();

  const toggleArea = (area: string) => {
    setFocusAreas(focusAreas.includes(area)
      ? focusAreas.filter(a => a !== area)
      : [...focusAreas, area]);
  };

  return (
    <div className="max-w-2xl mx-auto">
      <FirstRunExplainer mode={"interview" as never} />

      <div className="text-center mb-10">
        <Mic2 className="w-10 h-10 text-accentBlue mx-auto mb-3" />
        <h1 className="text-3xl font-bold text-textPrimary mb-2">Live Interview</h1>
        <p className="text-textSecondary text-sm">
          A conversational mock interview powered by AI. The interviewer adapts to your responses,
          follows up on weak areas, and covers multiple competency areas.
        </p>
      </div>

      {/* Track */}
      <div className="mb-8">
        <label className="block text-xs font-mono text-textMuted uppercase tracking-wider mb-3">Track</label>
        <div className="flex flex-wrap gap-2">
          {tracks.map(t => (
            <button key={t} onClick={() => setSelectedTrack(t)}
              className={clsx("px-4 py-2 rounded-lg text-sm font-medium border transition-colors",
                selectedTrack === t
                  ? "bg-accentBlue/10 border-accentBlue text-accentBlue"
                  : "border-border text-textSecondary hover:text-textPrimary hover:border-textMuted")}>
              {TRACK_LABELS[t] ?? t}
            </button>
          ))}
        </div>
      </div>

      {/* Level */}
      <div className="mb-8">
        <label className="block text-xs font-mono text-textMuted uppercase tracking-wider mb-3">Target Level</label>
        <div className="flex flex-wrap gap-2">
          {levels.map(l => {
            const def = getLevelDef(l);
            return (
              <button key={l} onClick={() => setSelectedLevel(l)}
                className={clsx("px-4 py-2 rounded-lg text-sm font-medium border transition-colors",
                  selectedLevel === l
                    ? "bg-accentBlue/10 border-accentBlue text-accentBlue"
                    : "border-border text-textSecondary hover:text-textPrimary hover:border-textMuted")}>
                {l} {def?.name ? `— ${def.name}` : ""}
              </button>
            );
          })}
        </div>
      </div>

      {/* Duration */}
      <div className="mb-8">
        <label className="block text-xs font-mono text-textMuted uppercase tracking-wider mb-3">Duration</label>
        <div className="flex gap-2">
          {DURATIONS.map((d, i) => (
            <button key={i} onClick={() => setSelectedDuration(i)}
              className={clsx("px-4 py-2 rounded-lg text-sm font-medium border transition-colors",
                selectedDuration === i
                  ? "bg-accentBlue/10 border-accentBlue text-accentBlue"
                  : "border-border text-textSecondary hover:text-textPrimary hover:border-textMuted")}>
              {d.label} ({d.minutes} min)
            </button>
          ))}
        </div>
      </div>

      {/* Focus areas */}
      <div className="mb-10">
        <label className="block text-xs font-mono text-textMuted uppercase tracking-wider mb-3">
          Focus Areas <span className="text-textMuted">(optional)</span>
        </label>
        <div className="flex flex-wrap gap-2">
          {areas.map(a => (
            <button key={a} onClick={() => toggleArea(a)}
              className={clsx("px-3 py-1.5 rounded text-xs font-medium border transition-colors",
                focusAreas.includes(a)
                  ? "bg-accentBlue/10 border-accentBlue text-accentBlue"
                  : "border-border text-textMuted hover:text-textSecondary hover:border-textMuted")}>
              {a}
            </button>
          ))}
        </div>
        {focusAreas.length === 0 && (
          <p className="text-[11px] text-textMuted mt-2">Leave empty to cover all areas.</p>
        )}
      </div>

      {/* Begin */}
      <div className="text-center">
        <button onClick={onBegin}
          className="px-8 py-3 rounded-xl bg-accentBlue text-white font-semibold text-lg hover:bg-accentBlue/90 transition-colors inline-flex items-center gap-2">
          <Play className="w-5 h-5" />
          Begin Interview
        </button>
      </div>
    </div>
  );
}

// ─── Feedback Phase ─────────────────────────────────────────────────────────

function FeedbackPhase({
  report,
  session,
  onNewInterview,
}: {
  report: InterviewReport;
  session: InterviewSession;
  onNewInterview: () => void;
}) {
  const [showTranscript, setShowTranscript] = useState(false);

  const ratingColor = (r: number) => {
    if (r >= 2.5) return "text-accentGreen";
    if (r >= 1.5) return "text-yellow-500";
    return "text-accentRed";
  };

  const ratingLabel = (r: number) => {
    if (r >= 2.5) return "Strong";
    if (r >= 1.5) return "Adequate";
    if (r > 0) return "Weak";
    return "Not covered";
  };

  return (
    <div className="max-w-2xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <CheckCircle2 className="w-10 h-10 text-accentGreen mx-auto mb-3" />
        <h1 className="text-2xl font-bold text-textPrimary mb-2">Interview Complete</h1>
        <p className="text-textSecondary text-sm mb-4">{report.summary}</p>
        <div className="inline-flex items-center gap-2 bg-surface border border-border rounded-full px-6 py-2">
          <span className="text-3xl font-bold text-textPrimary">{report.overallScore}</span>
          <span className="text-textMuted text-sm">/100</span>
        </div>
        <p className="text-xs text-textMuted mt-2">
          {report.questionsDiscussed} questions across {report.areaBreakdown.length} areas in {report.duration} min
        </p>
      </div>

      {/* Area breakdown */}
      <div className="mb-8">
        <h2 className="text-xs font-mono text-textMuted uppercase tracking-wider mb-3">Competency Coverage</h2>
        <div className="space-y-2">
          {report.areaBreakdown.map(a => (
            <div key={a.area} className="flex items-center gap-3">
              <span className="text-sm text-textSecondary w-28 truncate">{a.area}</span>
              <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
                <div className={clsx("h-full rounded-full transition-all", {
                  "bg-accentGreen": a.rating >= 2.5,
                  "bg-yellow-500": a.rating >= 1.5 && a.rating < 2.5,
                  "bg-accentRed": a.rating > 0 && a.rating < 1.5,
                  "bg-textMuted/20": a.rating === 0,
                })} style={{ width: `${Math.max((a.rating / 3) * 100, 5)}%` }} />
              </div>
              <span className={clsx("text-xs font-medium w-16 text-right", ratingColor(a.rating))}>
                {ratingLabel(a.rating)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Strengths */}
      {report.strengths.length > 0 && (
        <div className="mb-6">
          <h2 className="text-xs font-mono text-textMuted uppercase tracking-wider mb-2">Strengths</h2>
          <ul className="space-y-1">
            {report.strengths.map((s, i) => (
              <li key={i} className="text-sm text-accentGreen flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                <span>{s}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Weaknesses */}
      {report.weaknesses.length > 0 && (
        <div className="mb-6">
          <h2 className="text-xs font-mono text-textMuted uppercase tracking-wider mb-2">Areas to Improve</h2>
          <ul className="space-y-1">
            {report.weaknesses.map((w, i) => (
              <li key={i} className="text-sm text-yellow-600 flex items-start gap-2">
                <BarChart3 className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                <span>{w}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Practice recommendations */}
      {report.practiceRecs.length > 0 && (
        <div className="mb-8">
          <h2 className="text-xs font-mono text-textMuted uppercase tracking-wider mb-2">Practice These</h2>
          <div className="space-y-2">
            {report.practiceRecs.map((rec, i) => (
              <div key={i} className="p-3 rounded-lg border border-border bg-surface">
                <p className="text-sm text-textSecondary mb-1">{rec.reason}</p>
                <div className="flex flex-wrap gap-1.5">
                  {rec.questionIds.slice(0, 4).map(qId => (
                    <Link key={qId} href={`/practice?q=${qId}`}
                      className="text-xs px-2 py-1 rounded bg-accentBlue/10 text-accentBlue hover:bg-accentBlue/20 transition-colors">
                      {qId}
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Transcript toggle */}
      <div className="mb-8">
        <button onClick={() => setShowTranscript(s => !s)}
          className="text-xs font-mono text-textMuted hover:text-textSecondary flex items-center gap-1">
          {showTranscript ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          {showTranscript ? "Hide" : "View"} Full Transcript ({session.transcript.length} messages)
        </button>
        {showTranscript && (
          <div className="mt-3 space-y-3 max-h-96 overflow-y-auto">
            {session.transcript.map(entry => (
              <div key={entry.id} className={clsx("text-sm p-3 rounded-lg", {
                "bg-accentBlue/5 border border-accentBlue/10": entry.role === "candidate",
                "bg-surface border border-border": entry.role === "interviewer",
              })}>
                <span className="text-[10px] font-mono text-textMuted uppercase">
                  {entry.role === "interviewer" ? "Interviewer" : "You"}
                </span>
                <p className="mt-1 text-textSecondary">{entry.text}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex justify-center gap-3">
        <button onClick={onNewInterview}
          className="px-6 py-2.5 rounded-lg bg-accentBlue text-white font-medium inline-flex items-center gap-2">
          <RotateCcw className="w-4 h-4" />
          New Interview
        </button>
        <Link href="/progress"
          className="px-6 py-2.5 rounded-lg border border-border text-textSecondary hover:text-textPrimary inline-flex items-center gap-2">
          <BarChart3 className="w-4 h-4" />
          Progress
        </Link>
      </div>
    </div>
  );
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}
