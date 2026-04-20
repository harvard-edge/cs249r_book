"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion } from "framer-motion";
import {
  Crosshair, Play, Clock, CheckCircle2, XCircle, Terminal,
  RotateCcw, BarChart3, AlertTriangle, Share2, Check
} from "lucide-react";
import clsx from "clsx";
import Link from "next/link";
import {
  getTracks, getLevels, selectGauntletQuestions,
  getQuestionsByFilter, Question, cleanScenario
} from "@/lib/corpus";
import { getLevelDef } from "@/lib/levels";
import { buildReportUrl } from "@/lib/issue-url";
import { track } from "@/lib/analytics";
import { saveAttempt, saveGauntletResult, AttemptRecord, recordActivity, updateSRCard } from "@/lib/progress";
import { extractRubric, rubricToScore, RubricItem } from "@/lib/rubric";
import { useToast } from "@/components/Toast";
import NapkinMathDisplay from "@/components/NapkinMathDisplay";
import QuestionFeedback from "@/components/QuestionFeedback";
import HardwareRef from "@/components/HardwareRef";
import NapkinCalc from "@/components/NapkinCalc";
import AskInterviewer from "@/components/AskInterviewer";
import FirstRunExplainer from "@/components/FirstRunExplainer";

type Phase = "setup" | "active" | "review" | "results";

const DURATIONS = [
  { label: "Quick (5 Qs)", questions: 5, minutes: 10 },
  { label: "Standard (10 Qs)", questions: 10, minutes: 20 },
  { label: "Full (15 Qs)", questions: 15, minutes: 35 },
  { label: "Design (1 deep)", questions: 1, minutes: 15 },
];

export default function GauntletPage() {
  const { show: showToast } = useToast();
  const [phase, setPhase] = useState<Phase>("setup");
  const [mounted, setMounted] = useState(false);

  // Setup state
  const [selectedTrack, setSelectedTrack] = useState("cloud");
  const [selectedLevel, setSelectedLevel] = useState("L4");
  const [selectedDuration, setSelectedDuration] = useState(1); // index into DURATIONS
  const [availableCount, setAvailableCount] = useState(0);

  // Realism: how interview-like is the session?
  //   strict   = no Hardware Reference, no Napkin Calc, no Ask Interviewer
  //   standard = tools available, collapsed by default (current default)
  //   open     = tools available, expanded by default
  // Persisted in localStorage so the user's choice survives reloads.
  type Realism = "strict" | "standard" | "open";
  const [realism, setRealism] = useState<Realism>("standard");
  useEffect(() => {
    try {
      const saved = localStorage.getItem("staffml_gauntlet_realism");
      if (saved === "strict" || saved === "standard" || saved === "open") {
        setRealism(saved);
      }
    } catch { /* localStorage may be unavailable */ }
  }, []);
  const updateRealism = (r: Realism) => {
    setRealism(r);
    try { localStorage.setItem("staffml_gauntlet_realism", r); } catch { /* ignore */ }
  };

  // Active state
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [userAnswer, setUserAnswer] = useState("");
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Review state (self-assessment per question)
  const [scores, setScores] = useState<number[]>([]);
  const [copied, setCopied] = useState(false);
  const [rubricItems, setRubricItems] = useState<RubricItem[]>([]);

  // Per-question clarifications log — surfaced in the results phase as
  // "you asked N clarifications on this problem." Powers the metacognitive
  // post-mortem ritual for both Journal and Hosted AskInterviewer modes.
  const [clarifications, setClarifications] = useState<Record<string, string[]>>({});
  const recordClarification = (qId: string, q: string) => {
    setClarifications((prev) => ({
      ...prev,
      [qId]: [...(prev[qId] || []), q],
    }));
  };

  const tracks = getTracks().filter(t => t !== "global");
  const levels = getLevels();

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const count = getQuestionsByFilter({ track: selectedTrack, level: selectedLevel }).length;
    setAvailableCount(count);
  }, [selectedTrack, selectedLevel]);

  // Timer
  useEffect(() => {
    if (phase === "active" && timeRemaining > 0 && !showAnswer) {
      timerRef.current = setInterval(() => {
        setTimeRemaining(prev => {
          if (prev <= 1) {
            clearInterval(timerRef.current!);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
      return () => clearInterval(timerRef.current!);
    }
  }, [phase, showAnswer]);

  // Time expired — auto-finish
  useEffect(() => {
    if (phase === "active" && timeRemaining === 0) {
      // Score unanswered questions as 0 (skipped)
      const remaining = questions.length - scores.length;
      const finalScores = [...scores];
      for (let i = 0; i < remaining; i++) {
        const qIdx = scores.length + i;
        if (qIdx < questions.length) {
          const q = questions[qIdx];
          saveAttempt({
            questionId: q.id,
            competencyArea: q.competency_area,
            track: q.track,
            level: q.level,
            selfScore: 0,
            timestamp: Date.now(),
          });
          updateSRCard(q.id, 0);
          finalScores.push(0);
        }
      }
      setScores(finalScores);
      const dur = DURATIONS[selectedDuration];
      saveGauntletResult({
        id: `gauntlet-${Date.now()}`,
        track: selectedTrack,
        level: selectedLevel,
        questionCount: questions.length,
        duration: dur.minutes * 60,
        attempts: questions.map((q, i) => ({
          questionId: q.id,
          competencyArea: q.competency_area,
          track: q.track,
          level: q.level,
          selfScore: finalScores[i] ?? 0,
          timestamp: Date.now(),
        })),
        completedAt: Date.now(),
      });
      setPhase("results");
    }
  }, [timeRemaining, phase]);

  // Keyboard shortcuts: Cmd+Enter to reveal, 1-4 for scoring
  useEffect(() => {
    if (phase !== "active") return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLTextAreaElement) {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && !showAnswer) {
          e.preventDefault();
          revealAnswer();
        }
        return;
      }
      if (showAnswer && ['1', '2', '3', '4'].includes(e.key)) {
        scoreAndNext(parseInt(e.key) - 1);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [phase, showAnswer]);

  const isDesignMode = selectedDuration === 3; // "Design (1 deep)"

  const startGauntlet = useCallback(() => {
    const dur = DURATIONS[selectedDuration];
    let selected: Question[];
    if (isDesignMode) {
      // Design challenge: pick 1 question from design/evaluation/realization zones at L5+
      const designZones = ['design', 'evaluation', 'realization', 'specification', 'mastery'];
      const designLevels = ['L5', 'L6+'];
      const pool = getQuestionsByFilter({ track: selectedTrack })
        .filter(q => designZones.includes(q.zone) && designLevels.includes(q.level));
      if (pool.length === 0) {
        // Fallback to any L5+ question
        selected = selectGauntletQuestions(selectedTrack, 'L5', 1);
      } else {
        selected = [pool[Math.floor(Math.random() * pool.length)]];
      }
    } else {
      selected = selectGauntletQuestions(selectedTrack, selectedLevel, dur.questions);
    }
    if (selected.length === 0) {
      showToast({ type: 'info', title: 'No questions found', description: 'Try a different track or difficulty level.' });
      return;
    }
    setQuestions(selected);
    setCurrentIdx(0);
    setShowAnswer(false);
    setUserAnswer("");
    setScores([]);
    setClarifications({});
    setTimeRemaining(dur.minutes * 60);
    setPhase("active");
    track({ type: 'gauntlet_started', track: selectedTrack, level: selectedLevel, questionCount: selected.length });
  }, [selectedTrack, selectedLevel, selectedDuration]);

  const revealAnswer = () => {
    const q = questions[currentIdx];
    if (q) {
      const items = extractRubric(
        q.details.realistic_solution,
        q.details.common_mistake,
        q.details.napkin_math
      );
      setRubricItems(items);
    }
    setShowAnswer(true);
  };

  const scoreAndNext = (score: number) => {
    const q = questions[currentIdx];
    const attempt: AttemptRecord = {
      questionId: q.id,
      competencyArea: q.competency_area,
      track: q.track,
      level: q.level,
      selfScore: score,
      timestamp: Date.now(),
    };
    saveAttempt(attempt);
    updateSRCard(q.id, score);
    recordActivity();

    const newScores = [...scores, score];
    setScores(newScores);

    if (currentIdx < questions.length - 1) {
      setCurrentIdx(currentIdx + 1);
      setShowAnswer(false);
      setUserAnswer("");
      setRubricItems([]);
    } else {
      // All questions answered — show results
      clearInterval(timerRef.current!);
      const dur = DURATIONS[selectedDuration];
      const elapsed = dur.minutes * 60 - timeRemaining;
      saveGauntletResult({
        id: `gauntlet-${Date.now()}`,
        track: selectedTrack,
        level: selectedLevel,
        questionCount: questions.length,
        duration: elapsed,
        attempts: questions.map((q, i) => ({
          questionId: q.id,
          competencyArea: q.competency_area,
          track: q.track,
          level: q.level,
          selfScore: newScores[i] ?? 0,
          timestamp: Date.now(),
        })),
        completedAt: Date.now(),
      });
      const totalScore = newScores.reduce((a, b) => a + b, 0);
      const pctScore = Math.round((totalScore / (questions.length * 3)) * 100);
      track({ type: 'gauntlet_completed', track: selectedTrack, level: selectedLevel, pct: pctScore, questionCount: questions.length });
      setPhase("results");
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  if (!mounted) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Terminal className="w-6 h-6 text-textTertiary animate-pulse" />
      </div>
    );
  }

  // ─── SETUP PHASE ─────────────────────────────────────
  if (phase === "setup") {
    return (
      <div className="flex-1 flex flex-col">
        <div className="max-w-2xl w-full mx-auto px-6 pt-6">
          <FirstRunExplainer mode="gauntlet" />
        </div>
        <div className="flex-1 flex flex-col items-center justify-center px-6 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-xl w-full"
        >
          <div className="flex items-center gap-3 mb-8">
            <Crosshair className="w-8 h-8 text-accentRed" />
            <div>
              <h1 className="text-3xl font-extrabold text-textPrimary tracking-tight">The Gauntlet</h1>
              <p className="text-sm text-textSecondary">Timed mock interview across competency areas</p>
            </div>
          </div>

          {/* Track selection */}
          <div className="mb-6">
            <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-3">Track</label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {tracks.map(t => (
                <button
                  key={t}
                  onClick={() => setSelectedTrack(t)}
                  className={clsx(
                    "px-4 py-3 rounded-lg border text-sm font-medium transition-all text-center capitalize",
                    selectedTrack === t
                      ? "border-accentBlue bg-accentBlue/10 text-textPrimary"
                      : "border-border bg-surface text-textSecondary hover:border-borderHighlight"
                  )}
                >
                  {t === "tinyml" ? "TinyML" : t}
                </button>
              ))}
            </div>
          </div>

          {/* Level selection — hidden in design mode (always L5+) */}
          <div className={clsx("mb-6", isDesignMode && "opacity-40 pointer-events-none")}>
            <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-3">
              {isDesignMode ? "Difficulty (locked to L5+ for design)" : "Difficulty"}
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {levels.map(l => {
                const def = getLevelDef(l);
                return (
                  <button
                    key={l}
                    onClick={() => setSelectedLevel(l)}
                    className={clsx(
                      "px-4 py-3 rounded-lg border text-left transition-all",
                      selectedLevel === l
                        ? "border-accentBlue bg-accentBlue/10"
                        : "border-border bg-surface hover:border-borderHighlight"
                    )}
                  >
                    <div className="flex items-center gap-2 mb-0.5">
                      <div className="w-2 h-2 rounded-sm shrink-0" style={{ backgroundColor: def.color }} />
                      <span className="text-sm font-mono font-bold text-textPrimary">{l}</span>
                      <span className="text-[11px] text-textTertiary">{def.name}</span>
                    </div>
                    <span className="text-[10px] text-textMuted">{def.role}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Duration selection */}
          <div className="mb-8">
            <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-3">Format</label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {DURATIONS.map((d, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedDuration(i)}
                  className={clsx(
                    "px-4 py-3 rounded-lg border text-sm font-medium transition-all text-center",
                    selectedDuration === i
                      ? "border-accentBlue bg-accentBlue/10 text-textPrimary"
                      : "border-border bg-surface text-textSecondary hover:border-borderHighlight"
                  )}
                >
                  <div>{d.label}</div>
                  <div className="text-[10px] text-textTertiary mt-1">{d.minutes} min</div>
                </button>
              ))}
            </div>
            {isDesignMode && (
              <p className="text-[11px] text-textTertiary mt-3 leading-relaxed">
                One deep system design question at L5+ difficulty. Think through architecture, tradeoffs, and constraints — then compare against the model answer.
              </p>
            )}
          </div>

          {/* Realism — controls which helper tools appear during the interview */}
          <div className="mb-8">
            <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-3">Realism</label>
            <div className="grid grid-cols-3 gap-2">
              {([
                { id: "strict",   label: "Strict",   desc: "No tools. Closest to a real whiteboard." },
                { id: "standard", label: "Standard", desc: "Tools available, collapsed by default." },
                { id: "open",     label: "Open",     desc: "Tools open. Use any time." },
              ] as { id: Realism; label: string; desc: string }[]).map(opt => (
                <button
                  key={opt.id}
                  onClick={() => updateRealism(opt.id)}
                  aria-pressed={realism === opt.id}
                  className={clsx(
                    "px-3 py-3 rounded-lg border text-left transition-all",
                    realism === opt.id
                      ? "border-accentBlue bg-accentBlue/10"
                      : "border-border bg-surface hover:border-borderHighlight"
                  )}
                >
                  <div className="text-sm font-bold text-textPrimary">{opt.label}</div>
                  <div className="text-[10px] text-textTertiary mt-0.5 leading-relaxed">{opt.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Available count + start */}
          <div className="flex items-center justify-between">
            <span className="text-xs text-textTertiary font-mono">
              {availableCount} questions available for {selectedTrack.toUpperCase()} × {selectedLevel}
            </span>
            <button
              onClick={startGauntlet}
              disabled={availableCount < DURATIONS[selectedDuration].questions}
              className="inline-flex items-center gap-2 px-6 py-3 bg-textPrimary text-background font-bold rounded-lg hover:opacity-90 transition-all disabled:opacity-30 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(255,255,255,0.1)]"
            >
              Begin <Play className="w-4 h-4" />
            </button>
          </div>

          {availableCount < DURATIONS[selectedDuration].questions && availableCount > 0 && (
            <div className="mt-4 flex items-center gap-2 text-xs text-accentAmber">
              <AlertTriangle className="w-3.5 h-3.5" />
              Not enough questions. Try a shorter duration or different track/level.
            </div>
          )}
        </motion.div>
        </div>
      </div>
    );
  }

  // ─── ACTIVE PHASE ────────────────────────────────────
  if (phase === "active") {
    const q = questions[currentIdx];
    const progress = ((currentIdx + (showAnswer ? 1 : 0)) / questions.length) * 100;
    const isTimeLow = timeRemaining < 60 && timeRemaining > 0;

    return (
      <div className="flex-1 flex flex-col">
        {/* Progress bar + timer */}
        <div className="border-b border-border bg-surface/50 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span className="text-xs font-mono text-textTertiary">
              {currentIdx + 1}/{questions.length}
            </span>
            <div className="w-48 h-1.5 bg-border rounded-full overflow-hidden" role="progressbar" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100} aria-label="Gauntlet progress">
              <div
                className="h-full bg-accentBlue rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="text-[10px] font-mono text-textTertiary uppercase">
              {q.zone}
            </span>
          </div>
          <div className={clsx(
            "flex items-center gap-2 px-3 py-1 rounded-lg font-mono text-sm",
            isTimeLow ? "bg-accentRed/10 text-accentRed border border-accentRed/30" : "text-textSecondary"
          )}>
            <Clock className="w-3.5 h-3.5" />
            {formatTime(timeRemaining)}
          </div>
        </div>

        {/* Question */}
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          {/* Left: scenario */}
          <div className="flex-1 overflow-y-auto px-8 lg:px-12 py-10">
            <div className="max-w-3xl mx-auto">
              <motion.div
                key={q.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <h2 className="text-2xl lg:text-3xl font-bold text-textPrimary mb-6 tracking-tight">
                  {q.title}
                </h2>
                <div className="prose max-w-none">
                  <p className="text-textSecondary leading-relaxed text-base">
                    {cleanScenario(q.scenario)}
                  </p>
                </div>
              </motion.div>
            </div>
          </div>

          {/* Right: answer area */}
          <div className="w-full lg:w-[460px] border-t lg:border-t-0 lg:border-l border-border bg-surface/90 flex flex-col">
            <div className="h-10 border-b border-border flex items-center px-4 bg-background/50">
              <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest flex items-center gap-2">
                <Terminal className="w-3 h-3" /> your_answer.md
              </span>
            </div>

            {/* Tools available in all realism modes. `defaultOpen` still reflects
                the user's setup choice: Open = expanded, Standard/Strict = collapsed. */}
            <HardwareRef defaultOpen={realism === "open"} />
            <NapkinCalc defaultOpen={realism === "open"} />
            <AskInterviewer
              questionContext={q.scenario}
              defaultOpen={realism === "open"}
              onAsk={(question) => recordClarification(q.id, question)}
            />

            <div className="flex-1 p-5 flex flex-col overflow-y-auto">
              {!showAnswer ? (
                <>
                  <textarea
                    value={userAnswer}
                    onChange={(e) => setUserAnswer(e.target.value)}
                    placeholder="Type your answer, napkin math, or reasoning here..."
                    className="flex-1 min-h-[200px] w-full bg-background border border-border rounded-md p-5 font-mono text-[13px] text-textPrimary resize-none focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/50 leading-relaxed"
                    spellCheck="false"
                    autoFocus
                  />
                  <div className="flex items-center gap-3 mt-4">
                    <button
                      onClick={revealAnswer}
                      className="flex-1 bg-textPrimary text-background font-bold py-3 rounded-lg hover:opacity-90 transition-all flex items-center justify-center gap-2"
                    >
                      Reveal Answer <span className="text-[10px] opacity-50 ml-1">⌘↵</span>
                    </button>
                  </div>
                </>
              ) : (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-5"
                >
                  {/* User's answer — preserved for self-comparison against the model.
                      Collapsed by default so the model answer stays the visual focus. */}
                  {userAnswer.trim() && (
                    <details className="group" open>
                      <summary className="text-[10px] font-mono text-textTertiary uppercase cursor-pointer select-none flex items-center gap-1.5">
                        <span className="group-open:rotate-90 transition-transform text-[8px]">&#9654;</span>
                        Your answer
                      </summary>
                      <div className="mt-2 p-3 bg-background border border-border rounded-md font-mono text-[12px] text-textSecondary whitespace-pre-wrap leading-relaxed max-h-40 overflow-y-auto">
                        {userAnswer}
                      </div>
                    </details>
                  )}

                  {/* Model answer */}
                  {q.details.common_mistake && (
                    <div className="border-l-4 border-accentRed pl-4">
                      <span className="text-[10px] font-mono text-accentRed uppercase mb-1 block flex items-center gap-1">
                        <XCircle className="w-3 h-3" /> Common Mistake
                      </span>
                      <p className="text-sm text-textSecondary leading-relaxed">{q.details.common_mistake}</p>
                    </div>
                  )}
                  <div className="border-l-4 border-accentGreen pl-4">
                    <span className="text-[10px] font-mono text-accentGreen uppercase mb-1 block flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" /> Model Answer
                    </span>
                    <p className="text-sm text-textPrimary leading-relaxed">{q.details.realistic_solution}</p>
                  </div>
                  {q.details.napkin_math && (
                    <div className="bg-background border border-border p-4 rounded-lg">
                      <span className="text-[10px] font-mono text-accentBlue uppercase mb-3 block">Napkin Math</span>
                      <NapkinMathDisplay text={q.details.napkin_math} />
                    </div>
                  )}

                  {/* Rubric checkboxes */}
                  {rubricItems.length > 0 && (
                    <div className="border-t border-border pt-5">
                      <span className="text-[10px] font-mono text-textTertiary uppercase block mb-3">
                        Did your answer cover? <span className="text-textTertiary/50 ml-1">{rubricItems.filter(i => i.checked).length}/{rubricItems.length}</span>
                      </span>
                      <div className="space-y-2">
                        {rubricItems.map((item, idx) => (
                          <label
                            key={idx}
                            className={clsx(
                              "flex items-start gap-3 p-2.5 rounded-lg border cursor-pointer transition-all text-xs",
                              item.checked
                                ? "border-accentGreen/30 bg-accentGreen/5"
                                : "border-border hover:border-borderHighlight"
                            )}
                          >
                            <input
                              type="checkbox"
                              checked={item.checked}
                              onChange={() => {
                                const updated = [...rubricItems];
                                updated[idx] = { ...updated[idx], checked: !updated[idx].checked };
                                setRubricItems(updated);
                              }}
                              className="mt-0.5 accent-accentGreen"
                            />
                            <span className={clsx(
                              "leading-relaxed",
                              item.checked ? "text-textPrimary" : "text-textSecondary"
                            )}>
                              {item.text}
                            </span>
                          </label>
                        ))}
                      </div>
                      {rubricToScore(rubricItems) !== null && (
                        <div className="mt-2 text-[10px] font-mono text-textTertiary">
                          Rubric score: {rubricToScore(rubricItems)}/3 → {['Skip', 'Wrong', 'Partial', 'Nailed It'][rubricToScore(rubricItems)!]}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Self-assessment */}
                  {(() => {
                    const rubricScore = rubricItems.length > 0 ? rubricToScore(rubricItems) : null;
                    const effectiveMaxScore = rubricScore !== null ? rubricScore : 3;
                    return (
                      <div className="border-t border-border pt-5">
                        <span className="text-[10px] font-mono text-textTertiary uppercase block mb-3">
                          {rubricItems.length > 0 ? 'Confirm or override' : 'Rate yourself'}
                          <span className="text-textTertiary/50 ml-2">Press 1-4</span>
                        </span>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                          {[
                            { score: 0, label: "Skip", color: "border-border text-textTertiary hover:border-borderHighlight" },
                            { score: 1, label: "Wrong", color: "border-accentRed/30 text-accentRed hover:bg-accentRed/10" },
                            { score: 2, label: "Partial", color: "border-accentAmber/30 text-accentAmber hover:bg-accentAmber/10" },
                            { score: 3, label: "Nailed It", color: "border-accentGreen/30 text-accentGreen hover:bg-accentGreen/10" },
                          ].map(({ score, label, color }) => {
                            const disabled = score > effectiveMaxScore;
                            const isRubricSuggested = rubricScore !== null && score === rubricScore;
                            return (
                              <button
                                key={score}
                                onClick={() => scoreAndNext(Math.min(score, effectiveMaxScore))}
                                disabled={disabled}
                                aria-label={`Rate yourself: ${label}`}
                                className={clsx(
                                  `px-3 py-2.5 rounded-lg border text-xs font-medium transition-all`,
                                  disabled
                                    ? "opacity-30 cursor-not-allowed border-border text-textTertiary"
                                    : isRubricSuggested
                                    ? `${color} ring-1 ring-accentBlue/50`
                                    : color
                                )}
                              >
                                {label}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })()}
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ─── RESULTS PHASE ───────────────────────────────────
  if (phase === "results") {
    const totalScore = scores.reduce((a, b) => a + b, 0);
    const maxScore = questions.length * 3;
    const pct = Math.round((totalScore / maxScore) * 100);
    const byZone: Record<string, { total: number; score: number }> = {};
    questions.forEach((q, i) => {
      const zone = q.zone;
      if (!byZone[zone]) byZone[zone] = { total: 0, score: 0 };
      byZone[zone].total += 3;
      byZone[zone].score += scores[i] ?? 0;
    });

    return (
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-16">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-lg w-full text-center"
        >
          <div className={clsx(
            "w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 border-2",
            pct >= 70 ? "border-accentGreen bg-accentGreen/10" : pct >= 40 ? "border-accentAmber bg-accentAmber/10" : "border-accentRed bg-accentRed/10"
          )}>
            <span className={clsx(
              "text-2xl font-bold font-mono",
              pct >= 70 ? "text-accentGreen" : pct >= 40 ? "text-accentAmber" : "text-accentRed"
            )}>
              {pct}%
            </span>
          </div>

          <h2 className="text-2xl font-bold text-textPrimary mb-2">Gauntlet Complete</h2>
          <p className="text-sm text-textSecondary mb-2">
            {selectedTrack.toUpperCase()} × {selectedLevel} — {questions.length} questions
          </p>
          <p className="text-xs text-textTertiary font-mono mb-8">
            Completed in {formatTime(DURATIONS[selectedDuration].minutes * 60 - timeRemaining)}
          </p>

          {/* Zone performance profile */}
          <div className="text-left mb-8 space-y-2">
            {Object.entries(byZone).map(([zone, data]) => {
              const zonePct = Math.round((data.score / data.total) * 100);
              return (
                <div key={zone} className="flex items-center gap-3">
                  <span className="text-xs text-textSecondary capitalize w-28 truncate">{zone}</span>
                  <div className="flex-1 h-2 bg-border rounded-full overflow-hidden">
                    <div
                      className={clsx(
                        "h-full rounded-full transition-all",
                        zonePct >= 70 ? "bg-accentGreen" : zonePct >= 40 ? "bg-accentAmber" : "bg-accentRed"
                      )}
                      style={{ width: `${zonePct}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-textTertiary w-10 text-right">{zonePct}%</span>
                </div>
              );
            })}
          </div>

          {/* Per-question review */}
          <div className="text-left mb-8 space-y-2">
            <span className="text-[10px] font-mono text-textTertiary uppercase block mb-3">Question Review</span>
            {questions.map((q, i) => {
              const s = scores[i] ?? 0;
              const labels = ['Skipped', 'Wrong', 'Partial', 'Nailed'];
              const askedClarifications = clarifications[q.id] || [];
              return (
                <details key={q.id} className="group rounded-lg border border-borderSubtle bg-surface">
                  <summary className="flex items-center gap-3 px-3 py-2.5 cursor-pointer text-sm">
                    <span className={clsx(
                      "w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0",
                      s >= 2 ? "bg-accentGreen/20 text-accentGreen" : s === 1 ? "bg-accentRed/20 text-accentRed" : "bg-surface text-textTertiary"
                    )}>
                      {i + 1}
                    </span>
                    <span className="text-textPrimary truncate flex-1">{q.title}</span>
                    {askedClarifications.length > 0 && (
                      <span className="text-[9px] font-mono text-accentBlue shrink-0" title={`You asked ${askedClarifications.length} clarification${askedClarifications.length === 1 ? "" : "s"} on this problem`}>
                        ?{askedClarifications.length}
                      </span>
                    )}
                    <span className="text-[10px] font-mono text-textTertiary shrink-0">{labels[s]}</span>
                  </summary>
                  <div className="px-3 pb-3 pt-1 border-t border-borderSubtle space-y-2">
                    <p className="text-[12px] text-textSecondary leading-relaxed">{q.details.realistic_solution}</p>
                    {q.details.common_mistake && (
                      <p className="text-[11px] text-accentRed/80"><span className="font-bold">Common mistake:</span> {q.details.common_mistake}</p>
                    )}
                    {askedClarifications.length > 0 && (
                      <div className="pt-2 mt-2 border-t border-borderSubtle">
                        <p className="text-[10px] font-mono text-accentBlue uppercase mb-1.5">
                          Your clarifications ({askedClarifications.length})
                        </p>
                        <ul className="space-y-1">
                          {askedClarifications.map((c, j) => (
                            <li key={j} className="text-[11px] text-textSecondary leading-relaxed">
                              <span className="text-textTertiary mr-1.5">{j + 1}.</span>{c}
                            </li>
                          ))}
                        </ul>
                        <p className="text-[10px] text-textTertiary italic mt-2">
                          Senior interviewees typically ask 3–6 clarifications before solving. Compare against the model answer's assumptions.
                        </p>
                      </div>
                    )}
                    <QuestionFeedback question={{
                      id: q.id, title: q.title, level: q.level,
                      track: q.track, topic: q.topic, zone: q.zone,
                      competency_area: q.competency_area,
                    }} />
                  </div>
                </details>
              );
            })}
          </div>

          <div className="flex items-center gap-3 justify-center flex-wrap">
            <button
              onClick={() => {
                const zoneLines = Object.entries(byZone)
                  .map(([zone, data]) => {
                    const zp = Math.round((data.score / data.total) * 100);
                    const bar = zp >= 70 ? "🟩" : zp >= 40 ? "🟨" : "🟥";
                    return `${bar} ${zone}: ${zp}%`;
                  })
                  .join("\n");
                const text = [
                  `StaffML Gauntlet — ${pct}%`,
                  `${selectedTrack.toUpperCase()} × ${selectedLevel} — ${questions.length} questions`,
                  "",
                  zoneLines,
                  "",
                  "https://staffml.ai",
                ].join("\n");
                navigator.clipboard.writeText(text).then(() => {
                  setCopied(true);
                  setTimeout(() => setCopied(false), 2000);
                });
              }}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-accentBlue text-white font-bold rounded-lg hover:opacity-90 transition-all text-sm"
            >
              {copied ? <><Check className="w-4 h-4" /> Copied!</> : <><Share2 className="w-4 h-4" /> Share Score</>}
            </button>
            <button
              onClick={() => { setPhase("setup"); setScores([]); }}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg transition-colors text-sm"
            >
              <RotateCcw className="w-4 h-4" /> Try Again
            </button>
            <Link
              href="/progress"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg transition-colors text-sm"
            >
              <BarChart3 className="w-4 h-4" /> View Progress
            </Link>
          </div>
        </motion.div>
      </div>
    );
  }

  return null;
}
