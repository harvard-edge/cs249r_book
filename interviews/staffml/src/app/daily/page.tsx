"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Calendar, CheckCircle2, XCircle, Terminal, ChevronRight, Share2, Trophy
} from "lucide-react";
import clsx from "clsx";
import Link from "next/link";
import {
  getQuestions, Question, cleanScenario, checkNapkinMath,
  extractFinalNumber, NapkinResult
} from "@/lib/corpus";
import NapkinMathDisplay from "@/components/NapkinMathDisplay";
import { saveAttempt, recordActivity } from "@/lib/progress";
import { extractRubric, rubricToScore, RubricItem } from "@/lib/rubric";

// Deterministic daily question selection — same 3 for all users on a given day
function hashString(s: string): number {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    hash = ((hash << 5) - hash) + c;
    hash |= 0;
  }
  return Math.abs(hash);
}

function getDailyQuestions(): Question[] {
  const today = new Date().toISOString().slice(0, 10);
  const seed = hashString(today);
  const pool = getQuestions();
  if (pool.length === 0) return [];

  // Pick 3 from different indices to avoid collisions
  const primes = [1, 31, 97];
  const indices = primes.map(p => (seed * p) % pool.length);
  // Deduplicate
  const unique: number[] = [];
  indices.forEach(i => { if (!unique.includes(i)) unique.push(i); });
  while (unique.length < 3 && unique.length < pool.length) {
    const next = (unique[unique.length - 1] + 7) % pool.length;
    if (!unique.includes(next)) unique.push(next);
  }
  return unique.slice(0, 3).map(i => pool[i]);
}

function getTodayKey(): string {
  return new Date().toISOString().slice(0, 10);
}

function isDailyCompleted(): boolean {
  try {
    const data = JSON.parse(window.localStorage.getItem('staffml_daily') || '{}');
    return data[getTodayKey()] === true;
  } catch { return false; }
}

function markDailyCompleted(): void {
  try {
    const data = JSON.parse(window.localStorage.getItem('staffml_daily') || '{}');
    data[getTodayKey()] = true;
    // Keep only last 30 days
    const keys = Object.keys(data).sort().slice(-30);
    const trimmed: Record<string, boolean> = {};
    keys.forEach(k => trimmed[k] = data[k]);
    window.localStorage.setItem('staffml_daily', JSON.stringify(trimmed));
  } catch {}
}

export default function DailyPage() {
  const [mounted, setMounted] = useState(false);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [userAnswer, setUserAnswer] = useState("");
  const [napkinResult, setNapkinResult] = useState<(NapkinResult & { userNum: number; modelNum: number }) | null>(null);
  const [rubricItems, setRubricItems] = useState<RubricItem[]>([]);
  const [scores, setScores] = useState<number[]>([]);
  const [completed, setCompleted] = useState(false);
  const [alreadyDone, setAlreadyDone] = useState(false);

  useEffect(() => {
    setMounted(true);
    setQuestions(getDailyQuestions());
    setAlreadyDone(isDailyCompleted());
  }, []);

  const current = questions[currentIdx];

  const handleReveal = () => {
    if (!current) return;
    if (current.details.napkin_math && userAnswer.trim()) {
      const userNum = extractFinalNumber(userAnswer);
      const modelNum = extractFinalNumber(current.details.napkin_math);
      if (userNum !== null && modelNum !== null && modelNum > 0) {
        const result = checkNapkinMath(userNum, modelNum, current.track);
        setNapkinResult({ ...result, userNum, modelNum });
      }
    }
    const items = extractRubric(
      current.details.realistic_solution,
      current.details.common_mistake,
      current.details.napkin_math
    );
    setRubricItems(items);
    setShowAnswer(true);
  };

  const maxScore = napkinResult?.maxSelfScore ?? 3;

  const handleScore = (score: number) => {
    if (!current) return;
    const finalScore = Math.min(score, maxScore);
    saveAttempt({
      questionId: current.id,
      competencyArea: current.competency_area,
      track: current.track,
      level: current.level,
      selfScore: finalScore,
      timestamp: Date.now(),
    });
    recordActivity();

    const newScores = [...scores, finalScore];
    setScores(newScores);

    if (currentIdx < questions.length - 1) {
      setCurrentIdx(currentIdx + 1);
      setShowAnswer(false);
      setUserAnswer("");
      setNapkinResult(null);
      setRubricItems([]);
    } else {
      markDailyCompleted();
      setCompleted(true);
    }
  };

  if (!mounted) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Terminal className="w-6 h-6 text-textTertiary animate-pulse" />
      </div>
    );
  }

  // Already completed today
  if (alreadyDone && !completed) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md text-center"
        >
          <div className="w-16 h-16 rounded-full bg-accentGreen/10 border border-accentGreen/30 flex items-center justify-center mx-auto mb-6">
            <CheckCircle2 className="w-8 h-8 text-accentGreen" />
          </div>
          <h2 className="text-2xl font-bold text-textPrimary mb-2">Daily Complete</h2>
          <p className="text-sm text-textSecondary mb-6">
            You've already finished today's challenge. Come back tomorrow for 3 new questions.
          </p>
          <div className="flex items-center gap-3 justify-center">
            <Link
              href="/drill"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg transition-colors text-sm"
            >
              Keep Drilling
            </Link>
            <Link
              href="/heatmap"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-white text-black font-bold rounded-lg hover:bg-gray-100 transition-all text-sm"
            >
              View Progress
            </Link>
          </div>
        </motion.div>
      </div>
    );
  }

  // Completed just now — show summary
  if (completed) {
    const totalScore = scores.reduce((a, b) => a + b, 0);
    const maxTotal = scores.length * 3;
    const pct = maxTotal > 0 ? Math.round((totalScore / maxTotal) * 100) : 0;
    const today = new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' });

    return (
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-16">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-md text-center"
        >
          <div className={clsx(
            "w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 border-2",
            pct >= 70 ? "border-accentGreen bg-accentGreen/10" : pct >= 40 ? "border-accentAmber bg-accentAmber/10" : "border-accentRed bg-accentRed/10"
          )}>
            <Trophy className={clsx(
              "w-8 h-8",
              pct >= 70 ? "text-accentGreen" : pct >= 40 ? "text-accentAmber" : "text-accentRed"
            )} />
          </div>

          <h2 className="text-2xl font-bold text-textPrimary mb-1">Daily Challenge Complete</h2>
          <p className="text-xs text-textTertiary font-mono mb-4">{today}</p>

          <div className="flex items-center justify-center gap-6 mb-8">
            <div>
              <div className="text-3xl font-bold font-mono text-textPrimary">{pct}%</div>
              <div className="text-[10px] text-textTertiary uppercase">Score</div>
            </div>
            <div className="w-px h-10 bg-border" />
            <div>
              <div className="text-3xl font-bold font-mono text-textPrimary">{scores.length}</div>
              <div className="text-[10px] text-textTertiary uppercase">Questions</div>
            </div>
          </div>

          {/* Per-question summary */}
          <div className="text-left mb-8 space-y-2">
            {questions.map((q, i) => (
              <div key={q.id} className="flex items-center gap-3 text-sm">
                <span className={clsx(
                  "w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold",
                  (scores[i] ?? 0) >= 2 ? "bg-accentGreen/20 text-accentGreen" : "bg-accentRed/20 text-accentRed"
                )}>
                  {i + 1}
                </span>
                <span className="text-textSecondary truncate flex-1">{q.title}</span>
                <span className="text-[10px] font-mono text-textTertiary">{['Skip', 'Wrong', 'Partial', 'Nailed'][scores[i] ?? 0]}</span>
              </div>
            ))}
          </div>

          <div className="flex items-center gap-3 justify-center">
            <Link
              href="/drill"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg transition-colors text-sm"
            >
              Keep Drilling
            </Link>
            <Link
              href="/heatmap"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-white text-black font-bold rounded-lg hover:bg-gray-100 transition-all text-sm"
            >
              View Heat Map
            </Link>
          </div>
        </motion.div>
      </div>
    );
  }

  // Active daily challenge
  if (!current) {
    return (
      <div className="flex-1 flex items-center justify-center text-textTertiary">
        <p className="text-sm">No questions available.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* Progress header */}
      <div className="border-b border-border bg-surface/50 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Calendar className="w-4 h-4 text-accentAmber" />
          <span className="text-sm font-medium text-textPrimary">Daily Challenge</span>
          <span className="text-xs font-mono text-textTertiary">
            {currentIdx + 1} of {questions.length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {questions.map((_, i) => (
            <div
              key={i}
              className={clsx(
                "w-2.5 h-2.5 rounded-full transition-all",
                i < scores.length
                  ? (scores[i] >= 2 ? "bg-accentGreen" : "bg-accentRed")
                  : i === currentIdx
                  ? "bg-accentBlue"
                  : "bg-border"
              )}
            />
          ))}
        </div>
      </div>

      {/* Question + answer area — same layout as drill */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Question */}
        <div className="flex-1 overflow-y-auto px-8 lg:px-12 py-10">
          <div className="max-w-3xl mx-auto">
            <motion.div
              key={current.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <div className="flex items-center gap-3 mb-4">
                <span className="text-[10px] font-mono text-textTertiary uppercase px-2 py-0.5 rounded border border-border bg-surface">
                  {current.competency_area}
                </span>
                <span className="text-[10px] font-mono text-textTertiary">
                  {current.track} / {current.level}
                </span>
              </div>
              <h2 className="text-2xl lg:text-3xl font-bold text-textPrimary mb-6 tracking-tight">
                {current.title}
              </h2>
              <div className="prose max-w-none">
                <p className="text-textSecondary leading-relaxed text-base">
                  {cleanScenario(current.scenario)}
                </p>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Answer panel */}
        <div className="w-full lg:w-[460px] border-t lg:border-t-0 lg:border-l border-border bg-surface/90 flex flex-col">
          <div className="h-10 border-b border-border flex items-center px-4 bg-background/50">
            <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest flex items-center gap-2">
              <Terminal className="w-3 h-3" /> daily_{getTodayKey()}.md
            </span>
          </div>

          <div className="flex-1 p-5 flex flex-col overflow-y-auto">
            {!showAnswer ? (
              <>
                <textarea
                  value={userAnswer}
                  onChange={(e) => setUserAnswer(e.target.value)}
                  placeholder={
                    current.details.napkin_math
                      ? "Your napkin math...\n\n=> final answer"
                      : "Your answer..."
                  }
                  className="flex-1 min-h-[200px] w-full bg-background border border-border rounded-md p-5 font-mono text-[13px] text-textPrimary resize-none focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/40 leading-relaxed"
                  spellCheck="false"
                  autoFocus
                />
                <button
                  onClick={handleReveal}
                  className="mt-4 w-full bg-white text-black font-bold py-3 rounded-lg hover:bg-gray-100 transition-all flex items-center justify-center gap-2"
                >
                  Reveal <span className="text-[10px] opacity-50 ml-1">⌘↵</span>
                </button>
              </>
            ) : (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-5"
              >
                {/* Napkin result */}
                {napkinResult && (
                  <div className={clsx(
                    "p-3 rounded-lg border text-xs",
                    (napkinResult.grade === 'exact' || napkinResult.grade === 'close')
                      ? "bg-accentGreen/10 border-accentGreen/30 text-accentGreen"
                      : napkinResult.grade === 'ballpark'
                      ? "bg-accentAmber/10 border-accentAmber/30 text-accentAmber"
                      : "bg-accentRed/10 border-accentRed/30 text-accentRed"
                  )}>
                    <span className="font-bold">{napkinResult.label}</span>
                    <span className="text-textTertiary ml-2">
                      {napkinResult.userNum.toLocaleString()} vs {napkinResult.modelNum.toLocaleString()}
                    </span>
                  </div>
                )}

                {/* Model answer */}
                {current.details.common_mistake && (
                  <div className="border-l-4 border-accentRed pl-4">
                    <span className="text-[10px] font-mono text-accentRed uppercase mb-1 block">Common Mistake</span>
                    <p className="text-sm text-textSecondary">{current.details.common_mistake}</p>
                  </div>
                )}
                <div className="border-l-4 border-accentGreen pl-4">
                  <span className="text-[10px] font-mono text-accentGreen uppercase mb-1 block">Model Answer</span>
                  <p className="text-sm text-textPrimary">{current.details.realistic_solution}</p>
                </div>
                {current.details.napkin_math && (
                  <div className="bg-background border border-border p-4 rounded-lg">
                    <span className="text-[10px] font-mono text-accentBlue uppercase mb-3 block">Napkin Math</span>
                    <NapkinMathDisplay text={current.details.napkin_math} />
                  </div>
                )}

                {/* Rubric */}
                {rubricItems.length > 0 && (
                  <div className="border-t border-border pt-4 space-y-2">
                    <span className="text-[10px] font-mono text-textTertiary uppercase">Did you cover?</span>
                    {rubricItems.map((item, idx) => (
                      <label key={idx} className={clsx(
                        "flex items-start gap-2 p-2 rounded border cursor-pointer text-xs transition-all",
                        item.checked ? "border-accentGreen/30 bg-accentGreen/5" : "border-border"
                      )}>
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
                        <span className={item.checked ? "text-textPrimary" : "text-textSecondary"}>{item.text}</span>
                      </label>
                    ))}
                  </div>
                )}

                {/* Score */}
                <div className="border-t border-border pt-4">
                  <span className="text-[10px] font-mono text-textTertiary uppercase block mb-3">Rate yourself</span>
                  <div className="grid grid-cols-4 gap-2">
                    {[
                      { score: 0, label: "Skip", color: "border-border text-textTertiary hover:border-borderHighlight" },
                      { score: 1, label: "Wrong", color: "border-accentRed/30 text-accentRed hover:bg-accentRed/10" },
                      { score: 2, label: "Partial", color: "border-accentAmber/30 text-accentAmber hover:bg-accentAmber/10" },
                      { score: 3, label: "Nailed It", color: "border-accentGreen/30 text-accentGreen hover:bg-accentGreen/10" },
                    ].map(({ score, label, color }) => {
                      const disabled = score > maxScore;
                      return (
                        <button
                          key={score}
                          onClick={() => handleScore(Math.min(score, maxScore))}
                          disabled={disabled}
                          className={clsx(
                            "px-3 py-2.5 rounded-lg border text-xs font-medium transition-all",
                            disabled ? "opacity-30 cursor-not-allowed border-border text-textTertiary" : color
                          )}
                        >
                          {label}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
