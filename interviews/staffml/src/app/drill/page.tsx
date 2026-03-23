"use client";

import { Suspense, useState, useEffect, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Target, CheckCircle2, XCircle, Terminal, SkipForward,
  BookOpen, Calculator
} from "lucide-react";
import clsx from "clsx";
import HardwareRef from "@/components/HardwareRef";
import { useToast } from "@/components/Toast";
import {
  getTracks, getLevels, getCompetencyAreas, getArchetypes, getQuestionsByFilter,
  Question, checkNapkinMath, extractFinalNumber, cleanScenario,
  NapkinResult
} from "@/lib/corpus";
import { saveAttempt, getAttempts, updateSRCard, getDueQuestionIds, getDueCount, recordActivity } from "@/lib/progress";
import { extractRubric, rubricToScore, RubricItem } from "@/lib/rubric";
import { getQuestionById } from "@/lib/corpus";

export default function DrillPageWrapper() {
  return (
    <Suspense fallback={
      <div className="flex-1 flex items-center justify-center">
        <Terminal className="w-6 h-6 text-textTertiary animate-pulse" />
      </div>
    }>
      <DrillPage />
    </Suspense>
  );
}

function DrillPage() {
  const searchParams = useSearchParams();
  const { show: showToast } = useToast();
  const [mounted, setMounted] = useState(false);
  const [selectedTrack, setSelectedTrack] = useState("cloud");
  const [selectedLevel, setSelectedLevel] = useState("L4");
  const [selectedArea, setSelectedArea] = useState<string | null>(null);
  const [selectedArchetype, setSelectedArchetype] = useState<string | null>(null);

  const [pool, setPool] = useState<Question[]>([]);
  const [current, setCurrent] = useState<Question | null>(null);
  const [showAnswer, setShowAnswer] = useState(false);
  const [userAnswer, setUserAnswer] = useState("");
  const [napkinResult, setNapkinResult] = useState<(NapkinResult & { userNum: number; modelNum: number }) | null>(null);
  const [questionsAnswered, setQuestionsAnswered] = useState(0);
  const [weakestArea, setWeakestArea] = useState<{ area: string; pct: number } | null>(null);
  const [dueCount, setDueCount] = useState(0);
  const [reviewMode, setReviewMode] = useState(false); // true = SR review queue
  const [rubricItems, setRubricItems] = useState<RubricItem[]>([]);

  const tracks = getTracks().filter(t => t !== "global");
  const levels = getLevels();
  const areas = getCompetencyAreas();
  const archetypes = getArchetypes();

  useEffect(() => {
    setMounted(true);
    // Read URL params (from heat map click-through)
    const trackParam = searchParams.get('track');
    const areaParam = searchParams.get('area');
    if (trackParam && getTracks().includes(trackParam)) setSelectedTrack(trackParam);
    if (areaParam && getCompetencyAreas().includes(areaParam)) setSelectedArea(areaParam);

    // Compute weakest area from progress
    const attempts = getAttempts();
    if (attempts.length >= 3) {
      const byArea: Record<string, { total: number; correct: number }> = {};
      attempts.forEach(a => {
        if (!byArea[a.competencyArea]) byArea[a.competencyArea] = { total: 0, correct: 0 };
        byArea[a.competencyArea].total++;
        if (a.selfScore >= 2) byArea[a.competencyArea].correct++;
      });
      const scored = Object.entries(byArea)
        .filter(([, d]) => d.total >= 2)
        .map(([area, d]) => ({ area, pct: Math.round((d.correct / d.total) * 100) }))
        .sort((a, b) => a.pct - b.pct);
      if (scored.length > 0 && scored[0].pct < 70) {
        setWeakestArea(scored[0]);
      }
    }

    // Check for spaced repetition due cards
    setDueCount(getDueCount());
  }, []);

  // Update pool when filters change
  useEffect(() => {
    const filters: { track?: string; level?: string; competency_area?: string; company_archetype?: string } = {
      track: selectedTrack,
      level: selectedLevel,
    };
    if (selectedArea) filters.competency_area = selectedArea;
    if (selectedArchetype) filters.company_archetype = selectedArchetype;
    const q = getQuestionsByFilter(filters);
    setPool(q);
    if (q.length > 0) {
      pickRandom(q);
    } else {
      setCurrent(null);
    }
    setShowAnswer(false);
    setUserAnswer("");
    setNapkinResult(null);
  }, [selectedTrack, selectedLevel, selectedArea, selectedArchetype]);

  // Keyboard shortcuts: Enter to reveal, 1-4 for scoring, N to skip
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLTextAreaElement) {
        // Only Enter (without shift) to reveal when in textarea
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && !showAnswer && current) {
          e.preventDefault();
          handleReveal();
        }
        return;
      }
      if (showAnswer && ['1', '2', '3', '4'].includes(e.key)) {
        handleScore(parseInt(e.key) - 1);
      }
      if (e.key === 'n' || e.key === 'N') {
        pickRandom();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showAnswer, current]);

  const pickRandom = useCallback((fromPool?: Question[]) => {
    const p = fromPool || pool;
    if (p.length === 0) return;
    const idx = Math.floor(Math.random() * p.length);
    setCurrent(p[idx]);
    setShowAnswer(false);
    setUserAnswer("");
    setNapkinResult(null);
    setRubricItems([]);
  }, [pool]);

  const handleReveal = () => {
    // Try napkin math check if the question has napkin_math and user typed something
    if (current?.details.napkin_math && userAnswer.trim()) {
      const userNum = extractFinalNumber(userAnswer);
      const modelNum = extractFinalNumber(current.details.napkin_math);
      if (userNum !== null && modelNum !== null && modelNum > 0) {
        const result = checkNapkinMath(userNum, modelNum, current.track);
        setNapkinResult({ ...result, userNum, modelNum });
      }
    }

    // Generate rubric checkpoints from the answer
    if (current) {
      const items = extractRubric(
        current.details.realistic_solution,
        current.details.common_mistake,
        current.details.napkin_math
      );
      setRubricItems(items);
    }

    setShowAnswer(true);
  };

  // Constrain self-assessment: if napkin math was way off, cap the score
  const maxScore = napkinResult?.maxSelfScore ?? 3;
  // Auto-compute score from rubric if items are checked
  const rubricScore = rubricItems.length > 0 ? rubricToScore(rubricItems) : null;
  const effectiveMaxScore = Math.min(maxScore, rubricScore !== null ? rubricScore : 3);

  const handleScore = (score: number) => {
    if (current) {
      const finalScore = Math.min(score, maxScore);
      saveAttempt({
        questionId: current.id,
        competencyArea: current.competency_area,
        track: current.track,
        level: current.level,
        selfScore: finalScore,
        timestamp: Date.now(),
      });
      // Update spaced repetition card + streak
      updateSRCard(current.id, finalScore);
      const activity = recordActivity();
      if (activity.newMilestone) {
        showToast({
          type: 'badge',
          title: activity.newMilestone,
          description: `${activity.streak.currentStreak} day streak!`,
        });
      }
      setQuestionsAnswered(prev => prev + 1);
      setDueCount(getDueCount());
    }
    pickNext();
  };

  // Pick next question: from SR due queue if in review mode, else random
  const pickNext = useCallback(() => {
    if (reviewMode) {
      const dueIds = getDueQuestionIds();
      if (dueIds.length > 0) {
        const q = getQuestionById(dueIds[0]);
        if (q) {
          setCurrent(q);
          setShowAnswer(false);
          setUserAnswer("");
          setNapkinResult(null);
          setRubricItems([]);
          return;
        }
      }
      // No more due cards, exit review mode
      setReviewMode(false);
    }
    pickRandom();
  }, [reviewMode, pool]);

  if (!mounted) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Terminal className="w-6 h-6 text-textTertiary animate-pulse" />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col lg:flex-row">
      {/* Sidebar filters */}
      <aside className="w-full lg:w-64 border-b lg:border-b-0 lg:border-r border-border bg-surface/50 p-5 flex flex-col gap-6 lg:overflow-y-auto">
        <div className="flex items-center gap-2">
          <Target className="w-5 h-5 text-accentBlue" />
          <h2 className="text-lg font-bold text-white">Drill</h2>
          {questionsAnswered > 0 && (
            <span className="ml-auto text-[10px] font-mono text-textTertiary bg-surface px-2 py-0.5 rounded border border-border">
              {questionsAnswered} done
            </span>
          )}
        </div>

        {/* Spaced repetition review queue */}
        {dueCount > 0 && (
          <button
            onClick={() => {
              setReviewMode(!reviewMode);
              if (!reviewMode) {
                // Start review: pick first due card
                const dueIds = getDueQuestionIds();
                if (dueIds.length > 0) {
                  const q = getQuestionById(dueIds[0]);
                  if (q) {
                    setCurrent(q);
                    setShowAnswer(false);
                    setUserAnswer("");
                    setNapkinResult(null);
                    setRubricItems([]);
                  }
                }
              }
            }}
            className={clsx(
              "w-full text-left p-3 rounded-lg border transition-colors",
              reviewMode
                ? "bg-accentAmber/10 border-accentAmber/40"
                : "bg-accentAmber/5 border-accentAmber/20 hover:border-accentAmber/40"
            )}
          >
            <span className="text-[10px] font-mono text-accentAmber uppercase block mb-1">
              {reviewMode ? "Review Mode Active" : "Due for Review"}
            </span>
            <span className="text-sm text-white font-medium">{dueCount} questions</span>
            <span className="text-xs text-textTertiary ml-2">
              {reviewMode ? "click to exit" : "spaced repetition"}
            </span>
          </button>
        )}

        {/* Weakest area recommendation */}
        {weakestArea && !selectedArea && !reviewMode && (
          <button
            onClick={() => setSelectedArea(weakestArea.area)}
            className="w-full text-left p-3 rounded-lg bg-accentRed/5 border border-accentRed/20 hover:border-accentRed/40 transition-colors"
          >
            <span className="text-[10px] font-mono text-accentRed uppercase block mb-1">Weakest area</span>
            <span className="text-sm text-white capitalize font-medium">{weakestArea.area}</span>
            <span className="text-xs text-textTertiary ml-2">{weakestArea.pct}% accuracy</span>
          </button>
        )}

        {/* Track */}
        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Track</label>
          <div className="space-y-1">
            {tracks.map(t => (
              <button
                key={t}
                onClick={() => setSelectedTrack(t)}
                className={clsx(
                  "w-full text-left px-3 py-2 rounded-md text-sm font-medium capitalize transition-all",
                  selectedTrack === t
                    ? "bg-accentBlue/10 text-accentBlue"
                    : "text-textSecondary hover:bg-surfaceHover"
                )}
              >
                {t === "tinyml" ? "TinyML" : t}
              </button>
            ))}
          </div>
        </div>

        {/* Level */}
        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Level</label>
          <div className="grid grid-cols-3 gap-1">
            {levels.map(l => (
              <button
                key={l}
                onClick={() => setSelectedLevel(l)}
                className={clsx(
                  "px-2 py-1.5 rounded text-xs font-mono font-medium text-center transition-all",
                  selectedLevel === l
                    ? "bg-accentBlue/10 text-accentBlue border border-accentBlue/30"
                    : "text-textSecondary hover:bg-surfaceHover border border-transparent"
                )}
              >
                {l}
              </button>
            ))}
          </div>
        </div>

        {/* Competency Area */}
        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Competency</label>
          <div className="space-y-1 max-h-64 overflow-y-auto">
            <button
              onClick={() => setSelectedArea(null)}
              className={clsx(
                "w-full text-left px-3 py-1.5 rounded text-xs font-medium transition-all",
                !selectedArea
                  ? "bg-accentBlue/10 text-accentBlue"
                  : "text-textSecondary hover:bg-surfaceHover"
              )}
            >
              All areas
            </button>
            {areas.map(a => (
              <button
                key={a}
                onClick={() => setSelectedArea(a)}
                className={clsx(
                  "w-full text-left px-3 py-1.5 rounded text-xs font-medium capitalize transition-all",
                  selectedArea === a
                    ? "bg-accentBlue/10 text-accentBlue"
                    : "text-textSecondary hover:bg-surfaceHover"
                )}
              >
                {a}
              </button>
            ))}
          </div>
        </div>

        {/* Company Archetype */}
        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Interview Style</label>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            <button
              onClick={() => setSelectedArchetype(null)}
              className={clsx(
                "w-full text-left px-3 py-1.5 rounded text-xs font-medium transition-all",
                !selectedArchetype
                  ? "bg-accentBlue/10 text-accentBlue"
                  : "text-textSecondary hover:bg-surfaceHover"
              )}
            >
              All types
            </button>
            {archetypes.map(a => (
              <button
                key={a}
                onClick={() => setSelectedArchetype(a)}
                className={clsx(
                  "w-full text-left px-3 py-1.5 rounded text-xs font-medium transition-all",
                  selectedArchetype === a
                    ? "bg-accentBlue/10 text-accentBlue"
                    : "text-textSecondary hover:bg-surfaceHover"
                )}
              >
                {a}
              </button>
            ))}
          </div>
        </div>

        <div className="text-[10px] font-mono text-textTertiary mt-auto">
          {pool.length} questions in pool
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {current ? (
          <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
            {/* Question */}
            <div className="flex-1 overflow-y-auto px-8 lg:px-12 py-10">
              <div className="max-w-3xl mx-auto">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={current.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <span className="text-[10px] font-mono text-textTertiary uppercase px-2 py-0.5 rounded border border-border bg-surface">
                        {current.competency_area}
                      </span>
                      <span className="text-[10px] font-mono text-textTertiary uppercase">
                        {current.track} / {current.level}
                      </span>
                    </div>
                    <h2 className="text-2xl lg:text-3xl font-bold text-white mb-6 tracking-tight">
                      {current.title}
                    </h2>
                    <div className="prose max-w-none">
                      <p className="text-textSecondary leading-relaxed text-base">
                        {cleanScenario(current.scenario)}
                      </p>
                    </div>

                    {current.details.deep_dive_title && (
                      <a
                        href={current.details.deep_dive_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 mt-6 text-xs text-textTertiary hover:text-accentBlue transition-colors"
                      >
                        <BookOpen className="w-3.5 h-3.5" />
                        {current.details.deep_dive_title}
                      </a>
                    )}
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            {/* Answer panel */}
            <div className="w-full lg:w-[460px] border-t lg:border-t-0 lg:border-l border-border bg-surface/90 flex flex-col">
              <div className="h-10 border-b border-border flex items-center px-4 bg-background/50 justify-between">
                <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest flex items-center gap-2">
                  <Calculator className="w-3 h-3" /> {current.details.napkin_math ? "napkin_math.py" : "answer.md"}
                </span>
                <button
                  onClick={() => pickRandom()}
                  className="text-[10px] font-mono text-textTertiary hover:text-white transition-colors flex items-center gap-1"
                >
                  <SkipForward className="w-3 h-3" /> Skip
                </button>
              </div>

              <HardwareRef />
              <div className="flex-1 p-5 flex flex-col overflow-y-auto">
                {!showAnswer ? (
                  <>
                    <textarea
                      value={userAnswer}
                      onChange={(e) => setUserAnswer(e.target.value)}
                      placeholder={
                        current.details.napkin_math
                          ? "Type your napkin math here...\n\nExample:\nBandwidth: 3.35 TB/s\nModel size: 140 GB\nTime = 140 / 3350 ≈ 42 ms\n\n=> 42 ms   (mark your final answer with =>)"
                          : "Type your answer or reasoning here..."
                      }
                      className="flex-1 min-h-[200px] w-full bg-background border border-border rounded-md p-5 font-mono text-[13px] text-textPrimary resize-none focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/40 leading-relaxed"
                      spellCheck="false"
                      autoFocus
                    />
                    <button
                      onClick={handleReveal}
                      className="mt-4 w-full bg-white text-black font-bold py-3 rounded-lg hover:bg-gray-100 transition-all flex items-center justify-center gap-2"
                    >
                      Reveal Answer <span className="text-[10px] opacity-50 ml-1">⌘↵</span>
                    </button>
                  </>
                ) : (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-5"
                  >
                    {/* Napkin math result — gradient feedback */}
                    {napkinResult && (
                      <div className={clsx(
                        "p-4 rounded-lg border",
                        napkinResult.grade === 'exact' || napkinResult.grade === 'close'
                          ? "bg-accentGreen/10 border-accentGreen/30"
                          : napkinResult.grade === 'ballpark'
                          ? "bg-accentAmber/10 border-accentAmber/30"
                          : "bg-accentRed/10 border-accentRed/30"
                      )}>
                        <div className="flex items-center gap-2 mb-2">
                          {(napkinResult.grade === 'exact' || napkinResult.grade === 'close') ? (
                            <CheckCircle2 className="w-4 h-4 text-accentGreen" />
                          ) : napkinResult.grade === 'ballpark' ? (
                            <CheckCircle2 className="w-4 h-4 text-accentAmber" />
                          ) : (
                            <XCircle className="w-4 h-4 text-accentRed" />
                          )}
                          <span className={clsx(
                            "text-sm font-bold",
                            (napkinResult.grade === 'exact' || napkinResult.grade === 'close') ? "text-accentGreen"
                              : napkinResult.grade === 'ballpark' ? "text-accentAmber"
                              : "text-accentRed"
                          )}>
                            {napkinResult.label}
                          </span>
                        </div>
                        <p className="text-xs text-textSecondary font-mono">
                          Your answer: {napkinResult.userNum.toLocaleString()} |
                          Model answer: {napkinResult.modelNum.toLocaleString()} |
                          Off by: {(napkinResult.ratio * 100).toFixed(0)}%
                        </p>
                        {napkinResult.maxSelfScore < 3 && (
                          <p className="text-[10px] text-textTertiary mt-2">
                            Self-assessment capped at "{napkinResult.maxSelfScore === 2 ? 'Partial' : 'Wrong'}" based on napkin math accuracy
                          </p>
                        )}
                      </div>
                    )}

                    {current.details.common_mistake && (
                      <div className="border-l-4 border-accentRed pl-4">
                        <span className="text-[10px] font-mono text-accentRed uppercase mb-1 block flex items-center gap-1">
                          <XCircle className="w-3 h-3" /> Common Mistake
                        </span>
                        <p className="text-sm text-textSecondary leading-relaxed">{current.details.common_mistake}</p>
                      </div>
                    )}
                    <div className="border-l-4 border-accentGreen pl-4">
                      <span className="text-[10px] font-mono text-accentGreen uppercase mb-1 block flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" /> Model Answer
                      </span>
                      <p className="text-sm text-textPrimary leading-relaxed">{current.details.realistic_solution}</p>
                    </div>
                    {current.details.napkin_math && (
                      <div className="bg-background border border-border p-4 rounded-lg">
                        <span className="text-[10px] font-mono text-accentBlue uppercase mb-2 block">Napkin Math</span>
                        <pre className="font-mono text-xs text-textSecondary whitespace-pre-wrap">{current.details.napkin_math}</pre>
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
                        {rubricScore !== null && (
                          <div className="mt-2 text-[10px] font-mono text-textTertiary">
                            Rubric score: {rubricScore}/3 → {['Skip', 'Wrong', 'Partial', 'Nailed It'][rubricScore]}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Self-assessment */}
                    <div className="border-t border-border pt-5">
                      <span className="text-[10px] font-mono text-textTertiary uppercase block mb-3">
                        {rubricItems.length > 0 ? 'Confirm or override' : 'Rate yourself'}
                        <span className="text-textTertiary/50 ml-2">Press 1-4</span>
                      </span>
                      <div className="grid grid-cols-4 gap-2">
                        {[
                          { score: 0, label: "Skip", color: "border-border text-textTertiary hover:border-borderHighlight" },
                          { score: 1, label: "Wrong", color: "border-accentRed/30 text-accentRed hover:bg-accentRed/10" },
                          { score: 2, label: "Partial", color: "border-accentAmber/30 text-accentAmber hover:bg-accentAmber/10" },
                          { score: 3, label: "Nailed It", color: "border-accentGreen/30 text-accentGreen hover:bg-accentGreen/10" },
                        ].map(({ score, label, color }) => {
                          const disabled = score > maxScore;
                          const isRubricSuggested = rubricScore !== null && score === rubricScore;
                          return (
                            <button
                              key={score}
                              onClick={() => handleScore(Math.min(score, maxScore))}
                              disabled={disabled}
                              className={clsx(
                                "px-3 py-2.5 rounded-lg border text-xs font-medium transition-all",
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
                  </motion.div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-textTertiary">
            <p className="text-sm">No questions match your filters. Try adjusting track, level, or competency.</p>
          </div>
        )}
      </div>
    </div>
  );
}
