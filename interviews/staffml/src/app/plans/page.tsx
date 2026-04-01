"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Map, CheckCircle2, ArrowRight, Terminal, Clock, Target } from "lucide-react";
import clsx from "clsx";
import Link from "next/link";
import { STUDY_PLANS, getPlanQuestions, getPlanProgress, markPlanQuestionComplete, StudyPlan } from "@/lib/plans";
import { Question, cleanScenario, checkNapkinMath, extractFinalNumber, NapkinResult } from "@/lib/corpus";
import { saveAttempt, recordActivity, updateSRCard } from "@/lib/progress";
import NapkinMathDisplay from "@/components/NapkinMathDisplay";
import { extractRubric, rubricToScore, RubricItem } from "@/lib/rubric";
import { useToast } from "@/components/Toast";
import HardwareRef from "@/components/HardwareRef";

export default function PlansPage() {
  const [mounted, setMounted] = useState(false);
  const [activePlan, setActivePlan] = useState<StudyPlan | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [completedIds, setCompletedIds] = useState<string[]>([]);

  // Question state
  const [showAnswer, setShowAnswer] = useState(false);
  const [userAnswer, setUserAnswer] = useState("");
  const [napkinResult, setNapkinResult] = useState<(NapkinResult & { userNum: number; modelNum: number }) | null>(null);
  const [rubricItems, setRubricItems] = useState<RubricItem[]>([]);

  const { show: showToast } = useToast();

  useEffect(() => { setMounted(true); }, []);

  const startPlan = (plan: StudyPlan) => {
    const qs = getPlanQuestions(plan);
    const progress = getPlanProgress(plan.id);
    setActivePlan(plan);
    setQuestions(qs);
    setCompletedIds(progress.completedIds);
    // Find first uncompleted question
    const firstUncompleted = qs.findIndex(q => !progress.completedIds.includes(q.id));
    setCurrentIdx(firstUncompleted >= 0 ? firstUncompleted : 0);
    resetQuestionState();
  };

  const resetQuestionState = () => {
    setShowAnswer(false);
    setUserAnswer("");
    setNapkinResult(null);
    setRubricItems([]);
  };

  const current = questions[currentIdx];
  const maxScore = napkinResult?.maxSelfScore ?? 3;

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
    const items = extractRubric(current.details.realistic_solution, current.details.common_mistake, current.details.napkin_math);
    setRubricItems(items);
    setShowAnswer(true);
  };

  const handleScore = (score: number) => {
    if (!current || !activePlan) return;
    const finalScore = Math.min(score, maxScore);
    saveAttempt({
      questionId: current.id,
      competencyArea: current.competency_area,
      track: current.track,
      level: current.level,
      selfScore: finalScore,
      timestamp: Date.now(),
    });
    updateSRCard(current.id, finalScore);
    const activity = recordActivity();
    if (activity.newMilestone) {
      showToast({ type: 'badge', title: activity.newMilestone, description: `${activity.streak.currentStreak} day streak!` });
    }
    markPlanQuestionComplete(activePlan.id, current.id);
    const newCompleted = [...completedIds, current.id];
    setCompletedIds(newCompleted);

    // Move to next uncompleted
    let next = currentIdx + 1;
    while (next < questions.length && newCompleted.includes(questions[next].id)) {
      next++;
    }
    if (next < questions.length) {
      setCurrentIdx(next);
      resetQuestionState();
    } else {
      showToast({ type: 'success', title: 'Plan Complete!', description: `Finished ${activePlan.title}` });
      setActivePlan(null);
    }
  };

  if (!mounted) {
    return <div className="flex-1 flex items-center justify-center"><Terminal className="w-6 h-6 text-textTertiary animate-pulse" /></div>;
  }

  // Plan selection view
  if (!activePlan) {
    return (
      <div className="flex-1 flex flex-col px-6 py-10">
        <div className="max-w-4xl mx-auto w-full">
          <div className="flex items-center gap-3 mb-8">
            <Map className="w-8 h-8 text-accentBlue" />
            <div>
              <h1 className="text-3xl font-extrabold text-textPrimary tracking-tight">Study Plans</h1>
              <p className="text-sm text-textSecondary">Curated question sequences for targeted interview prep</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {STUDY_PLANS.map((plan, i) => {
              const progress = getPlanProgress(plan.id);
              const pct = plan.questionCount > 0
                ? Math.round((progress.completedIds.length / plan.questionCount) * 100)
                : 0;

              return (
                <motion.button
                  key={plan.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  onClick={() => startPlan(plan)}
                  className="text-left p-6 rounded-xl border border-border bg-surface/50 hover:border-borderHighlight hover:bg-surfaceHover transition-all group"
                >
                  <div className="flex items-start justify-between mb-3">
                    <span className="text-2xl">{plan.icon}</span>
                    <div className="flex items-center gap-2">
                      <Clock className="w-3 h-3 text-textTertiary" />
                      <span className="text-[10px] font-mono text-textTertiary">{plan.duration}</span>
                    </div>
                  </div>
                  <h3 className="text-lg font-bold text-textPrimary mb-1">{plan.title}</h3>
                  <p className="text-sm text-textSecondary mb-4 leading-relaxed">{plan.description}</p>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-mono text-textTertiary">{plan.questionCount} Qs</span>
                      <span className="text-xs font-mono text-textTertiary capitalize">{plan.track}</span>
                      <span className="text-xs font-mono text-textTertiary">{plan.levels.join('/')}</span>
                    </div>
                    {pct > 0 ? (
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-border rounded-full overflow-hidden">
                          <div className="h-full bg-accentGreen rounded-full" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="text-[10px] font-mono text-accentGreen">{pct}%</span>
                      </div>
                    ) : (
                      <span className="text-xs text-textTertiary group-hover:text-accentBlue transition-colors flex items-center gap-1">
                        Start <ArrowRight className="w-3 h-3" />
                      </span>
                    )}
                  </div>
                </motion.button>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  // Active plan — question view
  if (!current) {
    return <div className="flex-1 flex items-center justify-center text-textTertiary text-sm">No questions available for this plan.</div>;
  }

  const planProgress = Math.round((completedIds.length / questions.length) * 100);

  return (
    <div className="flex-1 flex flex-col">
      {/* Plan progress header */}
      <div className="border-b border-border bg-surface/50 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button onClick={() => setActivePlan(null)} className="text-xs text-textTertiary hover:text-textPrimary transition-colors">
            ← Plans
          </button>
          <span className="text-sm font-medium text-textPrimary">{activePlan.icon} {activePlan.title}</span>
          <span className="text-xs font-mono text-textTertiary">
            {completedIds.length}/{questions.length}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="w-32 h-1.5 bg-border rounded-full overflow-hidden" role="progressbar" aria-valuenow={planProgress} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full bg-accentBlue rounded-full transition-all duration-500" style={{ width: `${planProgress}%` }} />
          </div>
          <span className="text-xs font-mono text-textTertiary">{planProgress}%</span>
        </div>
      </div>

      {/* Question + answer — same split layout */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        <div className="flex-1 overflow-y-auto px-8 lg:px-12 py-10">
          <div className="max-w-3xl mx-auto">
            <motion.div key={current.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
              <div className="flex items-center gap-3 mb-4">
                <span className="text-[10px] font-mono text-textTertiary uppercase px-2 py-0.5 rounded border border-border bg-surface">{current.competency_area}</span>
                <span className="text-[10px] font-mono text-textTertiary uppercase px-2 py-0.5 rounded border border-accentBlue/20 bg-accentBlue/5">{current.zone}</span>
                <span className="text-[10px] font-mono text-textTertiary">{current.track} / {current.level}</span>
              </div>
              <h2 className="text-2xl lg:text-3xl font-bold text-textPrimary mb-6 tracking-tight">{current.title}</h2>
              <div className="prose max-w-none">
                <p className="text-textSecondary leading-relaxed text-base">{cleanScenario(current.scenario)}</p>
              </div>
            </motion.div>
          </div>
        </div>

        <div className="w-full lg:w-[460px] border-t lg:border-t-0 lg:border-l border-border bg-surface/90 flex flex-col">
          <div className="h-10 border-b border-border flex items-center px-4 bg-background/50">
            <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest flex items-center gap-2">
              <Target className="w-3 h-3" /> {activePlan.title} — Q{currentIdx + 1}
            </span>
          </div>
          <HardwareRef />
          <div className="flex-1 p-5 flex flex-col overflow-y-auto">
            {!showAnswer ? (
              <>
                <textarea
                  value={userAnswer}
                  onChange={(e) => setUserAnswer(e.target.value)}
                  placeholder={current.details.napkin_math ? "Your napkin math...\n\n=> final answer" : "Your answer..."}
                  className="flex-1 min-h-[200px] w-full bg-background border border-border rounded-md p-5 font-mono text-[13px] text-textPrimary resize-none focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/40 leading-relaxed"
                  spellCheck="false"
                  autoFocus
                />
                <button onClick={handleReveal} className="mt-4 w-full bg-textPrimary text-background font-bold py-3 rounded-lg hover:opacity-90 transition-all flex items-center justify-center gap-2">
                  Reveal <span className="text-[10px] opacity-50 ml-1">⌘↵</span>
                </button>
              </>
            ) : (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
                {napkinResult && (
                  <div className={clsx("p-3 rounded-lg border text-xs",
                    (napkinResult.grade === 'exact' || napkinResult.grade === 'close') ? "bg-accentGreen/10 border-accentGreen/30 text-accentGreen"
                    : napkinResult.grade === 'ballpark' ? "bg-accentAmber/10 border-accentAmber/30 text-accentAmber"
                    : "bg-accentRed/10 border-accentRed/30 text-accentRed"
                  )}>
                    <span className="font-bold">{napkinResult.label}</span>
                    <span className="text-textTertiary ml-2">{napkinResult.userNum.toLocaleString()} vs {napkinResult.modelNum.toLocaleString()}</span>
                  </div>
                )}
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
                {rubricItems.length > 0 && (
                  <div className="border-t border-border pt-4 space-y-2">
                    <span className="text-[10px] font-mono text-textTertiary uppercase">Did you cover?</span>
                    {rubricItems.map((item, idx) => (
                      <label key={idx} className={clsx("flex items-start gap-2 p-2 rounded border cursor-pointer text-xs transition-all",
                        item.checked ? "border-accentGreen/30 bg-accentGreen/5" : "border-border"
                      )}>
                        <input type="checkbox" checked={item.checked} onChange={() => {
                          const updated = [...rubricItems];
                          updated[idx] = { ...updated[idx], checked: !updated[idx].checked };
                          setRubricItems(updated);
                        }} className="mt-0.5 accent-accentGreen" />
                        <span className={item.checked ? "text-textPrimary" : "text-textSecondary"}>{item.text}</span>
                      </label>
                    ))}
                  </div>
                )}
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
                        <button key={score} onClick={() => handleScore(Math.min(score, maxScore))} disabled={disabled}
                          className={clsx("px-3 py-2.5 rounded-lg border text-xs font-medium transition-all",
                            disabled ? "opacity-30 cursor-not-allowed border-border text-textTertiary" : color
                          )}>
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
