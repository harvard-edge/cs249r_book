"use client";

import { Suspense, useState, useEffect, useCallback, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Target, CheckCircle2, XCircle, Terminal, SkipForward,
  Calculator, SlidersHorizontal, X, Sparkles, Lightbulb, Clock
} from "lucide-react";
import clsx from "clsx";
import HardwareRef from "@/components/HardwareRef";
import { ScenarioSkeleton } from "@/components/ScenarioSkeleton";
import NapkinCalc from "@/components/NapkinCalc";
// Legacy ID redirect map (cloud-balance-* → cloud-NNNN, etc.). Built by
// rename_legacy_ids.py on 2026-04-25; preserves shareable links to the
// 4,754 cohort-tagged IDs that were renamed to the clean form.
import idRedirects from "@/data/id-redirects.json";
import AskInterviewer from "@/components/AskInterviewer";
import MarkdownText from "@/components/MarkdownText";
import NapkinMathDisplay from "@/components/NapkinMathDisplay";
import LevelBadge from "@/components/LevelBadge";
import { useToast } from "@/components/Toast";
import {
  getTracks, getLevels, getCompetencyAreas, getZones, getQuestionsByFilter,
  getQuestions, getQuestionsByTopic, getTrackCount,
  Question, checkNapkinMath, extractFinalNumber, cleanScenario,
  NapkinResult
} from "@/lib/corpus";
import { saveAttempt, getAttempts, updateSRCard, getDueQuestionIds, getDueCount, recordActivity } from "@/lib/progress";
import { extractRubric, rubricToScore, RubricItem } from "@/lib/rubric";
import { getQuestionById } from "@/lib/corpus";
import { useFullQuestion } from "@/lib/hooks/useFullQuestion";
import { getTopicById, getZoneDefinition } from "@/lib/taxonomy";
import { getLevelDef } from "@/lib/levels";
import { getDailyQuestions, isDailyCompleted, markDailyCompleted } from "@/lib/daily";
import { shouldShowGate, incrementReveals, getRemainingReveals, isStarVerified } from "@/lib/star-gate";
import StarGate from "@/components/StarGate";
import { getChainForQuestion, getPrimaryChainForQuestion, ChainInfo } from "@/lib/corpus";
import ChainStrip from "@/components/ChainStrip";
import ChainBadge from "@/components/ChainBadge";
import { Calendar, ArrowLeft, Flag, LinkIcon } from "lucide-react";
import Link from "next/link";
import { buildReportUrl } from "@/lib/issue-url";
import QuestionFeedback from "@/components/QuestionFeedback";
import QuestionVisual from "@/components/QuestionVisual";
import { track } from "@/lib/analytics";

/**
 * Zone- and Bloom-aware fallback prompt for questions that have no
 * explicit `question` field yet AND no `?` in their scenario. Used by
 * the practice page to render a minimally-useful "Your task (inferred)"
 * callout during the 2026-04 backfill transition so readers aren't
 * left guessing the shape of the expected answer. Keep this short —
 * the goal is to orient, not to substitute for a properly authored
 * question.
 */
function inferTaskPrompt(zone: string | undefined, bloom: string | undefined): string {
  const z = (zone || "").toLowerCase();
  const b = (bloom || "").toLowerCase();
  // Zone-first mapping. Exact match against the 11 ikigai zones plus
  // their common morphological neighbours so we don't miss a variant.
  switch (z) {
    case "diagnosis":
      return "Identify the root cause suggested by the scenario and justify it with a specific mechanism.";
    case "specification":
      return "State the requirements or constraints the scenario imposes, then specify the design these dictate.";
    case "design":
      return "Propose a design that satisfies the scenario's constraints and explain the key trade-offs.";
    case "implement":
    case "realization":
      return "Sketch the implementation or the concrete steps needed to realize the scenario's goal.";
    case "evaluation":
      return "Evaluate the scenario's proposed approach — what works, what breaks, and at what cost?";
    case "optimization":
      return "Identify the dominant bottleneck and propose an optimization that addresses it.";
    case "fluency":
      return "Explain the core mechanism at play and why it behaves as the scenario describes.";
    case "analyze":
      return "Analyze the trade-offs the scenario presents and recommend an approach with justification.";
    case "recall":
      return "Identify the concept the scenario illustrates and name the principle it demonstrates.";
    case "mastery":
      return "Integrate the scenario's constraints, propose an approach, and justify it against the dominant trade-off.";
  }
  // Bloom fallback when zone is unclear.
  if (b === "remember" || b === "understand") return "Identify the concept the scenario illustrates and explain the underlying principle.";
  if (b === "apply") return "Apply the relevant principle to the scenario and compute or decide the outcome.";
  if (b === "analyze") return "Analyze the trade-offs the scenario presents and recommend an approach with justification.";
  if (b === "evaluate") return "Evaluate the scenario's setup — what succeeds, what fails, and why?";
  if (b === "create") return "Propose a design or plan that addresses the scenario and defend your choice.";
  return "Based on the scenario above, reason about the trade-offs and decide what approach you would take.";
}

function normalizePromptText(text: string | undefined): string {
  return (text || "")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function scenarioAlreadyContainsQuestion(
  scenario: string | undefined,
  question: string | undefined
): boolean {
  const normalizedQuestion = normalizePromptText(question);
  if (!normalizedQuestion) return false;
  return normalizePromptText(scenario).includes(normalizedQuestion);
}

export default function PracticePageWrapper() {
  return (
    <Suspense fallback={
      <div className="flex-1 flex items-center justify-center">
        <Terminal className="w-6 h-6 text-textTertiary animate-pulse" />
      </div>
    }>
      <PracticePage />
    </Suspense>
  );
}

function PracticePage() {
  const searchParams = useSearchParams();
  const { show: showToast } = useToast();
  const [mounted, setMounted] = useState(false);
  // Mobile-only filter drawer state. Desktop ignores this entirely (the
  // sidebar is always visible at lg+ breakpoints). On <lg viewports the
  // sidebar transforms into a slide-in drawer triggered by the FAB button
  // in the bottom-right corner of the main content area.
  const [filtersOpen, setFiltersOpen] = useState(false);
  const filterCloseBtnRef = useRef<HTMLButtonElement>(null);

  // Modal hygiene for the mobile filter drawer:
  //   - lock body scroll so the page underneath doesn't scroll while the
  //     drawer is open (otherwise iOS Safari produces a bouncy scroll
  //     chain that scrolls both the drawer and the main page)
  //   - close on Escape key for keyboard users
  //   - move focus to the close button on open so screen readers and
  //     tab navigation start inside the drawer (lightweight focus
  //     management — not a full focus trap, but better than nothing)
  // Desktop (lg+) is unaffected because filtersOpen never becomes true
  // when the drawer is hidden by the lg:translate-x-0 utility.
  useEffect(() => {
    if (!filtersOpen) return;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFiltersOpen(false);
    };
    document.addEventListener("keydown", onKey);
    // Defer focus to next tick so the slide-in animation has started.
    const focusTimer = window.setTimeout(() => {
      filterCloseBtnRef.current?.focus();
    }, 50);
    return () => {
      document.body.style.overflow = prevOverflow;
      document.removeEventListener("keydown", onKey);
      window.clearTimeout(focusTimer);
    };
  }, [filtersOpen]);

  const [selectedTrack, setSelectedTrack] = useState("cloud");
  const [selectedLevel, setSelectedLevel] = useState("L3");
  const [selectedArea, setSelectedArea] = useState<string | null>(null);
  const [selectedZone, setSelectedZone] = useState<string | null>(null);
  const [napkinOnly, setNapkinOnly] = useState(false);
  const [visualOnly, setVisualOnly] = useState(false);
  // notFoundQ holds the ?q= value when it resolves to no published question —
  // either it's a typo, a draft not yet promoted, or an archived id. We
  // surface a small banner instead of silently falling through to a random
  // question, so deep-links from chat / docs are auditable.
  const [notFoundQ, setNotFoundQ] = useState<string | null>(null);
  // "Chains only" restricts the pool to questions that are part of a
  // deepening chain (L1→L6+ on one topic). 890 chains cover ~30% of
  // the corpus. This toggle is the minimum-viable discoverability
  // surface — it exposes chain-membership as a filter without
  // prejudging the gated chain-browse page (see ChainBadge docstring).
  const [chainsOnly, setChainsOnly] = useState(false);

  const [pool, setPool] = useState<Question[]>([]);
  // `currentSummary` holds the lightweight record from the bundled summary
  // (no scenario/details). `current` is hydrated from the worker via
  // useFullQuestion — same shape, but scenario + details populated.
  const [currentSummary, setCurrentSummary] = useState<Question | null>(null);
  const { question: hydrated, status: hydrationStatus } = useFullQuestion(currentSummary);
  const current = hydrated ?? currentSummary;
  const setCurrent = setCurrentSummary;
  const skipFilterCount = useRef(0);
  const questionShownAt = useRef(Date.now());
  const [showAnswer, setShowAnswer] = useState(false);
  const [userAnswer, setUserAnswer] = useState("");
  const [napkinResult, setNapkinResult] = useState<(NapkinResult & { userNum: number; modelNum: number }) | null>(null);
  const [questionsAnswered, setQuestionsAnswered] = useState(0);
  const [weakestArea, setWeakestArea] = useState<{ area: string; pct: number } | null>(null);
  const [dueCount, setDueCount] = useState(0);
  const [reviewMode, setReviewMode] = useState(false); // true = SR review queue
  const [rubricItems, setRubricItems] = useState<RubricItem[]>([]);
  const [dailyDone, setDailyDone] = useState(false);
  const [sourceTopic, setSourceTopic] = useState<{ id: string; name: string } | null>(null);
  const [showStarGate, setShowStarGate] = useState(false);
  // Submit-gradient safeguard (per Chip Huyen's UX review).
  // Rationale: with the restructured layout, the Reveal button sits
  // directly below the answer textarea — removing the eye-travel
  // friction that previously enforced deliberation. Without this
  // guard, users type two sentences and reveal. We detect
  // low-effort reveals (<15s elapsed AND <50 chars typed) and
  // surface a one-time "Think longer?" confirm so the UX doesn't
  // silently erode learning depth. Once confirmed (or once the user
  // has demonstrated normal deliberation on any question), the guard
  // stays dismissed for the session.
  const [thinkConfirmOpen, setThinkConfirmOpen] = useState(false);
  const deliberationCalibrated = useRef(false);
  // Post-reveal model-answer anchor so we can scroll-into-view when
  // showAnswer flips. Prevents the model answer from rendering below
  // the fold on long scenarios.
  const modelAnswerRef = useRef<HTMLDivElement>(null);

  // Chain tracking — primary-first by default; ?chain=<id> URL param can
  // pin a specific chain (used by "more paths" deep-links into secondary
  // chains). Updates when current question changes.
  const chainParam = searchParams.get('chain');
  const chainInfo = current
    ? (chainParam
        ? getChainForQuestion(current.id, chainParam)
        : getPrimaryChainForQuestion(current.id))
    : null;

  // Pre-reveal chain sibling preview. Off by default; toggled open by
  // ChainBadge so the badge's "view chain siblings" affordance does
  // something visible without forcing answer reveal. Reset per question
  // so a new question starts clean.
  const [chainPreviewOpen, setChainPreviewOpen] = useState(false);
  useEffect(() => {
    setChainPreviewOpen(false);
  }, [current?.id]);

  const handleChainNavigate = useCallback((qId: string) => {
    const q = getQuestionById(qId);
    if (!q) return;
    skipFilterCount.current = 1;
    setCurrent(q);
    setShowAnswer(false);
    setUserAnswer("");
    setNapkinResult(null);
    setRubricItems([]);
  }, []);

  const tracks = getTracks().filter(t => t !== "global");
  const levels = getLevels();
  const areas = getCompetencyAreas();
  const zones = getZones();

  useEffect(() => {
    setMounted(true);
    setDailyDone(isDailyCompleted());

    // Auto-trigger daily challenge from ?daily=1 link
    const dailyParam = searchParams.get('daily');
    if (dailyParam === '1' && !isDailyCompleted()) {
      const dailyQs = getDailyQuestions();
      if (dailyQs.length > 0) {
        // React batches the state updates below into a single re-render,
        // which fires the filter-change effect ONCE — so skip exactly one
        // invocation, not three. Using 3 used to swallow the user's first
        // two real track/level clicks silently.
        skipFilterCount.current = 1;
        setPool(dailyQs);
        setCurrent(dailyQs[0]);
        return; // Skip other param handling
      }
    }

    // Default to L1 for brand-new users (no attempts yet)
    if (getAttempts().length === 0 && !searchParams.get('q') && !searchParams.get('topic') && !searchParams.get('level')) {
      setSelectedLevel("L1");
    }

    // Direct question link: ?q=<id> — load that specific question
    const qParam = searchParams.get('q');
    if (qParam) {
      let directQ = getQuestionById(qParam);
      // Fallback: legacy cohort-tagged IDs (cloud-fill-*, cloud-cell-*, ...)
      // were renamed to clean <track>-NNNN form on 2026-04-25. The redirect
      // map preserves access to old shared links — try resolving the param
      // through it before deciding the question is unknown.
      if (!directQ) {
        const redirects = (idRedirects as { redirects: Record<string, string> }).redirects;
        const newId = redirects[qParam];
        if (newId) {
          directQ = getQuestionById(newId);
        }
      }
      if (directQ) {
        // React batches track/level/area setState calls into ONE re-render,
        // so the filter-change effect fires exactly once from the mount
        // batch. Suppress one invocation, not three — the old value of 3
        // silently swallowed the user's first two track-filter clicks.
        skipFilterCount.current = 1;
        setCurrent(directQ);
        setSelectedTrack(directQ.track);
        setSelectedLevel(directQ.level);
        if (directQ.competency_area) setSelectedArea(directQ.competency_area);
        // Set pool to topic-mates so "next" stays in topic
        const topicPool = directQ.topic
          ? getQuestions().filter(q => q.topic === directQ.topic)
          : [directQ];
        setPool(topicPool);
        // Track source topic for back-navigation
        if (directQ.topic) {
          const t = getTopicById(directQ.topic);
          if (t) setSourceTopic({ id: t.id, name: t.name });
        }
        return; // Skip other param handling
      }
      // ?q=<unknown>: surface a banner. Don't silently fall through to the
      // default question (that hides the failure and breaks shared deep-links).
      setNotFoundQ(qParam);
      // Continue to default-flow setup so the page is still usable.
    }

    // Topic filter: ?topic=<concept>&level=<L3> — filter pool to that topic
    const topicParam = searchParams.get('topic');
    const levelParam = searchParams.get('level');
    if (topicParam) {
      const topicPool = getQuestions().filter(q => {
        if (q.topic !== topicParam) return false;
        if (levelParam && q.level !== levelParam) return false;
        return true;
      });
      if (topicPool.length > 0) {
        // See comment on the direct-link branch above: batched state
        // updates fire the filter effect once, so skip exactly one.
        skipFilterCount.current = 1;
        setPool(topicPool);
        setSelectedTrack(topicPool[0].track);
        if (levelParam) setSelectedLevel(levelParam);
        // Track source topic for back-navigation
        const t = getTopicById(topicParam);
        if (t) setSourceTopic({ id: t.id, name: t.name });
        pickRandom(topicPool);
        return; // Skip other param handling
      }
    }

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

  // Update pool when filters change — skip until mounted + direct link consumed
  useEffect(() => {
    if (!mounted) return;
    if (skipFilterCount.current > 0) {
      skipFilterCount.current--;
      return;
    }
    const filters: { track?: string; level?: string; competency_area?: string; zone?: string; chainsOnly?: boolean; visualOnly?: boolean } = {
      track: selectedTrack,
      level: selectedLevel,
    };
    if (selectedArea) filters.competency_area = selectedArea;
    if (selectedZone) filters.zone = selectedZone;
    if (chainsOnly) filters.chainsOnly = true;
    if (visualOnly) filters.visualOnly = true;
    let q = getQuestionsByFilter(filters);
    if (napkinOnly) q = q.filter(question => !!question.details.napkin_math);
    setPool(q);
    if (q.length > 0) {
      pickRandom(q);
    } else {
      setCurrent(null);
    }
    setShowAnswer(false);
    setUserAnswer("");
    setNapkinResult(null);
  }, [mounted, selectedTrack, selectedLevel, selectedArea, selectedZone, napkinOnly, chainsOnly, visualOnly]);

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
    // Track skip if there was a current question that wasn't scored
    if (current && !showAnswer) {
      track({ type: 'question_skipped', topic: current.topic, level: current.level });
    }
    const p = fromPool || pool;
    if (p.length === 0) return;
    const idx = Math.floor(Math.random() * p.length);
    setCurrent(p[idx]);
    questionShownAt.current = Date.now();
    setShowAnswer(false);
    setUserAnswer("");
    setNapkinResult(null);
    setRubricItems([]);
  }, [pool, current, showAnswer]);

  // Submit-gradient guard: intercept reveals that look like
  // "didn't really try." The restructured layout puts the Reveal
  // button directly below the textarea, so the eye-travel friction
  // that previously enforced deliberation is gone. Without a
  // deliberate pause, median time-before-reveal drops and
  // self-assessed scores inflate without actual learning. We surface
  // a one-time confirm, and once the user has demonstrated normal
  // deliberation on any question (≥20s OR ≥80 chars typed), the
  // guard calibrates itself off for the session. `force=true` skips
  // the check — used by the confirm dialog's "Reveal anyway" button.
  const handleReveal = (force: boolean = false) => {
    if (!force && !deliberationCalibrated.current) {
      const elapsedMs = Date.now() - questionShownAt.current;
      const charsTyped = userAnswer.trim().length;
      if (elapsedMs < 15000 && charsTyped < 50) {
        setThinkConfirmOpen(true);
        track({ type: 'think_guard_triggered' });
        return;
      }
      // Any reveal that passes the threshold marks this user as
      // deliberating normally — don't pester them again this session.
      if (elapsedMs >= 20000 || charsTyped >= 80) {
        deliberationCalibrated.current = true;
      }
    }
    setThinkConfirmOpen(false);

    // Star gate check
    if (shouldShowGate()) {
      setShowStarGate(true);
      track({ type: 'star_gate_shown' });
      return;
    }
    incrementReveals();

    // Try napkin math check if the question has napkin_math and user typed something
    let napkinGrade: string | undefined;
    if (current?.details.napkin_math && userAnswer.trim()) {
      const userNum = extractFinalNumber(userAnswer);
      const modelNum = extractFinalNumber(current.details.napkin_math);
      if (userNum !== null && modelNum !== null && modelNum > 0) {
        const result = checkNapkinMath(userNum, modelNum, current.track);
        setNapkinResult({ ...result, userNum, modelNum });
        napkinGrade = result.grade;
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
    // After the reveal state commits, smoothly scroll the model
    // answer into view. This is the primary pedagogical comparison
    // moment — the user's answer should still be visible above, the
    // model answer enters from below. Without this scroll, on long
    // scenarios the model answer renders below the fold and the
    // user has to hunt for it.
    setTimeout(() => {
      modelAnswerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 50);

    // Track answer reveal with response time and napkin grade
    if (current) {
      const hadUserAnswer = userAnswer.trim().length > 0;
      const responseTimeSec = Math.round((Date.now() - questionShownAt.current) / 1000);
      track({ type: 'answer_revealed', topic: current.topic, zone: current.zone, hadUserAnswer });
      if (responseTimeSec > 2) {
        track({
          type: 'answer_response_time',
          questionId: current.id,
          topic: current.topic,
          level: current.level,
          seconds: responseTimeSec,
          napkinGrade,
          hadUserAnswer,
        });
      }
    }
  };

  // Constrain self-assessment: if napkin math was way off, cap the score
  const maxScore = napkinResult?.maxSelfScore ?? 3;
  // Auto-compute score from rubric if items are checked
  const rubricScore = rubricItems.length > 0 ? rubricToScore(rubricItems) : null;
  const effectiveMaxScore = Math.min(maxScore, rubricScore !== null ? rubricScore : 3);

  const handleScore = (score: number) => {
    if (current) {
      const finalScore = Math.min(score, effectiveMaxScore);
      saveAttempt({
        questionId: current.id,
        competencyArea: current.competency_area,
        track: current.track,
        level: current.level,
        selfScore: finalScore,
        timestamp: Date.now(),
      });
      // Update spaced repetition card + streak + analytics
      updateSRCard(current.id, finalScore);
      track({ type: 'question_scored', questionId: current.id, topic: current.topic, zone: current.zone, level: current.level, track: current.track, score: finalScore });
      const activity = recordActivity();
      if (activity.newMilestone) {
        showToast({
          type: 'badge',
          title: activity.newMilestone,
          description: `${activity.streak.currentStreak} day streak!`,
        });
      } else if (questionsAnswered === 0) {
        // First question — reassure user data is saved locally
        showToast({
          type: 'success',
          title: 'Saved to your browser',
          description: 'No account needed. Your progress is stored locally.',
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
    <div className="flex-1 flex flex-col lg:flex-row relative">
      {/*
        Mobile drawer backdrop. Renders only when filtersOpen is true and only
        below the lg breakpoint. Tapping the backdrop closes the drawer.
      */}
      {filtersOpen && (
        <div
          className="lg:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-[1px]"
          onClick={() => setFiltersOpen(false)}
          aria-hidden="true"
        />
      )}
      {/*
        Sidebar — at lg+ this is the inline left column (w-64, in-flow).
        Below lg it transforms into a fixed-position slide-in drawer that
        slides in from the left when the user taps the Filters FAB. The
        drawer occupies 85% of the viewport width (max 320px) so the user
        can still see a sliver of the underlying question — a standard
        affordance from iOS Files / Notion mobile.
      */}
      <aside
        id="practice-filters"
        aria-label="Practice filters"
        role="dialog"
        aria-modal={filtersOpen}
        className={clsx(
          "border-border bg-surface/50 p-4 lg:p-5 flex flex-col gap-4 lg:gap-6",
          // Desktop layout (lg+): inline left column, in-flow, with right border.
          "lg:relative lg:w-64 lg:border-r lg:border-b-0 lg:translate-x-0 lg:overflow-y-auto",
          // Mobile layout (<lg): fixed full-height left drawer, slides in/out.
          "fixed inset-y-0 left-0 z-50 w-[85%] max-w-[320px] overflow-y-auto",
          "transition-transform duration-200 ease-out shadow-2xl lg:shadow-none",
          filtersOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        {/*
          Mobile-only drawer close button. Hidden on lg+ where the sidebar
          is always visible. Receives focus on drawer open via the
          useEffect above so keyboard / screen-reader users start inside
          the drawer.
        */}
        <button
          ref={filterCloseBtnRef}
          onClick={() => setFiltersOpen(false)}
          className="lg:hidden self-end -mt-1 -mr-1 p-2 text-textTertiary hover:text-textPrimary rounded-lg"
          aria-label="Close filters"
        >
          <X className="w-5 h-5" />
        </button>
        {/* Back to vault link */}
        {sourceTopic && (
          <Link
            href="/"
            className="flex items-center gap-1.5 text-[12px] text-textTertiary hover:text-textSecondary transition-colors -mt-1 mb-2"
          >
            <ArrowLeft className="w-3 h-3" />
            Back to {sourceTopic.name}
          </Link>
        )}

        <div className="flex items-center gap-2">
          <Target className="w-5 h-5 text-accentBlue" />
          <h2 className="text-lg font-bold text-textPrimary">Practice</h2>
          {questionsAnswered > 0 && (
            <span className="ml-auto text-[10px] font-mono text-textTertiary bg-surface px-2 py-0.5 rounded border border-border">
              {questionsAnswered} done
            </span>
          )}
        </div>

        {/* Daily Challenge banner */}
        {!dailyDone && (
          <button
            onClick={() => {
              const dailyQs = getDailyQuestions();
              if (dailyQs.length > 0) {
                skipFilterCount.current = 1;
                setPool(dailyQs);
                setCurrent(dailyQs[0]);
                setShowAnswer(false);
                setUserAnswer("");
                setNapkinResult(null);
                setRubricItems([]);
              }
            }}
            className="w-full text-left p-3 rounded-lg bg-accentAmber/5 border border-accentAmber/20 hover:border-accentAmber/40 transition-colors"
          >
            <div className="flex items-center gap-2 mb-1">
              <Calendar className="w-3.5 h-3.5 text-accentAmber" />
              <span className="text-[10px] font-mono text-accentAmber uppercase">Today&apos;s Challenge</span>
            </div>
            <span className="text-sm text-textPrimary font-medium">3 questions</span>
            <span className="text-xs text-textTertiary ml-2">same for everyone</span>
          </button>
        )}
        {dailyDone && (
          <div className="p-2.5 rounded-lg bg-accentGreen/5 border border-accentGreen/20 flex items-center gap-2">
            <CheckCircle2 className="w-3.5 h-3.5 text-accentGreen" />
            <span className="text-[12px] text-accentGreen font-medium">Daily complete</span>
          </div>
        )}

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
            <span className="text-sm text-textPrimary font-medium">{dueCount} questions</span>
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
            <span className="text-sm text-textPrimary capitalize font-medium">{weakestArea.area}</span>
            <span className="text-xs text-textTertiary ml-2">{weakestArea.pct}% accuracy</span>
          </button>
        )}

        {/* Track */}
        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Track</label>
          <div className="flex flex-wrap gap-1 lg:flex-col lg:gap-1">
            {tracks.map(t => (
              <button
                key={t}
                onClick={() => setSelectedTrack(t)}
                className={clsx(
                  "px-3 py-1.5 lg:py-2 lg:w-full rounded-md text-sm font-medium capitalize transition-all flex items-center gap-1.5",
                  "lg:text-left",
                  selectedTrack === t
                    ? "bg-accentBlue/10 text-accentBlue"
                    : "text-textSecondary hover:bg-surfaceHover"
                )}
              >
                <span className="flex-1">{t === "tinyml" ? "TinyML" : t}</span>
                <span
                  className={clsx(
                    "font-mono text-[10px]",
                    selectedTrack === t ? "opacity-70" : "text-textTertiary"
                  )}
                >
                  {getTrackCount(t).toLocaleString()}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Level */}
        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Difficulty</label>
          <div className="grid grid-cols-2 gap-1">
            {levels.map(l => {
              const def = getLevelDef(l);
              return (
                <button
                  key={l}
                  onClick={() => setSelectedLevel(l)}
                  className={clsx(
                    "px-2 py-1.5 rounded text-xs font-medium text-left transition-all flex items-center gap-1.5",
                    selectedLevel === l
                      ? "border"
                      : "text-textSecondary hover:bg-surfaceHover border border-transparent"
                  )}
                  style={selectedLevel === l ? {
                    color: def.color,
                    backgroundColor: `${def.color}10`,
                    borderColor: `${def.color}30`,
                  } : undefined}
                >
                  <div className="w-2 h-2 rounded-sm shrink-0" style={{ backgroundColor: def.color }} />
                  <span className="font-mono">{def.id}</span> {def.name}
                </button>
              );
            })}
          </div>
        </div>

        {/* Competency Area — collapsible on mobile, open on desktop */}
        <details className="group" open>
          <summary className="text-[10px] font-mono text-textTertiary uppercase tracking-widest mb-2 cursor-pointer select-none flex items-center gap-1 list-none">
            Competency
            <span className="text-[8px] text-textMuted group-open:rotate-90 transition-transform">&#9654;</span>
            {selectedArea && <span className="ml-auto text-[9px] text-accentBlue capitalize font-medium">{selectedArea}</span>}
          </summary>
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
        </details>

        {/* Zone filter */}
        <div>
          <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Zone</label>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            <button
              onClick={() => setSelectedZone(null)}
              className={clsx(
                "w-full text-left px-3 py-1.5 rounded text-xs font-medium transition-all",
                !selectedZone
                  ? "bg-accentBlue/10 text-accentBlue"
                  : "text-textSecondary hover:bg-surfaceHover"
              )}
            >
              All zones
            </button>
            {zones.map(z => {
              const def = getZoneDefinition(z);
              return (
                <button
                  key={z}
                  onClick={() => setSelectedZone(z)}
                  title={def?.description || z}
                  className={clsx(
                    "w-full text-left px-3 py-1.5 rounded text-xs font-medium capitalize transition-all",
                    selectedZone === z
                      ? "bg-accentBlue/10 text-accentBlue"
                      : "text-textSecondary hover:bg-surfaceHover"
                  )}
                >
                  {z}
                  {def && (
                    <span className="block text-[9px] font-normal text-textMuted mt-0.5 normal-case">{def.description}</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Napkin Math toggle */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={napkinOnly}
            onChange={() => setNapkinOnly(!napkinOnly)}
            className="accent-accentBlue"
          />
          <span className="text-[11px] text-textSecondary font-medium">Napkin math only</span>
        </label>

        {/* Visual-only toggle — useful for diagram/topology/timing practice. */}
        <label className="flex items-center gap-2 cursor-pointer" title="Restrict pool to questions with diagrams or visual prompts">
          <input
            type="checkbox"
            checked={visualOnly}
            onChange={() => setVisualOnly(!visualOnly)}
            className="accent-accentBlue"
          />
          <span className="text-[11px] text-textSecondary font-medium">Visual questions only</span>
        </label>

        {/* Chains-only toggle — discoverability affordance for the 890
            curated chain sequences (L1→L6+ on one topic). Separate from
            the gated `/chains` browse page — this is the minimum
            filter-side intervention. */}
        <label className="flex items-center gap-2 cursor-pointer" title="Restrict pool to questions that belong to a deepening L1→L6+ chain">
          <input
            type="checkbox"
            checked={chainsOnly}
            onChange={() => setChainsOnly(!chainsOnly)}
            className="accent-accentBlue"
          />
          <LinkIcon className="w-3 h-3 text-textTertiary" aria-hidden="true" />
          <span className="text-[11px] text-textSecondary font-medium">Chained questions only</span>
        </label>

        <div className="text-[10px] font-mono text-textTertiary mt-auto">
          {pool.length} questions in pool
          {chainsOnly && <span className="ml-1 text-accentBlue">· chains</span>}
          {visualOnly && <span className="ml-1 text-accentBlue">· visuals</span>}
        </div>
      </aside>

      {/*
        Floating Filters button — visible only below the lg breakpoint where
        the sidebar is hidden by default. The bottom offset uses
        env(safe-area-inset-bottom) so the FAB clears the iOS home
        indicator on modern iPhones (otherwise the gesture bar overlaps
        the button on iPhone X+). max() guarantees a minimum 1rem gap
        even on devices with no safe-area inset. Aria-controls ties it to
        the drawer for screen readers.
      */}
      <button
        onClick={() => setFiltersOpen(true)}
        aria-controls="practice-filters"
        aria-expanded={filtersOpen}
        className="lg:hidden fixed right-4 z-30 flex items-center gap-2 px-4 py-3 bg-accentBlue text-white font-bold rounded-full shadow-lg shadow-accentBlue/30 hover:opacity-90 active:scale-95 transition-all"
        style={{ bottom: "max(1rem, env(safe-area-inset-bottom, 1rem))" }}
      >
        <SlidersHorizontal className="w-4 h-4" />
        <span className="text-sm">Filters</span>
      </button>

      {/*
        Main content — restructured 2026-04-24 based on UX feedback from
        Emma (beginner), David (power user), and Chip Huyen (practitioner).
        Key shifts:
          - LEFT column = problem + answer + reveal + post-reveal (one
            vertical reading/typing flow). No more 500px eye-travel from
            scenario to textarea.
          - RIGHT column = tools panel (Ask Interviewer, Hardware ref,
            Napkin calc) — each a collapsible card so non-AI users don't
            see wasted pixels shouting "use the AI."
          - Your-task callout is STICKY at the top of the left scroll
            container: scroll the scenario away, the question stays
            visible (David's fix for long-scenario context loss).
          - Submit-gradient safeguard: reveals triggered with <15s
            elapsed AND <50 chars typed pop a "Think longer?" confirm.
            Once the user demonstrates normal deliberation, the guard
            self-calibrates off for the session (Chip's warning about
            the new layout making premature reveals too frictionless).
          - Post-reveal model answer scrolls into view via modelAnswerRef
            so it doesn't render below the fold.
          - "Stuck? Ask the Interviewer →" nudge beneath the textarea
            so beginners discover the AI without it dominating the page
            (Emma's scaffolding request + Chip's "don't advertise
            unused features" rule).
      */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/*
          Not-found banner — fires when ?q=<id> resolves to no published
          question. Surfaces the failure instead of silently swapping to
          a random question (which broke shared deep-links to drafts or
          archived items in the past).
        */}
        {notFoundQ && (
          <div
            role="alert"
            className="mx-8 lg:mx-12 mt-4 px-4 py-3 rounded-lg border border-accentRed/30 bg-accentRed/5 flex items-start gap-3"
          >
            <span className="text-accentRed font-bold shrink-0 mt-0.5">!</span>
            <div className="text-sm text-textSecondary">
              <span className="font-medium text-accentRed">
                Question <code className="font-mono">{notFoundQ}</code> isn&rsquo;t in the published bundle.
              </span>{" "}
              It may be a draft awaiting review, an archived duplicate, or a
              typo. The default question pool is shown below.{" "}
              <button
                onClick={() => setNotFoundQ(null)}
                className="underline hover:no-underline text-textPrimary"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}
        {current ? (
          <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
            {/* ── LEFT: problem + answer + reveal + post-reveal ── */}
            <div className="flex-1 flex flex-col min-h-0">
              <div className="flex-1 overflow-y-auto px-8 lg:px-12 py-10">
                <div className="max-w-3xl mx-auto">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={current.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                    >
                      {/* Badges + copy-link + report */}
                      <div className="flex items-center gap-3 mb-4">
                        <LevelBadge level={current.level} />
                        <span className="text-[10px] font-mono text-textTertiary uppercase px-2 py-0.5 rounded border border-border bg-surface">
                          {current.competency_area}
                        </span>
                        <span className="text-[10px] font-mono text-textTertiary uppercase px-2 py-0.5 rounded border border-accentBlue/20 bg-accentBlue/5">
                          {current.zone}
                        </span>
                        <span className="text-[10px] font-mono text-textTertiary uppercase">
                          {current.track}
                        </span>
                        <span className="flex-1" />
                        <button
                          onClick={() => {
                            const url = `${window.location.origin}/practice?q=${current.id}`;
                            navigator.clipboard.writeText(url);
                            showToast({ type: 'badge', title: 'Link copied', description: 'Share this question with others' });
                          }}
                          className="text-textMuted hover:text-textSecondary transition-colors p-2 -m-2"
                          aria-label="Copy question link"
                          title="Copy question link"
                        >
                          <LinkIcon className="w-3.5 h-3.5" />
                        </button>
                        <a
                          href={buildReportUrl(current)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-[11px] text-textSecondary hover:text-accentRed transition-colors"
                          title="Report an issue with this question"
                        >
                          <Flag className="w-3.5 h-3.5" /> Report
                        </a>
                      </div>

                      {/* Chain badge (pre-reveal only) */}
                      {chainInfo && !showAnswer && (
                        <div className="mb-3">
                          <ChainBadge
                            chainId={chainInfo.chainId}
                            position={chainInfo.position + 1}
                            total={chainInfo.total}
                            tier={chainInfo.tier}
                            onClick={() => setChainPreviewOpen((v) => !v)}
                          />
                        </div>
                      )}

                      {/* Title */}
                      <h2 className="text-2xl lg:text-3xl font-bold text-textPrimary mb-6 tracking-tight">
                        {current.title}
                      </h2>

                      {hydrationStatus === "error" && (
                        <div className="mb-4 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                          Could not load the full question details. The
                          question prompt is shown, but scenario and answer
                          notes are unavailable. Reload to retry.
                        </div>
                      )}

                      {/* Scenario prose */}
                      <div className="prose max-w-none mt-6">
                        {current.scenario ? (
                          <p className="text-textSecondary leading-relaxed text-base">
                            {cleanScenario(current.scenario)}
                          </p>
                        ) : (
                          <ScenarioSkeleton />
                        )}
                      </div>

                      {/*
                        STICKY Your-task callout. Pins to the top of the
                        scroll container so the question stays visible
                        while the user scrolls. Moved below Scenario
                        to provide context before the ask.
                      */}
                      <div className="sticky top-0 z-20 -mx-8 lg:-mx-12 px-8 lg:px-12 py-3 bg-background border-b border-border mt-8">
                        {current.question && !scenarioAlreadyContainsQuestion(current.scenario, current.question) ? (
                          <div className="p-4 rounded-lg border-l-4 border-accentBlue bg-accentBlue/5">
                            <div className="flex items-center gap-2 mb-1.5">
                              <Target className="w-3.5 h-3.5 text-accentBlue" />
                              <span className="text-[10px] font-mono text-accentBlue uppercase tracking-widest">Your task</span>
                            </div>
                            <p className="text-textPrimary leading-relaxed text-base font-medium">
                              <MarkdownText text={current.question} />
                            </p>
                          </div>
                        ) : !current.question && current.scenario && !current.scenario.trim().endsWith("?") ? (
                          <div className="p-4 rounded-lg border border-dashed border-border bg-surface/40">
                            <div className="flex items-center gap-2 mb-1.5">
                              <Target className="w-3.5 h-3.5 text-textTertiary" />
                              <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest">
                                Your task <span className="text-textMuted normal-case">(inferred)</span>
                              </span>
                            </div>
                            <p className="text-textSecondary leading-relaxed text-sm">
                              {inferTaskPrompt(current.zone, current.bloom_level)}
                            </p>
                          </div>
                        ) : (
                          /* Scenario ends with ?; no callout needed but reserve minimal spacing */
                          <div className="flex items-center gap-2 text-[10px] font-mono text-textTertiary uppercase tracking-widest">
                            <Target className="w-3.5 h-3.5" />
                            Your task — see scenario above
                          </div>
                        )}
                      </div>

                      {/* Visual diagram (optional) */}
                      {current.visual && (
                        <QuestionVisual track={current.track} visual={current.visual} />
                      )}

                      {/* Pre-reveal chain sibling preview (toggle from ChainBadge) */}
                      {chainInfo && !showAnswer && chainPreviewOpen && (
                        <div className="mt-6" data-testid="chain-preview-prereveal">
                          <ChainStrip chain={chainInfo} onNavigate={handleChainNavigate} />
                        </div>
                      )}

                      {/* ── ANSWER + REVEAL (pre-reveal) ── */}
                      {!showAnswer ? (
                        <div className="mt-8">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest flex items-center gap-1.5">
                              <Calculator className="w-3 h-3" />
                              {current.details.napkin_math ? "Your napkin math" : "Your answer"}
                            </span>
                          </div>
                          <textarea
                            value={userAnswer}
                            onChange={(e) => setUserAnswer(e.target.value)}
                            placeholder={
                              current.details.napkin_math
                                ? "Type your napkin math here...\n\nExample:\nBandwidth: 3.35 TB/s\nModel size: 140 GB\nTime = 140 / 3350 ≈ 42 ms\n\n=> 42 ms   (mark your final answer with =>)"
                                : "Type your answer or reasoning here..."
                            }
                            className="w-full min-h-[220px] bg-surface border border-border rounded-md p-5 font-mono text-[13px] text-textPrimary resize-y focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/40 leading-relaxed"
                            spellCheck="false"
                          />
                          <button
                            onClick={() => handleReveal()}
                            className="mt-4 w-full bg-textPrimary text-background font-bold py-3 rounded-lg hover:opacity-90 transition-all flex items-center justify-center gap-2"
                          >
                            Reveal Answer <span className="text-[10px] opacity-50 ml-1">⌘↵</span>
                          </button>

                          {/*
                            Beginner scaffolding nudge (Emma's UX review).
                            Subtle link below the textarea that points
                            discoverable users toward the Ask Interviewer
                            feature in the right column. Arrow implies
                            spatial direction. Kept small so it doesn't
                            compete with the Reveal button for attention.
                          */}
                          <div className="mt-3 flex items-center justify-center gap-1.5 text-[11px] text-textTertiary">
                            <Lightbulb className="w-3 h-3" />
                            <span>Stuck? Use <span className="text-accentBlue font-medium">Ask Interviewer</span> in the tools panel →</span>
                          </div>
                        </div>
                      ) : (
                        /* ── POST-REVEAL — model answer lands below the user's input ── */
                        <motion.div
                          ref={modelAnswerRef}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-8 space-y-5 scroll-mt-24"
                        >
                          {userAnswer.trim() && (
                            <details className="group" open>
                              <summary className="text-[10px] font-mono text-textTertiary uppercase cursor-pointer select-none flex items-center gap-1.5">
                                <span className="group-open:rotate-90 transition-transform text-[8px]">&#9654;</span>
                                Your answer
                              </summary>
                              <div className="mt-2 p-3 bg-surface border border-border rounded-md font-mono text-[12px] text-textSecondary whitespace-pre-wrap leading-relaxed max-h-56 overflow-y-auto">
                                {userAnswer}
                              </div>
                            </details>
                          )}

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
                              <span className="text-[10px] font-mono text-accentRed uppercase mb-1 flex items-center gap-1">
                                <XCircle className="w-3 h-3" /> Common Mistake
                              </span>
                              <div className="text-sm text-textSecondary leading-relaxed whitespace-pre-wrap">
                                <MarkdownText text={current.details.common_mistake} />
                              </div>
                            </div>
                          )}

                          <div className="border-l-4 border-accentGreen pl-4">
                            <span className="text-[10px] font-mono text-accentGreen uppercase mb-1 flex items-center gap-1">
                              <CheckCircle2 className="w-3 h-3" /> Model Answer
                            </span>
                            <div className="text-sm text-textPrimary leading-relaxed whitespace-pre-wrap">
                              <MarkdownText text={current.details.realistic_solution} />
                            </div>
                          </div>

                          {current.details.napkin_math && (
                            <div className="bg-surface border border-border p-4 rounded-lg">
                              <span className="text-[10px] font-mono text-accentBlue uppercase mb-3 block">Napkin Math</span>
                              <NapkinMathDisplay text={current.details.napkin_math} />
                            </div>
                          )}

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
                                const disabled = score > effectiveMaxScore;
                                const isRubricSuggested = rubricScore !== null && score === rubricScore;
                                return (
                                  <button
                                    key={score}
                                    onClick={() => handleScore(Math.min(score, effectiveMaxScore))}
                                    disabled={disabled}
                                    aria-label={`Rate yourself: ${label} (${score} of 3)`}
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
                            <QuestionFeedback question={current} />
                            {chainInfo && (
                              <ChainStrip chain={chainInfo} onNavigate={handleChainNavigate} />
                            )}
                          </div>
                        </motion.div>
                      )}
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>

              {/* Sticky bottom bar: pool count + Next Question */}
              <div className="shrink-0 border-t border-border bg-background px-8 lg:px-12 py-3 flex items-center justify-between">
                <span className="text-[11px] font-mono text-textTertiary">
                  {pool.length} in pool
                </span>
                <button
                  onClick={() => pickRandom()}
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-textSecondary hover:text-textPrimary bg-surface border border-border rounded-lg transition-colors"
                >
                  Next Question <SkipForward className="w-3.5 h-3.5" />
                  <kbd className="text-[10px] text-textMuted bg-background border border-border px-1.5 py-0.5 rounded ml-1">N</kbd>
                </button>
              </div>
            </div>

            {/* ── RIGHT: tools panel ── */}
            <div className="w-full lg:w-[400px] border-t lg:border-t-0 lg:border-l border-border bg-surface flex flex-col overflow-y-auto">
              <div className="h-10 border-b border-border flex items-center px-4 bg-background/50 justify-between shrink-0">
                <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest flex items-center gap-2">
                  <Sparkles className="w-3 h-3" /> Tools
                </span>
                <button
                  onClick={() => pickRandom()}
                  className="text-[10px] font-mono text-textTertiary hover:text-textPrimary transition-colors flex items-center gap-1 py-2 -my-2 px-2 -mx-2"
                  aria-label="Skip to a random question"
                >
                  <SkipForward className="w-3 h-3" /> Skip
                </button>
              </div>

              {/*
                Right-column order:
                  - Hardware reference first and open: it is the stable
                    reference table students need while reading the problem.
                  - Napkin calculator stays closed: it is an invoked tool,
                    not persistent context.
                  - Ask Interviewer / Ask Tutor stays last and open: it can
                    grow as a chat transcript without pushing references out
                    of the initial viewport, while remaining visible enough
                    to teach users the clarification/tutoring workflow. The
                    `key` prop force-remounts on persona swap so pre-reveal
                    and post-reveal transcripts stay separate.
              */}
              <HardwareRef defaultOpen={true} />
              <NapkinCalc defaultOpen={false} />
              <AskInterviewer
                key={`${current.id}-${showAnswer ? "study" : "interview"}`}
                questionContext={current.scenario}
                defaultOpen={true}
                mode={showAnswer ? "study" : "interview"}
                canonicalAnswer={showAnswer ? current.details.realistic_solution : undefined}
              />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-textTertiary">
            <p className="text-sm">No questions match your filters. Try adjusting track, level, competency, or zone.</p>
          </div>
        )}
      </div>

      {/*
        Submit-gradient confirm dialog (per Chip Huyen's UX review).
        Fires when a user clicks Reveal with <15s elapsed AND <50
        chars typed — indicating they likely didn't really try. One
        dismissal (either choice) calibrates the guard off for the
        rest of the session so power users are not pestered. Uses a
        full-viewport overlay so the dialog is unmissable.
      */}
      {thinkConfirmOpen && (
        <div
          className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-6"
          onClick={() => setThinkConfirmOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="think-confirm-title"
        >
          <div
            className="bg-surface border border-border rounded-xl p-6 max-w-md w-full shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 mb-3">
              <Clock className="w-4 h-4 text-accentAmber" />
              <h3 id="think-confirm-title" className="text-sm font-bold text-textPrimary uppercase tracking-widest">Think longer?</h3>
            </div>
            <p className="text-sm text-textSecondary leading-relaxed mb-5">
              You&apos;ve been on this question for less than 15 seconds and haven&apos;t typed much. The practice works better when you commit to an answer first — even a wrong one sharpens what you learn from the reveal.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setThinkConfirmOpen(false)}
                className="flex-1 py-2.5 rounded-lg border border-accentBlue/30 bg-accentBlue/5 text-accentBlue font-medium text-sm hover:bg-accentBlue/10 transition-colors"
              >
                Keep thinking
              </button>
              <button
                onClick={() => {
                  deliberationCalibrated.current = true;
                  handleReveal(true);
                }}
                className="flex-1 py-2.5 rounded-lg border border-border text-textSecondary font-medium text-sm hover:bg-surfaceHover transition-colors"
              >
                Reveal anyway
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Star gate overlay */}
      {showStarGate && (
        <StarGate onVerified={() => { setShowStarGate(false); track({ type: 'star_gate_verified' }); }} />
      )}
    </div>
  );
}
