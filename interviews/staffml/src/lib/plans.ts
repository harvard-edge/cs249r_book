// Curated study plans + custom learning paths
import { getQuestionsByFilter, getCompetencyAreas, Question } from './corpus';

export interface StudyPlan {
  id: string;
  title: string;
  description: string;
  duration: string;
  questionCount: number;
  icon: string; // emoji
  track: string;
  levels: string[];
  // Question selection: competency areas to cover in order
  competencySequence: string[];
}

export const STUDY_PLANS: StudyPlan[] = [
  {
    id: 'mle-sprint',
    title: 'MLE Interview Sprint',
    description: 'Fastest path to interview readiness. 5 questions/day across core competencies.',
    duration: '2 weeks',
    questionCount: 50,
    icon: '⚡',
    track: 'cloud',
    levels: ['L4', 'L5'],
    competencySequence: [
      'compute', 'memory', 'latency', 'parallelism', 'optimization',
      'architecture', 'deployment', 'reliability', 'networking', 'data',
    ],
  },
  {
    id: 'staff-deep',
    title: 'Staff Engineer Deep Dive',
    description: 'Comprehensive coverage for L6+ candidates. Systems design focus.',
    duration: '4 weeks',
    questionCount: 100,
    icon: '🏗️',
    track: 'cloud',
    levels: ['L5', 'L6+'],
    competencySequence: [
      'architecture', 'compute', 'memory', 'parallelism', 'networking',
      'latency', 'optimization', 'reliability', 'deployment', 'data',
      'cross-cutting', 'precision',
    ],
  },
  {
    id: 'edge-specialist',
    title: 'Edge ML Specialist',
    description: 'Real-time inference, sensor pipelines, power-constrained deployment.',
    duration: '2 weeks',
    questionCount: 40,
    icon: '🤖',
    track: 'edge',
    levels: ['L3', 'L4', 'L5'],
    competencySequence: [
      'compute', 'memory', 'latency', 'power', 'deployment',
      'optimization', 'architecture',
    ],
  },
  {
    id: 'mobile-ai',
    title: 'Mobile AI Engineer',
    description: 'On-device ML, quantization, battery-efficient inference.',
    duration: '10 days',
    questionCount: 30,
    icon: '📱',
    track: 'mobile',
    levels: ['L3', 'L4'],
    competencySequence: [
      'compute', 'memory', 'optimization', 'deployment', 'power', 'latency',
    ],
  },
  {
    id: '72hr-blitz',
    title: '72-Hour Blitz',
    description: 'Interview tomorrow? 30 highest-yield questions across all areas.',
    duration: '3 days',
    questionCount: 30,
    icon: '🔥',
    track: 'cloud',
    levels: ['L4', 'L5'],
    competencySequence: [
      'compute', 'memory', 'latency', 'parallelism', 'architecture',
      'optimization',
    ],
  },
  {
    id: 'gpu-systems',
    title: 'GPU Systems Engineer',
    description: 'CUDA, memory hierarchy, kernel optimization, multi-GPU scaling. For roles building on accelerated hardware.',
    duration: '3 weeks',
    questionCount: 80,
    icon: '🎮',
    track: 'cloud',
    levels: ['L4', 'L5', 'L6+'],
    competencySequence: [
      'compute', 'memory', 'optimization', 'parallelism',
      'architecture', 'latency', 'precision', 'networking',
    ],
  },
  {
    id: 'llm-inference',
    title: 'LLM Inference Engineer',
    description: 'KV-cache, batching, quantization, serving at scale. For roles deploying large language models.',
    duration: '2 weeks',
    questionCount: 60,
    icon: '🧠',
    track: 'cloud',
    levels: ['L4', 'L5', 'L6+'],
    competencySequence: [
      'memory', 'latency', 'compute', 'optimization', 'architecture',
      'deployment', 'precision', 'reliability',
    ],
  },
];

// Generate the ordered question list for a study plan
export function getPlanQuestions(plan: StudyPlan): Question[] {
  const result: Question[] = [];
  const usedIds = new Set<string>();
  const perArea = Math.ceil(plan.questionCount / plan.competencySequence.length);

  for (const area of plan.competencySequence) {
    for (const level of plan.levels) {
      const pool = getQuestionsByFilter({
        track: plan.track,
        level,
        competency_area: area,
      }).filter(q => !usedIds.has(q.id));

      const take = Math.min(pool.length, perArea - result.filter(q => q.competency_area === area).length);
      // Sort by topic for conceptual grouping (deterministic ordering)
      pool.sort((a, b) => a.topic.localeCompare(b.topic) || a.id.localeCompare(b.id));
      const selected = pool.slice(0, take);
      selected.forEach(q => {
        result.push(q);
        usedIds.add(q.id);
      });

      if (result.length >= plan.questionCount) break;
    }
    if (result.length >= plan.questionCount) break;
  }

  return result.slice(0, plan.questionCount);
}

// Get user's progress on a plan
const PLAN_PROGRESS_KEY = 'staffml_plan_progress';

export interface PlanProgress {
  planId: string;
  completedIds: string[];
  startedAt: number;
}

export function getPlanProgress(planId: string): PlanProgress {
  try {
    const all = JSON.parse(window.localStorage.getItem(PLAN_PROGRESS_KEY) || '{}');
    return all[planId] || { planId, completedIds: [], startedAt: 0 };
  } catch {
    return { planId, completedIds: [], startedAt: 0 };
  }
}

export function markPlanQuestionComplete(planId: string, questionId: string): void {
  try {
    const all = JSON.parse(window.localStorage.getItem(PLAN_PROGRESS_KEY) || '{}');
    if (!all[planId]) {
      all[planId] = { planId, completedIds: [], startedAt: Date.now() };
    }
    if (!all[planId].completedIds.includes(questionId)) {
      all[planId].completedIds.push(questionId);
    }
    window.localStorage.setItem(PLAN_PROGRESS_KEY, JSON.stringify(all));
  } catch {}
}

// ─── Custom Learning Paths ──────────────────────

const CUSTOM_PATHS_KEY = 'staffml_custom_paths';
const LEVEL_ORDER = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6+'];

export interface CustomPath {
  id: string;
  title: string;
  track: string;
  area: string | null;
  startLevel: string;
  targetDate?: number;
  dailyQuota?: number;
  createdAt: number;
}

export function getCustomPaths(): CustomPath[] {
  try {
    const raw = window.localStorage.getItem(CUSTOM_PATHS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveCustomPath(path: CustomPath): void {
  try {
    const paths = getCustomPaths().filter(p => p.id !== path.id);
    paths.unshift(path);
    window.localStorage.setItem(CUSTOM_PATHS_KEY, JSON.stringify(paths));
  } catch {}
}

export function deleteCustomPath(id: string): void {
  try {
    const paths = getCustomPaths().filter(p => p.id !== id);
    window.localStorage.setItem(CUSTOM_PATHS_KEY, JSON.stringify(paths));
    const all = JSON.parse(window.localStorage.getItem(PLAN_PROGRESS_KEY) || '{}');
    delete all[id];
    window.localStorage.setItem(PLAN_PROGRESS_KEY, JSON.stringify(all));
  } catch {}
}

export function generateCustomPathQuestions(path: CustomPath): Question[] {
  const startIdx = LEVEL_ORDER.indexOf(path.startLevel);
  const levels = startIdx >= 0 ? LEVEL_ORDER.slice(startIdx) : LEVEL_ORDER;
  const areas = path.area ? [path.area] : getCompetencyAreas();
  const result: Question[] = [];
  const usedIds = new Set<string>();

  for (const level of levels) {
    for (const area of areas) {
      const pool = getQuestionsByFilter({ track: path.track, level, competency_area: area })
        .filter(q => !usedIds.has(q.id));
      pool.sort((a, b) => a.topic.localeCompare(b.topic) || a.id.localeCompare(b.id));
      for (const q of pool) {
        result.push(q);
        usedIds.add(q.id);
      }
    }
  }

  return result;
}

export function countCustomPathQuestions(track: string, area: string | null, startLevel: string): number {
  const startIdx = LEVEL_ORDER.indexOf(startLevel);
  const levels = startIdx >= 0 ? LEVEL_ORDER.slice(startIdx) : LEVEL_ORDER;
  const areas = area ? [area] : getCompetencyAreas();
  let count = 0;
  for (const level of levels) {
    for (const a of areas) {
      count += getQuestionsByFilter({ track, level, competency_area: a }).length;
    }
  }
  return count;
}

// ─── Interview Prep Stats ───────────────────────

export interface PrepStats {
  totalQuestions: number;
  completed: number;
  daysRemaining: number;
  dailyTarget: number;
  todayCompleted: number;
  onTrack: boolean;
  projectedCompletion: number;
}

export function getPrepStats(path: CustomPath, questions: Question[]): PrepStats | null {
  if (!path.targetDate) return null;
  const progress = getPlanProgress(path.id);
  const completed = progress.completedIds.length;
  const total = questions.length;
  const remaining = total - completed;
  const msRemaining = path.targetDate - Date.now();
  const daysRemaining = Math.max(0, Math.ceil(msRemaining / 86400000));
  const dailyTarget = daysRemaining > 0 ? Math.ceil(remaining / daysRemaining) : remaining;

  const today = new Date().toISOString().split('T')[0];
  let todayCompleted = 0;
  try {
    const { getAttempts } = require('./progress') as typeof import('./progress');
    const attempts = getAttempts();
    const completedSet = new Set(progress.completedIds);
    todayCompleted = attempts.filter(a =>
      new Date(a.timestamp).toISOString().split('T')[0] === today &&
      completedSet.has(a.questionId)
    ).length;
  } catch {}

  const daysSinceStart = Math.max(1, Math.ceil((Date.now() - (progress.startedAt || path.createdAt)) / 86400000));
  const pace = completed / daysSinceStart;
  const projectedDays = pace > 0 ? Math.ceil(remaining / pace) : Infinity;
  const projectedCompletion = Date.now() + projectedDays * 86400000;

  const totalDays = Math.max(1, Math.ceil((path.targetDate - (progress.startedAt || path.createdAt)) / 86400000));
  const expectedByNow = Math.round((daysSinceStart / totalDays) * total);
  const onTrack = completed >= expectedByNow;

  return { totalQuestions: total, completed, daysRemaining, dailyTarget, todayCompleted, onTrack, projectedCompletion };
}
