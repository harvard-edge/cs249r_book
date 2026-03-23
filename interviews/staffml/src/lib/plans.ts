// Curated study plans — ordered question sequences for targeted prep
import { getQuestionsByFilter, Question } from './corpus';

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
