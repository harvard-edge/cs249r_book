import type { ChainSummary, ChainMember } from "./corpus";

// ─── Session configuration ──────────────────────────────────────────────────

export interface InterviewConfig {
  track: string;
  level: string;
  duration: number;
  competencyFocus?: string[];
  roleDescription?: string;
}

// ─── Conductor AI metadata ──────────────────────────────────────────────────

export type ConductorIntent =
  | "greeting"
  | "present_scenario"
  | "follow_up"
  | "probe"
  | "napkin_math"
  | "hint"
  | "transition"
  | "closing";

export type ConductorNextAction =
  | "advance_chain"
  | "retreat_chain"
  | "switch_area"
  | "conclude"
  | null;

export interface ConductorMeta {
  intent: ConductorIntent;
  questionRef?: string;
  chainRef?: string;
  performanceNote?: string;
  nextAction?: ConductorNextAction;
  areaRatings?: Record<string, number>;
}

export interface ConductorResponse {
  message: string;
  meta: ConductorMeta;
  provider: string;
  vendorLabel: string;
  modelLabel: string;
  privacyNote: string;
}

// ─── Performance tracking ───────────────────────────────────────────────────

export interface PerformanceSignal {
  questionId: string;
  questionTitle: string;
  signal: "strong" | "partial" | "weak" | "no_response";
  note: string;
  timestamp: number;
}

export interface AreaAssessment {
  area: string;
  questionsAsked: number;
  rating: number;
  evidenceNotes: string[];
  signals: PerformanceSignal[];
}

// ─── Transcript ─────────────────────────────────────────────────────────────

export interface TranscriptEntry {
  id: string;
  role: "candidate" | "interviewer";
  text: string;
  timestamp: number;
  meta?: ConductorMeta;
}

// ─── Session state ──────────────────────────────────────────────────────────

export interface InterviewSession {
  id: string;
  config: InterviewConfig;
  startedAt: number;
  status: "active" | "completed" | "abandoned";
  transcript: TranscriptEntry[];
  chainPool: ChainSummary[];
  currentChainId: string | null;
  currentQuestionId: string | null;
  currentChainPosition: number;
  coveredAreas: string[];
  coveredChainIds: string[];
  areaAssessments: Record<string, AreaAssessment>;
  elapsedSeconds: number;
  report?: InterviewReport;
}

// ─── Report ─────────────────────────────────────────────────────────────────

export interface InterviewReport {
  overallScore: number;
  summary: string;
  areaBreakdown: AreaAssessment[];
  strengths: string[];
  weaknesses: string[];
  practiceRecs: PracticeRecommendation[];
  duration: number;
  questionsDiscussed: number;
}

export interface PracticeRecommendation {
  area: string;
  topic: string;
  questionIds: string[];
  reason: string;
}

// ─── Worker request payload ─────────────────────────────────────────────────

export interface InterviewRequestPayload {
  candidateMessage: string;
  conductorContext: {
    track: string;
    level: string;
    duration: number;
    timeExpired: boolean;
    currentQuestion: {
      id: string;
      title: string;
      level: string;
      zone: string;
      scenario: string;
      canonicalAnswer: string;
      napkinMath?: string;
    } | null;
    chainContext: {
      chainId: string;
      position: number;
      total: number;
      topic: string;
      area: string;
    } | null;
    coveredAreas: string[];
    uncoveredAreas: string[];
    areaRatings: Record<string, number>;
  };
  transcript: { role: "user" | "assistant"; content: string }[];
}
