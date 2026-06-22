import {
  getChainsForInterview,
  getChainEntryPoint,
  getChainsByArea,
  getCompetencyAreas,
  getQuestionFullDetail,
  type ChainSummary,
  type ChainMember,
  type Question,
} from "./corpus";
import type {
  InterviewConfig,
  InterviewSession,
  InterviewReport,
  InterviewRequestPayload,
  ConductorResponse,
  ConductorMeta,
  AreaAssessment,
  TranscriptEntry,
  PracticeRecommendation,
} from "./interview-types";

const LEVEL_ADJACENT: Record<string, string[]> = {
  L3: ["L2", "L3", "L4"],
  L4: ["L3", "L4", "L5"],
  L5: ["L4", "L5", "L6+"],
  "L6+": ["L5", "L6+"],
};

// ─── Session lifecycle ──────────────────────────────────────────────────────

export function createSession(config: InterviewConfig): InterviewSession {
  const levels = LEVEL_ADJACENT[config.level] ?? [config.level];
  const chainPool = getChainsForInterview(config.track, levels, config.competencyFocus);

  const allAreas = config.competencyFocus?.length
    ? config.competencyFocus
    : getCompetencyAreas();

  const areaAssessments: Record<string, AreaAssessment> = {};
  for (const area of allAreas) {
    areaAssessments[area] = {
      area,
      questionsAsked: 0,
      rating: 0,
      evidenceNotes: [],
      signals: [],
    };
  }

  const first = selectFirstChain(chainPool, config.level);

  return {
    id: crypto.randomUUID(),
    config,
    startedAt: Date.now(),
    status: "active",
    transcript: [],
    chainPool,
    currentChainId: first?.chainId ?? null,
    currentQuestionId: first?.questionId ?? null,
    currentChainPosition: first?.position ?? 0,
    coveredAreas: [],
    coveredChainIds: [],
    areaAssessments,
    elapsedSeconds: 0,
  };
}

function selectFirstChain(
  pool: ChainSummary[],
  targetLevel: string,
): { chainId: string; questionId: string; position: number } | null {
  if (pool.length === 0) return null;
  const chain = pool[0];
  const entry = getChainEntryPoint(chain, targetLevel);
  if (!entry) return null;
  return { chainId: chain.chainId, questionId: entry.id, position: entry.position };
}

// ─── Chain navigation ───────────────────────────────────────────────────────

export function advanceInChain(session: InterviewSession): InterviewSession | null {
  const chain = session.chainPool.find(c => c.chainId === session.currentChainId);
  if (!chain) return null;

  const nextPos = session.currentChainPosition + 1;
  const next = chain.members.find(m => m.position === nextPos);
  if (!next) return null;

  return {
    ...session,
    currentQuestionId: next.id,
    currentChainPosition: nextPos,
  };
}

export function retreatInChain(session: InterviewSession): InterviewSession | null {
  const chain = session.chainPool.find(c => c.chainId === session.currentChainId);
  if (!chain) return null;

  const prevPos = session.currentChainPosition - 1;
  const prev = chain.members.find(m => m.position === prevPos);
  if (!prev) return null;

  return {
    ...session,
    currentQuestionId: prev.id,
    currentChainPosition: prevPos,
  };
}

export function selectNextChain(
  session: InterviewSession,
): { chainId: string; questionId: string; position: number; reason: string } | null {
  const coveredSet = new Set(session.coveredChainIds);
  const coveredAreaSet = new Set(session.coveredAreas);

  const uncoveredAreaChains = session.chainPool.filter(
    c => !coveredSet.has(c.chainId) && !coveredAreaSet.has(c.area),
  );

  if (uncoveredAreaChains.length > 0) {
    const chain = uncoveredAreaChains[0];
    const entry = getChainEntryPoint(chain, session.config.level);
    if (entry) {
      return {
        chainId: chain.chainId,
        questionId: entry.id,
        position: entry.position,
        reason: `Switching to ${chain.area} (uncovered area)`,
      };
    }
  }

  const unusedChains = session.chainPool.filter(c => !coveredSet.has(c.chainId));
  if (unusedChains.length > 0) {
    const chain = unusedChains[0];
    const entry = getChainEntryPoint(chain, session.config.level);
    if (entry) {
      return {
        chainId: chain.chainId,
        questionId: entry.id,
        position: entry.position,
        reason: `Continuing with ${chain.area} / ${chain.topic}`,
      };
    }
  }

  return null;
}

// ─── Payload assembly ───────────────────────────────────────────────────────

export async function buildWorkerPayload(
  session: InterviewSession,
  candidateMessage: string,
  currentQuestion: Question | null,
  timeExpired: boolean,
): Promise<InterviewRequestPayload> {
  const chain = session.chainPool.find(c => c.chainId === session.currentChainId);
  const allAreas = Object.keys(session.areaAssessments);
  const coveredSet = new Set(session.coveredAreas);

  const areaRatings: Record<string, number> = {};
  for (const [area, assessment] of Object.entries(session.areaAssessments)) {
    if (assessment.questionsAsked > 0) {
      areaRatings[area] = assessment.rating;
    }
  }

  let questionContext: InterviewRequestPayload["conductorContext"]["currentQuestion"] = null;
  if (currentQuestion && currentQuestion.scenario) {
    questionContext = {
      id: currentQuestion.id,
      title: currentQuestion.title,
      level: currentQuestion.level,
      zone: currentQuestion.zone,
      scenario: currentQuestion.scenario,
      canonicalAnswer: currentQuestion.details?.realistic_solution ?? "",
      napkinMath: currentQuestion.details?.napkin_math ?? undefined,
    };
  }

  let chainContext: InterviewRequestPayload["conductorContext"]["chainContext"] = null;
  if (chain) {
    chainContext = {
      chainId: chain.chainId,
      position: session.currentChainPosition,
      total: chain.members.length,
      topic: chain.topic,
      area: chain.area,
    };
  }

  const transcript = windowTranscript(session.transcript, 48_000);

  return {
    candidateMessage,
    conductorContext: {
      track: session.config.track,
      level: session.config.level,
      duration: session.config.duration,
      timeExpired,
      currentQuestion: questionContext,
      chainContext,
      coveredAreas: session.coveredAreas,
      uncoveredAreas: allAreas.filter(a => !coveredSet.has(a)),
      areaRatings,
    },
    transcript: transcript.map(t => ({
      role: t.role === "interviewer" ? "assistant" as const : "user" as const,
      content: t.text,
    })),
  };
}

// ─── Response handling ──────────────────────────────────────────────────────

export function applyResponse(
  session: InterviewSession,
  response: ConductorResponse,
  candidateMessage: string,
): InterviewSession {
  const now = Date.now();

  const candidateEntry: TranscriptEntry = {
    id: crypto.randomUUID(),
    role: "candidate",
    text: candidateMessage,
    timestamp: now - 1,
  };

  const interviewerEntry: TranscriptEntry = {
    id: crypto.randomUUID(),
    role: "interviewer",
    text: response.message,
    timestamp: now,
    meta: response.meta,
  };

  const newTranscript = candidateMessage
    ? [...session.transcript, candidateEntry, interviewerEntry]
    : [...session.transcript, interviewerEntry];

  const chain = session.chainPool.find(c => c.chainId === session.currentChainId);
  const coveredAreas = [...session.coveredAreas];
  if (chain && !coveredAreas.includes(chain.area)) {
    coveredAreas.push(chain.area);
  }

  const coveredChainIds = [...session.coveredChainIds];
  if (session.currentChainId && !coveredChainIds.includes(session.currentChainId)) {
    coveredChainIds.push(session.currentChainId);
  }

  const areaAssessments = { ...session.areaAssessments };
  if (chain && areaAssessments[chain.area]) {
    const assessment = { ...areaAssessments[chain.area] };
    if (response.meta.performanceNote) {
      assessment.evidenceNotes = [...assessment.evidenceNotes, response.meta.performanceNote];
    }
    if (response.meta.areaRatings && response.meta.areaRatings[chain.area] !== undefined) {
      assessment.rating = response.meta.areaRatings[chain.area];
    }
    if (response.meta.intent === "present_scenario" || response.meta.intent === "follow_up") {
      assessment.questionsAsked = assessment.questionsAsked + 1;
    }
    areaAssessments[chain.area] = assessment;
  }

  return {
    ...session,
    transcript: newTranscript,
    coveredAreas,
    coveredChainIds,
    areaAssessments,
  };
}

// ─── Report generation ──────────────────────────────────────────────────────

export function generateReport(session: InterviewSession): InterviewReport {
  const assessed = Object.values(session.areaAssessments).filter(a => a.questionsAsked > 0);
  const strengths: string[] = [];
  const weaknesses: string[] = [];
  const practiceRecs: PracticeRecommendation[] = [];

  for (const a of assessed) {
    if (a.rating >= 2.5) {
      strengths.push(`${a.area}: ${a.evidenceNotes[a.evidenceNotes.length - 1] || "strong performance"}`);
    } else if (a.rating <= 1.5 && a.questionsAsked > 0) {
      weaknesses.push(`${a.area}: ${a.evidenceNotes[a.evidenceNotes.length - 1] || "needs improvement"}`);

      const areaChains = getChainsByArea(a.area, session.config.track);
      const recIds = areaChains
        .slice(0, 2)
        .flatMap(c => c.members.slice(0, 2).map(m => m.id));

      if (recIds.length > 0) {
        practiceRecs.push({
          area: a.area,
          topic: areaChains[0]?.topic ?? a.area,
          questionIds: recIds,
          reason: `Strengthen ${a.area} fundamentals`,
        });
      }
    }
  }

  const totalRating = assessed.length > 0
    ? assessed.reduce((sum, a) => sum + a.rating, 0) / assessed.length
    : 0;
  const overallScore = Math.round((totalRating / 3) * 100);

  const elapsed = session.elapsedSeconds || Math.round((Date.now() - session.startedAt) / 1000);

  return {
    overallScore,
    summary: buildSummary(overallScore, strengths, weaknesses),
    areaBreakdown: assessed,
    strengths,
    weaknesses,
    practiceRecs,
    duration: Math.round(elapsed / 60),
    questionsDiscussed: assessed.reduce((sum, a) => sum + a.questionsAsked, 0),
  };
}

function buildSummary(score: number, strengths: string[], weaknesses: string[]): string {
  if (score >= 80) return "Strong performance across assessed areas with clear systems thinking and accurate estimation.";
  if (score >= 60) return "Solid foundation with some areas showing depth. Targeted practice on weaker areas would strengthen the overall profile.";
  if (score >= 40) return "Partial understanding demonstrated. Several core areas need deeper study before a staff-level interview.";
  return "Significant gaps across multiple areas. Recommend focused study using the practice recommendations below.";
}

// ─── Transcript windowing ───────────────────────────────────────────────────

export function windowTranscript(
  transcript: TranscriptEntry[],
  maxBytes: number,
): TranscriptEntry[] {
  if (transcript.length === 0) return [];

  const full = JSON.stringify(transcript);
  if (full.length <= maxBytes) return transcript;

  const result: TranscriptEntry[] = [];
  let currentSize = 2; // for [] brackets

  for (let i = transcript.length - 1; i >= 0; i--) {
    const entryJson = JSON.stringify(transcript[i]);
    if (currentSize + entryJson.length + 1 > maxBytes) break;
    result.unshift(transcript[i]);
    currentSize += entryJson.length + 1;
  }

  if (result.length < transcript.length && result.length > 0) {
    const skipped = transcript.length - result.length;
    const summary: TranscriptEntry = {
      id: "summary",
      role: "interviewer",
      text: `[Earlier: ${skipped} exchanges covering ${new Set(
        transcript
          .slice(0, skipped)
          .filter(t => t.meta?.chainRef)
          .map(t => t.meta!.chainRef)
      ).size} topics]`,
      timestamp: transcript[0].timestamp,
    };
    result.unshift(summary);
  }

  return result;
}
