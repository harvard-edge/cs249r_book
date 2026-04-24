import corpusData from '../data/corpus-summary.json';

/**
 * Question shape matching vault schema v1.0.
 *
 * As of 2026-04-22 the bundle is SUMMARY-ONLY: the heavy `scenario` and
 * `details` fields live on the Cloudflare Worker at
 * https://staffml-vault.mlsysbook-ai-account.workers.dev and are fetched
 * lazily via `getQuestionFullDetail()` (async) or the `useFullQuestion()`
 * React hook. This took the bundled data from 20.5 MiB to 2.9 MiB.
 *
 * For callers that access `scenario` or `details.*` synchronously, the
 * fields are defined as optional on this interface — they will be `undefined`
 * until the worker fetch resolves. Use the `useFullQuestion` hook to get
 * a hydrated record as a drop-in replacement for the summary.
 */
export interface Question {
  id: string;
  track: string;
  level: string;
  title: string;
  /**
   * Explicit one-sentence interrogative derived from (scenario,
   * realistic_solution). Ships in the summary bundle so the practice
   * page can render it synchronously as a "Your task" callout. Optional
   * while the backfill is in progress — if absent, the render falls
   * back to a zone-based inferred-task label.
   */
  question?: string;
  topic: string;            // one of 87 curated topic IDs
  zone: string;             // one of 11 ikigai zones
  competency_area: string;  // one of 13 canonical areas
  bloom_level?: string;     // remember | understand | apply | analyze | evaluate | create
  phase?: string;           // training | inference | both
  status?: string;          // draft | published | flagged | archived | deleted
  chain_ids?: string[];
  chain_positions?: Record<string, number>;

  // ── Heavy fields (bundled as empty stubs; hydrated from worker) ──
  // The summary bundle ships scenario: "" and details with empty strings
  // for common_mistake / realistic_solution / napkin_math. Hydration via
  // `useFullQuestion(q)` or `getQuestionFullDetail(q.id)` fills them with
  // real content from the worker. MCQ options/correct_index ARE bundled
  // (scoring uses them synchronously).
  scenario: string;
  details: {
    common_mistake: string;
    realistic_solution: string;
    napkin_math?: string;
    resources?: Resource[];
    options?: string[];
    correct_index?: number;
  };

  // ── Trust signals (bundled; populated when YAMLs are regenerated) ──
  /** LLM validation pass (Gemini). */
  validated?: boolean;
  /** Second-pass LLM math check. */
  math_verified?: boolean;
  /** Human verification, distinct from LLM stamps. */
  human_reviewed?: {
    status: string;         // not-reviewed | verified | flagged | needs-rework
    by?: string | null;
    date?: string | null;
  };
}

/** Author-curated external reference attached to a question. */
export interface Resource {
  name: string;
  url: string;
}

const questions = corpusData as unknown as Question[];

export function getQuestions(): Question[] {
  return questions;
}

/**
 * Marketing-friendly question count string. Rounds the live corpus length
 * down to the nearest thousand and appends a `+` so the headline never goes
 * stale until the next 1,000-question milestone is crossed.
 */
export const QUESTION_COUNT_DISPLAY = `${(Math.floor(questions.length / 1000) * 1000).toLocaleString("en-US")}+`;

export function getQuestionById(id: string): Question | undefined {
  return questions.find((q) => q.id === id);
}

export function getTracks(): string[] {
  const tracks = new Set(questions.map((q) => q.track));
  return Array.from(tracks).sort();
}

// Memoize per-track counts so re-renders don't re-scan 9k+ questions.
const _trackCounts: Record<string, number> = (() => {
  const counts: Record<string, number> = {};
  for (const q of questions) counts[q.track] = (counts[q.track] ?? 0) + 1;
  return counts;
})();

/** Total question count for a single track, or the full corpus when omitted. */
export function getTrackCount(track?: string): number {
  return track ? (_trackCounts[track] ?? 0) : questions.length;
}

export function getLevels(): string[] {
  const order = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6+'];
  const levels = new Set(questions.map((q) => q.level));
  return order.filter(l => levels.has(l));
}

export function getCompetencyAreas(): string[] {
  const areas = new Set(questions.map((q) => q.competency_area));
  return Array.from(areas).sort();
}

export function getZones(): string[] {
  const zones = new Set(questions.map((q) => q.zone));
  return Array.from(zones).sort();
}

export function getTopics(): string[] {
  const topics = new Set(questions.map((q) => q.topic));
  return Array.from(topics).sort();
}

export function getTopicsByArea(area: string): string[] {
  const topics = new Set(
    questions.filter(q => q.competency_area === area).map(q => q.topic)
  );
  return Array.from(topics).sort();
}

export function getQuestionsByFilter(filters: {
  track?: string;
  level?: string;
  competency_area?: string;
  topic?: string;
  zone?: string;
}): Question[] {
  return questions.filter((q) => {
    if (filters.track && q.track !== filters.track) return false;
    if (filters.level && q.level !== filters.level) return false;
    if (filters.competency_area && q.competency_area !== filters.competency_area) return false;
    if (filters.topic && q.topic !== filters.topic) return false;
    if (filters.zone && q.zone !== filters.zone) return false;
    return true;
  });
}

/**
 * Fallback full-text search — used only when the Worker `/search` endpoint
 * (FTS5, via `corpus-provider.vaultSearch`) is unreachable. Because the
 * bundle is now summary-only (no scenario/details), this fallback searches
 * titles + topics only. The worker path is always preferred and gives
 * real FTS5 ranking over all fields.
 */
export function searchQuestions(query: string, limit = 50): Question[] {
  const q = query.toLowerCase().trim();
  if (!q) return [];

  const terms = q.split(/\s+/).filter(t => t.length >= 2);
  if (terms.length === 0) return [];

  const scored: { question: Question; score: number }[] = [];

  for (const question of questions) {
    let score = 0;
    const title = question.title.toLowerCase();
    const topic = question.topic.toLowerCase();

    for (const term of terms) {
      if (title.includes(term)) score += 10;
      if (topic.includes(term)) score += 3;
    }

    if (score > 0) {
      scored.push({ question, score });
    }
  }

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(s => s.question);
}

export function getQuestionsByTopic(topicId: string, level?: string): Question[] {
  return questions.filter((q) => {
    if (q.topic !== topicId) return false;
    if (level && q.level !== level) return false;
    return true;
  });
}

export function getQuestionsByZone(zone: string): Question[] {
  return questions.filter(q => q.zone === zone);
}

// Gauntlet: select N questions ensuring competency breadth, with warm-up
export function selectGauntletQuestions(
  track: string,
  level: string,
  count: number
): Question[] {
  const pool = questions.filter(q => q.track === track && q.level === level);
  if (pool.length === 0) return [];

  // Add warm-up: pick one easier question if target level is L4+
  const warmUpLevels: Record<string, string> = { 'L4': 'L2', 'L5': 'L3', 'L6+': 'L3' };
  const warmUpLevel = warmUpLevels[level];
  let warmUp: Question | null = null;
  if (warmUpLevel) {
    const warmUpPool = questions.filter(q => q.track === track && q.level === warmUpLevel);
    if (warmUpPool.length > 0) {
      warmUp = warmUpPool[Math.floor(Math.random() * warmUpPool.length)];
    }
  }

  // Group by zone for breadth across competency zones
  const byZone: Record<string, Question[]> = {};
  pool.forEach(q => {
    const zone = q.zone || 'recall';
    if (!byZone[zone]) byZone[zone] = [];
    byZone[zone].push(q);
  });

  const zones = Object.keys(byZone);
  const selected: Question[] = [];
  const usedIds = new Set<string>();

  // Round-robin across zones
  let zoneIdx = 0;
  while (selected.length < count && selected.length < pool.length) {
    const zone = zones[zoneIdx % zones.length];
    const available = byZone[zone].filter(q => !usedIds.has(q.id));
    if (available.length > 0) {
      const pick = available[Math.floor(Math.random() * available.length)];
      selected.push(pick);
      usedIds.add(pick.id);
    }
    zoneIdx++;
    if (zoneIdx > zones.length * count) break;
  }

  // Shuffle the main selection
  for (let i = selected.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [selected[i], selected[j]] = [selected[j], selected[i]];
  }

  // Prepend warm-up question at position 0
  if (warmUp && !usedIds.has(warmUp.id)) {
    selected.unshift(warmUp);
    if (selected.length > count) selected.pop();
  }

  return selected;
}

// Napkin math result grades
export type NapkinGrade = 'exact' | 'close' | 'ballpark' | 'off' | 'way_off';

export interface NapkinResult {
  grade: NapkinGrade;
  ratio: number; // how far off (0 = exact)
  tolerance: number;
  label: string;
  maxSelfScore: number; // caps self-assessment
}

export function checkNapkinMath(
  userAnswer: number,
  modelAnswer: number,
  track: string
): NapkinResult {
  const tolerances: Record<string, number> = {
    cloud: 0.25,
    edge: 0.15,
    mobile: 0.10,
    tinyml: 0.05,
  };
  const tolerance = tolerances[track] || 0.25;
  const ratio = Math.abs(userAnswer - modelAnswer) / Math.max(modelAnswer, 1);

  if (ratio <= tolerance * 0.5) {
    return { grade: 'exact', ratio, tolerance, label: 'Spot on', maxSelfScore: 3 };
  }
  if (ratio <= tolerance) {
    return { grade: 'close', ratio, tolerance, label: 'Within tolerance', maxSelfScore: 3 };
  }
  if (ratio <= 1.0) {
    return { grade: 'ballpark', ratio, tolerance, label: 'Right ballpark', maxSelfScore: 2 };
  }
  if (ratio <= 5.0) {
    return { grade: 'off', ratio, tolerance, label: `Off by ${ratio.toFixed(1)}×`, maxSelfScore: 1 };
  }
  return { grade: 'way_off', ratio, tolerance, label: `Off by ${ratio.toFixed(0)}×`, maxSelfScore: 1 };
}

// Clean scenario text: strip markdown interviewer prefix and stray quotes
export function cleanScenario(text: string): string {
  return text
    .replace(/^-\s*\*\*Interviewer:\*\*\s*/i, '')
    .replace(/^"/,'')
    .replace(/"$/,'')
    .trim();
}

// ─── Answer-type inference ──────────────────────────────────
// TODO(answer_type): replace this heuristic with an explicit
// `answer_type: 'numeric' | 'recall' | 'conceptual' | 'design'`
// field on every question in the corpus. Until then, infer from
// the scenario text. The bias is conservative: only classify as
// numeric when we're confident, so the grader can never fire on
// a recall question like "What does NPU stand for?".
const QUANT_VERBS = /\b(estimate|calculate|compute|how (?:many|much|long|fast|big)|what(?:'s| is) the (?:size|bandwidth|throughput|latency|memory|cost|time|number|ratio)|derive|approximate)\b/i;
const HAS_DIGIT = /\d/;
const QUESTION_MARK = /\?/;

export function isNumericQuestion(question: { scenario: string; details: { napkin_math?: string } }): boolean {
  // Required: the corpus has napkin math AND that napkin math contains a number
  if (!question.details.napkin_math) return false;
  if (extractFinalNumber(question.details.napkin_math) === null) return false;

  // Required: the prompt itself either contains a quantitative verb,
  // or shows a digit alongside a question mark (a typical numeric ask).
  const scenario = question.scenario || '';
  if (QUANT_VERBS.test(scenario)) return true;
  if (HAS_DIGIT.test(scenario) && QUESTION_MARK.test(scenario)) return true;

  // Defensive default: not numeric. Falls back to self-rate.
  return false;
}

// Extract the user's final answer number
export function extractFinalNumber(text: string): number | null {
  const markerMatch = text.match(/(?:^|\n)\s*(?:=>|answer:|final:)\s*([\d,]+(?:\.\d+)?)/im);
  if (markerMatch) {
    const num = Number(markerMatch[1].replace(/,/g, ''));
    if (!isNaN(num) && isFinite(num)) return num;
  }

  const numbers = text.match(/[\d,]+(?:\.\d+)?/g)?.map(s => Number(s.replace(/,/g, ''))) || [];
  const valid = numbers.filter(n => !isNaN(n) && isFinite(n));
  return valid.length > 0 ? valid[valid.length - 1] : null;
}

// ─── Chain helpers ──────────────────────────────────────────
// Chains are deepening question sequences on a topic (L1 → L6+)

export interface ChainInfo {
  chainId: string;
  position: number;       // 0-indexed position of current question
  total: number;          // total questions in chain
  questions: { id: string; title: string; level: string; position: number }[];
}

// Build chain index once
const _chainIndex = new Map<string, { id: string; title: string; level: string; position: number }[]>();
for (const q of questions) {
  if (!q.chain_ids || !q.chain_positions) continue;
  for (const chainId of q.chain_ids) {
    const pos = q.chain_positions[chainId];
    if (pos === undefined) continue;
    if (!_chainIndex.has(chainId)) _chainIndex.set(chainId, []);
    _chainIndex.get(chainId)!.push({
      id: q.id,
      title: q.title,
      level: q.level,
      position: pos,
    });
  }
}
// Sort each chain by position
_chainIndex.forEach((qs) => {
  qs.sort((a, b) => a.position - b.position);
});

/** Get chain info for a question, or null if not in a chain */
export function getChainForQuestion(questionId: string): ChainInfo | null {
  const q = questions.find(x => x.id === questionId);
  if (!q || !q.chain_ids || !q.chain_positions) return null;

  // Use the first chain this question belongs to
  const chainId = q.chain_ids[0];
  if (!chainId) return null;
  const pos = q.chain_positions[chainId];
  if (pos === undefined) return null;

  const chain = _chainIndex.get(chainId);
  if (!chain || chain.length <= 1) return null;

  return {
    chainId,
    position: pos,
    total: chain.length,
    questions: chain,
  };
}

// ─── Async worker fetchers (for scenario/details, post-bundle-shrink) ──────

/** URL of the Cloudflare Worker that serves full question data. */
const VAULT_API = process.env.NEXT_PUBLIC_VAULT_API
  ?? "https://staffml-vault.mlsysbook-ai-account.workers.dev";

// In-memory cache for hydrated questions during one session.
const _detailsCache = new Map<string, Question>();

/**
 * Fetch the FULL question (with `scenario` and `details.*`) from the
 * Cloudflare Worker. Returns the summary-only record on network failure
 * so the UI can still render id/title/level/zone.
 */
export async function getQuestionFullDetail(id: string): Promise<Question | undefined> {
  const cached = _detailsCache.get(id);
  if (cached?.scenario && cached.details?.realistic_solution) return cached;

  const summary = questions.find(q => q.id === id);
  if (!summary) return undefined;

  try {
    const res = await fetch(`${VAULT_API}/questions/${encodeURIComponent(id)}`, {
      signal: AbortSignal.timeout(5_000),
    });
    if (!res.ok) return summary;
    // Worker returns a DENORMALIZED row (flat fields straight from the D1
    // questions table) — common_mistake / realistic_solution / napkin_math
    // live at the top level, NOT under `details`. Re-nest to match the
    // site's Question shape before returning, otherwise callers get
    // `current.details.napkin_math` → TypeError on an undefined details.
    const full = await res.json() as {
      scenario?: string;
      common_mistake?: string;
      realistic_solution?: string;
      napkin_math?: string;
      details?: Question["details"];   // future-proof if worker changes
    };
    const workerDetails = full.details ?? {
      common_mistake: full.common_mistake ?? "",
      realistic_solution: full.realistic_solution ?? "",
      napkin_math: full.napkin_math ?? "",
    };
    const merged: Question = {
      ...summary,
      scenario: full.scenario ?? summary.scenario,
      details: {
        // Preserve MCQ options/correct_index that came in the summary.
        ...summary.details,
        ...workerDetails,
      },
    };
    _detailsCache.set(id, merged);
    return merged;
  } catch {
    // Worker unreachable → serve summary. Callers should handle missing
    // scenario/details gracefully (skeleton UI, hide sections, etc.).
    return summary;
  }
}

/**
 * Pre-warm the details cache for a batch of IDs (e.g., gauntlet session).
 * Fires fetches in parallel, resolves when all complete (or time out).
 */
export async function prefetchQuestionDetails(ids: string[]): Promise<void> {
  await Promise.all(ids.map(id => getQuestionFullDetail(id)));
}
