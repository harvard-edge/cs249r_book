import corpusData from '../data/corpus-summary.json';
import { QUESTION_COUNT } from './stats';

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
  /**
   * Optional diagram attached to the question. The SVG file lives at
   * `/question-visuals/<track>/<visual.path>` — copied from the vault
   * by the build step. Rendered between the scenario and the Your
   * task callout so the reading order is context → diagram → ask.
   */
  visual?: {
    kind: "svg";               // closed enum as of v0.1.2 (mermaid retired)
    path: string;              // bare filename, resolves under /question-visuals/<track>/
    alt: string;                // a11y-required description, ≥10 chars
    caption: string;            // required as of v0.1.2, ≥5 chars
  };
  topic: string;            // one of 87 curated topic IDs
  zone: string;             // one of 11 ikigai zones
  competency_area: string;  // one of 13 canonical areas
  bloom_level?: string;     // remember | understand | apply | analyze | evaluate | create
  phase?: string;           // training | inference | both
  status?: string;          // draft | published | flagged | archived | deleted
  chain_ids?: string[];
  chain_positions?: Record<string, number>;
  /**
   * Per-membership tier label. "primary" chains came out of the strict
   * Bloom-progression sweep and are surfaced by default; "secondary"
   * chains came out of the lenient second-pass coverage build and are
   * deprioritized in default UI surfaces. Mirrors chain_positions in
   * shape — one entry per chain_id this question is in. See
   * CHAIN_ROADMAP.md Phase 1/2 for the mechanism.
   */
  chain_tiers?: Record<string, "primary" | "secondary">;

  /**
   * "Go deeper" pointers back into the textbook, derived at build time from
   * the question's `topic` via `schema/topic_chapter_map.yaml` (one mapping
   * per topic, inherited by every question that carries it). This is a
   * recommended-reading pointer tied to the topic, NOT an answer key — render
   * it AFTER the attempt as a "Learn more" affordance. Ships in the summary
   * bundle so the card renders synchronously (no worker fetch). Absent for
   * the handful of topics with no chapter mapped.
   */
  book_refs?: BookRef[];

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

/**
 * Build-derived textbook chapter pointer (from topic_chapter_map.yaml). The
 * `primary` ref carries an authored `why` line connecting the topic to the
 * chapter; `also_see` refs are secondary reading. `url` is a chapter-level
 * mlsysbook.ai link (section anchors are deferred — they regenerate on
 * rebuild). See `BookRefResolver` in vault-cli and cs249r_book#1822.
 */
export interface BookRef {
  vol: number;
  chapter: string;
  title: string;
  url: string;
  role: "primary" | "also_see";
  why?: string;
}

const questions = corpusData as unknown as Question[];

export function getQuestions(): Question[] {
  return questions;
}

/**
 * Marketing-friendly question count string, e.g. "8,000+". Derived from the
 * authoritative manifest count (QUESTION_COUNT in stats.ts — the single source
 * of truth for user-visible counts), NOT from the runtime bundle length, which
 * can be a partial/filtered corpus in local dev. Rounds DOWN to a sensible
 * granularity and appends `+` so the headline never goes stale or overstates.
 *
 * The granularity adapts to magnitude so a small or partial corpus never
 * renders the old "0+" bug (Math.floor(900/1000)*1000 = 0).
 */
export function roundedFloorForDisplay(n: number): number {
  if (n >= 1000) return Math.floor(n / 1000) * 1000;
  if (n >= 100) return Math.floor(n / 100) * 100;
  if (n >= 10) return Math.floor(n / 10) * 10;
  return Math.max(0, Math.floor(n)); // tiny corpus: show the exact floor
}
export const QUESTION_COUNT_DISPLAY = `${roundedFloorForDisplay(QUESTION_COUNT).toLocaleString("en-US")}+`;

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
  /** When true, restrict results to questions that are part of a chain. */
  chainsOnly?: boolean;
  /** When true, restrict results to questions with an attached visual. */
  visualOnly?: boolean;
}): Question[] {
  return questions.filter((q) => {
    if (filters.track && q.track !== filters.track) return false;
    if (filters.level && q.level !== filters.level) return false;
    if (filters.competency_area && q.competency_area !== filters.competency_area) return false;
    if (filters.topic && q.topic !== filters.topic) return false;
    if (filters.zone && q.zone !== filters.zone) return false;
    if (filters.chainsOnly && (!q.chain_ids || q.chain_ids.length === 0)) return false;
    if (filters.visualOnly && !q.visual) return false;
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
  // Relative error against the magnitude of the model answer. The old code
  // divided by Math.max(modelAnswer, 1), which collapsed the denominator to 1
  // for any sub-unit answer — so a model answer of 0.5 vs. a user answer of 1.0
  // (a 2× miss) scored as ratio 0.5 ("ballpark") instead of 1.0. ML napkin
  // answers are frequently < 1 (fractions of a GB, sub-second latencies), so
  // this systematically under-penalized small magnitudes.
  const denom = Math.abs(modelAnswer);
  const ratio =
    denom > 0
      ? Math.abs(userAnswer - modelAnswer) / denom
      : userAnswer === 0
        ? 0
        : Infinity; // model answer is exactly 0; any nonzero guess is "way off"

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
  // ratio can be Infinity when the model answer is 0; guard the label.
  const offLabel = Number.isFinite(ratio) ? `Off by ${ratio.toFixed(0)}×` : 'Way off';
  return { grade: 'way_off', ratio, tolerance, label: offLabel, maxSelfScore: 1 };
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

// Extract the user's final answer number.
//
// Every numeric token must START with a digit (`\d[\d,]*...`). The old pattern
// `[\d,]+` matched a lone comma, so `Number("".replace(/,/g,""))` → 0 was
// accepted as a valid answer — a stray comma in the prose silently became the
// "final number". Requiring a leading digit rejects that. We deliberately do
// NOT accept a leading minus: a range like "10-20 ms" must read as 20, not -20.
const NUMERIC_TOKEN = /\d[\d,]*(?:\.\d+)?/g;

export function extractFinalNumber(text: string): number | null {
  // Explicit marker form ("=> 42", "answer: 1,024", "final: 3.5") wins.
  const markerMatch = text.match(/(?:^|\n)\s*(?:=>|answer:|final:)\s*(\d[\d,]*(?:\.\d+)?)/im);
  if (markerMatch) {
    const num = Number(markerMatch[1].replace(/,/g, ''));
    if (Number.isFinite(num)) return num;
  }

  // Otherwise take the LAST numeric token in the text.
  const numbers = text.match(NUMERIC_TOKEN)?.map(s => Number(s.replace(/,/g, ''))) ?? [];
  const valid = numbers.filter(n => Number.isFinite(n));
  return valid.length > 0 ? valid[valid.length - 1] : null;
}

// ─── Unit-aware numeric grading (js-quantities / Pint-equivalent) ───────────
//
// `extractFinalNumber` above is intentionally unit-blind and pins the
// last-token convention (tested in napkin-grading.test.ts). The functions
// below add unit-awareness as a NEW layer on top of it, using the
// `js-quantities` library — the JS analogue of Python's Pint that this
// project already uses server-side in mlsysim. The library, not a hand-rolled
// table, validates whether a token is a real unit, handles SI prefixes
// (mW ≠ MW), dimensional analysis, and conversion.
//
// SSR-safe import: js-quantities is a pure-JS CommonJS module with no DOM or
// Node-only dependencies, so a top-level ESM import is safe under Next.js
// server rendering. We type it loosely (the `@types/js-quantities` shapes
// vary across versions) and only touch the small surface we need.
import Qty from "js-quantities";

/** Instance type returned by a Qty(...) call (the @types decl exposes the
 *  callable as Qty.Type, so we derive the instance via ReturnType). */
type QtyInstance = ReturnType<typeof Qty>;

/** A parsed quantity plus the exact display string the user/model wrote. */
export interface ParsedQuantity {
  qty: QtyInstance;
  /** Original "number + unit" substring, e.g. "42 ms", for UI display. */
  display: string;
  /** The unit token as written, e.g. "ms", "GB/s". */
  unit: string;
}

// A candidate is a number (optionally with thousands commas / decimals /
// scientific notation) immediately followed by a run of unit-ish characters.
// The unit run accepts letters, the micro signs (µ U+00B5 / μ U+03BC), and the
// `/` and `·`/`*` separators that appear in compound units like GB/s, FLOP/s.
// We do NOT require the library to recognize the token here — we hand every
// candidate to Qty() and let the library accept or reject it. That rejection
// is exactly what fixes the last-token bug: "40 layers" / "40 epochs" fail to
// parse and are skipped.
const QTY_CANDIDATE =
  /(\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-zµμ]+(?:\s*\/\s*[A-Za-z]+)?)/g;

// Marker line that names the final answer, mirroring extractFinalNumber's
// convention ("=> 42 ms", "answer: 1,024 GB", "final: 3.5 s").
const QTY_MARKER = /(?:^|\n)\s*(?:=>|answer:|final:)\s*(.+)$/im;

function tryParseQty(numRaw: string, unitRaw: string): ParsedQuantity | null {
  const num = numRaw.replace(/,/g, "");
  const unit = unitRaw.replace(/\s+/g, "");
  try {
    const qty = Qty(`${num} ${unit}`);
    // Guard against degenerate parses (NaN scalar, etc.).
    if (!Number.isFinite(qty.scalar)) return null;
    return { qty, display: `${numRaw.trim()} ${unit}`, unit };
  } catch {
    return null;
  }
}

/**
 * Extract the user's final ANSWER as a unit-bearing quantity.
 *
 * Strategy mirrors `extractFinalNumber`:
 *   1. If a marker line ("=>", "answer:", "final:") exists, parse the LAST
 *      parseable quantity on that line first.
 *   2. Otherwise scan the whole text and return the LAST candidate that the
 *      units library successfully parses.
 *
 * Returns null when no candidate parses to a real unit. Non-unit trailing
 * tokens ("40 layers", "40 epochs") are skipped because Qty() rejects them —
 * so "1.31 MB across 40 layers" yields 1.31 MB, not 40.
 */
export function extractFinalQuantity(text: string): ParsedQuantity | null {
  const scan = (segment: string): ParsedQuantity | null => {
    let last: ParsedQuantity | null = null;
    // Array.from avoids requiring downlevelIteration for the matchAll iterator.
    for (const m of Array.from(segment.matchAll(QTY_CANDIDATE))) {
      const parsed = tryParseQty(m[1], m[2]);
      if (parsed) last = parsed;
    }
    return last;
  };

  const markerMatch = text.match(QTY_MARKER);
  if (markerMatch) {
    const fromMarker = scan(markerMatch[1]);
    if (fromMarker) return fromMarker;
  }
  return scan(text);
}

/** Result of unit-aware grading: a NapkinResult plus display metadata. */
export interface NapkinGradeResult extends NapkinResult {
  /** Number used for grading (base-unit magnitude when units were involved). */
  userNum: number;
  modelNum: number;
  /** Original display string the user wrote, e.g. "42 ms" (null if bare). */
  userDisplay: string | null;
  modelDisplay: string | null;
  /** True when both sides parsed as compatible quantities and were converted. */
  unitAware: boolean;
}

/**
 * Grade a user's napkin-math answer against the model answer, unit-aware.
 *
 * - If BOTH sides parse to quantities:
 *     - compatible dimensions → convert to common base, grade via checkNapkinMath.
 *     - incompatible dimensions → way_off, labelled "Wrong quantity (a vs b)".
 * - Otherwise → fall back to the legacy bare-number path (extractFinalNumber +
 *   checkNapkinMath), preserving all existing behavior.
 *
 * checkNapkinMath's relative-error logic is reused verbatim — it is never
 * reimplemented here.
 */
export function gradeNapkinAnswer(
  userText: string,
  modelText: string,
  track: string,
): NapkinGradeResult | null {
  const userQty = extractFinalQuantity(userText);
  const modelQty = extractFinalQuantity(modelText);

  if (userQty && modelQty) {
    if (userQty.qty.isCompatible(modelQty.qty)) {
      // Convert both to base units so the magnitudes are directly comparable
      // (42 ms and 42000 µs both become 0.042 s).
      const userBase = userQty.qty.toBase().scalar;
      const modelBase = modelQty.qty.toBase().scalar;
      const result = checkNapkinMath(userBase, modelBase, track);
      return {
        ...result,
        userNum: userBase,
        modelNum: modelBase,
        userDisplay: userQty.display,
        modelDisplay: modelQty.display,
        unitAware: true,
      };
    }
    // Incompatible dimensions: a bandwidth answer to a capacity question, etc.
    return {
      grade: "way_off",
      ratio: Infinity,
      tolerance: 0,
      label: `Wrong quantity (${userQty.unit} vs ${modelQty.unit})`,
      maxSelfScore: 1,
      userNum: userQty.qty.scalar,
      modelNum: modelQty.qty.scalar,
      userDisplay: userQty.display,
      modelDisplay: modelQty.display,
      unitAware: true,
    };
  }

  // Legacy bare-number fallback (no parseable units on one or both sides).
  const userNum = extractFinalNumber(userText);
  const modelNum = extractFinalNumber(modelText);
  if (userNum === null || modelNum === null) return null;
  const result = checkNapkinMath(userNum, modelNum, track);
  return {
    ...result,
    userNum,
    modelNum,
    userDisplay: userQty?.display ?? null,
    modelDisplay: modelQty?.display ?? null,
    unitAware: false,
  };
}

// ─── Chain helpers ──────────────────────────────────────────
// Chains are deepening question sequences on a topic (L1 → L6+)

export type ChainTier = "primary" | "secondary";

export interface ChainInfo {
  chainId: string;
  position: number;       // 0-indexed position of current question
  total: number;          // total questions in chain
  /**
   * "primary" — surface by default (clean Bloom progression).
   * "secondary" — deprioritized in default surfaces; reachable via the
   * "more paths" UI or explicit ?chain= URL routing.
   */
  tier: ChainTier;
  questions: { id: string; title: string; level: string; position: number }[];
}

// Build chain index once. Tier is a chain-level attribute (every member
// of a chain shares the same tier), so we keep it in a sibling map rather
// than embedding it in each question record.
const _chainIndex = new Map<string, { id: string; title: string; level: string; position: number }[]>();
const _chainTier = new Map<string, ChainTier>();
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
    if (!_chainTier.has(chainId)) {
      const t = q.chain_tiers?.[chainId];
      _chainTier.set(chainId, t === "secondary" ? "secondary" : "primary");
    }
  }
}
// Sort each chain by position
_chainIndex.forEach((qs) => {
  qs.sort((a, b) => a.position - b.position);
});

function _tierOf(chainId: string): ChainTier {
  return _chainTier.get(chainId) ?? "primary";
}

/** Get chain info for a question, or null if not in a chain.
 *
 * When a question belongs to multiple chains (multi-membership pattern —
 * a foundational L1/L2 question anchoring two distinct progressions),
 * caller can disambiguate by passing `preferredChainId`. If omitted or
 * not a match, falls back to the first chain. Tier-aware callers should
 * prefer ``getPrimaryChainForQuestion`` for the default surface.
 */
export function getChainForQuestion(
  questionId: string,
  preferredChainId?: string,
): ChainInfo | null {
  const q = questions.find(x => x.id === questionId);
  if (!q || !q.chain_ids || !q.chain_positions) return null;

  // Pick preferred if it's actually one of this question's chains; else first.
  const chainId =
    (preferredChainId && q.chain_ids.includes(preferredChainId))
      ? preferredChainId
      : q.chain_ids[0];
  if (!chainId) return null;
  const pos = q.chain_positions[chainId];
  if (pos === undefined) return null;

  const chain = _chainIndex.get(chainId);
  if (!chain || chain.length <= 1) return null;

  return {
    chainId,
    position: pos,
    total: chain.length,
    tier: _tierOf(chainId),
    questions: chain,
  };
}

/** Return ALL chains a question is in (size ≥ 2 only). Empty array if none.
 *
 * Order: primary chains first (in their original chain_ids order), then
 * secondary chains. Callers that want primary-only should filter the
 * result on ``c.tier === "primary"``.
 */
export function getAllChainsForQuestion(questionId: string): ChainInfo[] {
  const q = questions.find(x => x.id === questionId);
  if (!q || !q.chain_ids || !q.chain_positions) return [];
  const out: ChainInfo[] = [];
  for (const chainId of q.chain_ids) {
    const pos = q.chain_positions[chainId];
    if (pos === undefined) continue;
    const chain = _chainIndex.get(chainId);
    if (!chain || chain.length <= 1) continue;
    out.push({
      chainId,
      position: pos,
      total: chain.length,
      tier: _tierOf(chainId),
      questions: chain,
    });
  }
  // Stable: primary first, then secondary; preserves intra-tier order.
  out.sort((a, b) => {
    if (a.tier === b.tier) return 0;
    return a.tier === "primary" ? -1 : 1;
  });
  return out;
}

/** Return the question's primary chain if it has one; otherwise the
 * first secondary; otherwise null. The default-surface helper for UI
 * components that want to render one chain badge / one strip per question.
 */
export function getPrimaryChainForQuestion(questionId: string): ChainInfo | null {
  const all = getAllChainsForQuestion(questionId);
  if (all.length === 0) return null;
  return all.find(c => c.tier === "primary") ?? all[0];
}

// ─── Chain selection for interview conductor ────────────────────────────────

export interface ChainMember {
  id: string;
  title: string;
  level: string;
  zone: string;
  position: number;
}

export interface ChainSummary {
  chainId: string;
  tier: ChainTier;
  topic: string;
  area: string;
  members: ChainMember[];
}

const _chainSummaries: ChainSummary[] = [];
_chainIndex.forEach((members, chainId) => {
  if (members.length <= 1) return;
  const firstQ = questions.find(q => q.id === members[0].id);
  if (!firstQ) return;
  _chainSummaries.push({
    chainId,
    tier: _tierOf(chainId),
    topic: firstQ.topic,
    area: firstQ.competency_area,
    members: members.map((m: { id: string; title: string; level: string; position: number }) => {
      const q = questions.find(x => x.id === m.id);
      return { id: m.id, title: m.title, level: m.level, zone: q?.zone ?? "recall", position: m.position };
    }),
  });
});

const LEVEL_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6+"];

export function getChainsByArea(area: string, track?: string): ChainSummary[] {
  return _chainSummaries.filter(c => {
    if (c.area !== area) return false;
    if (track) {
      const q = questions.find(x => x.id === c.members[0].id);
      if (q && q.track !== track) return false;
    }
    return true;
  });
}

export function getChainsByTopic(topic: string, track?: string): ChainSummary[] {
  return _chainSummaries.filter(c => {
    if (c.topic !== topic) return false;
    if (track) {
      const q = questions.find(x => x.id === c.members[0].id);
      if (q && q.track !== track) return false;
    }
    return true;
  });
}

export function getChainsForInterview(
  track: string,
  levels: string[],
  areas?: string[],
): ChainSummary[] {
  const levelSet = new Set(levels);
  return _chainSummaries
    .filter(c => {
      const q = questions.find(x => x.id === c.members[0].id);
      if (!q || q.track !== track) return false;
      if (areas && !areas.includes(c.area)) return false;
      return c.members.some(m => levelSet.has(m.level));
    })
    .sort((a, b) => {
      if (a.tier !== b.tier) return a.tier === "primary" ? -1 : 1;
      return b.members.length - a.members.length;
    });
}

export function getChainEntryPoint(
  chain: ChainSummary,
  targetLevel: string,
): ChainMember | null {
  const targetIdx = LEVEL_ORDER.indexOf(targetLevel);
  if (targetIdx < 0) return chain.members[0] ?? null;
  let best: ChainMember | null = null;
  let bestDist = Infinity;
  for (const m of chain.members) {
    const mIdx = LEVEL_ORDER.indexOf(m.level);
    const dist = targetIdx - mIdx;
    if (dist >= 0 && dist < bestDist) {
      best = m;
      bestDist = dist;
    }
  }
  return best ?? chain.members[0] ?? null;
}

// ─── Async worker fetchers (for scenario/details, post-bundle-shrink) ──────

import { getVaultApiBase, getVaultMode } from "./vault-config";
import { vaultFetchJson } from "./vault-fetch";

// In-memory cache for hydrated questions during one session.
const _detailsCache = new Map<string, Question>();
// IDs we have successfully hydrated. Tracked explicitly rather than inferred
// from `details.realistic_solution` being truthy — a recall/MCQ question can
// have a legitimately empty realistic_solution, and inferring hydration from
// it caused those questions to re-fetch from the worker on every access.
const _hydratedIds = new Set<string>();
let _staticDetailsCache: Map<string, Question> | null = null;

async function getStaticFullDetail(id: string, summary: Question): Promise<Question | undefined> {
  if (!_staticDetailsCache) {
    // Fetch corpus.json from /data/corpus.json (served from public/). This
    // file is written by `vault build --local-json` and exists only in local
    // dev. Production deploys neither emit nor bundle it; the worker fetch
    // path handles those. If the file is missing at runtime the fetch fails
    // and the caller surfaces an error to the UI.
    const res = await fetch("/data/corpus.json");
    if (!res.ok) {
      throw new Error(
        `Static corpus.json not available at /data/corpus.json (status ${res.status}). ` +
        "Run 'vault build --local-json' from the repo root to regenerate it.",
      );
    }
    const data = (await res.json()) as Question[];
    _staticDetailsCache = new Map(data.map((q) => [q.id, q]));
  }
  const full = _staticDetailsCache.get(id);
  if (!full) return undefined;
  const merged: Question = {
    ...summary,
    ...full,
    details: {
      ...summary.details,
      ...full.details,
    },
  };
  _detailsCache.set(id, merged);
  _hydratedIds.add(id);
  return merged;
}

/**
 * Fetch the FULL question (with `scenario` and `details.*`) from the
 * Cloudflare Worker. Returns the cache-merged Question on success.
 * Throws on Worker error — useFullQuestion catches and renders the
 * "details unavailable" state. (Static fallback is opt-in via
 * NEXT_PUBLIC_VAULT_FALLBACK=static and is handled earlier.)
 */
export async function getQuestionFullDetail(id: string): Promise<Question | undefined> {
  // Hydration is tracked explicitly (see _hydratedIds): a question whose
  // realistic_solution is legitimately empty is still fully hydrated and must
  // not re-fetch on every access.
  if (_hydratedIds.has(id)) return _detailsCache.get(id);

  const summary = questions.find(q => q.id === id);
  if (!summary) return undefined;

  if (getVaultMode() === "static") {
    return getStaticFullDetail(id, summary);
  }

  const apiBase = getVaultApiBase();
  if (!apiBase) throw new Error("vault worker base URL unavailable");

  // Worker returns a DENORMALIZED row (flat fields straight from the D1
  // questions table) — common_mistake / realistic_solution / napkin_math
  // live at the top level, NOT under `details`. Re-nest to match the
  // site's Question shape before returning, otherwise callers get
  // `current.details.napkin_math` → TypeError on an undefined details.
  const full = await vaultFetchJson<{
    scenario?: string;
    common_mistake?: string;
    realistic_solution?: string;
    napkin_math?: string;
    details?: Question["details"];   // future-proof if worker changes
  }>(`${apiBase}/questions/${encodeURIComponent(id)}`);
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
  _hydratedIds.add(id);
  return merged;
}

/**
 * Pre-warm the details cache for a batch of IDs (e.g., gauntlet session).
 * Fires fetches in parallel; individual failures don't reject the batch.
 */
export async function prefetchQuestionDetails(ids: string[]): Promise<void> {
  await Promise.allSettled(ids.map(id => getQuestionFullDetail(id)));
}
