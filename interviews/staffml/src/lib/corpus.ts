import corpusData from '../data/corpus.json';

export interface Question {
  id: string;
  track: string;
  scope: string;
  level: string;
  title: string;
  topic: string;
  scenario: string;
  competency_area: string;
  company_archetype?: string;
  taxonomy_concept?: string;
  // v5.3 Taxonomy (6-axis classification)
  reasoning_competency?: string;   // RC-1 through RC-13
  knowledge_area?: string;          // A1 through F1 (35 areas)
  reasoning_mode?: string;          // 7 modes
  concept_tags?: string[];          // ~132 tags, multi-label
  primary_concept?: string;         // preserved taxonomy_concept
  chain_ids?: string;           // chain this question belongs to
  chain_positions?: string;     // position in chain (as string number)
  details: {
    common_mistake: string;
    realistic_solution: string;
    napkin_math?: string;
    deep_dive_title?: string;
    deep_dive_url?: string;
    options?: string[];
    correct_index?: number;
  };
}

const questions = corpusData as Question[];

export function getQuestions(): Question[] {
  return questions;
}

export function getQuestionById(id: string): Question | undefined {
  return questions.find((q) => q.id === id);
}

export function getTracks(): string[] {
  const tracks = new Set(questions.map((q) => q.track));
  return Array.from(tracks).sort();
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

export function getArchetypes(): string[] {
  const archetypes = new Set(questions.map((q) => q.company_archetype).filter((a): a is string => !!a));
  return Array.from(archetypes).sort();
}

export function getQuestionsByFilter(filters: {
  track?: string;
  level?: string;
  competency_area?: string;
  company_archetype?: string;
  // v5.3 faceted filters
  reasoning_competency?: string;
  knowledge_area?: string;
  reasoning_mode?: string;
  concept_tag?: string;
}): Question[] {
  return questions.filter((q) => {
    if (filters.track && q.track !== filters.track) return false;
    if (filters.level && q.level !== filters.level) return false;
    if (filters.competency_area && q.competency_area !== filters.competency_area) return false;
    if (filters.company_archetype && q.company_archetype !== filters.company_archetype) return false;
    if (filters.reasoning_competency && q.reasoning_competency !== filters.reasoning_competency) return false;
    if (filters.knowledge_area && q.knowledge_area !== filters.knowledge_area) return false;
    if (filters.reasoning_mode && q.reasoning_mode !== filters.reasoning_mode) return false;
    if (filters.concept_tag && !(q.concept_tags || []).includes(filters.concept_tag)) return false;
    return true;
  });
}

// v5.3 Taxonomy getters
export function getReasoningCompetencies(): string[] {
  const rcs = new Set(questions.map((q) => q.reasoning_competency).filter((v): v is string => !!v));
  return Array.from(rcs).sort();
}

export function getKnowledgeAreas(): string[] {
  const kas = new Set(questions.map((q) => q.knowledge_area).filter((v): v is string => !!v));
  return Array.from(kas).sort();
}

export function getReasoningModes(): string[] {
  const modes = new Set(questions.map((q) => q.reasoning_mode).filter((v): v is string => !!v));
  return Array.from(modes).sort();
}

export function getConceptTags(): string[] {
  const tags = new Set(questions.flatMap((q) => q.concept_tags || []));
  return Array.from(tags).sort();
}

export function getQuestionsByTopic(topicId: string, level?: string): Question[] {
  return questions.filter((q) => {
    if (q.taxonomy_concept !== topicId) return false;
    if (level && q.level !== level) return false;
    return true;
  });
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

  // Group by competency area
  const byArea: Record<string, Question[]> = {};
  pool.forEach(q => {
    const area = q.competency_area || 'general';
    if (!byArea[area]) byArea[area] = [];
    byArea[area].push(q);
  });

  const areas = Object.keys(byArea);
  const selected: Question[] = [];
  const usedIds = new Set<string>();

  // Round-robin across competency areas
  let areaIdx = 0;
  while (selected.length < count && selected.length < pool.length) {
    const area = areas[areaIdx % areas.length];
    const available = byArea[area].filter(q => !usedIds.has(q.id));
    if (available.length > 0) {
      const pick = available[Math.floor(Math.random() * available.length)];
      selected.push(pick);
      usedIds.add(pick.id);
    }
    areaIdx++;
    // Safety: if we've gone through all areas without adding, break
    if (areaIdx > areas.length * count) break;
  }

  // Shuffle the main selection
  for (let i = selected.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [selected[i], selected[j]] = [selected[j], selected[i]];
  }

  // Prepend warm-up question (easier level) at position 0
  if (warmUp && !usedIds.has(warmUp.id)) {
    selected.unshift(warmUp);
    // Trim to maintain requested count
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

// Extract the user's final answer number
// Priority: lines starting with => or "answer:" or "final:", then last number
export function extractFinalNumber(text: string): number | null {
  // Check for explicit answer markers: => 83.6 or "answer: 83.6" or "final: 83.6"
  const markerMatch = text.match(/(?:^|\n)\s*(?:=>|answer:|final:)\s*([\d,]+(?:\.\d+)?)/im);
  if (markerMatch) {
    const num = Number(markerMatch[1].replace(/,/g, ''));
    if (!isNaN(num) && isFinite(num)) return num;
  }

  // Fallback: last number in the text
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
  const chainId = q.chain_ids;
  const chainPos = q.chain_positions;
  if (chainId && chainPos !== undefined) {
    if (!_chainIndex.has(chainId)) _chainIndex.set(chainId, []);
    _chainIndex.get(chainId)!.push({
      id: q.id,
      title: q.title,
      level: q.level,
      position: parseInt(chainPos, 10),
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
  if (!q) return null;
  const chainId = q.chain_ids;
  const chainPos = q.chain_positions;
  if (!chainId || chainPos === undefined) return null;
  const chain = _chainIndex.get(chainId);
  if (!chain || chain.length <= 1) return null; // skip single-question "chains"
  return {
    chainId,
    position: parseInt(chainPos, 10),
    total: chain.length,
    questions: chain,
  };
}
