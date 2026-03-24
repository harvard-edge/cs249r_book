import taxonomyData from "../data/taxonomy.json";
import corpusData from "../data/corpus.json";
import chapterUrls from "../data/chapter-urls.json";

// ─── Types ─────────────────────────────────────────────────────

export interface QuestionSummary {
  id: string;
  title: string;
  level: string;
  track: string;
  scenario: string;
}

export interface Topic {
  id: string;
  name: string;
  description: string;
  questionCount: number;
  levels: Record<string, number>;
  tracks: string[];
  /** Questions grouped by level for drill-down */
  questionsByLevel: Record<string, QuestionSummary[]>;
  /** Chapter name (human-readable) */
  chapterName?: string;
  /** URL to textbook chapter */
  chapterUrl?: string;
}

export interface CompetencyArea {
  id: string;
  name: string;
  questionCount: number;
  topicCount: number;
  topics: Topic[];
}

interface Concept {
  id: string;
  name: string;
  description: string;
  tracks: string[];
  source_chapters: string[];
  question_count: number;
  level_distribution: Record<string, number>;
  textbook_url?: string;
}

interface RawQuestion {
  id: string;
  track: string;
  level: string;
  title: string;
  scenario: string;
  competency_area: string;
  taxonomy_concept?: string;
  details: {
    [key: string]: unknown;
  };
}

// ─── Chapter URL + name helpers ────────────────────────────────

const chapterUrlMap = chapterUrls as Record<string, string>;

function formatChapterName(ch: string): string {
  return ch
    .replace("vol1_", "")
    .replace("vol2_", "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ─── Build index ───────────────────────────────────────────────

const concepts = (taxonomyData as { concepts: Concept[] }).concepts;
const questions = corpusData as RawQuestion[];
const conceptMap = new Map(concepts.map((c) => [c.id, c]));

// Group questions by competency_area → taxonomy_concept
const areaTopicQs: Record<string, Record<string, RawQuestion[]>> = {};
for (const q of questions) {
  const area = q.competency_area;
  const tc = q.taxonomy_concept || "_unmapped";
  if (!areaTopicQs[area]) areaTopicQs[area] = {};
  if (!areaTopicQs[area][tc]) areaTopicQs[area][tc] = [];
  areaTopicQs[area][tc].push(q);
}

// Build CompetencyArea[] with sorted topics
const _areas: CompetencyArea[] = Object.entries(areaTopicQs)
  .map(([areaId, topicMap]) => {
    const topics: Topic[] = Object.entries(topicMap)
      .filter(([tc]) => tc !== "_unmapped")
      .map(([tc, qs]) => {
        const concept = conceptMap.get(tc);
        const levels: Record<string, number> = {};
        const trackSet = new Set<string>();
        const questionsByLevel: Record<string, QuestionSummary[]> = {};

        for (const q of qs) {
          levels[q.level] = (levels[q.level] || 0) + 1;
          trackSet.add(q.track);
          if (!questionsByLevel[q.level]) questionsByLevel[q.level] = [];
          questionsByLevel[q.level].push({
            id: q.id,
            title: q.title,
            level: q.level,
            track: q.track,
            scenario: q.scenario,
          });
        }

        // Get chapter info from taxonomy concept (not from question deep_dive)
        const sourceChapter = concept?.source_chapters?.[0];
        const chapterUrl = sourceChapter
          ? chapterUrlMap[sourceChapter]
          : undefined;
        const chapterName = sourceChapter
          ? formatChapterName(sourceChapter)
          : undefined;

        return {
          id: tc,
          name:
            concept?.name ||
            tc
              .replace(/-/g, " ")
              .replace(/\b\w/g, (c) => c.toUpperCase()),
          description: concept?.description || "",
          questionCount: qs.length,
          levels,
          tracks: Array.from(trackSet).sort(),
          questionsByLevel,
          chapterName,
          chapterUrl,
        };
      })
      .sort((a, b) => b.questionCount - a.questionCount);

    return {
      id: areaId,
      name:
        areaId.charAt(0).toUpperCase() +
        areaId.slice(1).replace(/-/g, " "),
      questionCount: topics.reduce((s, t) => s + t.questionCount, 0),
      topicCount: topics.length,
      topics,
    };
  })
  .sort((a, b) => b.questionCount - a.questionCount);

// ─── Public API ────────────────────────────────────────────────

export function getAreas(): CompetencyArea[] {
  return _areas;
}

export function getAreaById(id: string): CompetencyArea | undefined {
  return _areas.find((a) => a.id === id);
}

export function getTopicById(id: string): Topic | undefined {
  for (const area of _areas) {
    const t = area.topics.find((t) => t.id === id);
    if (t) return t;
  }
  return undefined;
}

export function getAreaForTopic(topicId: string): CompetencyArea | undefined {
  return _areas.find((a) => a.topics.some((t) => t.id === topicId));
}

export function searchTopics(query: string): Topic[] {
  const q = query.toLowerCase().trim();
  if (!q) return _areas.flatMap((a) => a.topics);
  return _areas.flatMap((a) =>
    a.topics.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.id.includes(q) ||
        t.description.toLowerCase().includes(q)
    )
  );
}

export function getVaultStats() {
  return {
    totalQuestions: questions.length,
    totalTopics: _areas.reduce((s, a) => s + a.topicCount, 0),
    totalAreas: _areas.length,
    tracks: ["cloud", "edge", "mobile", "tinyml"],
  };
}

// ─── Area colors ───────────────────────────────────────────────

const AREA_COLORS: Record<string, string> = {
  memory: "#3b82f6",
  compute: "#f59e0b",
  deployment: "#22c55e",
  architecture: "#a855f7",
  latency: "#ef4444",
  "cross-cutting": "#06b6d4",
  data: "#14b8a6",
  networking: "#f97316",
  power: "#eab308",
  optimization: "#8b5cf6",
  precision: "#ec4899",
  reliability: "#6366f1",
  parallelism: "#10b981",
};

export function getAreaColor(areaId: string): string {
  return AREA_COLORS[areaId] || "#6b7280";
}
