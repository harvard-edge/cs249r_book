import taxonomyData from "../data/taxonomy.json";
import corpusData from "../data/corpus.json";
import chapterUrls from "../data/chapter-urls.json";
import {
  HardDrive, Cpu, Rocket, Layers, Timer, Shuffle,
  Database, Network, Zap, Gauge, Binary,
  Shield, GitBranch,
  type LucideIcon,
} from "lucide-react";

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
  questionsByLevel: Record<string, QuestionSummary[]>;
  chapterName?: string;
  chapterUrl?: string;
}

export interface CompetencyArea {
  id: string;
  name: string;
  questionCount: number;
  topicCount: number;
  topics: Topic[];
  /** Aggregate level distribution across all topics */
  levels: Record<string, number>;
}

export interface AreaStyle {
  primary: string;
  bg: string;
  border: string;
  icon: LucideIcon;
}

// ─── Internals ─────────────────────────────────────────────────

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
  details: { [key: string]: unknown };
}

const chapterUrlMap = chapterUrls as Record<string, string>;

function formatChapterName(ch: string): string {
  return ch.replace("vol1_", "").replace("vol2_", "").replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ─── Build index ───────────────────────────────────────────────

const concepts = (taxonomyData as { concepts: Concept[] }).concepts;
const questions = corpusData as RawQuestion[];
const conceptMap = new Map(concepts.map((c) => [c.id, c]));

const areaTopicQs: Record<string, Record<string, RawQuestion[]>> = {};
for (const q of questions) {
  const area = q.competency_area;
  const tc = q.taxonomy_concept || "_unmapped";
  if (!areaTopicQs[area]) areaTopicQs[area] = {};
  if (!areaTopicQs[area][tc]) areaTopicQs[area][tc] = [];
  areaTopicQs[area][tc].push(q);
}

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
            id: q.id, title: q.title, level: q.level,
            track: q.track, scenario: q.scenario,
          });
        }

        const sourceChapter = concept?.source_chapters?.[0];
        return {
          id: tc,
          name: concept?.name || tc.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
          description: concept?.description || "",
          questionCount: qs.length,
          levels, tracks: Array.from(trackSet).sort(), questionsByLevel,
          chapterName: sourceChapter ? formatChapterName(sourceChapter) : undefined,
          chapterUrl: sourceChapter ? chapterUrlMap[sourceChapter] : undefined,
        };
      })
      .sort((a, b) => b.questionCount - a.questionCount);

      // Aggregate level distribution
      const areaLevels: Record<string, number> = {};
      for (const t of topics) {
        for (const [lv, cnt] of Object.entries(t.levels)) {
          areaLevels[lv] = (areaLevels[lv] || 0) + Number(cnt);
        }
      }

      return {
        id: areaId,
        name: areaId.charAt(0).toUpperCase() + areaId.slice(1).replace(/-/g, " "),
        questionCount: topics.reduce((s, t) => s + t.questionCount, 0),
        topicCount: topics.length,
        topics,
        levels: areaLevels,
      };
  })
  .sort((a, b) => a.name.localeCompare(b.name));

// ─── Area styles ───────────────────────────────────────────────

const AREA_STYLES: Record<string, AreaStyle> = {
  memory:         { primary: "#60a5fa", bg: "#60a5fa12", border: "#60a5fa30", icon: HardDrive },
  compute:        { primary: "#fbbf24", bg: "#fbbf2412", border: "#fbbf2430", icon: Cpu },
  deployment:     { primary: "#4ade80", bg: "#4ade8012", border: "#4ade8030", icon: Rocket },
  architecture:   { primary: "#c084fc", bg: "#c084fc12", border: "#c084fc30", icon: Layers },
  latency:        { primary: "#f87171", bg: "#f8717112", border: "#f8717130", icon: Timer },
  "cross-cutting":{ primary: "#22d3ee", bg: "#22d3ee12", border: "#22d3ee30", icon: Shuffle },
  data:           { primary: "#2dd4bf", bg: "#2dd4bf12", border: "#2dd4bf30", icon: Database },
  networking:     { primary: "#fb923c", bg: "#fb923c12", border: "#fb923c30", icon: Network },
  power:          { primary: "#e8b83d", bg: "#e8b83d12", border: "#e8b83d30", icon: Zap },
  optimization:   { primary: "#a78bfa", bg: "#a78bfa12", border: "#a78bfa30", icon: Gauge },
  precision:      { primary: "#f472b6", bg: "#f472b612", border: "#f472b630", icon: Binary },
  reliability:    { primary: "#818cf8", bg: "#818cf812", border: "#818cf830", icon: Shield },
  parallelism:    { primary: "#34d399", bg: "#34d39912", border: "#34d39930", icon: GitBranch },
};

// ─── Public API ────────────────────────────────────────────────

export function getAreas(): CompetencyArea[] { return _areas; }
export function getAreaById(id: string) { return _areas.find((a) => a.id === id); }
export function getTopicById(id: string): Topic | undefined {
  for (const area of _areas) { const t = area.topics.find((t) => t.id === id); if (t) return t; }
}
export function getAreaForTopic(topicId: string) {
  return _areas.find((a) => a.topics.some((t) => t.id === topicId));
}

export function searchTopics(query: string): Topic[] {
  const q = query.toLowerCase().trim();
  if (!q) return _areas.flatMap((a) => a.topics);
  return _areas.flatMap((a) => a.topics.filter((t) =>
    t.name.toLowerCase().includes(q) || t.id.includes(q) || t.description.toLowerCase().includes(q)
  ));
}

export function getVaultStats() {
  return {
    totalQuestions: questions.length,
    totalTopics: _areas.reduce((s, a) => s + a.topicCount, 0),
    totalAreas: _areas.length,
  };
}

export function getAreaStyle(areaId: string): AreaStyle {
  return AREA_STYLES[areaId] || { primary: "#6b7280", bg: "#6b728012", border: "#6b728030", icon: Layers };
}
