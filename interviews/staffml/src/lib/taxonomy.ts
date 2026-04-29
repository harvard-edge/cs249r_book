import taxonomyData from "../data/taxonomy.json";
import corpusData from "../data/corpus-summary.json";
import zonesData from "../data/zones.json";
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
  zone: string;
}

export interface Topic {
  id: string;
  name: string;
  description: string;
  area: string;
  prerequisites: string[];
  questionCount: number;
  levels: Record<string, number>;
  tracks: string[];
  zones: Record<string, number>;
  questionsByLevel: Record<string, QuestionSummary[]>;
}

export interface CompetencyArea {
  id: string;
  name: string;
  questionCount: number;
  topicCount: number;
  topics: Topic[];
  levels: Record<string, number>;
  zones: Record<string, number>;
}

export interface ZoneDefinition {
  id: string;
  skills: string[];
  description: string;
  levels: string[];
}

export interface AreaStyle {
  primary: string;
  bg: string;
  border: string;
  icon: LucideIcon;
}

// ─── Internals ─────────────────────────────────────────────────

interface RawConcept {
  id: string;
  name: string;
  description: string;
  area: string;
  prerequisites: string[];
  tracks: string[];
  question_count: number;
  level_distribution?: Record<string, number>;
}

interface RawQuestion {
  id: string;
  track: string;
  level: string;
  title: string;
  scenario: string;
  topic: string;
  zone: string;
  competency_area: string;
  details: { [key: string]: unknown };
}

// ─── Build index ───────────────────────────────────────────────

const concepts = (taxonomyData as unknown as { concepts: RawConcept[] }).concepts;
const questions = corpusData as RawQuestion[];
const conceptMap = new Map(concepts.map((c) => [c.id, c]));

// Group questions by area → topic
const areaTopicQs: Record<string, Record<string, RawQuestion[]>> = {};
for (const q of questions) {
  const area = q.competency_area;
  const topic = q.topic;
  if (!areaTopicQs[area]) areaTopicQs[area] = {};
  if (!areaTopicQs[area][topic]) areaTopicQs[area][topic] = [];
  areaTopicQs[area][topic].push(q);
}

const _areas: CompetencyArea[] = Object.entries(areaTopicQs)
  .map(([areaId, topicMap]) => {
    const topics: Topic[] = Object.entries(topicMap)
      .map(([topicId, qs]) => {
        const concept = conceptMap.get(topicId);
        const levels: Record<string, number> = {};
        const zones: Record<string, number> = {};
        const trackSet = new Set<string>();
        const questionsByLevel: Record<string, QuestionSummary[]> = {};

        for (const q of qs) {
          levels[q.level] = (levels[q.level] || 0) + 1;
          zones[q.zone] = (zones[q.zone] || 0) + 1;
          trackSet.add(q.track);
          if (!questionsByLevel[q.level]) questionsByLevel[q.level] = [];
          questionsByLevel[q.level].push({
            id: q.id, title: q.title, level: q.level,
            track: q.track, scenario: q.scenario, zone: q.zone,
          });
        }

        return {
          id: topicId,
          name: concept?.name || topicId.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
          description: concept?.description || "",
          area: areaId,
          prerequisites: concept?.prerequisites || [],
          questionCount: qs.length,
          levels, zones, tracks: Array.from(trackSet).sort(), questionsByLevel,
        };
      })
      .sort((a, b) => b.questionCount - a.questionCount);

    // Aggregate level + zone distribution
    const areaLevels: Record<string, number> = {};
    const areaZones: Record<string, number> = {};
    for (const t of topics) {
      for (const [lv, cnt] of Object.entries(t.levels)) {
        areaLevels[lv] = (areaLevels[lv] || 0) + Number(cnt);
      }
      for (const [z, cnt] of Object.entries(t.zones)) {
        areaZones[z] = (areaZones[z] || 0) + Number(cnt);
      }
    }

    return {
      id: areaId,
      name: areaId.charAt(0).toUpperCase() + areaId.slice(1).replace(/-/g, " "),
      questionCount: topics.reduce((s, t) => s + t.questionCount, 0),
      topicCount: topics.length,
      topics,
      levels: areaLevels,
      zones: areaZones,
    };
  })
  .sort((a, b) => a.name.localeCompare(b.name));

// ─── Zone definitions ─────────────────────────────────────────

const _zones: ZoneDefinition[] = Object.entries(
  (zonesData as { zones: Record<string, { skills: string[]; description: string; levels: string[] }> }).zones
).map(([id, def]) => ({
  id,
  skills: def.skills,
  description: def.description,
  levels: def.levels,
}));

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

export function getZoneDefinitions(): ZoneDefinition[] { return _zones; }
export function getZoneDefinition(zoneId: string): ZoneDefinition | undefined {
  return _zones.find(z => z.id === zoneId);
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
    totalZones: _zones.length,
  };
}

export function getAreaStyle(areaId: string): AreaStyle {
  return AREA_STYLES[areaId] || { primary: "#6b7280", bg: "#6b728012", border: "#6b728030", icon: Layers };
}

/** Get prerequisite chain for a topic (ordered learning path) */
export function getPrerequisiteChain(topicId: string): Topic[] {
  const visited = new Set<string>();
  const chain: Topic[] = [];

  function walk(id: string) {
    if (visited.has(id)) return;
    visited.add(id);
    const topic = getTopicById(id);
    if (!topic) return;
    for (const prereq of topic.prerequisites) {
      walk(prereq);
    }
    chain.push(topic);
  }

  walk(topicId);
  return chain;
}
