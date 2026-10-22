import taxonomyLabels from "../data/mlsys-taxonomy.json";
import corpusData from "../data/corpus-summary.json";
import {
  Boxes, Cpu, Workflow, ShieldCheck, Bot, Layers,
  type LucideIcon,
} from "lucide-react";
import type { AreaStyle, CompetencyArea, QuestionSummary, Topic } from "./taxonomy";

/**
 * The ML Systems Taxonomy as StaffML's user-facing classification (2026-10).
 *
 * Every question carries `taxonomy_subject` (one subject) and
 * `taxonomy_concepts` (the concepts it tests). This module groups the corpus
 * into taxonomy areas → subjects in the same shapes the Vault components
 * already render (`CompetencyArea` holding `Topic`s), so a "topic" card or
 * drawer can show a subject unchanged. Legacy `topic` / `competency_area`
 * stay on every question and keep driving chains, plans, the interviewer, and
 * textbook pointers; see `taxonomy.ts`.
 */

interface LabelArea {
  id: string; label: string; short: string; group: string; group_code: string;
  question: string; subjects: string[];
}
interface Labels {
  version: string;
  explorer: string;
  areas: LabelArea[];
  subjects: Record<string, { code: string; label: string; short: string; area: string; area_id: string; definition: string }>;
  concepts: Record<string, { label: string; subject: string }>;
}

interface RawQuestion {
  id: string; track: string; level: string; title: string; scenario: string;
  question?: string; zone: string; taxonomy_subject?: string; taxonomy_concepts?: string[];
}

export interface SubjectConcept {
  id: string;
  label: string;
  questionCount: number;
}

/** A taxonomy subject in the Vault's `Topic` shape, plus taxonomy fields. */
export interface Subject extends Topic {
  code: string;
  fullLabel: string;
  concepts: SubjectConcept[];
}

/** A taxonomy area in the Vault's `CompetencyArea` shape. */
export interface TaxArea extends CompetencyArea {
  topics: Subject[];
  group: string;
  groupCode: string;
  question: string;
}

const labels = taxonomyLabels as Labels;
const questions = corpusData as RawQuestion[];

export const TAXONOMY_EXPLORER = labels.explorer;
export const TAXONOMY_VERSION = labels.version;

const subjectOfQuestion = new Map<string, string>();
const conceptsOfQuestion = new Map<string, string[]>();
const bySubject = new Map<string, RawQuestion[]>();
for (const q of questions) {
  const s = q.taxonomy_subject;
  if (!s || !labels.subjects[s]) continue;
  subjectOfQuestion.set(q.id, s);
  conceptsOfQuestion.set(q.id, (q.taxonomy_concepts ?? []).filter((c) => labels.concepts[c]));
  if (!bySubject.has(s)) bySubject.set(s, []);
  bySubject.get(s)!.push(q);
}

function buildSubject(id: string, qs: RawQuestion[]): Subject {
  const meta = labels.subjects[id];
  const levels: Record<string, number> = {};
  const zones: Record<string, number> = {};
  const trackSet = new Set<string>();
  const questionsByLevel: Record<string, QuestionSummary[]> = {};
  const conceptCounts = new Map<string, number>();
  for (const q of qs) {
    levels[q.level] = (levels[q.level] || 0) + 1;
    zones[q.zone] = (zones[q.zone] || 0) + 1;
    trackSet.add(q.track);
    (questionsByLevel[q.level] ??= []).push({
      id: q.id, title: q.title, level: q.level, track: q.track,
      scenario: q.scenario, question: q.question, zone: q.zone,
    });
    for (const c of conceptsOfQuestion.get(q.id) ?? []) conceptCounts.set(c, (conceptCounts.get(c) || 0) + 1);
  }
  const concepts = Array.from(conceptCounts.entries())
    .map(([cid, n]) => ({ id: cid, label: labels.concepts[cid].label, questionCount: n }))
    .sort((a, b) => b.questionCount - a.questionCount || a.label.localeCompare(b.label));
  return {
    id,
    name: meta.short,
    fullLabel: meta.label,
    code: meta.code,
    description: meta.definition,
    area: meta.area_id,
    prerequisites: [],
    questionCount: qs.length,
    levels, zones, tracks: Array.from(trackSet).sort(), questionsByLevel,
    concepts,
    practiceParam: "subject",
    explorerUrl: `${labels.explorer}#subject=${encodeURIComponent(id)}`,
  };
}

const _areas: TaxArea[] = labels.areas
  .map((a) => {
    const topics = a.subjects
      .filter((sid) => bySubject.has(sid))
      .map((sid) => buildSubject(sid, bySubject.get(sid)!));
    const levels: Record<string, number> = {};
    const zones: Record<string, number> = {};
    for (const t of topics) {
      for (const [k, n] of Object.entries(t.levels)) levels[k] = (levels[k] || 0) + n;
      for (const [k, n] of Object.entries(t.zones)) zones[k] = (zones[k] || 0) + n;
    }
    return {
      id: a.id,
      name: a.short,
      code: a.id,
      questionCount: topics.reduce((s, t) => s + t.questionCount, 0),
      topicCount: topics.length,
      topics,
      levels, zones,
      group: a.group,
      groupCode: a.group_code,
      question: a.question,
    };
  })
  .filter((a) => a.topicCount > 0);

const _subjects = new Map(_areas.flatMap((a) => a.topics.map((t) => [t.id, t] as const)));

// One hue per group, so the 20 areas read as five families.
const GROUP_STYLES: Record<string, { primary: string; icon: LucideIcon }> = {
  F: { primary: "#a78bfa", icon: Layers },
  S: { primary: "#60a5fa", icon: Cpu },
  P: { primary: "#4ade80", icon: Workflow },
  Q: { primary: "#fbbf24", icon: ShieldCheck },
  T: { primary: "#f472b6", icon: Bot },
};

// ─── Public API ────────────────────────────────────────────────

export function getTaxAreas(): TaxArea[] { return _areas; }
export function getTaxAreaById(id: string): TaxArea | undefined { return _areas.find((a) => a.id === id); }
export function getSubjectById(id: string): Subject | undefined { return _subjects.get(id); }
export function getTaxAreaForSubject(subjectId: string): TaxArea | undefined {
  const s = _subjects.get(subjectId);
  return s ? _areas.find((a) => a.id === s.area) : undefined;
}

export function getTaxAreaStyle(areaId: string): AreaStyle {
  const style = GROUP_STYLES[areaId.charAt(0)];
  if (!style) return { primary: "#6b7280", bg: "#6b728012", border: "#6b728030", icon: Boxes };
  return { primary: style.primary, bg: `${style.primary}12`, border: `${style.primary}30`, icon: style.icon };
}

/** Subject identity of a question, looked up from its id (never stored in progress). */
export function getSubjectIdForQuestion(questionId: string): string | undefined {
  return subjectOfQuestion.get(questionId);
}
export function getConceptIdsForQuestion(questionId: string): string[] {
  return conceptsOfQuestion.get(questionId) ?? [];
}
export function getConceptLabel(conceptId: string): string | undefined {
  return labels.concepts[conceptId]?.label;
}
export function getSubjectLabel(subjectId: string): string | undefined {
  return labels.subjects[subjectId]?.label;
}

/** Subjects whose label, code, definition, or concepts match the query. */
export function searchSubjects(query: string): Subject[] {
  const q = query.toLowerCase().trim();
  const all = _areas.flatMap((a) => a.topics);
  if (!q) return all;
  return all.filter((t) =>
    t.name.toLowerCase().includes(q) || t.fullLabel.toLowerCase().includes(q) ||
    t.code.toLowerCase() === q || t.id.includes(q) || t.description.toLowerCase().includes(q) ||
    t.concepts.some((c) => c.label.toLowerCase().includes(q)),
  );
}

export function getTaxonomyStats() {
  return {
    areas: _areas.length,
    subjects: _subjects.size,
    concepts: new Set(Array.from(conceptsOfQuestion.values()).flat()).size,
    taggedQuestions: subjectOfQuestion.size,
  };
}
