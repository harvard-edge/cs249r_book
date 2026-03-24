import taxonomyData from "../data/taxonomy.json";

export interface Concept {
  id: string;
  name: string;
  description: string;
  tracks: string[];
  source_chapters: string[];
  prerequisites: string[];
  dependents: string[];
  question_count: number;
  level_distribution: Record<string, number>;
  role: "foundational" | "competency" | "contextual";
}

export interface Edge {
  source: string;
  target: string;
}

interface TaxonomyData {
  version: string;
  synced_at: string;
  total_concepts: number;
  total_edges: number;
  total_questions: number;
  concepts: Concept[];
  edges: Edge[];
}

const data = taxonomyData as TaxonomyData;
const conceptMap = new Map<string, Concept>(data.concepts.map((c) => [c.id, c]));

// ─── Accessors ─────────────────────────────────────────────────

export function getConcepts(): Concept[] {
  return data.concepts;
}

export function getConceptById(id: string): Concept | undefined {
  return conceptMap.get(id);
}

export function getEdges(): Edge[] {
  return data.edges;
}

export function getTaxonomyStats() {
  return {
    version: data.version,
    syncedAt: data.synced_at,
    totalConcepts: data.total_concepts,
    totalEdges: data.total_edges,
    totalQuestions: data.total_questions,
  };
}

// ─── Queries ───────────────────────────────────────────────────

export function getPrereqs(id: string): Concept[] {
  const concept = conceptMap.get(id);
  if (!concept) return [];
  return concept.prerequisites
    .map((pid) => conceptMap.get(pid))
    .filter((c): c is Concept => !!c);
}

export function getDependents(id: string): Concept[] {
  const concept = conceptMap.get(id);
  if (!concept) return [];
  return concept.dependents
    .map((did) => conceptMap.get(did))
    .filter((c): c is Concept => !!c);
}

export function search(query: string): Concept[] {
  const q = query.toLowerCase().trim();
  if (!q) return data.concepts;
  return data.concepts.filter(
    (c) =>
      c.id.includes(q) ||
      c.name.toLowerCase().includes(q) ||
      c.description.toLowerCase().includes(q) ||
      c.tracks.some((t) => t.includes(q)) ||
      c.source_chapters.some((ch) => ch.includes(q))
  );
}

export function getChapters(): string[] {
  const chapters = new Set<string>();
  data.concepts.forEach((c) =>
    c.source_chapters.forEach((ch) => chapters.add(ch))
  );
  return Array.from(chapters).sort();
}

export function getTracks(): string[] {
  const tracks = new Set<string>();
  data.concepts.forEach((c) => c.tracks.forEach((t) => tracks.add(t)));
  return Array.from(tracks).sort();
}

export function getRoles(): Array<Concept["role"]> {
  return ["foundational", "competency", "contextual"];
}

// ─── Formatting helpers ────────────────────────────────────────

export function formatChapter(ch: string): string {
  return ch
    .replace("vol1_", "V1: ")
    .replace("vol2_", "V2: ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
