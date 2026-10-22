import { describe, expect, it } from "vitest";
import corpus from "../../data/corpus-summary.json";
import labels from "../../data/mlsys-taxonomy.json";
import {
  getConceptIdsForQuestion, getSubjectById, getSubjectIdForQuestion, getTaxAreaForSubject,
  getTaxAreas, getTaxAreaStyle, getTaxonomyStats, searchSubjects,
} from "../mlsys-taxonomy";

type Q = { id: string; taxonomy_subject?: string; taxonomy_concepts?: string[] };
const questions = corpus as Q[];

describe("ML Systems Taxonomy grouping", () => {
  it("places every tagged question in exactly one subject of one area", () => {
    const counted = getTaxAreas().reduce((n, a) => n + a.questionCount, 0);
    const tagged = questions.filter((q) => q.taxonomy_subject).length;
    expect(tagged).toBe(questions.length);
    expect(counted).toBe(tagged);
    expect(getTaxonomyStats().taggedQuestions).toBe(tagged);
  });

  it("keeps the taxonomy's area order and codes", () => {
    const order = (labels as { areas: { id: string }[] }).areas.map((a) => a.id);
    const shown = getTaxAreas().map((a) => a.id);
    expect(shown).toEqual(order.filter((id) => shown.includes(id)));
    for (const area of getTaxAreas()) {
      expect(area.code).toBe(area.id);
      for (const s of area.topics) {
        expect(s.code.startsWith(area.id + ".")).toBe(true);
        expect(getTaxAreaForSubject(s.id)?.id).toBe(area.id);
        expect(s.practiceParam).toBe("subject");
      }
    }
  });

  it("looks up subject and concepts from a question id", () => {
    const q = questions.find((x) => (x.taxonomy_concepts ?? []).length > 0)!;
    expect(getSubjectIdForQuestion(q.id)).toBe(q.taxonomy_subject);
    expect(getConceptIdsForQuestion(q.id)).toEqual(q.taxonomy_concepts);
    const subject = getSubjectById(q.taxonomy_subject!)!;
    expect(subject.concepts.map((c) => c.id)).toEqual(expect.arrayContaining(q.taxonomy_concepts!));
  });

  it("finds subjects by label, code, and concept", () => {
    expect(searchSubjects("KV cache").map((s) => s.id)).toContain("llm-inference");
    const code = getSubjectById("llm-inference")!.code.toLowerCase();
    expect(searchSubjects(code).map((s) => s.id)).toEqual(["llm-inference"]);
  });

  it("styles areas by group", () => {
    expect(getTaxAreaStyle("S1").primary).toBe(getTaxAreaStyle("S4").primary);
    expect(getTaxAreaStyle("S1").primary).not.toBe(getTaxAreaStyle("Q1").primary);
  });
});
