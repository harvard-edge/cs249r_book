import glossaryData from "@/data/glossary.json";

export interface GlossaryEntry {
  term: string;
  display: string;
  definition: string;
  acronym: string | null;
}

const entries: GlossaryEntry[] = glossaryData as GlossaryEntry[];

const byTerm = new Map<string, GlossaryEntry>();
const acronymToEntry = new Map<string, GlossaryEntry>();

for (const e of entries) {
  byTerm.set(e.term, e);
  byTerm.set(e.display.toLowerCase(), e);

  if (/^[A-Z][A-Z0-9/.+-]{1,}$/.test(e.display)) {
    acronymToEntry.set(e.display, e);
  }

  const parenMatch = e.display.match(/\(([A-Z][A-Z0-9/.+-]+)\)\s*$/);
  if (parenMatch) {
    acronymToEntry.set(parenMatch[1], e);
  }
}

const acronymPattern = buildAcronymPattern();

function buildAcronymPattern(): RegExp | null {
  const acronyms = Array.from(acronymToEntry.keys())
    .filter((a) => a.length >= 2)
    .sort((a, b) => b.length - a.length);
  if (acronyms.length === 0) return null;
  const escaped = acronyms.map((c) => c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return new RegExp(`\\b(${escaped.join("|")})\\b`, "g");
}

export function lookupTerm(term: string): GlossaryEntry | undefined {
  return byTerm.get(term.toLowerCase());
}

export function findAcronymsInText(text: string): { start: number; end: number; entry: GlossaryEntry }[] {
  if (!acronymPattern) return [];
  const matches: { start: number; end: number; entry: GlossaryEntry }[] = [];
  acronymPattern.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = acronymPattern.exec(text)) !== null) {
    const entry = acronymToEntry.get(m[1]) ?? byTerm.get(m[1].toLowerCase());
    if (entry) {
      matches.push({ start: m.index, end: m.index + m[1].length, entry });
    }
  }
  return matches;
}

export function getAllEntries(): GlossaryEntry[] {
  return entries;
}
