#!/usr/bin/env node
/**
 * Sync periodic-table/table.yml -> interviews/staffml/src/data/periodicTable.ts
 *
 * This is a one-way generator. Edit the YAML, never edit the .ts file
 * (it has a @generated header to remind you). The script is wired into
 * the StaffML package.json as a `predev` and `prebuild` hook so the
 * data file is always fresh before the dev server or production build runs.
 *
 * Usage: npm run sync:periodic-table   (from interviews/staffml)
 *        node interviews/staffml/scripts/sync-periodic-table.mjs   (from anywhere)
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..", "..", "..");
const YAML_PATH = path.join(REPO, "periodic-table", "table.yml");
const TS_PATH = path.join(REPO, "interviews", "staffml", "src", "data", "periodicTable.ts");

// ── 1. Load and validate ────────────────────────────────────────────────
const doc = yaml.load(fs.readFileSync(YAML_PATH, "utf8"));
validate(doc);
const KNOWN_SYMS = new Set(doc.elements.map((e) => e.sym));

// ── 2. Parse every compound formula into a token list ──────────────────
// This happens at sync time so the runtime React renderer stays simple
// (it just iterates the pre-parsed tokens, no parser at runtime).
const parsedCompounds = doc.compounds.map((section) => ({
  ...section,
  items: section.items.map((item) => ({
    name: item.name,
    formula: parseFormula(item.formula, KNOWN_SYMS),
  })),
}));

// ── 3. Emit the TypeScript file ─────────────────────────────────────────
const out = renderTs(doc, parsedCompounds);
fs.writeFileSync(TS_PATH, out);

console.log(`Wrote ${TS_PATH}`);
console.log(`  ${doc.elements.length} elements`);
console.log(`  ${parsedCompounds.reduce((n, s) => n + s.items.length, 0)} compounds across ${parsedCompounds.length} sections`);

// ════════════════════════════════════════════════════════════════════════
// Validation (mirrors build-html.mjs)
// ════════════════════════════════════════════════════════════════════════
function validate(doc) {
  if (!doc || typeof doc !== "object") throw new Error("table.yml is empty or malformed");
  if (!Array.isArray(doc.elements) || doc.elements.length === 0) throw new Error("No elements defined");

  const issues = [];
  const knownSyms = new Set(doc.elements.map((e) => e.sym));
  const cellSeen = new Map();

  for (const e of doc.elements) {
    const k = `${e.row},${e.col}`;
    if (cellSeen.has(k)) {
      issues.push(`Cell collision at (${k}): #${cellSeen.get(k)} and #${e.id}`);
    } else {
      cellSeen.set(k, e.id);
    }
  }
  for (const e of doc.elements) {
    for (const b of e.bonds || []) {
      if (!knownSyms.has(b)) issues.push(`Element #${e.id} ${e.sym}: unresolved bond '${b}'`);
    }
  }
  const symCount = {};
  for (const e of doc.elements) symCount[e.sym] = (symCount[e.sym] || 0) + 1;
  const declared = new Set((doc.known_collisions || []).map((c) => c.sym));
  for (const [sym, count] of Object.entries(symCount)) {
    if (count > 1 && !declared.has(sym)) {
      issues.push(`Undocumented symbol collision: ${sym} appears ${count} times — add to known_collisions`);
    }
  }
  for (const section of doc.compounds || []) {
    for (const item of section.items || []) {
      const refs = extractFormulaSymbols(item.formula);
      for (const ref of refs) {
        if (!knownSyms.has(ref)) {
          issues.push(`Compound "${item.name}" references unknown symbol '${ref}': ${item.formula}`);
        }
      }
    }
  }
  if (issues.length > 0) {
    console.error("VALIDATION FAILED:");
    issues.forEach((i) => console.error("  " + i));
    process.exit(1);
  }
}

function extractFormulaSymbols(formula) {
  const cleaned = formula.replace(/_[A-Za-z]+/g, "");
  const re = /(?<![A-Za-z])([A-Z][a-z])(?![A-Za-z])/g;
  const out = [];
  let m;
  while ((m = re.exec(cleaned)) !== null) out.push(m[1]);
  return out;
}

// ════════════════════════════════════════════════════════════════════════
// Formula parser: string -> FormulaToken[]
// ════════════════════════════════════════════════════════════════════════
/**
 * Walks the formula left to right. A "symbol token" is a two-character
 * substring [A-Z][a-z] that (a) appears in `knownSyms`, (b) is bordered
 * by non-letters on both sides, and (c) optionally has a `_subscript`
 * trailing it. Anything else accumulates into op tokens (sequential
 * connectives, brackets, parens, the ᴺ marker, etc.).
 *
 * Bracket/paren subscripts (`]ᴺ_enc`) become an op token containing the
 * literal underscore + word — the React renderer treats them as inline
 * text. (The standalone HTML emitter promotes them to <sub> tags; the
 * React renderer doesn't, but the visual difference is minor and we can
 * upgrade later.)
 */
function parseFormula(formula, knownSyms) {
  const tokens = [];
  let pendingOp = "";
  let i = 0;
  while (i < formula.length) {
    const two = formula.slice(i, i + 2);
    const isSym =
      two.length === 2 &&
      /^[A-Z][a-z]$/.test(two) &&
      knownSyms.has(two) &&
      (i === 0 || !/[A-Za-z]/.test(formula[i - 1])) &&
      (i + 2 >= formula.length || !/[A-Za-z]/.test(formula[i + 2]));

    if (isSym) {
      if (pendingOp) {
        tokens.push({ kind: "op", text: pendingOp });
        pendingOp = "";
      }
      let sub;
      if (formula[i + 2] === "_") {
        const subMatch = formula.slice(i + 3).match(/^[A-Za-z]+/);
        if (subMatch) {
          sub = subMatch[0];
          tokens.push({ kind: "sym", sym: two, sub });
          i += 3 + sub.length;
          continue;
        }
      }
      tokens.push({ kind: "sym", sym: two });
      i += 2;
    } else {
      pendingOp += formula[i];
      i += 1;
    }
  }
  if (pendingOp) tokens.push({ kind: "op", text: pendingOp });
  return tokens;
}

// ════════════════════════════════════════════════════════════════════════
// TypeScript code generation
// ════════════════════════════════════════════════════════════════════════
function renderTs(doc, parsedCompounds) {
  let out = "";
  out += "// @generated — DO NOT EDIT BY HAND\n";
  out += "// Source of truth: periodic-table/table.yml\n";
  out += "// Regenerate: npm run sync:periodic-table   (from interviews/staffml)\n";
  out += "//\n";
  out += "// Edits to this file will be overwritten on the next sync. To change\n";
  out += "// any element, compound, color, or label, edit table.yml and re-run\n";
  out += "// the sync script (or just run `npm run dev` — predev hook handles it).\n";
  out += "\n";

  // Types
  out += `export type BlockKey = ${doc.blocks.map((b) => `"${b.key}"`).join(" | ")};\n\n`;

  out += "export interface Block {\n";
  out += "  key: BlockKey;\n";
  out += "  name: string;\n";
  out += "  sub: string;\n";
  out += "  color: string;\n";
  out += "  cols: number[];\n";
  out += "}\n\n";

  // blocks (as a Record so the existing React code keeps working — it does
  // `blocks[el.block]` lookups by key)
  out += "export const blocks: Record<BlockKey, Block> = {\n";
  for (const b of doc.blocks) {
    out += `  ${b.key}: { key: "${b.key}", name: ${tsStr(b.name)}, sub: ${tsStr(b.sub)}, color: ${tsStr(b.color)}, cols: [${b.cols.join(", ")}] },\n`;
  }
  out += "};\n\n";

  // rowLabels (a flat string array — what the React page imports)
  out += "export const rowLabels = [\n";
  for (const r of doc.rows) {
    out += `  ${tsStr(r.name)},\n`;
  }
  out += "];\n\n";

  // Element type
  out += "export interface Element {\n";
  out += "  num: number;\n";
  out += "  sym: string;\n";
  out += "  name: string;\n";
  out += "  block: BlockKey;\n";
  out += "  row: number;\n";
  out += "  col: number;\n";
  out += "  year: string;\n";
  out += "  desc: string;\n";
  out += "  bonds: string[];\n";
  out += "  whyHere: string;\n";
  out += "}\n\n";

  // elements
  out += "export const elements: Element[] = [\n";
  for (const e of doc.elements) {
    const year = e.year === null || e.year === undefined ? '"—"' : tsStr(e.year);
    const bonds = e.bonds.length === 0 ? "[]" : `[${e.bonds.map(tsStr).join(", ")}]`;
    out += `  { num: ${e.id}, sym: ${tsStr(e.sym)}, name: ${tsStr(e.name)}, block: "${e.block}", row: ${e.row}, col: ${e.col}, year: ${year}, desc: ${tsStr(e.desc)}, bonds: ${bonds}, whyHere: ${tsStr(e.why)} },\n`;
  }
  out += "];\n\n";

  // elMap — last-write-wins lookup, matching the original behavior
  out += "// Lookup map by symbol — last write wins (matches the original\n";
  out += "// behavior for documented symbol collisions like Sm/Sp/Ro/En).\n";
  out += "export const elMap: Record<string, Element> = {};\n";
  out += "elements.forEach((e) => { elMap[e.sym] = e; });\n\n";

  // Compound types
  out += "// ── Compounds ─────────────────────────────────────────────────────────────\n";
  out += "// Each formula is a list of typed tokens parsed from the YAML's formula\n";
  out += "// string at sync time, so the React renderer never has to parse anything.\n";
  out += "//   sym tokens reference an element by symbol, with optional subscript.\n";
  out += "//   op tokens are literal connective text (→ ∥ ⇌ ↺ [ ] ( ) ?).\n";
  out += "export type FormulaToken =\n";
  out += '  | { kind: "sym"; sym: string; sub?: string }\n';
  out += '  | { kind: "op"; text: string };\n\n';

  out += "export interface Compound {\n";
  out += "  name: string;\n";
  out += "  formula: FormulaToken[];\n";
  out += "}\n\n";

  out += "export interface CompoundSection {\n";
  out += "  title: string;\n";
  out += "  hint?: string;\n";
  out += "  items: Compound[];\n";
  out += "}\n\n";

  // compounds (parsed)
  out += "export const compounds: CompoundSection[] = [\n";
  for (const section of parsedCompounds) {
    out += `  {\n`;
    out += `    title: ${tsStr(section.section)},\n`;
    if (section.hint) out += `    hint: ${tsStr(section.hint)},\n`;
    out += `    items: [\n`;
    for (const item of section.items) {
      out += `      { name: ${tsStr(item.name)}, formula: ${renderTokenList(item.formula)} },\n`;
    }
    out += `    ],\n`;
    out += `  },\n`;
  }
  out += "];\n";

  return out;
}

function renderTokenList(tokens) {
  if (tokens.length === 0) return "[]";
  const parts = tokens.map((t) => {
    if (t.kind === "sym") {
      return t.sub
        ? `{ kind: "sym", sym: ${tsStr(t.sym)}, sub: ${tsStr(t.sub)} }`
        : `{ kind: "sym", sym: ${tsStr(t.sym)} }`;
    }
    return `{ kind: "op", text: ${tsStr(t.text)} }`;
  });
  return "[" + parts.join(", ") + "]";
}

function tsStr(s) {
  if (s === null || s === undefined) return '""';
  // Use double quotes; escape backslash, double quote, and special chars.
  return '"' + String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n") + '"';
}
