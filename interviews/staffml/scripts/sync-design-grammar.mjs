#!/usr/bin/env node
/**
 * Sync the design-grammar YAML -> interviews/staffml/src/data/designGrammar.ts
 *
 * This is a one-way generator. Edit the YAML, never edit the .ts file
 * (it has a @generated header to remind you). The script is wired into
 * the StaffML package.json as a `predev` and `prebuild` hook so the
 * data file is always fresh before the dev server or production build runs.
 *
 * Usage: npm run sync:design-grammar   (from interviews/staffml)
 *        node interviews/staffml/scripts/sync-design-grammar.mjs   (from anywhere)
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..", "..", "..");
const YAML_PATH = path.join(REPO, "design-grammar", "grammar.yml");
const TS_PATH = path.join(REPO, "interviews", "staffml", "src", "data", "designGrammar.ts");

// ── 1. Load and validate ────────────────────────────────────────────────
const doc = yaml.load(fs.readFileSync(YAML_PATH, "utf8"));
validate(doc);
const KNOWN_SYMS = new Set(doc.primitives.map((e) => e.sym));

// ── 2. Parse every assembly expression into a token list ──────────────────
// This happens at sync time so the runtime React renderer stays simple
// (it just iterates the pre-parsed tokens, no parser at runtime).
const parsedAssemblies = doc.assemblies.map((section) => ({
  ...section,
  items: section.items.map((item) => ({
    name: item.name,
    expression: parseExpression(item.expression, KNOWN_SYMS),
  })),
}));

// ── 3. Emit the TypeScript file ─────────────────────────────────────────
const out = renderTs(doc, parsedAssemblies);
fs.writeFileSync(TS_PATH, out);

console.log(`Wrote ${TS_PATH}`);
console.log(`  ${doc.primitives.length} primitives`);
console.log(`  ${parsedAssemblies.reduce((n, s) => n + s.items.length, 0)} assemblies across ${parsedAssemblies.length} sections`);

// ════════════════════════════════════════════════════════════════════════
// Validation (mirrors build-html.mjs)
// ════════════════════════════════════════════════════════════════════════
function validate(doc) {
  if (!doc || typeof doc !== "object") throw new Error("grammar.yml is empty or malformed");
  if (!Array.isArray(doc.primitives) || doc.primitives.length === 0) throw new Error("No primitives defined");

  const issues = [];
  const knownSyms = new Set(doc.primitives.map((e) => e.sym));
  const cellSeen = new Map();

  for (const e of doc.primitives) {
    const k = `${e.layer},${e.col}`;
    if (cellSeen.has(k)) {
      issues.push(`Cell collision at (${k}): #${cellSeen.get(k)} and #${e.id}`);
    } else {
      cellSeen.set(k, e.id);
    }
  }
  for (const e of doc.primitives) {
    for (const b of e.composition_links || []) {
      if (!knownSyms.has(b)) issues.push(`Primitive #${e.id} ${e.sym}: unresolved composition link '${b}'`);
    }
  }
  const symCount = {};
  for (const e of doc.primitives) symCount[e.sym] = (symCount[e.sym] || 0) + 1;
  const declared = new Set((doc.known_collisions || []).map((c) => c.sym));
  for (const [sym, count] of Object.entries(symCount)) {
    if (count > 1 && !declared.has(sym)) {
      issues.push(`Undocumented symbol collision: ${sym} appears ${count} times — add to known_collisions`);
    }
  }
  for (const section of doc.assemblies || []) {
    for (const item of section.items || []) {
      const refs = extractExpressionSymbols(item.expression);
      for (const ref of refs) {
        if (!knownSyms.has(ref)) {
          issues.push(`Assembly "${item.name}" references unknown symbol '${ref}': ${item.expression}`);
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

function extractExpressionSymbols(expression) {
  const cleaned = expression.replace(/_[A-Za-z]+/g, "");
  const re = /(?<![A-Za-z])([A-Z][a-z])(?![A-Za-z])/g;
  const out = [];
  let m;
  while ((m = re.exec(cleaned)) !== null) out.push(m[1]);
  return out;
}

// ════════════════════════════════════════════════════════════════════════
// Expression parser: string -> ExpressionToken[]
// ════════════════════════════════════════════════════════════════════════
/**
 * Walks the expression left to right. A "symbol token" is a two-character
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
function parseExpression(expression, knownSyms) {
  const tokens = [];
  let pendingOp = "";
  let i = 0;
  while (i < expression.length) {
    const two = expression.slice(i, i + 2);
    const isSym =
      two.length === 2 &&
      /^[A-Z][a-z]$/.test(two) &&
      knownSyms.has(two) &&
      (i === 0 || !/[A-Za-z]/.test(expression[i - 1])) &&
      (i + 2 >= expression.length || !/[A-Za-z]/.test(expression[i + 2]));

    if (isSym) {
      if (pendingOp) {
        tokens.push({ kind: "op", text: pendingOp });
        pendingOp = "";
      }
      let sub;
      if (expression[i + 2] === "_") {
        const subMatch = expression.slice(i + 3).match(/^[A-Za-z]+/);
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
      pendingOp += expression[i];
      i += 1;
    }
  }
  if (pendingOp) tokens.push({ kind: "op", text: pendingOp });
  return tokens;
}

// ════════════════════════════════════════════════════════════════════════
// TypeScript code generation
// ════════════════════════════════════════════════════════════════════════
function renderTs(doc, parsedAssemblies) {
  let out = "";
  out += "// @generated — DO NOT EDIT BY HAND\n";
  out += "// Source of truth: design-grammar catalog data (design-grammar/grammar.yml)\n";
  out += "// Regenerate: npm run sync:design-grammar   (from interviews/staffml)\n";
  out += "//\n";
  out += "// Edits to this file will be overwritten on the next sync. To change\n";
  out += "// any primitive, assembly, color, or label, edit grammar.yml and re-run\n";
  out += "// the sync script (or just run `npm run dev` — predev hook handles it).\n";
  out += "\n";

  // Types
  out += `export type RoleKey = ${doc.roles.map((b) => `"${b.key}"`).join(" | ")};\n\n`;

  out += "export interface Role {\n";
  out += "  key: RoleKey;\n";
  out += "  name: string;\n";
  out += "  sub: string;\n";
  out += "  color: string;\n";
  out += "  cols: number[];\n";
  out += "}\n\n";

  // roles (as a Record so the existing React code keeps working — it does
  // `roles[el.role]` lookups by key)
  out += "export const roles: Record<RoleKey, Role> = {\n";
  for (const b of doc.roles) {
    out += `  ${b.key}: { key: "${b.key}", name: ${tsStr(b.name)}, sub: ${tsStr(b.sub)}, color: ${tsStr(b.color)}, cols: [${b.cols.join(", ")}] },\n`;
  }
  out += "};\n\n";

  // layerLabels (a flat string array — what the React page imports)
  out += "export const layerLabels = [\n";
  for (const r of doc.layers) {
    out += `  ${tsStr(r.name)},\n`;
  }
  out += "];\n\n";

  // Primitive type (keeps the Primitive name for existing React imports)
  out += "export interface Primitive {\n";
  out += "  num: number;\n";
  out += "  sym: string;\n";
  out += "  name: string;\n";
  out += "  role: RoleKey;\n";
  out += "  layer: number;\n";
  out += "  col: number;\n";
  out += "  year: string;\n";
  out += "  description: string;\n";
  out += "  composition_links: string[];\n";
  out += "  rationale: string;\n";
  out += "}\n\n";

  // primitives
  out += "export const primitives: Primitive[] = [\n";
  for (const e of doc.primitives) {
    const year = e.year === null || e.year === undefined ? '"—"' : tsStr(e.year);
    const composition_links = e.composition_links.length === 0 ? "[]" : `[${e.composition_links.map(tsStr).join(", ")}]`;
    out += `  { num: ${e.id}, sym: ${tsStr(e.sym)}, name: ${tsStr(e.name)}, role: "${e.role}", layer: ${e.layer}, col: ${e.col}, year: ${year}, description: ${tsStr(e.description)}, composition_links: ${composition_links}, rationale: ${tsStr(e.rationale)} },\n`;
  }
  out += "];\n\n";

  // primitiveMap — last-write-wins lookup, matching the original behavior
  out += "// Lookup map by symbol — last write wins (matches the original\n";
  out += "// behavior for documented symbol collisions like Sm/Sp/Ro/En).\n";
  out += "export const primitiveMap: Record<string, Primitive> = {};\n";
  out += "primitives.forEach((e) => { primitiveMap[e.sym] = e; });\n\n";

  // Assembly types (keep Assembly names for existing React imports)
  out += "// ── Assemblies ────────────────────────────────────────────────────────────\n";
  out += "// Each expression is a list of typed tokens parsed from the YAML's expression\n";
  out += "// string at sync time, so the React renderer never has to parse anything.\n";
  out += "//   sym tokens reference a primitive by symbol, with optional subscript.\n";
  out += "//   op tokens are literal connective text (→ ∥ ⇌ ↺ [ ] ( ) ?).\n";
  out += "export type ExpressionToken =\n";
  out += '  | { kind: "sym"; sym: string; sub?: string }\n';
  out += '  | { kind: "op"; text: string };\n\n';

  out += "export interface Assembly {\n";
  out += "  name: string;\n";
  out += "  expression: ExpressionToken[];\n";
  out += "}\n\n";

  out += "export interface AssemblySection {\n";
  out += "  title: string;\n";
  out += "  hint?: string;\n";
  out += "  items: Assembly[];\n";
  out += "}\n\n";

  // assemblies (parsed)
  out += "export const assemblies: AssemblySection[] = [\n";
  for (const section of parsedAssemblies) {
    out += `  {\n`;
    out += `    title: ${tsStr(section.section)},\n`;
    if (section.hint) out += `    hint: ${tsStr(section.hint)},\n`;
    out += `    items: [\n`;
    for (const item of section.items) {
      out += `      { name: ${tsStr(item.name)}, expression: ${renderTokenList(item.expression)} },\n`;
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
