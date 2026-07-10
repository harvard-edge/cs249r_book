#!/usr/bin/env node
/**
 * Validate design-grammar/grammar.yml against the schema and the cross-reference
 * invariants (cell collisions, composition-link resolution, assembly-expression resolution, undocumented
 * symbol collisions). Exits 0 on success, 1 on failure.
 *
 * Usage: node design-grammar/scripts/validate.mjs
 *        make validate                 (from design-grammar/)
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const YAML_PATH = path.join(__dirname, "..", "grammar.yml");
const REWRITE_RULES_PATH = path.join(__dirname, "..", "rewrite-rules.yml");

const doc = yaml.load(fs.readFileSync(YAML_PATH, "utf8"));
const issues = [];

if (!doc || typeof doc !== "object") {
  console.error("FAIL: grammar.yml is empty or malformed");
  process.exit(1);
}
if (!Array.isArray(doc.primitives) || doc.primitives.length === 0) {
  console.error("FAIL: no primitives defined");
  process.exit(1);
}

// ── Cross-reference invariants ──────────────────────────────────────────
const knownSyms = new Set(doc.primitives.map((e) => e.sym));
const cellSeen = new Map();

for (const e of doc.primitives) {
  // Schema-ish field validation (in case the user hand-edited and broke shape)
  if (typeof e.id !== "number") issues.push(`Primitive ${e.sym || "?"}: id must be a number`);
  if (typeof e.sym !== "string" || !/^[A-Z][a-z]$/.test(e.sym)) issues.push(`Primitive #${e.id}: sym '${e.sym}' must match [A-Z][a-z]`);
  if (typeof e.layer !== "number" || e.layer < 1 || e.layer > 8) issues.push(`Primitive #${e.id} ${e.sym}: layer ${e.layer} out of range 1-8`);
  if (typeof e.col !== "number" || e.col < 1 || e.col > 18) issues.push(`Primitive #${e.id} ${e.sym}: col ${e.col} out of range 1-18`);
  if (!["R", "C", "X", "K", "M"].includes(e.role)) issues.push(`Primitive #${e.id} ${e.sym}: role '${e.role}' must be one of R/C/X/K/M`);

  // Cell collisions
  const k = `${e.layer},${e.col}`;
  if (cellSeen.has(k)) {
    const prev = cellSeen.get(k);
    issues.push(`Cell collision at (${k}): #${prev.id} ${prev.sym} and #${e.id} ${e.sym}`);
  } else {
    cellSeen.set(k, e);
  }
}

// Composition-link resolution
for (const e of doc.primitives) {
  for (const b of e.composition_links || []) {
    if (!knownSyms.has(b)) issues.push(`Primitive #${e.id} ${e.sym}: composition link '${b}' references unknown primitive`);
  }
}

// Symbol collisions: must be in known_collisions
const symCount = {};
for (const e of doc.primitives) symCount[e.sym] = (symCount[e.sym] || 0) + 1;
const declared = new Set((doc.known_collisions || []).map((c) => c.sym));
for (const [sym, count] of Object.entries(symCount)) {
  if (count > 1 && !declared.has(sym)) {
    issues.push(`Undocumented symbol collision: '${sym}' appears ${count} times — add to known_collisions`);
  }
}

// known_collisions notes must actually match the primitives they describe.
// Previously only `sym` presence was checked, so the free-text `note` field
// (row/col call-outs) could drift from the real data with nothing to catch it.
const byId = new Map(doc.primitives.map((e) => [e.id, e]));
for (const c of doc.known_collisions || []) {
  const cited = new Map();
  const re = /#(\d+)[^)]*?row\s*(\d+)\s*col\s*(\d+)/g;
  let m;
  while ((m = re.exec(c.note || "")) !== null) {
    cited.set(Number(m[1]), { layer: Number(m[2]), col: Number(m[3]) });
  }
  for (const id of c.ids || []) {
    const actual = byId.get(id);
    const claimed = cited.get(id);
    if (!actual) {
      issues.push(`known_collisions '${c.sym}': id ${id} does not exist`);
    } else if (!claimed) {
      issues.push(`known_collisions '${c.sym}': note does not cite row/col for #${id}`);
    } else if (claimed.layer !== actual.layer || claimed.col !== actual.col) {
      issues.push(
        `known_collisions '${c.sym}': note claims #${id} is row ${claimed.layer} col ${claimed.col}, ` +
        `but it is actually row ${actual.layer} col ${actual.col}`
      );
    }
  }
}

// Expression resolution: every two-letter [A-Z][a-z] token in any expression must
// resolve to a known primitive symbol.
for (const section of doc.assemblies || []) {
  for (const item of section.items || []) {
    const cleaned = item.expression.replace(/_[A-Za-z]+/g, "");
    const re = /(?<![A-Za-z])([A-Z][a-z])(?![A-Za-z])/g;
    let m;
    while ((m = re.exec(cleaned)) !== null) {
      if (!knownSyms.has(m[1])) {
        issues.push(`Assembly "${item.name}" references unknown symbol '${m[1]}': ${item.expression}`);
      }
    }
  }
}

// ── rewrite-rules.yml ────────────────────────────────────────────────────
// Previously never read by this validator at all, so drift here was silent
// and permanent. Check that every rule's `relieves` entries are drawn from
// the file's own declared constraint vocabulary.
if (!fs.existsSync(REWRITE_RULES_PATH)) {
  issues.push(`rewrite-rules.yml not found at ${REWRITE_RULES_PATH}`);
} else {
  const rewriteDoc = yaml.load(fs.readFileSync(REWRITE_RULES_PATH, "utf8"));
  const knownConstraints = new Set(rewriteDoc.constraints || []);
  const rules = (rewriteDoc && rewriteDoc.rules) || [];
  for (const rule of rules) {
    for (const constraint of rule.relieves || []) {
      if (!knownConstraints.has(constraint)) {
        issues.push(
          `rewrite-rules.yml rule '${rule.key}': relieves '${constraint}' is not in the declared constraint vocabulary`
        );
      }
    }
  }
}

if (issues.length > 0) {
  console.error(`✗ grammar.yml VALIDATION FAILED — ${issues.length} issue(s):`);
  for (const i of issues) console.error("  " + i);
  process.exit(1);
}

const assemblyCount = (doc.assemblies || []).reduce((n, s) => n + (s.items || []).length, 0);
console.log(`✓ grammar.yml is valid`);
console.log(`  ${doc.primitives.length} primitives, ${assemblyCount} assemblies`);
console.log(`  ${(doc.known_collisions || []).length} documented intentional symbol collisions`);
console.log(`  0 cell collisions, 0 unresolved composition links, 0 unresolved assembly references`);
