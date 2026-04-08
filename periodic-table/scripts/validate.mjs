#!/usr/bin/env node
/**
 * Validate periodic-table/table.yml against the schema and the cross-reference
 * invariants (cell collisions, bond resolution, formula resolution, undocumented
 * symbol collisions). Exits 0 on success, 1 on failure.
 *
 * Usage: node periodic-table/scripts/validate.mjs
 *        make validate                 (from periodic-table/)
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const YAML_PATH = path.join(__dirname, "..", "table.yml");

const doc = yaml.load(fs.readFileSync(YAML_PATH, "utf8"));
const issues = [];

if (!doc || typeof doc !== "object") {
  console.error("FAIL: table.yml is empty or malformed");
  process.exit(1);
}
if (!Array.isArray(doc.elements) || doc.elements.length === 0) {
  console.error("FAIL: no elements defined");
  process.exit(1);
}

// ── Cross-reference invariants ──────────────────────────────────────────
const knownSyms = new Set(doc.elements.map((e) => e.sym));
const cellSeen = new Map();

for (const e of doc.elements) {
  // Schema-ish field validation (in case the user hand-edited and broke shape)
  if (typeof e.id !== "number") issues.push(`Element ${e.sym || "?"}: id must be a number`);
  if (typeof e.sym !== "string" || !/^[A-Z][a-z]$/.test(e.sym)) issues.push(`Element #${e.id}: sym '${e.sym}' must match [A-Z][a-z]`);
  if (typeof e.row !== "number" || e.row < 1 || e.row > 8) issues.push(`Element #${e.id} ${e.sym}: row ${e.row} out of range 1-8`);
  if (typeof e.col !== "number" || e.col < 1 || e.col > 18) issues.push(`Element #${e.id} ${e.sym}: col ${e.col} out of range 1-18`);
  if (!["R", "C", "X", "K", "M"].includes(e.block)) issues.push(`Element #${e.id} ${e.sym}: block '${e.block}' must be one of R/C/X/K/M`);

  // Cell collisions
  const k = `${e.row},${e.col}`;
  if (cellSeen.has(k)) {
    const prev = cellSeen.get(k);
    issues.push(`Cell collision at (${k}): #${prev.id} ${prev.sym} and #${e.id} ${e.sym}`);
  } else {
    cellSeen.set(k, e);
  }
}

// Bond resolution
for (const e of doc.elements) {
  for (const b of e.bonds || []) {
    if (!knownSyms.has(b)) issues.push(`Element #${e.id} ${e.sym}: bond '${b}' references unknown element`);
  }
}

// Symbol collisions: must be in known_collisions
const symCount = {};
for (const e of doc.elements) symCount[e.sym] = (symCount[e.sym] || 0) + 1;
const declared = new Set((doc.known_collisions || []).map((c) => c.sym));
for (const [sym, count] of Object.entries(symCount)) {
  if (count > 1 && !declared.has(sym)) {
    issues.push(`Undocumented symbol collision: '${sym}' appears ${count} times — add to known_collisions`);
  }
}

// Formula resolution: every two-letter [A-Z][a-z] token in any formula must
// resolve to a known element symbol.
for (const section of doc.compounds || []) {
  for (const item of section.items || []) {
    const cleaned = item.formula.replace(/_[A-Za-z]+/g, "");
    const re = /(?<![A-Za-z])([A-Z][a-z])(?![A-Za-z])/g;
    let m;
    while ((m = re.exec(cleaned)) !== null) {
      if (!knownSyms.has(m[1])) {
        issues.push(`Compound "${item.name}" references unknown symbol '${m[1]}': ${item.formula}`);
      }
    }
  }
}

if (issues.length > 0) {
  console.error(`✗ table.yml VALIDATION FAILED — ${issues.length} issue(s):`);
  for (const i of issues) console.error("  " + i);
  process.exit(1);
}

const compoundCount = (doc.compounds || []).reduce((n, s) => n + (s.items || []).length, 0);
console.log(`✓ table.yml is valid`);
console.log(`  ${doc.elements.length} elements, ${compoundCount} compounds`);
console.log(`  ${(doc.known_collisions || []).length} documented intentional symbol collisions`);
console.log(`  0 cell collisions, 0 unresolved bonds, 0 unresolved formula references`);
