#!/usr/bin/env node
/**
 * One-shot migration: extract the live data from periodic-table/index.html
 * and emit periodic-table/table.yml as the new source of truth.
 *
 * Usage:
 *   node periodic-table/scripts/migrate-html-to-yaml.mjs
 *
 * After this runs, the YAML is canonical. Re-running the migrator is safe
 * but will overwrite the YAML — only do that if you're recovering from a
 * corrupt YAML, since any hand-edits to the YAML will be lost.
 */

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..", "..");
const HTML_PATH = path.join(REPO, "periodic-table", "index.html");
const YAML_PATH = path.join(REPO, "periodic-table", "table.yml");

// ── 1. Extract `blocks`, `rowLabels`, `elements` from the script tag ──
const html = fs.readFileSync(HTML_PATH, "utf8");
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) throw new Error("No <script> block found in index.html");
const scriptText = scriptMatch[1];

// Take only the data declarations (not the render code, which references DOM)
const dataMatch = scriptText.match(/^const blocks[\s\S]*?^\];$/m);
if (!dataMatch) throw new Error("Could not locate `const blocks ... const elements [...]` block");

const ctx = vm.createContext({});
vm.runInContext(
  dataMatch[0] + "\nthis.blocks = blocks; this.rowLabels = rowLabels; this.elements = elements;",
  ctx,
);
const blocksRaw = ctx.blocks;
const rowLabelsRaw = ctx.rowLabels;
const elementsRaw = ctx.elements;

console.log(`Extracted: ${elementsRaw.length} elements, ${Object.keys(blocksRaw).length} blocks, ${rowLabelsRaw.length} row labels`);

// ── 2. Translate the JS literals into YAML-friendly typed structures ──

// Block-key order should match the visual left-to-right order, which the
// original `cols` arrays imply. Sort by min(cols).
const blockKeyOrder = Object.entries(blocksRaw)
  .sort((a, b) => Math.min(...a[1].cols) - Math.min(...b[1].cols))
  .map(([k]) => k);

const blocks = blockKeyOrder.map((k) => ({
  key: k,
  name: blocksRaw[k].name,
  sub: blocksRaw[k].sub,
  color: blocksRaw[k].color,
  cols: blocksRaw[k].cols,
}));

// Convert the row labels from a flat string array into objects with index/key.
// `key` is a slug derived from the name.
const rows = rowLabelsRaw.map((name, i) => ({
  index: i + 1,
  key: name.toLowerCase().replace(/[^a-z]/g, "_"),
  name,
}));

// Each tuple is [num, sym, name, block, row, col, year, desc, bonds, why]
const elements = elementsRaw.map((tuple) => {
  const [id, sym, name, block, row, col, year, desc, bonds, why] = tuple;
  return {
    id,
    sym,
    name,
    block,
    row,
    col,
    year: year === "—" || year === "" ? null : year,
    desc,
    bonds: bonds || [],
    why,
  };
});

// Sort elements: by row, then col, then id (stable for ties).
elements.sort((a, b) => a.row - b.row || a.col - b.col || a.id - b.id);

// ── 3. Extract compounds from the .c-card markup ──
// The structure is:
//   <h3>Section Name <span class="isomer-hint">hint text</span></h3>
//   <div class="compound-grid">
//     <div class="c-card">
//       <div class="c-name">CompoundName</div>
//       <div class="c-formula"><span>Sym</span><sub>sub</sub> ... </div>
//     </div>
//     ...
//   </div>

// Pull the entire compounds container so we can walk it section by section.
const compoundsBlockMatch = html.match(/<h2>Molecular ML[\s\S]*?<\/div>\s*<\/div>\s*\n*<div class="overlay"/);
if (!compoundsBlockMatch) throw new Error("Could not locate the Molecular ML compounds block");
const compoundsBlock = compoundsBlockMatch[0];

// Walk h3 by h3 to identify sections
const sectionRegex = /<h3>([^<]+?)(?:\s*<span class="isomer-hint">([^<]+)<\/span>)?\s*<\/h3>([\s\S]*?)(?=<h3>|<\/div>\s*<\/div>\s*\n*<div class="overlay")/g;
const compounds = [];
let m;
while ((m = sectionRegex.exec(compoundsBlock)) !== null) {
  const sectionName = m[1].trim();
  const hint = m[2] ? m[2].trim() : undefined;
  const sectionBody = m[3];

  // Now find each c-card inside the section body
  const cardRegex = /<div class="c-card">([\s\S]*?)<\/div>\s*<\/div>/g;
  // Actually the structure has nested divs. Easier: match c-name + c-formula pairs.
  const itemRegex = /<div class="c-name">([\s\S]*?)<\/div>\s*<div class="c-formula">([\s\S]*?)<\/div>/g;
  const items = [];
  let im;
  while ((im = itemRegex.exec(sectionBody)) !== null) {
    const name = stripHtml(im[1]).trim();
    const formula = formulaMarkupToString(im[2]);
    items.push({ name, formula });
  }
  if (items.length > 0) {
    compounds.push({ section: sectionName, ...(hint ? { hint } : {}), items });
  }
}

console.log(`Extracted: ${compounds.length} compound sections, ${compounds.reduce((n, s) => n + s.items.length, 0)} compounds total`);

// ── 4. Identify symbol collisions for the known_collisions block ──
const symGroups = {};
for (const e of elements) {
  (symGroups[e.sym] ||= []).push(e);
}
const knownCollisions = Object.entries(symGroups)
  .filter(([, list]) => list.length > 1)
  .map(([sym, list]) => ({
    sym,
    ids: list.map((e) => e.id).sort((a, b) => a - b),
    note: list.map((e) => `${e.name} (#${e.id}, row ${e.row} col ${e.col}, ${e.block})`).join(" + "),
  }));

console.log(`Documented ${knownCollisions.length} intentional symbol collisions`);

// ── 5. Validate before writing ──
const issues = [];

// Cell collisions (should be zero given the current data)
const cellGroups = {};
for (const e of elements) {
  const key = `${e.row},${e.col}`;
  (cellGroups[key] ||= []).push(e);
}
for (const [key, list] of Object.entries(cellGroups)) {
  if (list.length > 1) {
    issues.push(`Cell collision at (${key}): ${list.map((e) => `#${e.id} ${e.sym}`).join(", ")}`);
  }
}

// Bond resolution
const knownSyms = new Set(elements.map((e) => e.sym));
for (const e of elements) {
  for (const b of e.bonds) {
    if (!knownSyms.has(b)) issues.push(`Element #${e.id} ${e.sym}: unresolved bond '${b}'`);
  }
}

// Formula resolution — every two-letter [A-Z][a-z] substring in a formula
// must reference a known element symbol.
for (const section of compounds) {
  for (const item of section.items) {
    const refs = extractFormulaSymbols(item.formula);
    for (const ref of refs) {
      if (!knownSyms.has(ref)) {
        issues.push(`Compound "${item.name}" references unknown symbol '${ref}' in formula: ${item.formula}`);
      }
    }
  }
}

if (issues.length > 0) {
  console.error("VALIDATION FAILED:");
  issues.forEach((i) => console.error("  " + i));
  process.exit(1);
}
console.log("Validation passed (cells, bonds, formula references)");

// ── 6. Emit YAML by hand ──
const out = emitYaml({
  version: "0.2",
  title: "The Periodic Table of Machine Learning Systems",
  subtitle: "Two fundamental axes — abstraction layer and information-processing role — organize ML concepts the way electron shells and valence organize chemistry.",
  blocks,
  rows,
  elements,
  compounds,
  known_collisions: knownCollisions,
});

fs.writeFileSync(YAML_PATH, out);
console.log(`\nWrote ${YAML_PATH}`);
console.log(`  ${elements.length} elements`);
console.log(`  ${compounds.reduce((n, s) => n + s.items.length, 0)} compounds across ${compounds.length} sections`);
console.log(`  ${knownCollisions.length} documented symbol collisions`);

// ════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════

function stripHtml(s) {
  return s.replace(/<[^>]+>/g, "");
}

/**
 * Convert a .c-formula's inner HTML back into the string form used in
 * table.yml. Examples:
 *
 *   "<span>Eb</span> → [(<span>At</span> ∥ <span>Mk</span>) → ..."
 *   →  "Eb → [(At ∥ Mk) → ..."
 *
 *   "<span>Tk</span><sub>patch</sub> → <span>Eb</span>"
 *   →  "Tk_patch → Eb"
 *
 *   "<span>Tk</span><sub>img</sub> ∥ <span>Tk</span><sub>txt</sub>"
 *   →  "Tk_img ∥ Tk_txt"
 *
 * Strategy: walk the markup left to right. When we see <span>X</span>,
 * emit X. When we see <sub>...</sub> immediately after a <span>, emit
 * "_<sub-content>". Otherwise emit literal characters.
 */
function formulaMarkupToString(markup) {
  // Normalize whitespace inside the markup but preserve significant spaces
  let s = markup.replace(/\s+/g, " ").trim();
  // Replace <span>X</span> -> X
  s = s.replace(/<span[^>]*>([^<]*)<\/span>/g, "$1");
  // Replace <sub>X</sub> -> _X (subscript marker)
  s = s.replace(/<sub[^>]*>([^<]*)<\/sub>/g, "_$1");
  // Strip any other HTML
  s = s.replace(/<[^>]+>/g, "");
  // Collapse double spaces around operators
  s = s.replace(/\s+/g, " ").trim();
  return s;
}

/**
 * Inverse of the parser used downstream — extract every symbol reference
 * from a formula string. A symbol is a two-letter token of the form
 * `[A-Z][a-z]` that's not immediately preceded or followed by another
 * letter (so we don't mis-tokenize words inside subscripts).
 */
function extractFormulaSymbols(formula) {
  const out = [];
  // Strip subscripts first: `Tk_patch` → keep `Tk`, drop `_patch`
  // We do this by splitting on `_`: anything after an underscore until
  // the next non-word char is subscript text.
  const cleaned = formula.replace(/_[A-Za-z]+/g, "");
  const re = /(?<![A-Za-z])([A-Z][a-z])(?![A-Za-z])/g;
  let m;
  while ((m = re.exec(cleaned)) !== null) out.push(m[1]);
  return out;
}

// ────────────────────────────────────────────────────────────────────────
// Tiny hand-written YAML emitter. Only handles the shape we produce.
// ────────────────────────────────────────────────────────────────────────

function emitYaml(doc) {
  let out = "";
  out += `# Periodic Table of ML Systems — single source of truth.\n`;
  out += `# Generated from periodic-table/index.html by scripts/migrate-html-to-yaml.mjs\n`;
  out += `# Edit this file directly. Run \`make all\` (in periodic-table/) to regenerate\n`;
  out += `# index.html and the StaffML React data file from this YAML.\n\n`;
  out += `version: ${quote(doc.version)}\n`;
  out += `title: ${quote(doc.title)}\n`;
  out += `subtitle: ${quote(doc.subtitle)}\n\n`;

  out += "# ─── Blocks (information-processing roles, columns) ───────────\n";
  out += "blocks:\n";
  for (const b of doc.blocks) {
    out += `  - key: ${b.key}\n`;
    out += `    name: ${quote(b.name)}\n`;
    out += `    sub: ${quote(b.sub)}\n`;
    out += `    color: ${quote(b.color)}\n`;
    out += `    cols: [${b.cols.join(", ")}]\n`;
  }
  out += "\n";

  out += "# ─── Rows (abstraction layers) ────────────────────────────────\n";
  out += "rows:\n";
  for (const r of doc.rows) {
    out += `  - index: ${r.index}\n`;
    out += `    key: ${quote(r.key)}\n`;
    out += `    name: ${quote(r.name)}\n`;
  }
  out += "\n";

  out += `# ─── Elements (${doc.elements.length} total) ─────────────────────────────\n`;
  out += "elements:\n";
  for (const e of doc.elements) {
    out += `  - id: ${e.id}\n`;
    out += `    sym: ${quote(e.sym)}\n`;
    out += `    name: ${quote(e.name)}\n`;
    out += `    block: ${e.block}\n`;
    out += `    row: ${e.row}\n`;
    out += `    col: ${e.col}\n`;
    out += `    year: ${e.year === null ? "null" : quote(e.year)}\n`;
    out += `    desc: ${quote(e.desc)}\n`;
    if (e.bonds.length === 0) {
      out += `    bonds: []\n`;
    } else {
      out += `    bonds: [${e.bonds.map(quote).join(", ")}]\n`;
    }
    out += `    why: ${quote(e.why)}\n`;
  }
  out += "\n";

  const compoundCount = doc.compounds.reduce((n, s) => n + s.items.length, 0);
  out += `# ─── Compounds (${compoundCount} total across ${doc.compounds.length} sections) ──────────────\n`;
  out += "# Formulas are written in the same notation as the paper:\n";
  out += "#   Sym -> two-letter element reference (resolves to elements[*].sym)\n";
  out += "#   _xxx -> subscript on the preceding token (e.g. Tk_patch, ]ᴺ_enc)\n";
  out += "#   ->   sequential composition\n";
  out += "#   ‖    parallel\n";
  out += "#   ?    conditional\n";
  out += "#   ⇌    adversarial\n";
  out += "#   ↺    feedback loop\n";
  out += "#   [...]ᴺ  repeated block\n";
  out += "compounds:\n";
  for (const section of doc.compounds) {
    out += `  - section: ${quote(section.section)}\n`;
    if (section.hint) out += `    hint: ${quote(section.hint)}\n`;
    out += `    items:\n`;
    for (const item of section.items) {
      out += `      - name: ${quote(item.name)}\n`;
      out += `        formula: ${quote(item.formula)}\n`;
    }
  }
  out += "\n";

  if (doc.known_collisions.length > 0) {
    out += "# ─── Documented intentional symbol collisions ─────────────────\n";
    out += "# Lookup behavior is last-wins; consumers may disambiguate by id\n";
    out += "# or by (row, col). The validator allows only collisions listed here.\n";
    out += "known_collisions:\n";
    for (const c of doc.known_collisions) {
      out += `  - sym: ${quote(c.sym)}\n`;
      out += `    ids: [${c.ids.join(", ")}]\n`;
      out += `    note: ${quote(c.note)}\n`;
    }
  }

  return out;
}

/**
 * Conservative YAML string quoter. Always uses double quotes and escapes
 * the few characters that matter inside double-quoted YAML strings.
 */
function quote(s) {
  if (s === null || s === undefined) return "null";
  const str = String(s);
  return '"' + str.replace(/\\/g, "\\\\").replace(/"/g, '\\"') + '"';
}
