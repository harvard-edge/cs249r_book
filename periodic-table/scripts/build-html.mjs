#!/usr/bin/env node
/**
 * Build periodic-table/index.html from periodic-table/table.yml.
 *
 * Strategy: the existing index.html is the template. Data sections are
 * marked with sentinel HTML comments (`<!-- @gen:blocks -->` etc.). On
 * each run we replace the contents between the sentinels with freshly
 * emitted markup; everything else (CSS, render JS, prose) is preserved.
 *
 * On the FIRST run (before sentinels exist) we use a one-time bootstrap
 * that finds the current data definitions by their familiar `const ... =`
 * patterns and inserts the sentinel comments around them.
 *
 * Usage: node periodic-table/scripts/build-html.mjs
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..", "..");
const HTML_PATH = path.join(REPO, "periodic-table", "index.html");
const YAML_PATH = path.join(REPO, "periodic-table", "table.yml");

// ── 1. Load and validate ────────────────────────────────────────────────
const doc = yaml.load(fs.readFileSync(YAML_PATH, "utf8"));
validate(doc);

const KNOWN_SYMS = new Set(doc.elements.map((e) => e.sym));

// ── 2. Render the data sections from YAML ───────────────────────────────
const renderedBlocks = renderBlocksJs(doc.blocks, doc.rows);
const renderedElements = renderElementsJs(doc.elements);
const renderedCompounds = renderCompoundsHtml(doc.compounds, KNOWN_SYMS);

// ── 3. Read the current HTML, bootstrap sentinels if needed, patch ──────
let html = fs.readFileSync(HTML_PATH, "utf8");
html = bootstrapSentinels(html);

html = replaceBetween(html, "@gen:blocks", renderedBlocks);
html = replaceBetween(html, "@gen:elements", renderedElements);
html = replaceBetween(html, "@gen:compounds", renderedCompounds);

fs.writeFileSync(HTML_PATH, html);

console.log(`Wrote ${HTML_PATH}`);
console.log(`  ${doc.elements.length} elements`);
console.log(`  ${doc.compounds.reduce((n, s) => n + s.items.length, 0)} compounds across ${doc.compounds.length} sections`);
console.log(`  ${(doc.known_collisions || []).length} documented symbol collisions`);

// ════════════════════════════════════════════════════════════════════════
// Validation
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
// Renderers
// ════════════════════════════════════════════════════════════════════════
function renderBlocksJs(blocks, rows) {
  let out = "const blocks = {\n";
  for (const b of blocks) {
    out += `  ${b.key}: { name:${jsStr(b.name)}, sub:${jsStr(b.sub)}, color:${jsStr(b.color)}, cols:[${b.cols.join(",")}] },\n`;
  }
  out += "};\n";
  out += `const rowLabels = [${rows.map((r) => jsStr(r.name)).join(",")}];`;
  return out;
}

function renderElementsJs(elements) {
  let out = "// [num, sym, name, block, row, col, year, desc, bonds[], whyHere]\n";
  out += "const elements = [\n";

  const byRow = {};
  for (const e of elements) (byRow[e.row] ||= []).push(e);

  const rowComments = {
    1: "// Row 0: Data (The Raw Material)",
    2: "// Row 1: Math (The Theoretical Bedrock)",
    3: "// Row 2: Algorithms (The Operations)",
    4: "// Row 3: Architecture (The Topologies)",
    5: "// Row 4: Optimization (The Physics of Efficiency)",
    6: "// Row 5: Runtime (Software Execution Primitives)",
    7: "// Row 6: Hardware (Silicon Primitives)",
    8: "// Row 7: Production (Fleet Primitives)",
  };

  const rowKeys = Object.keys(byRow).map(Number).sort((a, b) => a - b);
  for (const row of rowKeys) {
    out += `  ${rowComments[row] || `// Row ${row}`}\n`;
    for (const e of byRow[row]) {
      const yearLit = e.year === null || e.year === undefined ? `'—'` : jsStr(e.year);
      const bondsLit = `[${(e.bonds || []).map(jsStr).join(",")}]`;
      out += `  [${e.id},${jsStr(e.sym)},${jsStr(e.name)},${jsStr(e.block)},${e.row},${e.col},${yearLit},${jsStr(e.desc)},${bondsLit},${jsStr(e.why)}],\n`;
    }
    out += "\n";
  }
  // Drop the trailing blank line and trailing comma to keep diffs minimal.
  out = out.replace(/,\n\n$/, "\n");
  out += "];";
  return out;
}

function renderCompoundsHtml(compounds, knownSyms) {
  let out = "";
  for (const section of compounds) {
    out += `\n  <h3>${escapeHtml(section.section)}`;
    if (section.hint) {
      out += ` <span class="isomer-hint">${escapeHtml(section.hint)}</span>`;
    }
    out += "</h3>\n";
    out += `  <div class="compound-grid">\n`;
    for (const item of section.items) {
      out += `    <div class="c-card">\n`;
      out += `      <div class="c-name">${escapeHtml(item.name)}</div>\n`;
      out += `      <div class="c-formula">${formulaToHtml(item.formula, knownSyms)}</div>\n`;
      out += `    </div>\n`;
    }
    out += `  </div>\n`;
  }
  return out;
}

/**
 * Convert a formula string into the markup the existing CSS/JS expects:
 *   <span>Sym</span> for a known symbol
 *   <span>Sym</span><sub>xxx</sub> for a symbol with subscript
 *   literal text for everything else
 */
function formulaToHtml(formula, knownSyms) {
  let out = "";
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
      if (formula[i + 2] === "_") {
        const subMatch = formula.slice(i + 3).match(/^[A-Za-z]+/);
        if (subMatch) {
          out += `<span>${two}</span><sub>${subMatch[0]}</sub>`;
          i += 3 + subMatch[0].length;
          continue;
        }
      }
      out += `<span>${two}</span>`;
      i += 2;
    } else if (formula[i] === "_") {
      // Subscript on a non-symbol position (like `]ᴺ_enc`, `]ᴺ_dec`).
      // Original HTML used <sub> here too, so emit a <sub> tag.
      const subMatch = formula.slice(i + 1).match(/^[A-Za-z]+/);
      if (subMatch) {
        out += `<sub>${subMatch[0]}</sub>`;
        i += 1 + subMatch[0].length;
        continue;
      }
      out += escapeHtml(formula[i]);
      i += 1;
    } else {
      out += escapeHtml(formula[i]);
      i += 1;
    }
  }
  return out;
}

// ════════════════════════════════════════════════════════════════════════
// Sentinel bootstrap & replace-between
// ════════════════════════════════════════════════════════════════════════
function bootstrapSentinels(html) {
  // Blocks + rowLabels
  if (!html.includes("@gen:blocks")) {
    const blocksRe = /(const blocks = \{[\s\S]*?\};\s*\nconst rowLabels = \[[^\]]*\];)/;
    if (!blocksRe.test(html)) throw new Error("Bootstrap: could not locate `const blocks ... const rowLabels` to wrap with sentinel");
    html = html.replace(blocksRe, "/* @gen:blocks */\n$1\n/* @end:blocks */");
  }
  // Elements
  if (!html.includes("@gen:elements")) {
    const elemsRe = /(\/\/ \[num, sym, name, block[\s\S]*?const elements = \[[\s\S]*?^\];)/m;
    if (!elemsRe.test(html)) throw new Error("Bootstrap: could not locate `const elements = [...]` to wrap with sentinel");
    html = html.replace(elemsRe, "/* @gen:elements */\n$1\n/* @end:elements */");
  }
  // Compounds: wrap from the first <h3> after the .c-legend through the
  // last </div> before .container.compounds closes.
  if (!html.includes("@gen:compounds")) {
    const compoundsRe = /(<\/div>\s*\n\s*)(<h3>[\s\S]*?<\/div>)(\s*<\/div>\s*\n*<div class="overlay")/;
    if (!compoundsRe.test(html)) throw new Error("Bootstrap: could not locate compound sections to wrap with sentinel");
    html = html.replace(compoundsRe, "$1<!-- @gen:compounds -->\n$2\n  <!-- @end:compounds -->$3");
  }
  return html;
}

function replaceBetween(html, name, replacement) {
  // JS data sections use /* */ wrapping; HTML compound section uses <!-- -->
  const isJs = name === "@gen:blocks" || name === "@gen:elements";
  const start = isJs ? `/* ${name} */` : `<!-- ${name} -->`;
  const endName = name.replace("@gen:", "@end:");
  const end = isJs ? `/* ${endName} */` : `<!-- ${endName} -->`;

  const startIdx = html.indexOf(start);
  if (startIdx < 0) throw new Error(`Sentinel start not found: ${start}`);
  const endIdx = html.indexOf(end, startIdx + start.length);
  if (endIdx < 0) throw new Error(`Sentinel end not found: ${end}`);

  const before = html.slice(0, startIdx + start.length);
  const after = html.slice(endIdx);
  return before + "\n" + replacement + "\n" + after;
}

// ════════════════════════════════════════════════════════════════════════
// String helpers
// ════════════════════════════════════════════════════════════════════════
function jsStr(s) {
  return "'" + String(s).replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n") + "'";
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
