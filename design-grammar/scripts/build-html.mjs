#!/usr/bin/env node
/**
 * Build design-grammar/index.html from design-grammar/grammar.yml.
 *
 * Strategy: the existing index.html is the template. Data sections are
 * marked with sentinel HTML comments (`<!-- @gen:roles -->` etc.). On
 * each run we replace the contents between the sentinels with freshly
 * emitted markup; everything else (CSS, render JS, prose) is preserved.
 *
 * On the FIRST run (before sentinels exist) we use a one-time bootstrap
 * that finds the current data definitions by their familiar `const ... =`
 * patterns and inserts the sentinel comments around them.
 *
 * Usage: node design-grammar/scripts/build-html.mjs
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..", "..");
const HTML_PATH = path.join(REPO, "design-grammar", "index.html");
const YAML_PATH = path.join(REPO, "design-grammar", "grammar.yml");

// ── 1. Load and validate ────────────────────────────────────────────────
const doc = yaml.load(fs.readFileSync(YAML_PATH, "utf8"));
validate(doc);

const KNOWN_SYMS = new Set(doc.primitives.map((e) => e.sym));

// ── 2. Render the data sections from YAML ───────────────────────────────
const renderedRoles = renderRolesJs(doc.roles, doc.layers);
const renderedPrimitives = renderPrimitivesJs(doc.primitives);
const renderedAssemblies = renderAssembliesHtml(doc.assemblies, KNOWN_SYMS);

// ── 3. Read the current HTML, bootstrap sentinels if needed, patch ──────
let html = fs.readFileSync(HTML_PATH, "utf8");
html = bootstrapSentinels(html);

html = html.replace(/<h1>.*?<\/h1>/, `<h1>${escapeHtml(doc.title)}</h1>`);
html = html.replace(/<p class="lead">.*?<\/p>/, `<p class="lead">${escapeHtml(doc.subtitle)}</p>`);
html = replaceBetween(html, "@gen:roles", renderedRoles);
html = replaceBetween(html, "@gen:primitives", renderedPrimitives);
html = replaceBetween(html, "@gen:assemblies", renderedAssemblies);

fs.writeFileSync(HTML_PATH, html);

console.log(`Wrote ${HTML_PATH}`);
console.log(`  ${doc.primitives.length} primitives`);
console.log(`  ${doc.assemblies.reduce((n, s) => n + s.items.length, 0)} assemblies across ${doc.assemblies.length} sections`);
console.log(`  ${(doc.known_collisions || []).length} documented symbol collisions`);

// ════════════════════════════════════════════════════════════════════════
// Validation
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
// Renderers
// ════════════════════════════════════════════════════════════════════════
function renderRolesJs(roles, layers) {
  let out = "const roles = {\n";
  for (const b of roles) {
    out += `  ${b.key}: { name:${jsStr(b.name)}, sub:${jsStr(b.sub)}, color:${jsStr(b.color)}, cols:[${b.cols.join(",")}] },\n`;
  }
  out += "};\n";
  out += `const layerLabels = [${layers.map((r) => jsStr(r.name)).join(",")}];`;
  return out;
}

function renderPrimitivesJs(primitives) {
  let out = "// [num, sym, name, role, layer, col, year, links[], rationale]\n";
  out += "const primitives = [\n";

  const byLayer = {};
  for (const e of primitives) (byLayer[e.layer] ||= []).push(e);

  const layerComments = {
    1: "// Layer 0: Data (The Raw Material)",
    2: "// Layer 1: Math (The Theoretical Bedrock)",
    3: "// Layer 2: Algorithms (The Operations)",
    4: "// Layer 3: Architecture (The Topologies)",
    5: "// Layer 4: Optimization (The Physics of Efficiency)",
    6: "// Layer 5: Runtime (Software Execution Primitives)",
    7: "// Layer 6: Hardware (Silicon Primitives)",
    8: "// Layer 7: Production (Fleet Primitives)",
  };

  const layerKeys = Object.keys(byLayer).map(Number).sort((a, b) => a - b);
  for (const layer of layerKeys) {
    out += `  ${layerComments[layer] || `// Layer ${layer}`}\n`;
    for (const e of byLayer[layer]) {
      const yearLit = e.year === null || e.year === undefined ? `'—'` : jsStr(e.year);
      const linksLit = `[${(e.composition_links || []).map(jsStr).join(",")}]`;
      out += `  [${e.id},${jsStr(e.sym)},${jsStr(e.name)},${jsStr(e.role)},${e.layer},${e.col},${yearLit},${jsStr(e.description)},${linksLit},${jsStr(e.rationale)}],\n`;
    }
    out += "\n";
  }
  // Drop the trailing blank line and trailing comma to keep diffs minimal.
  out = out.replace(/,\n\n$/, "\n");
  out += "];";
  return out;
}

function renderAssembliesHtml(assemblies, knownSyms) {
  let out = "";
  for (const section of assemblies) {
    out += `\n  <h3>${escapeHtml(section.section)}`;
    if (section.hint) {
      out += ` <span class="isomer-hint">${escapeHtml(section.hint)}</span>`;
    }
    out += "</h3>\n";
    out += `  <div class="assembly-grid">\n`;
    for (const item of section.items) {
      out += `    <div class="assembly-card">\n`;
      out += `      <div class="assembly-name">${escapeHtml(item.name)}</div>\n`;
      out += `      <div class="assembly-expression">${expressionToHtml(item.expression, knownSyms)}</div>\n`;
      out += `    </div>\n`;
    }
    out += `  </div>\n`;
  }
  return out;
}

/**
 * Convert a expression string into the markup the existing CSS/JS expects:
 *   <span>Sym</span> for a known symbol
 *   <span>Sym</span><sub>xxx</sub> for a symbol with subscript
 *   literal text for everything else
 */
function expressionToHtml(expression, knownSyms) {
  let out = "";
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
      if (expression[i + 2] === "_") {
        const subMatch = expression.slice(i + 3).match(/^[A-Za-z]+/);
        if (subMatch) {
          out += `<span>${two}</span><sub>${subMatch[0]}</sub>`;
          i += 3 + subMatch[0].length;
          continue;
        }
      }
      out += `<span>${two}</span>`;
      i += 2;
    } else if (expression[i] === "_") {
      // Subscript on a non-symbol position (like `]ᴺ_enc`, `]ᴺ_dec`).
      // Original HTML used <sub> here too, so emit a <sub> tag.
      const subMatch = expression.slice(i + 1).match(/^[A-Za-z]+/);
      if (subMatch) {
        out += `<sub>${subMatch[0]}</sub>`;
        i += 1 + subMatch[0].length;
        continue;
      }
      out += escapeHtml(expression[i]);
      i += 1;
    } else {
      out += escapeHtml(expression[i]);
      i += 1;
    }
  }
  return out;
}

// ════════════════════════════════════════════════════════════════════════
// Sentinel bootstrap & replace-between
// ════════════════════════════════════════════════════════════════════════
function bootstrapSentinels(html) {
  // Roles + layerLabels
  if (!html.includes("@gen:roles")) {
    const rolesRe = /(const roles = \{[\s\S]*?\};\s*\nconst layerLabels = \[[^\]]*\];)/;
    if (!rolesRe.test(html)) throw new Error("Bootstrap: could not locate `const roles ... const layerLabels` to wrap with sentinel");
    html = html.replace(rolesRe, "/* @gen:roles */\n$1\n/* @end:roles */");
  }
  // Primitives
  if (!html.includes("@gen:primitives")) {
    const elemsRe = /(\/\/ \[num, sym, name, role[\s\S]*?const primitives = \[[\s\S]*?^\];)/m;
    if (!elemsRe.test(html)) throw new Error("Bootstrap: could not locate `const primitives = [...]` to wrap with sentinel");
    html = html.replace(elemsRe, "/* @gen:primitives */\n$1\n/* @end:primitives */");
  }
  // Assemblies: wrap from the first <h3> after the assembly legend through the
  // last </div> before .container.assemblies closes.
  if (!html.includes("@gen:assemblies")) {
    const assembliesRe = /(<\/div>\s*\n\s*)(<h3>[\s\S]*?<\/div>)(\s*<\/div>\s*\n*<div class="overlay")/;
    if (!assembliesRe.test(html)) throw new Error("Bootstrap: could not locate assembly sections to wrap with sentinel");
    html = html.replace(assembliesRe, "$1<!-- @gen:assemblies -->\n$2\n  <!-- @end:assemblies -->$3");
  }
  return html;
}

function replaceBetween(html, name, replacement) {
  // JS data sections use /* */ wrapping; HTML assembly section uses <!-- -->
  const isJs = name === "@gen:roles" || name === "@gen:primitives";
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
