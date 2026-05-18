#!/usr/bin/env node
/**
 * One-shot migration: extract the live data from design-grammar/index.html
 * and emit design-grammar/grammar.yml as the new source of truth.
 *
 * Usage:
 *   node design-grammar/scripts/migrate-html-to-yaml.mjs
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
const HTML_PATH = path.join(REPO, "design-grammar", "index.html");
const YAML_PATH = path.join(REPO, "design-grammar", "grammar.yml");

// ── 1. Extract `roles`, `layerLabels`, `primitives` from the script tag ──
const html = fs.readFileSync(HTML_PATH, "utf8");
// Case-insensitive, permissive </script ...> close (HTML5 tokenization); avoid
// naive /<script>/ which misses <SCRIPT> and odd close tags.
const scriptMatch = html.match(/<script\b[^>]*>([\s\S]*?)<\/script\b[^>]*>/i);
if (!scriptMatch) throw new Error("No <script> tag found in index.html");
const scriptText = scriptMatch[1];

// Take only the data declarations (not the render code, which references DOM)
const dataMatch = scriptText.match(/^const roles[\s\S]*?^\];$/m);
if (!dataMatch) throw new Error("Could not locate `const roles ... const primitives [...]` data");

const ctx = vm.createContext({});
vm.runInContext(
  dataMatch[0] + "\nthis.roles = roles; this.layerLabels = layerLabels; this.primitives = primitives;",
  ctx,
);
const rolesRaw = ctx.roles;
const layerLabelsRaw = ctx.layerLabels;
const primitivesRaw = ctx.primitives;

console.log(`Extracted: ${primitivesRaw.length} primitives, ${Object.keys(rolesRaw).length} roles, ${layerLabelsRaw.length} layer labels`);

// ── 2. Translate the JS literals into YAML-friendly typed structures ──

// Role-key order should match the visual left-to-right order, which the
// original `cols` arrays imply. Sort by min(cols).
const roleKeyOrder = Object.entries(rolesRaw)
  .sort((a, b) => Math.min(...a[1].cols) - Math.min(...b[1].cols))
  .map(([k]) => k);

const roles = roleKeyOrder.map((k) => ({
  key: k,
  name: rolesRaw[k].name,
  sub: rolesRaw[k].sub,
  color: rolesRaw[k].color,
  cols: rolesRaw[k].cols,
}));

// Convert the layer labels from a flat string array into objects with index/key.
// `key` is a slug derived from the name.
const layers = layerLabelsRaw.map((name, i) => ({
  index: i + 1,
  key: name.toLowerCase().replace(/[^a-z]/g, "_"),
  name,
}));

// Each tuple is [num, sym, name, role, layer, col, year, description, links, rationale]
const primitives = primitivesRaw.map((tuple) => {
  const [id, sym, name, role, layer, col, year, description, composition_links, rationale] = tuple;
  return {
    id,
    sym,
    name,
    role,
    layer,
    col,
    year: year === "—" || year === "" ? null : year,
    description,
    composition_links: composition_links || [],
    rationale,
  };
});

// Sort primitives: by layer, then col, then id (stable for ties).
primitives.sort((a, b) => a.layer - b.layer || a.col - b.col || a.id - b.id);

// ── 3. Extract assemblies from the .assembly-card markup ──
// The structure is:
//   <h3>Section Name <span class="isomer-hint">hint text</span></h3>
//   <div class="assembly-grid">
//     <div class="assembly-card">
//       <div class="assembly-name">AssemblyName</div>
//       <div class="assembly-expression"><span>Sym</span><sub>sub</sub> ... </div>
//     </div>
//     ...
//   </div>

// Pull the entire assemblies container so we can walk it section by section.
const assembliesBlockMatch = html.match(/<h2>System Assemblies<\/h2>[\s\S]*?<\/div>\s*<\/div>\s*\n*<div class="overlay"/);
if (!assembliesBlockMatch) throw new Error("Could not locate the system assemblies block");
const assembliesBlock = assembliesBlockMatch[0];

// Walk h3 by h3 to identify sections
const sectionRegex = /<h3>([^<]+?)(?:\s*<span class="isomer-hint">([^<]+)<\/span>)?\s*<\/h3>([\s\S]*?)(?=<h3>|<\/div>\s*<\/div>\s*\n*<div class="overlay")/g;
const assemblies = [];
let m;
while ((m = sectionRegex.exec(assembliesBlock)) !== null) {
  const sectionName = m[1].trim();
  const hint = m[2] ? m[2].trim() : undefined;
  const sectionBody = m[3];

  // Match assembly-name + assembly-expression pairs.
  const itemRegex = /<div class="assembly-name">([\s\S]*?)<\/div>\s*<div class="assembly-expression">([\s\S]*?)<\/div>/g;
  const items = [];
  let im;
  while ((im = itemRegex.exec(sectionBody)) !== null) {
    const name = stripHtml(im[1]).trim();
    const expression = expressionMarkupToString(im[2]);
    items.push({ name, expression });
  }
  if (items.length > 0) {
    assemblies.push({ section: sectionName, ...(hint ? { hint } : {}), items });
  }
}

console.log(`Extracted: ${assemblies.length} assembly sections, ${assemblies.reduce((n, s) => n + s.items.length, 0)} assemblies total`);

// ── 4. Identify symbol collisions for the known_collisions role ──
const symGroups = {};
for (const e of primitives) {
  (symGroups[e.sym] ||= []).push(e);
}
const knownCollisions = Object.entries(symGroups)
  .filter(([, list]) => list.length > 1)
  .map(([sym, list]) => ({
    sym,
    ids: list.map((e) => e.id).sort((a, b) => a - b),
    note: list.map((e) => `${e.name} (#${e.id}, layer ${e.layer} col ${e.col}, ${e.role})`).join(" + "),
  }));

console.log(`Documented ${knownCollisions.length} intentional symbol collisions`);

// ── 5. Validate before writing ──
const issues = [];

// Cell collisions (should be zero given the current data)
const cellGroups = {};
for (const e of primitives) {
  const key = `${e.layer},${e.col}`;
  (cellGroups[key] ||= []).push(e);
}
for (const [key, list] of Object.entries(cellGroups)) {
  if (list.length > 1) {
    issues.push(`Cell collision at (${key}): ${list.map((e) => `#${e.id} ${e.sym}`).join(", ")}`);
  }
}

// Composition-link resolution
const knownSyms = new Set(primitives.map((e) => e.sym));
for (const e of primitives) {
  for (const b of e.composition_links) {
    if (!knownSyms.has(b)) issues.push(`Primitive #${e.id} ${e.sym}: unresolved composition link '${b}'`);
  }
}

// Expression resolution: every two-letter [A-Z][a-z] substring in a expression
// must reference a known primitive symbol.
for (const section of assemblies) {
  for (const item of section.items) {
    const refs = extractExpressionSymbols(item.expression);
    for (const ref of refs) {
      if (!knownSyms.has(ref)) {
        issues.push(`Assembly "${item.name}" references unknown symbol '${ref}' in expression: ${item.expression}`);
      }
    }
  }
}

if (issues.length > 0) {
  console.error("VALIDATION FAILED:");
  issues.forEach((i) => console.error("  " + i));
  process.exit(1);
}
console.log("Validation passed (cells, composition links, assembly references)");

// ── 6. Emit YAML by hand ──
const out = emitYaml({
  version: "0.2",
  title: "ML Systems Design Grammar",
  subtitle: "Stable primitives, physical constraints, and reusable rewrite rules for deriving ML systems from first principles.",
  roles,
  layers,
  primitives,
  assemblies,
  known_collisions: knownCollisions,
});

fs.writeFileSync(YAML_PATH, out);
console.log(`\nWrote ${YAML_PATH}`);
console.log(`  ${primitives.length} primitives`);
console.log(`  ${assemblies.reduce((n, s) => n + s.items.length, 0)} assemblies across ${assemblies.length} sections`);
console.log(`  ${knownCollisions.length} documented symbol collisions`);

// ════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════

function stripHtml(s) {
  return stripTagsUntilStable(s);
}

/** Remove HTML-like tags; repeat until stable so nested / partial tags cannot re-form. */
function stripTagsUntilStable(s) {
  let prev;
  do {
    prev = s;
    s = s.replace(/<[^>]+>/g, "");
  } while (s !== prev);
  return s;
}

/**
 * Convert an .assembly-expression's inner HTML back into the string form used in
 * grammar.yml. Examples:
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
function expressionMarkupToString(markup) {
  // Normalize whitespace inside the markup but preserve significant spaces
  let s = markup.replace(/\s+/g, " ").trim();
  // Replace <span>X</span> -> X
  s = s.replace(/<span[^>]*>([^<]*)<\/span>/g, "$1");
  // Replace <sub>X</sub> -> _X (subscript marker)
  s = s.replace(/<sub[^>]*>([^<]*)<\/sub>/g, "_$1");
  // Strip any other HTML (loop: single pass can re-expose tag-like text)
  s = stripTagsUntilStable(s);
  // Collapse double spaces around operators
  s = s.replace(/\s+/g, " ").trim();
  return s;
}

/**
 * Inverse of the parser used downstream — extract every symbol reference
 * from a expression string. A symbol is a two-letter token of the form
 * `[A-Z][a-z]` that's not immediately preceded or followed by another
 * letter (so we don't mis-tokenize words inside subscripts).
 */
function extractExpressionSymbols(expression) {
  const out = [];
  // Strip subscripts first: `Tk_patch` → keep `Tk`, drop `_patch`
  // We do this by splitting on `_`: anything after an underscore until
  // the next non-word char is subscript text.
  const cleaned = expression.replace(/_[A-Za-z]+/g, "");
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
  out += `# ML Systems Design Grammar -- single source of truth.\n`;
  out += `# Generated from design-grammar/index.html by scripts/migrate-html-to-yaml.mjs\n`;
  out += `# Edit this file directly. Run \`make all\` (in design-grammar/) to regenerate\n`;
  out += `# index.html and the StaffML React data file from this YAML.\n\n`;
  out += `version: ${quote(doc.version)}\n`;
  out += `title: ${quote(doc.title)}\n`;
  out += `subtitle: ${quote(doc.subtitle)}\n\n`;

  out += "# ─── Roles (information-processing roles, columns) ───────────\n";
  out += "roles:\n";
  for (const b of doc.roles) {
    out += `  - key: ${b.key}\n`;
    out += `    name: ${quote(b.name)}\n`;
    out += `    sub: ${quote(b.sub)}\n`;
    out += `    color: ${quote(b.color)}\n`;
    out += `    cols: [${b.cols.join(", ")}]\n`;
  }
  out += "\n";

  out += "# ─── Layers (abstraction layers) ────────────────────────────────\n";
  out += "layers:\n";
  for (const r of doc.layers) {
    out += `  - index: ${r.index}\n`;
    out += `    key: ${quote(r.key)}\n`;
    out += `    name: ${quote(r.name)}\n`;
  }
  out += "\n";

  out += `# ─── Primitives (${doc.primitives.length} total; schema key: primitives) ─────────────────────────────\n`;
  out += "primitives:\n";
  for (const e of doc.primitives) {
    out += `  - id: ${e.id}\n`;
    out += `    sym: ${quote(e.sym)}\n`;
    out += `    name: ${quote(e.name)}\n`;
    out += `    role: ${e.role}\n`;
    out += `    layer: ${e.layer}\n`;
    out += `    col: ${e.col}\n`;
    out += `    year: ${e.year === null ? "null" : quote(e.year)}\n`;
    out += `    description: ${quote(e.description)}\n`;
    if (e.composition_links.length === 0) {
      out += `    composition_links: []\n`;
    } else {
      out += `    composition_links: [${e.composition_links.map(quote).join(", ")}]\n`;
    }
    out += `    rationale: ${quote(e.rationale)}\n`;
  }
  out += "\n";

  const assemblyCount = doc.assemblies.reduce((n, s) => n + s.items.length, 0);
  out += `# ─── Assemblies (${assemblyCount} total across ${doc.assemblies.length} sections; schema key: assemblies) ──────────────\n`;
  out += "# Expressions are written in the same notation as the paper:\n";
  out += "#   Sym -> two-letter primitive reference (resolves to primitives[*].sym)\n";
  out += "#   _xxx -> subscript on the preceding token (e.g. Tk_patch, ]ᴺ_enc)\n";
  out += "#   ->   sequential composition\n";
  out += "#   ‖    parallel\n";
  out += "#   ?    conditional\n";
  out += "#   ⇌    adversarial\n";
  out += "#   ↺    feedback loop\n";
  out += "#   [...]ᴺ  repeated role\n";
  out += "assemblies:\n";
  for (const section of doc.assemblies) {
    out += `  - section: ${quote(section.section)}\n`;
    if (section.hint) out += `    hint: ${quote(section.hint)}\n`;
    out += `    items:\n`;
    for (const item of section.items) {
      out += `      - name: ${quote(item.name)}\n`;
      out += `        expression: ${quote(item.expression)}\n`;
    }
  }
  out += "\n";

  if (doc.known_collisions.length > 0) {
    out += "# ─── Documented intentional symbol collisions ─────────────────\n";
    out += "# Lookup behavior is last-wins; consumers may disambiguate by id\n";
    out += "# or by (layer, col). The validator allows only collisions listed here.\n";
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
