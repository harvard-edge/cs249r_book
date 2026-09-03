/**
 * Pure QMD validation functions — no VS Code API dependency.
 *
 * Each function accepts the full text of a `.qmd` file and returns
 * an array of {@link CheckResult} objects describing any issues found.
 * All checks are designed to run in <100 ms per file.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CheckResult {
  /** 0-based line number where the issue was found. */
  line: number;
  /** Human-readable description of the issue. */
  message: string;
  /** Severity level for UI display. */
  severity: 'error' | 'warning' | 'info';
  /** Machine-readable check identifier (e.g. 'duplicate-label'). */
  checkId: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** All Quarto label prefixes we care about. */
const LABEL_PREFIXES = ['fig', 'tbl', 'lst', 'sec', 'eq', 'thm', 'lem', 'cor', 'def', 'exm', 'exr'];

/** Matches label definitions: {#fig-xxx}, {#tbl-xxx}, {#sec-xxx}, etc. */
const LABEL_DEF_RE = new RegExp(
  `\\{#((?:${LABEL_PREFIXES.join('|')})-[\\w-]+)\\}`,
  'g',
);

/** Matches #| label: fig-xxx in Python code blocks. */
const CODE_LABEL_RE = /^#\|\s*label:\s*((?:fig|tbl|lst)-[\w-]+)/;

/** Matches @ref cross-references in prose. */
const REF_RE = new RegExp(
  `@((?:${LABEL_PREFIXES.join('|')})-[\\w-]+)`,
  'g',
);

/** Matches div fence openers: ::: or :::: (with optional class). */
const DIV_OPEN_RE = /^(:{3,})\s*\{/;

/** Matches div fence closers: ::: or :::: on a line by themselves. */
const DIV_CLOSE_RE = /^(:{3,})\s*$/;

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

/**
 * Detect duplicate label definitions within a single file.
 *
 * Scans for both attribute-style labels (`{#fig-xxx}`) and
 * code-block labels (`#| label: fig-xxx`).
 */
export function checkDuplicateLabels(text: string): CheckResult[] {
  const results: CheckResult[] = [];
  const seen = new Map<string, number>(); // label → first line (0-based)
  const lines = text.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Attribute-style labels
    LABEL_DEF_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = LABEL_DEF_RE.exec(line)) !== null) {
      const label = m[1];
      if (seen.has(label)) {
        results.push({
          line: i,
          message: `Duplicate label '#${label}' (first defined on line ${seen.get(label)! + 1})`,
          severity: 'error',
          checkId: 'duplicate-label',
        });
      } else {
        seen.set(label, i);
      }
    }

    // Code-block labels
    const cm = CODE_LABEL_RE.exec(line);
    if (cm) {
      const label = cm[1];
      if (seen.has(label)) {
        results.push({
          line: i,
          message: `Duplicate label '${label}' (first defined on line ${seen.get(label)! + 1})`,
          severity: 'error',
          checkId: 'duplicate-label',
        });
      } else {
        seen.set(label, i);
      }
    }
  }

  return results;
}

/**
 * Detect unclosed or mismatched div fences (`::: { }` / `::::`).
 *
 * Tracks a stack of openers and reports unmatched fences.
 */
export function checkUnclosedDivs(text: string): CheckResult[] {
  const results: CheckResult[] = [];
  const lines = text.split('\n');

  // Stack entries: [colon-count, line-number]
  const stack: Array<[number, number]> = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    const openMatch = DIV_OPEN_RE.exec(line);
    if (openMatch) {
      stack.push([openMatch[1].length, i]);
      continue;
    }

    const closeMatch = DIV_CLOSE_RE.exec(line);
    if (closeMatch) {
      const closeLen = closeMatch[1].length;
      if (stack.length === 0) {
        results.push({
          line: i,
          message: `Closing div fence '${''.padStart(closeLen, ':')}' has no matching opener`,
          severity: 'warning',
          checkId: 'unclosed-div',
        });
      } else {
        // Pop the most recent opener — Quarto allows ::: to close ::::
        stack.pop();
      }
    }
  }

  // Any remaining openers are unclosed
  for (const [colons, lineNum] of stack) {
    results.push({
      line: lineNum,
      message: `Div fence '${''.padStart(colons, ':')}' opened here is never closed`,
      severity: 'warning',
      checkId: 'unclosed-div',
    });
  }

  return results;
}

/**
 * Detect figure code blocks missing `fig-alt` metadata.
 *
 * Scans `{python}` blocks that have a `#| label: fig-*` directive
 * and warns if no `#| fig-alt:` directive is present.
 */
export function checkMissingAltText(text: string): CheckResult[] {
  const results: CheckResult[] = [];
  const lines = text.split('\n');

  let inPythonBlock = false;
  let blockStartLine = -1;
  let hasFigLabel = false;
  let hasAltText = false;
  let figLabel = '';

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (/^```\{python\}/.test(line)) {
      inPythonBlock = true;
      blockStartLine = i;
      hasFigLabel = false;
      hasAltText = false;
      figLabel = '';
      continue;
    }

    if (inPythonBlock && /^```\s*$/.test(line)) {
      // End of block — check
      if (hasFigLabel && !hasAltText) {
        results.push({
          line: blockStartLine,
          message: `Figure '${figLabel}' is missing '#| fig-alt:' accessibility text`,
          severity: 'warning',
          checkId: 'missing-alt-text',
        });
      }
      inPythonBlock = false;
      continue;
    }

    if (inPythonBlock) {
      const labelMatch = /^#\|\s*label:\s*(fig-[\w-]+)/.exec(line);
      if (labelMatch) {
        hasFigLabel = true;
        figLabel = labelMatch[1];
      }
      if (/^#\|\s*fig-alt:/.test(line)) {
        hasAltText = true;
      }
    }
  }

  return results;
}

/**
 * Detect cross-references (`@fig-xxx`, `@tbl-xxx`, etc.) that point to
 * labels not defined anywhere in the same file.
 *
 * Note: cross-chapter references are valid in Quarto and will produce
 * false positives here, so these are reported as `info` severity.
 */
export function checkBrokenInFileRefs(text: string): CheckResult[] {
  const results: CheckResult[] = [];
  const lines = text.split('\n');

  // First pass: collect all defined labels
  const definedLabels = new Set<string>();

  for (const line of lines) {
    LABEL_DEF_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = LABEL_DEF_RE.exec(line)) !== null) {
      definedLabels.add(m[1]);
    }

    const cm = CODE_LABEL_RE.exec(line);
    if (cm) {
      definedLabels.add(cm[1]);
    }
  }

  // Second pass: find references not in the defined set
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Skip lines inside code blocks (rough heuristic: starts with #|)
    if (/^\s*#\|/.test(line)) { continue; }
    // Skip YAML frontmatter-like lines
    if (/^\s*-\s/.test(line) && i < 20) { continue; }

    REF_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = REF_RE.exec(line)) !== null) {
      const ref = m[1];
      if (!definedLabels.has(ref)) {
        // Only flag sec- refs as info (almost always cross-chapter)
        const severity = ref.startsWith('sec-') ? 'info' as const : 'info' as const;
        results.push({
          line: i,
          message: `Reference '@${ref}' has no matching label in this file (may be cross-chapter)`,
          severity,
          checkId: 'unresolved-ref',
        });
      }
    }
  }

  return results;
}

/**
 * Run all fast checks on a QMD file's text.
 */
export function runAllChecks(text: string): CheckResult[] {
  return [
    ...checkDuplicateLabels(text),
    ...checkUnclosedDivs(text),
    ...checkMissingAltText(text),
    ...checkBrokenInFileRefs(text),
  ];
}
