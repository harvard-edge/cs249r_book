import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import { getRepoRoot } from '../utils/workspace';

/**
 * Output channel for Python resolver diagnostics.
 * Visible in Output panel as "MLSysBook Python".
 */
let outputChannel: vscode.OutputChannel | undefined;

function getOutputChannel(): vscode.OutputChannel {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel('MLSysBook Python');
  }
  return outputChannel;
}

function log(msg: string): void {
  getOutputChannel().appendLine(`[${new Date().toISOString()}] ${msg}`);
}

/**
 * Cached result for a single QMD file's Python variable resolution.
 */
interface ResolvedFile {
  /** Hash of the Python code blocks (to detect changes) */
  codeHash: string;
  /** Map of variable name → resolved string value */
  values: Map<string, string>;
  /** Timestamp of last resolution */
  resolvedAt: number;
  /** Whether resolution failed */
  error?: string;
}

/**
 * Extracts all ```{python} code blocks from a QMD document text.
 * Returns the concatenated Python source and a simple hash.
 */
function extractPythonBlocks(text: string): { source: string; hash: string } {
  const lines = text.split('\n');
  const blocks: string[] = [];
  let inBlock = false;

  for (const line of lines) {
    if (/^```\{python\}/.test(line)) {
      inBlock = true;
      continue;
    }
    if (inBlock && /^```\s*$/.test(line)) {
      inBlock = false;
      blocks.push(''); // separator between blocks
      continue;
    }
    if (inBlock) {
      blocks.push(line);
    }
  }

  const source = blocks.join('\n');
  // Simple hash: length + a checksum-like value
  let hash = `${source.length}:`;
  let sum = 0;
  for (let i = 0; i < source.length; i++) {
    sum = ((sum << 5) - sum + source.charCodeAt(i)) | 0;
  }
  hash += sum.toString(36);
  return { source, hash };
}

/**
 * Extracts all inline `{python} var_name` references from a document.
 */
export function extractInlineRefs(text: string): string[] {
  const regex = /`\{python\}\s+(\w+(?:\.\w+)?)`/g;
  const vars: string[] = [];
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    const varName = match[1];
    if (!vars.includes(varName)) {
      vars.push(varName);
    }
  }
  return vars;
}

/**
 * Builds a Python script that executes the code blocks and prints
 * the values of all referenced variables as JSON.
 */
function buildResolverScript(pythonSource: string, varNames: string[]): string {
  if (varNames.length === 0) {
    return '';
  }

  // Build each variable capture as a top-level try/except (no indentation).
  // We use _resolve_val() to unwrap IPython.display.Markdown objects
  // (returned by md(), md_math(), md_frac()) into their .data string,
  // and also unwrap dicts with a "str" key (from display_value() etc.).
  const captures = varNames.map(v => [
    `try:`,
    `    _results["${v}"] = _resolve_val(${v})`,
    `except Exception as _e:`,
    `    _results["${v}"] = f"<error: {_e}>"`,
  ].join('\n')).join('\n');

  // Assemble the full script with clear section breaks.
  // We force matplotlib to the non-interactive Agg backend so plots
  // are built silently without opening windows or rendering images.
  // We also suppress plt.show() so it's a no-op.
  const lines = [
    'import sys, json, os',
    '# Force non-interactive backend BEFORE any matplotlib import.',
    '# Setting the env var ensures even lazy imports pick up Agg.',
    'os.environ["MPLBACKEND"] = "Agg"',
    'os.environ["MATPLOTLIB_BACKEND"] = "Agg"',
    'try:',
    '    import matplotlib',
    '    matplotlib.use("Agg", force=True)',
    '    import matplotlib.pyplot as _plt',
    '    _plt.switch_backend("Agg")',
    '    _plt.show = lambda *a, **k: None',
    '    _plt.pause = lambda *a, **k: None',
    '    # Close any figures that might have been created during import',
    '    _plt.close("all")',
    'except ImportError:',
    '    pass',
    '',
    '# Helper to unwrap display objects into readable strings',
    'def _resolve_val(_v):',
    '    # IPython.display.Markdown (from md(), md_math(), md_frac())',
    '    if hasattr(_v, "data") and type(_v).__name__ == "Markdown":',
    '        return _v.data',
    '    # dict with "str" key (from display_value(), display_percent(), etc.)',
    '    if isinstance(_v, dict) and "str" in _v:',
    '        return _v["str"]',
    '    return str(_v)',
    '',
    '# Execute the QMD Python blocks',
    pythonSource,
    '',
    '# Close any figures created during execution',
    'try:',
    '    import matplotlib.pyplot as _plt2',
    '    _plt2.close("all")',
    'except Exception:',
    '    pass',
    '',
    '# Collect variable values',
    '_results = {}',
    captures,
    '',
    '# Output as JSON',
    'print(json.dumps(_results))',
  ];

  return lines.join('\n');
}

/**
 * Locate python3 binary. Tries several strategies:
 * 1. User-configured path
 * 2. `python3` on PATH (via shell)
 * 3. Common macOS/Linux locations
 */
async function findPython3(): Promise<string> {
  // Check user setting first
  const configured = vscode.workspace
    .getConfiguration('mlsysbook')
    .get<string>('python3Path', '');
  if (configured && configured.trim().length > 0) {
    return configured.trim();
  }

  // Try to find python3 via shell (resolves user PATH)
  try {
    const result = await new Promise<string>((resolve, reject) => {
      cp.exec('which python3', { timeout: 5000 }, (error, stdout) => {
        if (error) { reject(error); return; }
        const p = stdout.trim();
        if (p.length > 0) { resolve(p); } else { reject(new Error('empty')); }
      });
    });
    return result;
  } catch {
    // Fall through to common paths
  }

  // Common locations
  const candidates = [
    '/usr/local/bin/python3',
    '/usr/bin/python3',
    '/opt/homebrew/bin/python3',
    '/Library/Frameworks/Python.framework/Versions/Current/bin/python3',
  ];
  const fs = await import('fs');
  for (const candidate of candidates) {
    try {
      await fs.promises.access(candidate, fs.constants.X_OK);
      return candidate;
    } catch {
      // Try next
    }
  }

  // Last resort — hope it's on PATH
  return 'python3';
}

export class QmdPythonValueResolver implements vscode.Disposable {
  private readonly cache = new Map<string, ResolvedFile>();
  private readonly disposables: vscode.Disposable[] = [];
  private pendingResolves = new Map<string, Promise<Map<string, string>>>();
  private python3Path: string | undefined;
  private readonly onDidResolveEmitter = new vscode.EventEmitter<vscode.Uri>();
  /** Fires when a file's Python values have been resolved (for ghost text refresh). */
  readonly onDidResolve = this.onDidResolveEmitter.event;

  constructor() {
    // Create output channel eagerly so it appears in the dropdown immediately
    getOutputChannel();
    log('QmdPythonValueResolver initialized');
  }

  dispose(): void {
    this.cache.clear();
    this.onDidResolveEmitter.dispose();
    this.disposables.forEach(d => d.dispose());
    outputChannel?.dispose();
    outputChannel = undefined;
  }

  /**
   * Get resolved values for a document. Returns cached results if
   * the Python blocks haven't changed, otherwise triggers resolution.
   */
  async getValues(document: vscode.TextDocument): Promise<Map<string, string>> {
    const key = document.uri.toString();
    const text = document.getText();
    const { source, hash } = extractPythonBlocks(text);
    const varNames = extractInlineRefs(text);

    if (varNames.length === 0 || source.trim().length === 0) {
      log(`No inline refs or no Python blocks in ${document.uri.fsPath.split('/').pop()}`);
      return new Map();
    }

    log(`getValues called for ${document.uri.fsPath.split('/').pop()} — ${varNames.length} vars, hash=${hash}`);

    // Check cache
    const cached = this.cache.get(key);
    if (cached && cached.codeHash === hash) {
      return cached.values;
    }

    // Check if we already have a pending resolve for this file
    const pending = this.pendingResolves.get(key);
    if (pending) {
      return pending;
    }

    // Resolve
    const promise = this.resolve(key, document.uri, source, hash, varNames);
    this.pendingResolves.set(key, promise);
    try {
      return await promise;
    } finally {
      this.pendingResolves.delete(key);
    }
  }

  /**
   * Synchronous cache lookup — returns immediately if cached, undefined otherwise.
   */
  getCachedValues(document: vscode.TextDocument): Map<string, string> | undefined {
    const key = document.uri.toString();
    const text = document.getText();
    const { hash } = extractPythonBlocks(text);
    const cached = this.cache.get(key);
    if (cached && cached.codeHash === hash) {
      return cached.values;
    }
    return undefined;
  }

  /**
   * Invalidate cache for a document.
   */
  invalidate(uri: vscode.Uri): void {
    this.cache.delete(uri.toString());
  }

  /**
   * Trigger background resolution for a document without awaiting.
   */
  triggerResolve(document: vscode.TextDocument): void {
    log(`triggerResolve called for ${document.uri.fsPath.split('/').pop()}`);
    this.getValues(document).catch(err => {
      log(`Background resolve failed: ${err}`);
    });
  }

  private async resolve(
    key: string,
    uri: vscode.Uri,
    pythonSource: string,
    hash: string,
    varNames: string[],
  ): Promise<Map<string, string>> {
    const script = buildResolverScript(pythonSource, varNames);
    if (!script) {
      return new Map();
    }

    const root = getRepoRoot();
    if (!root) {
      log('Cannot resolve: repo root not found');
      return new Map();
    }

    const quartoDir = path.join(root, 'book', 'quarto');

    try {
      if (!this.python3Path) {
        this.python3Path = await findPython3();
        log(`Using python3 at: ${this.python3Path}`);
      }

      const result = await this.execPython(script, quartoDir);
      const parsed = JSON.parse(result) as Record<string, string>;
      const values = new Map<string, string>();
      for (const [k, v] of Object.entries(parsed)) {
        values.set(k, v);
      }

      log(`Resolved ${values.size} variables for ${path.basename(uri.fsPath)}`);

      this.cache.set(key, {
        codeHash: hash,
        values,
        resolvedAt: Date.now(),
      });

      this.onDidResolveEmitter.fire(uri);
      return values;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      log(`Resolution failed for ${path.basename(uri.fsPath)}: ${errorMsg}`);
      this.cache.set(key, {
        codeHash: hash,
        values: new Map(),
        resolvedAt: Date.now(),
        error: errorMsg,
      });
      return new Map();
    }
  }

  private execPython(script: string, cwd: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const pythonPath = this.python3Path ?? 'python3';
      // Use shell: true so the user's PATH is inherited from their shell
      // environment. This is critical on macOS where python3 may be in
      // /Library/Frameworks/ or /opt/homebrew/ which aren't on the
      // default VSCode extension host PATH.
      const env = { ...process.env, PYTHONPATH: cwd };
      cp.exec(
        `"${pythonPath}" -c ${shellEscape(script)}`,
        { cwd, env, timeout: 15000, maxBuffer: 1024 * 1024 },
        (error, stdout, stderr) => {
          if (error) {
            reject(new Error(`Python execution failed:\n${stderr || error.message}`));
            return;
          }
          // Find the JSON line (last non-empty line of stdout)
          const lines = stdout.trim().split('\n');
          const jsonLine = lines[lines.length - 1];
          if (!jsonLine || !jsonLine.startsWith('{')) {
            reject(new Error(`Unexpected Python output: ${stdout.slice(0, 200)}`));
            return;
          }
          resolve(jsonLine);
        },
      );
    });
  }
}

/**
 * Escape a string for safe inclusion in a shell command as a single argument.
 * Uses single quotes with proper escaping of embedded single quotes.
 */
function shellEscape(s: string): string {
  // Replace single quotes with '\'' (end quote, escaped quote, start quote)
  return "'" + s.replace(/'/g, "'\\''") + "'";
}
