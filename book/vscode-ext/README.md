# MLSysBook VS Code Extension

The MLSysBook extension provides editor-first workflows for building, debugging, and validating the book without leaving VS Code.

## Loading the extension (so you see the latest code)

The editor runs **one** copy of the extension. Reload only reloads that copy. If that copy isn’t from this repo, you won’t see new changes (e.g. Reset Quarto config) until you point the editor at this repo’s extension.

**Option A – Run from repo (recommended when developing)**
1. Open the **repo root** (`mlsysbook-vols`) in Cursor/VS Code.
2. Run **`npm run compile`** in `book/vscode-ext` (or run it from the repo root: `cd book/vscode-ext && npm run compile`).
3. Press **F5** (or Run → Start Debugging → “Run MLSysBook Extension”).
4. A **new window** opens; that window is running the extension from `book/vscode-ext`. Use that window for the book (Reset, Build, etc.).
5. After code changes: run **compile** again, then in that new window use **Developer: Reload Window** so it picks up the new code.

**Option B – Install from this folder**
1. Command Palette → **Developer: Install Extension from Location…**
2. Choose the folder: **`mlsysbook-vols/book/vscode-ext`**.
3. Reload the window when prompted.
4. After code changes: run **`npm run compile`** in `book/vscode-ext`, then **Developer: Reload Window**. No need to install again unless you remove the extension.

## Binder-First Integration

This extension treats `./book/binder` as the operational backend.

- Build and debug commands execute Binder subcommands.
- Validation actions execute `binder validate ...`.
- Maintenance actions execute `binder maintain ...`.

## Work vs Configuration

The extension separates daily work from configuration:

- Work views: Build, Debug, Recent Runs, Navigator, Validate, Publish/Maintenance
- Configuration view: execution mode, navigator preset, visual preset, and extension settings

This keeps daily actions focused while preserving easy access to tuning controls.

## Why This Matters

- One public CLI contract for terminal, extension, and CI.
- Reduced drift between script behavior and editor behavior.
- Better diagnostics with optional machine-readable output (`--json`) from Binder validation.

## Core Workflows

- Build chapter: `./book/binder build html <chapter>` / `./book/binder build pdf <chapter>`
- Debug failures: `./book/binder debug <pdf|html|epub> --vol1|--vol2`
- Validate content: `./book/binder validate all`
- Maintenance checks: `./book/binder maintain repo-health`

## Parallel builds (Test All Chapters)

**Build All Chapters (Parallel)** in the Debug view runs each chapter in a separate git worktree so multiple builds run at once. You choose **volume** (Vol I / Vol II) and **format** (PDF, HTML, or EPUB).

**Requirements:**

1. **Open the repo root** — Open the folder that contains `book/` (e.g. `mlsysbook-vols`), not a subfolder like `book/` or `book/quarto/`. The extension needs `book/binder` to be present to register commands and run parallel jobs.
2. **Git in PATH** — Parallel mode uses `git worktree add --detach`; ensure `git` is available in the environment where the extension runs.
3. **Optional settings** (VS Code/Cursor → Settings → MLSysBook):
   - `mlsysbook.parallelDebugWorkers` — number of concurrent builds (default: 4).
   - `mlsysbook.parallelDebugRoot` — directory for worktrees under repo root (default: `.mlsysbook/worktrees`).
   - `mlsysbook.keepFailedWorktrees` — keep worktrees for failed chapters for inspection (default: true).

If worktree creation fails, check the **MLSysBook Parallel Debug** output channel; the first failure will include a tip about repo root and git.

## Quarto Visual Customization

The extension includes QMD-focused visual highlighting to improve scanability in long Quarto documents.

Recommended baseline:

- `mlsysbook.enableQmdChunkHighlight = true`
- `mlsysbook.qmdVisualPreset = balanced`
- `mlsysbook.highlightInlineReferences = true`
- `mlsysbook.highlightLabelDefinitions = true`
- `mlsysbook.highlightDivFenceMarkers = true`

What gets emphasized:

- Inline references like `@fig-...`, `@tbl-...`, `@sec-...`
- Label definitions like `{#fig-...}` and `#| label: ...`
- Quarto div fence markers (`:::` / `::::`) and region blocks (callouts/divs/code fences)

## Notes

- Legacy script entrypoints still exist for compatibility in some areas.
- For Binder-covered tasks, direct script invocation is soft-deprecated.
