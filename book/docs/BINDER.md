# Book Binder CLI

The **Book Binder** is a self-contained, lightning-fast development CLI for the MLSysBook project. It provides streamlined commands for building, previewing, and managing the book in both HTML and PDF formats.

## Public API Policy

Binder is the public automation interface for this repository.

- Use `./binder ...` commands in local workflows, CI, and editor integrations.
- Scripts under `book/tools/scripts/` are internal implementation details.
- Direct script invocation is soft-deprecated for workflows now covered by Binder.
- The VS Code extension should call Binder subcommands, not scripts directly.

## Quick Start

```bash
# First time setup
./binder setup

# Welcome and overview
./binder hello

# Build a single chapter (HTML)
./binder build intro

# Build multiple chapters together (HTML)
./binder build intro,ml_systems

# Preview a chapter (builds and opens in browser)
./binder preview intro

# Build the complete book (HTML)
./binder build

# Build the complete book (PDF)
./binder build pdf

# Build a single chapter (PDF) - SELECTIVE BUILD
./binder build pdf intro
# ↳ Automatically comments out all chapters except index.qmd and introduction.qmd

# Publish the book
./binder publish

# Get help
./binder help

# Run native validation suite
./binder validate all

# Run maintenance namespace
./binder maintain repo-health
```

## Installation

The binder is a Python script located in the project root. Make sure it's executable:

```bash
chmod +x binder
```

**Dependencies**: Python 3.6+ (uses only standard library modules)

## Command Reference

### ⚡ Core Commands

Intuitive commands that work on both individual chapters and the entire book.

| Command | Description | Example |
|---------|-------------|---------|
| `build [chapter[,ch2,...]]` | Build book or chapter(s) in HTML | `./binder build intro,ml_systems` |
| `preview [chapter[,ch2,...]]` | Preview book or chapter(s) | `./binder preview ops` |
| `build pdf [chapter[,ch2,...]]` | Build book or chapter(s) in PDF | `./binder build pdf intro` |

**Smart defaults**: No target = entire book, with target = specific chapter(s)

### 📚 Full Book Examples

| Command | Description | Example |
|---------|-------------|---------|
| `build` | Build complete book (HTML) | `./binder build` |
| `preview` | Preview complete book | `./binder preview` |
| `build pdf` | Build complete book (PDF) | `./binder build pdf` |
| `publish` | Build and publish book | `./binder publish` |

### 🔧 Management Commands

| Command | Description | Example |
|---------|-------------|---------|
| `validate <subcommand>` | Run Binder-native validation checks | `./binder validate inline-refs` |
| `maintain <topic> ...` | Run Binder-native maintenance utilities | `./binder maintain glossary build` |
| `setup` | Configure environment | `./binder setup` |
| `clean` | Clean configs & artifacts | `./binder clean` |
| `switch <format>` | Switch active config | `./binder switch pdf` |
| `status` | Show current status | `./binder status` |
| `list` | List available chapters | `./binder list` |
| `doctor` | Run comprehensive health check | `./binder doctor` |
| `about` | Show project information | `./binder about` |
| `help` | Show help information | `./binder help` |

### ✅ Validation Namespace

Use Binder-native checks instead of direct script calls:

- `./binder validate inline-python`
- `./binder validate refs`
- `./binder validate citations`
- `./binder validate references` — validate .bib entries vs academic DBs (hallucinator; install: `pip install -e ".[reference-check]"` or `pip install -r book/tools/dependencies/requirements.txt`)
  - `-f BIB` / `--file BIB` — .bib file(s) to check (can repeat)
  - `-o FILE` / `--output FILE` — write report to FILE
  - `--limit N` — check only first N refs (quick test)
- `./binder validate duplicate-labels`
- `./binder validate unreferenced-labels`
- `./binder validate inline-refs`
- `./binder validate all`

#### Bibliography consistency (`.bib` + pre-commit)

Committed `.bib` files go through **pre-commit** in this order:

1. **`bib-apply-mechanical`** — `book/tools/bib_apply_mechanical_fixes.py` (safe §5 field fixes; only **changed** `.bib` in the commit; exits **1** if it rewrote a file so you re-stage)
2. **`bibtex-tidy`** — layout
3. **`./book/binder check bib --scope hygiene`** — same as `book/tools/bib_lint.py` **errors** (see `book/tools/bib_lint_baseline.json` for grandfathering; warnings do not block)

To run the same pipeline **by hand** on the whole tree (e.g. before a big merge), from the **repository root**:

```bash
./book/binder bib normalize              # all git-tracked *.bib
./book/binder bib normalize --vol1        # only under book/quarto/contents/vol1
```

This runs, in order: `book/tools/bib_apply_mechanical_fixes.py` (safe §5 field fixes) → `pre-commit run bibtex-tidy --all-files` (until it passes) → `python3 book/tools/bib_lint.py --all --check`. Optional metadata refresh from Crossref is separate: **`./book/binder bib update`**.

Machine-readable output is available for editor/CI integration:

```bash
./binder validate all --json
```

Exit semantics:

- `0` success (no issues)
- `1` validation failures or command failure

### 🧰 Maintenance Namespace

- `./binder maintain glossary build [--vol1|--vol2]`
- `./binder maintain images compress [-f <file> ... | --all] [--smart-compression] [--apply]`
- `./binder maintain repo-health [--json] [--min-size-mb N]`

### 🚀 Shortcuts

All commands have convenient shortcuts:

| Shortcut | Command |
|----------|---------|
| `b` | `build` |
| `p` | `preview` |
| `pdf` | `build pdf` |
| `epub` | `build epub` |
| `l` | `list` |
| `s` | `status` |
| `d` | `doctor` |
| `h` | `help` |

## Chapter Names

Chapters can be referenced by their short names. Common examples:

- `intro` → Introduction chapter
- `ml_systems` → Machine Learning Systems chapter
- `nn_computation` → Neural Computation chapter
- `training` → Training chapter
- `ops` → MLOps chapter

Use `./binder list` to see all available chapters.

## Build Outputs

| Format | Output Location | Description |
|--------|-----------------|-------------|
| HTML | `build/html/` | Website format with navigation |
| PDF | `build/pdf/` | Academic book format |

## 🚀 Publishing

The `publish` command provides two modes based on how you call it:

### 1. Interactive Mode (Default)

When called without arguments, `publish` runs the interactive wizard:

```bash
# Interactive publishing wizard
./binder publish
```

**What interactive mode does:**

1. **🔍 Pre-flight checks** - Verifies git status and branch
2. **🧹 Cleans** - Removes previous builds
3. **📚 Builds HTML** - Creates web version
4. **📄 Builds PDF** - Creates downloadable version
5. **📦 Copies PDF** - Moves PDF to assets directory
6. **💾 Commits** - Adds PDF to git
7. **🚀 Pushes** - Triggers GitHub Actions deployment

### 2. Command-Line Trigger Mode

When called with arguments, `publish` triggers the GitHub Actions workflow directly:

```bash
# Trigger GitHub Actions workflow
./binder publish "Description" [COMMIT_HASH]

# With options
./binder publish "Add new chapter" abc123def --type patch --no-ai
```

**What command-line mode does:**

1. **🔍 Validates environment** - Checks GitHub CLI, authentication, branch
2. **✅ Validates commit** - Ensures the dev commit exists (if provided)
3. **🚀 Triggers workflow** - Uses GitHub CLI to trigger the publish-live workflow
4. **📊 Provides feedback** - Shows monitoring links and next steps

**Options:**
- `--type patch|minor|major` - Release type (default: minor)
- `--no-ai` - Disable AI release notes
- `--yes` - Skip confirmation prompts

**Requirements:**
- GitHub CLI installed and authenticated (`gh auth login`)
- Must be on main or dev branch
- Dev commit must exist (if provided)

### Publishing Workflow:

```bash
# Development workflow
./binder preview intro          # Preview a chapter
./binder build                  # Build complete HTML
./binder build pdf              # Build complete PDF
./binder publish                # Publish to the world
```

### After Publishing:

- **🌐 Web version**: Available at https://harvard-edge.github.io/cs249r_book
- **📄 PDF download**: Available at https://harvard-edge.github.io/cs249r_book/assets/downloads/Machine-Learning-Systems.pdf
- **📈 GitHub Actions**: Monitors build progress at https://github.com/harvard-edge/cs249r_book/actions

### Requirements:

- Must be on `main` branch
- No uncommitted changes
- Git repository properly configured

## Advanced Features

### Unified Multi-Chapter Builds

The binder supports building multiple chapters together in a single Quarto render command:

```bash
# Build multiple chapters together (HTML)
./binder build intro,ml_systems

# Build multiple chapters together (PDF)
./binder build pdf intro,ml_systems

# Preview multiple chapters together
./binder preview intro,ml_systems
```

**Benefits:**
- ✅ **Faster builds**: Single Quarto process instead of multiple
- ✅ **Shared context**: Dependencies loaded once
- ✅ **Unified processing**: Cross-references and quizzes processed together
- ✅ **Better UX**: Single browser window opens with complete site

### Fast Build Mode

Fast builds use selective rendering to only build essential files plus target chapters:

**HTML Fast Build** (project.render):
```yaml
render:
  - index.qmd
  - 404.qmd
  - contents/frontmatter/
  - contents/core/target-chapter.qmd
```

**PDF Fast Build** (comments out unused chapters):
```yaml
chapters:
  - index.qmd
  - contents/frontmatter/foreword.qmd
  - contents/core/target-chapter.qmd
  # - contents/core/other-chapter.qmd  # Commented for fast build
```

#### Selective PDF Chapter Building

When you run `./binder build pdf intro`, the system automatically:

1. **Creates a backup** of the original PDF configuration
2. **Comments out all chapters** except the target chapter and essential files
3. **Builds only the selected content**:
   - ✅ `index.qmd` (always included)
   - ✅ `contents/core/introduction/introduction.qmd` (target chapter)
   - ❌ `contents/backmatter/glossary/glossary.qmd` (commented out)
   - ❌ `contents/backmatter/references.qmd` (commented out)
4. **Restores the original configuration** after build completion

**Example output:**
```bash
./binder build pdf intro

📄 Building chapter(s) as PDF: intro
🚀 Building 1 chapters (pdf)
⚡ Setting up fast build mode...
📋 Files to build: 2 files
✓ - index.qmd
✓ - contents/core/introduction/introduction.qmd
✓ Fast build mode configured (PDF/EPUB)
```

This ensures that in Binder environments, you get exactly what you need: a PDF containing only the index and your target chapter, with all other chapters automatically commented out during the build process.

#### Cloud Binder Compatibility

The selective PDF build system works seamlessly in cloud environments like [mybinder.org](https://mybinder.org):

**For cloud Binder users:**
```bash
# In a Jupyter terminal or notebook cell
!./binder build pdf intro

# Or using the Python CLI directly
!python binder build pdf intro
```

**Key benefits for cloud environments:**
- ✅ **Reduced memory usage** - Only builds essential chapters
- ✅ **Faster build times** - Skips unnecessary content
- ✅ **Automatic cleanup** - Restores configuration after build
- ✅ **No manual editing** - Everything is automated

**What gets built:**
- Always includes `index.qmd` for proper book structure
- Includes your target chapter (e.g., `introduction.qmd`)
- Comments out all other chapters automatically
- Comments out backmatter (glossary, references) for minimal builds

### Configuration Management

The binder automatically manages Quarto configurations:

- **`_quarto-html.yml`**: Website build configuration
- **`_quarto-pdf.yml`**: Academic PDF build configuration
- **`_quarto.yml`**: **Symlink** to active configuration (currently → `config/_quarto-html.yml`)

**Important**: The `_quarto.yml` file is a symlink that points to the active configuration. This allows the binder to quickly switch between HTML and PDF build modes without copying files.

**Quarto Executable**: The system quarto executable (`/Applications/quarto/bin/quarto`) is NOT a symlink - it's a regular executable file.

Use `./binder switch <format>` to change the active configuration symlink.

## Development Workflow

### Typical Chapter Development

```bash
# 1. Start development on a chapter
./binder preview intro

# 2. Make edits, save files (auto-rebuild in preview mode)

# 3. Build multiple related chapters together
./binder build intro,ml_systems html

# 4. Check full book before committing
./binder build * pdf
```

### Before Committing

```bash
# Clean up any build artifacts
./binder clean

# Run health check
./binder doctor

# Build full book to ensure everything works
./binder build
./binder build pdf
```

## Troubleshooting

### Common Issues

**"Chapter not found"**
- Use `./binder list` to see available chapters
- Check that the chapter QMD file exists
- Verify the chapter path in configuration files

**"Build artifacts detected"**
- Run `./binder clean` to remove temporary files
- Use `./binder doctor` to verify system health

**"Config not clean"**
- The binder detected a previous fast build configuration
- Run `./binder clean` to restore normal configuration

**"Symlink issues"**
- If `_quarto.yml` is not a symlink: `ln -sf config/_quarto-html.yml book/_quarto.yml`
- Check current symlink target: `ls -la book/_quarto.yml`
- The symlink should point to either `config/_quarto-html.yml` or `config/_quarto-pdf.yml`

### Performance Tips

- Use fast builds (`./binder build chapter html`) for development
- Use unified builds (`./binder build ch1,ch2 html`) for multiple chapters
- Only use full builds (`./binder build * format`) for final verification
- Preview mode auto-rebuilds on file changes

## 🚀 Publishing

The `publish` command provides a complete publishing workflow:

```bash
# One-command publishing
./binder publish
```

**What it does:**
1. **Validates environment** - Checks Git status, tools, and dependencies
2. **Manages branches** - Merges `dev` to `main` with confirmation
3. **Plans release** - Suggests version bump (patch/minor/major)
4. **Builds everything** - PDF first, then HTML (ensures PDF is available)
5. **Creates release** - Git tag, AI-generated release notes, GitHub release
6. **Deploys** - Copies PDF to assets, commits, pushes to production

**Features:**
- 🤖 **AI-powered release notes** (requires Ollama)
- 📊 **Smart version suggestions** based on changes
- 🛡️ **Safety checks** and confirmations
- 🎯 **Step-by-step wizard** with clear progress

For more details, see:
- [BUILD.md](BUILD.md) - Complete build instructions
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup and workflow
