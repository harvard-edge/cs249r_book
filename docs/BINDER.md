# Book Binder CLI

The **Book Binder** is a self-contained, lightning-fast development CLI for the MLSysBook project. It provides streamlined commands for building, previewing, and managing the book in both HTML and PDF formats.

## Quick Start

```bash
# First time setup
./binder setup

# Welcome and overview
./binder hello

# Build a single chapter
./binder build intro html

# Build multiple chapters together 
./binder build intro,ml_systems html

# Preview a chapter (builds and opens in browser)
./binder preview intro

# Build the complete book
./binder build * pdf

# Publish the book
./binder publish

# Get help
./binder help
```

## Installation

The binder is a Python script located in the project root. Make sure it's executable:

```bash
chmod +x binder
```

**Dependencies**: Python 3.6+ (uses only standard library modules)

## Command Reference

### ⚡ Fast Chapter Commands

Fast builds focus on individual chapters with minimal overhead, perfect for development and iteration.

| Command | Description | Example |
|---------|-------------|---------|
| `build <chapter[,ch2,...]> <format>` | Build one or more chapters | `./binder build intro,ml_systems html` |
| `preview <chapter>` | Build and preview a chapter | `./binder preview ops` |

**Supported formats**: `html`, `pdf` (format is required)

### 📚 Full Book Commands

Full builds render the complete book with all chapters, parts, and cross-references.

| Command | Description | Example |
|---------|-------------|---------|
| `build * <format>` | Build complete book | `./binder build * pdf` |
| `preview-full` | Preview complete book | `./binder preview-full` |
| `publish` | Build and publish book | `./binder publish` |

### 🔧 Management Commands

| Command | Description | Example |
|---------|-------------|---------|
| `setup` | Configure environment | `./binder setup` |
| `hello` | Welcome and overview | `./binder hello` |
| `clean` | Clean configs & artifacts | `./binder clean` |
| `check` | Check for build artifacts | `./binder check` |
| `switch <format>` | Switch active config | `./binder switch pdf` |
| `status` | Show current status | `./binder status` |
| `list` | List available chapters | `./binder list` |
| `help` | Show help information | `./binder help` |

### 🚀 Shortcuts

All commands have single-letter shortcuts:

| Shortcut | Command |
|----------|---------|
| `b` | `build` |
| `p` | `preview` |
| `pf` | `preview-full` |
| `pub` | `publish` |
| `se` | `setup` |
| `he` | `hello` |
| `c` | `clean` |
| `ch` | `check` |
| `s` | `switch` |
| `st` | `status` |
| `l` | `list` |
| `h` | `help` |

## Chapter Names

Chapters can be referenced by their short names. Common examples:

- `intro` → Introduction chapter
- `ml_systems` → Machine Learning Systems chapter  
- `dl_primer` → Deep Learning Primer chapter
- `training` → Training chapter
- `ops` → MLOps chapter

Use `./binder list` to see all available chapters.

## Build Outputs

| Format | Output Location | Description |
|--------|-----------------|-------------|
| HTML | `build/html/` | Website format with navigation |
| PDF | `build/pdf/` | Academic book format |

## 🚀 Publishing

The `publish` command handles the complete publication workflow:

```bash
# Publish the book (build + deploy)
./binder publish
```

### What `publish` does:

1. **🔍 Pre-flight checks** - Verifies git status and branch
2. **🧹 Cleans** - Removes previous builds
3. **📚 Builds HTML** - Creates web version
4. **📄 Builds PDF** - Creates downloadable version
5. **📦 Copies PDF** - Moves PDF to assets directory
6. **💾 Commits** - Adds PDF to git
7. **🚀 Pushes** - Triggers GitHub Actions deployment

### Publishing Workflow:

```bash
# Development workflow
./binder preview intro          # Preview a chapter
./binder build - html          # Build complete HTML
./binder build - pdf           # Build complete PDF
./binder publish               # Publish to the world
```

### After Publishing:

- **🌐 Web version**: Available at https://harvard-edge.github.io/cs249r_book
- **📄 PDF download**: Available at https://harvard-edge.github.io/cs249r_book/assets/Machine-Learning-Systems.pdf
- **📈 GitHub Actions**: Monitors build progress at https://github.com/harvard-edge/cs249r_book/actions

### Requirements:

- Must be on `main` branch
- No uncommitted changes
- Git repository properly configured

## Advanced Features

### Unified Multi-Chapter Builds

The binder supports building multiple chapters together in a single Quarto render command:

```bash
# Old behavior: Sequential builds (slower)
# ./binder build intro    # Build intro alone  
# ./binder build ml_systems  # Build ml_systems alone

# New behavior: Unified build (faster)
./binder build intro,ml_systems html  # Build both together
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

# Verify no artifacts remain
./binder check

# Build full book to ensure everything works
./binder build * html
./binder build * pdf
```

## Troubleshooting

### Common Issues

**"Chapter not found"**
- Use `./binder list` to see available chapters
- Check that the chapter QMD file exists
- Verify the chapter path in configuration files

**"Build artifacts detected"**
- Run `./binder clean` to remove temporary files
- Use `./binder check` to verify cleanup

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
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup
- [MAINTENANCE_GUIDE.md](MAINTENANCE_GUIDE.md) - Advanced maintenance 