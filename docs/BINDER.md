# Book Binder CLI

The **Book Binder** is a self-contained, lightning-fast development CLI for the MLSysBook project. It provides streamlined commands for building, previewing, and managing the book in both HTML and PDF formats.

## Quick Start

```bash
# Build a single chapter
./binder build intro

# Build multiple chapters together 
./binder build intro,ml_systems html

# Preview a chapter (builds and opens in browser)
./binder preview intro

# Build the complete book
./binder build-full pdf

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
| `build <chapter[,ch2,...]> [format]` | Build one or more chapters | `./binder build intro,ml_systems html` |
| `preview <chapter>` | Build and preview a chapter | `./binder preview ops` |

**Supported formats**: `html` (default), `pdf`

### 📚 Full Book Commands

Full builds render the complete book with all chapters, parts, and cross-references.

| Command | Description | Example |
|---------|-------------|---------|
| `build-full [format]` | Build complete book | `./binder build-full pdf` |
| `preview-full` | Preview complete book | `./binder preview-full` |

### 🔧 Management Commands

| Command | Description | Example |
|---------|-------------|---------|
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
| `bf` | `build-full` |
| `pf` | `preview-full` |
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
- **`_quarto.yml`**: Symlink to active configuration

Use `./binder switch <format>` to change the active configuration.

## Development Workflow

### Typical Chapter Development

```bash
# 1. Start development on a chapter
./binder preview intro

# 2. Make edits, save files (auto-rebuild in preview mode)

# 3. Build multiple related chapters together
./binder build intro,ml_systems html

# 4. Check full book before committing
./binder build-full pdf
```

### Before Committing

```bash
# Clean up any build artifacts
./binder clean

# Verify no artifacts remain
./binder check

# Build full book to ensure everything works
./binder build-full html
./binder build-full pdf
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

### Performance Tips

- Use fast builds (`./binder build chapter`) for development
- Use unified builds (`./binder build ch1,ch2`) for multiple chapters
- Only use full builds (`./binder build-full`) for final verification
- Preview mode auto-rebuilds on file changes

## Integration

The binder integrates with:

- **Quarto**: Primary rendering engine
- **GitHub Actions**: Automated builds and publishing
- **Pre-commit hooks**: Artifact detection and cleanup
- **VS Code**: Can be used as build tasks

For more details, see:
- [BUILD.md](BUILD.md) - Complete build instructions
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup
- [MAINTENANCE_GUIDE.md](MAINTENANCE_GUIDE.md) - Advanced maintenance 