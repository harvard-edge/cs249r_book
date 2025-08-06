# MLSysBook Development Guide

This guide covers the development workflow, automated cleanup system, and best practices for contributing to the Machine Learning Systems book.

## 🚀 Quick Start

```bash
# First time setup
./binder setup      # Configure environment and tools
./binder hello      # Welcome and overview

# Daily development
./binder preview intro    # Preview a chapter
./binder build - html     # Build complete book
./binder publish          # Publish to the world
```

## 🧹 Automated Cleanup System

This project includes an **automated cleanup system** that runs before every commit to ensure a clean repository.

### What Gets Cleaned Automatically

The cleanup system removes:
- **Build artifacts**: `*.html`, `*.pdf`, `*.tex`, `*.aux`, `*.log`, `*.toc`
- **Cache directories**: `.quarto/`, `site_libs/`, `index_files/` (legacy)
- **Python artifacts**: `__pycache__/`, `*.pyc`, `*.pyo`
- **System files**: `.DS_Store`, `Thumbs.db`, `*.swp`
- **Editor files**: `*~`, `.#*`
- **Debug files**: `debug.log`, `error.log`

### Manual Cleanup Commands

```bash
# Regular cleanup (recommended before commits)
make clean
./tools/scripts/build/clean.sh

# See what would be cleaned (safe preview)
make clean-dry
./tools/scripts/build/clean.sh --dry-run

# Deep clean (removes caches, virtual environments)
make clean-deep
./tools/scripts/build/clean.sh --deep

# Quiet cleanup (minimal output)
./tools/scripts/build/clean.sh --quiet
```

### Pre-Commit Hook

The git pre-commit hook automatically:
1. 🔍 **Scans for build artifacts** in staged files
2. 🧹 **Runs cleanup** if artifacts are detected
3. ⚠️ **Warns about large files** (>1MB)
4. 🚨 **Blocks commits** with potential secrets
5. ✅ **Allows clean commits** to proceed

#### Bypassing the Hook (Emergency)

```bash
# Only if absolutely necessary
git commit --no-verify -m "Emergency commit"
```

## 🔨 Building the Book

### Build Commands

```bash
# Using binder (recommended)
./binder build - html          # Build HTML version
./binder build - pdf           # Build PDF version
./binder publish               # Build and publish

# Using make (legacy)
make build                     # HTML version
make build-pdf                 # PDF version
make build-all                 # All formats
```

### Development Workflow

```bash
# Preview a chapter (fastest)
./binder preview intro

# Build complete book
./binder build - html

# Publish to the world
./binder publish
```

### Environment Setup

The `./binder setup` command provides a complete environment configuration:

**What it does:**
1. **Checks environment** - Verifies all required tools and versions
2. **Installs dependencies** - Auto-installs missing tools (Quarto, GitHub CLI, Ollama)
3. **Configures Git** - Sets up user name, email, and GitHub username
4. **Sets preferences** - Configures build format and browser behavior
5. **Tests setup** - Builds a test chapter to verify everything works

**Features:**
- 🛠️ **Automatic tool installation** (Homebrew, apt, pip)
- 👤 **Interactive Git configuration**
- ⚙️ **User preference setup**
- 🧪 **Built-in testing** to verify setup

```bash
# Run setup
./binder setup

# Get welcome and overview
./binder hello
```

### Development Server

```bash
# Start live preview server
make preview
cd book && quarto preview

# The server will automatically reload when you save changes
```

### Build Outputs

- **HTML**: `build/html/index.html` (main output directory)
- **PDF**: `build/pdf/` (PDF output directory)
- **PDF**: `book/index.pdf` (in book directory)
- **Artifacts**: Automatically cleaned by git hooks

## 🚀 Publishing

The `./binder publish` command provides a complete publishing workflow:

**Step-by-step process:**
1. **Environment validation** - Checks Git status, tools, and dependencies
2. **Branch management** - Merges `dev` to `main` with confirmation
3. **Release planning** - Suggests version bump based on changes
4. **Build process** - PDF first, then HTML (ensures PDF availability)
5. **Release creation** - Git tag, AI-generated release notes, GitHub release
6. **Deployment** - Copies PDF to assets, commits, pushes to production

**Features:**
- 🤖 **AI-powered release notes** (requires Ollama)
- 📊 **Smart version suggestions** (patch/minor/major)
- 🛡️ **Safety checks** and confirmations
- 🎯 **Step-by-step wizard** with clear progress

```bash
# One-command publishing
./binder publish
```

### Manual Publishing Steps

If you prefer to do it step by step:

```bash
# 1. Ensure you're on main branch
git checkout main
git merge dev

# 2. Build both formats
./binder build - html
./binder build - pdf

# 3. Copy PDF to assets
cp build/pdf/Machine-Learning-Systems.pdf assets/

# 4. Commit and push
git add assets/downloads/Machine-Learning-Systems.pdf
git commit -m "Add PDF to assets"
git push origin main
```

### Publishing Requirements

- ✅ Must be on `main` branch
- ✅ No uncommitted changes
- ✅ All builds successful
- ✅ Git repository properly configured

### After Publishing

The GitHub Actions workflow will:
- 🔄 Run quality checks
- 🏗️ Build all formats (Linux + Windows)
- 🚀 Deploy to GitHub Pages
- 📦 Create release assets

**Monitor progress**: https://github.com/harvard-edge/cs249r_book/actions

## 🔍 Project Health Checks

### Quick Status Check

```bash
make check          # Overall project health
make status         # Detailed project status
git status          # Git repository status
```

### Comprehensive Testing

```bash
make test           # Run validation tests
make lint           # Check for common issues
quarto check        # Validate Quarto configuration
```

### Example Health Check Output

```
🔍 Checking project health...

📊 Project Structure:
  QMD files: 45
  Bibliography files: 20
  Quiz files: 18

🗂️ Git Status:
  Repository is clean

📦 Dependencies:
  ✅ Quarto: 1.4.x
  ✅ Python: 3.x
```

## 📝 Content Development

### Chapter Structure

```
book/contents/
├── core/                    # Main content chapters
│   ├── introduction/
│   │   ├── introduction.qmd
│   │   ├── introduction.bib
│   │   └── introduction_quizzes.json
│   └── ...
├── frontmatter/            # Preface, about, etc.
├── backmatter/             # References, appendices
└── labs/                   # Hands-on exercises
```

### Working with Minimal Configuration

For faster development, you can work with a minimal set of chapters:

1. **Edit `book/_quarto-html.yml`**: Comment out chapters you're not working on
2. **Edit bibliography section**: Comment out unused `.bib` files
3. **Build faster**: Only active chapters will be processed

```yaml
chapters:
  - index.qmd
  - contents/core/introduction/introduction.qmd
  # - contents/core/ml_systems/ml_systems.qmd    # Commented out
  # - contents/core/dl_primer/dl_primer.qmd      # Commented out
```

### Restoring Full Configuration

Simply uncomment the chapters and bibliography entries you want to restore.

## 🔧 Troubleshooting

### Common Issues

1. **Build fails with missing files**
   ```bash
   make clean          # Clean artifacts
   make check          # Verify structure
   ```

2. **Git hook blocks commit**
   ```bash
   make clean          # Remove artifacts
   git status          # Check what's staged
   ```

3. **Slow builds**
   ```bash
   make clean-deep     # Full cleanup
   # Use minimal configuration
   ```

4. **Permission denied on scripts**
   ```bash
   make setup-hooks    # Fix permissions
   ```

### Getting Help

```bash
make help           # Show all commands
make help-clean     # Detailed cleanup help
make help-build     # Detailed build help
```

## 🎯 Best Practices

### Before Starting Work

```bash
git pull            # Get latest changes
make clean          # Clean workspace
make check          # Verify health
```

### Daily Development Workflow

```bash
# 1. Clean and build
make clean build

# 2. Start development server
make preview

# 3. Make changes to .qmd files
# 4. Preview updates automatically

# 5. When ready to commit
git add .           # Pre-commit hook runs automatically
git commit -m "Your message"
```

### Before Major Changes

```bash
make clean-deep     # Full cleanup
make full-clean-build  # Clean build from scratch
make test           # Run all tests
```

### Release Preparation

```bash
make release-check  # Comprehensive validation
make build-all      # Build all formats
make check          # Final health check
```

## ⚙️ Configuration Files

- **`book/_quarto-html.yml`**: HTML website configuration
- **`book/_quarto-pdf.yml`**: PDF book configuration
- **`Makefile`**: Development commands
- **`tools/scripts/build/clean.sh`**: Cleanup script
- **`.git/hooks/pre-commit`**: Automated cleanup hook
- **`.gitignore`**: Ignored file patterns

## 🗂️ Scripts Organization

The `tools/scripts/` directory is organized into logical categories:

```
tools/scripts/
├── build/           # Build and development scripts (clean.sh, etc.)
├── content/         # Content management tools
├── maintenance/     # System maintenance scripts
├── testing/         # Test and validation scripts
├── utilities/       # General utility scripts
├── docs/            # Script documentation
├── genai/           # AI and generation tools
├── cross_refs/      # Cross-reference management
├── quarto_publish/  # Publishing workflows
└── ai_menu/         # AI menu tools
```

Each directory has its own README.md with specific usage instructions.

## 🤝 Contributing

1. **Fork and clone** the repository
2. **Run setup**: `make setup-hooks && make install`
3. **Make changes** with the development workflow above
4. **Test thoroughly**: `make test && make build-all`
5. **Submit pull request** with clean commits

The automated cleanup system ensures that your commits will be clean and won't include build artifacts, making code reviews easier and keeping the repository tidy.

## 📞 Support

If you encounter issues with the development workflow:
1. Check this guide first
2. Run `make check` for diagnostics
3. Review the cleanup script output with `make clean-dry`
4. Ask for help in project discussions 