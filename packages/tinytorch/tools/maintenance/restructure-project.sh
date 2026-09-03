#!/bin/bash
# TinyTorch Professional Restructure
# This script reorganizes the project following industry conventions

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "🏗️  TinyTorch Professional Restructure"
echo "======================================"
echo ""
echo "This will reorganize the project structure."
echo "A backup will be created before any changes."
echo ""

# Confirm
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create backup
BACKUP_DIR="../TinyTorch-backup-$(date +%Y%m%d-%H%M%S)"
echo "📦 Creating backup at: $BACKUP_DIR"
cp -r . "$BACKUP_DIR"
echo "✅ Backup complete"
echo ""

# Phase 1: Create new directory structure
echo "📁 Phase 1: Creating directory structure..."

mkdir -p tools/dev
mkdir -p tools/build
mkdir -p tools/maintenance
mkdir -p docs/_static/demos/scripts

echo "✅ Directories created"
echo ""

# Phase 2: Move GIF generation scripts
echo "🎬 Phase 2: Moving GIF generation scripts..."

if [ -f "scripts/generate-demo-gifs.sh" ]; then
    mv scripts/generate-demo-gifs.sh docs/_static/demos/scripts/generate.sh
    echo "  ✅ generate-demo-gifs.sh → docs/_static/demos/scripts/generate.sh"
fi

if [ -f "scripts/optimize-gifs.sh" ]; then
    mv scripts/optimize-gifs.sh docs/_static/demos/scripts/optimize.sh
    echo "  ✅ optimize-gifs.sh → docs/_static/demos/scripts/optimize.sh"
fi

if [ -f "scripts/validate-gifs.sh" ]; then
    mv scripts/validate-gifs.sh docs/_static/demos/scripts/validate.sh
    echo "  ✅ validate-gifs.sh → docs/_static/demos/scripts/validate.sh"
fi

echo ""

# Phase 3: Move developer tools
echo "🛠️  Phase 3: Moving developer tools..."

if [ -f "setup-dev.sh" ]; then
    mv setup-dev.sh tools/dev/setup.sh
    echo "  ✅ setup-dev.sh → tools/dev/setup.sh"
fi

if [ -f "scripts/generate_student_notebooks.py" ]; then
    mv scripts/generate_student_notebooks.py tools/build/generate_notebooks.py
    echo "  ✅ generate_student_notebooks.py → tools/build/generate_notebooks.py"
fi

if [ -f "scripts/generate_module_metadata.py" ]; then
    mv scripts/generate_module_metadata.py tools/build/generate_metadata.py
    echo "  ✅ generate_module_metadata.py → tools/build/generate_metadata.py"
fi

if [ -f "scripts/cleanup_repo_history.sh" ]; then
    mv scripts/cleanup_repo_history.sh tools/maintenance/cleanup_history.sh
    echo "  ✅ cleanup_repo_history.sh → tools/maintenance/cleanup_history.sh"
fi

echo ""

# Phase 4: Rename site → docs (if not already done)
echo "📚 Phase 4: Checking docs structure..."

if [ -d "site" ] && [ ! -d "docs" ]; then
    echo "  Renaming site/ → docs/"
    mv site docs
    echo "  ✅ site/ → docs/"
elif [ -d "site" ] && [ -d "docs" ]; then
    echo "  ⚠️  Both site/ and docs/ exist. Manual merge required."
    echo "  Skipping automatic rename."
else
    echo "  ✅ docs/ already exists"
fi

echo ""

# Phase 5: Move old docs content
echo "📝 Phase 5: Organizing documentation..."

if [ -d "docs/development" ]; then
    echo "  ✅ docs/development/ already organized"
else
    echo "  ⚠️  docs/development/ not found. May need manual organization."
fi

if [ -d "instructor" ]; then
    echo "  Moving instructor/ → docs/instructor/"
    mkdir -p docs/instructor
    cp -r instructor/* docs/instructor/
    echo "  ✅ Instructor content moved"
fi

if [ -f "INSTRUCTOR.md" ]; then
    mv INSTRUCTOR.md docs/instructor/README.md
    echo "  ✅ INSTRUCTOR.md → docs/instructor/README.md"
fi

if [ -f "TA_GUIDE.md" ]; then
    mv TA_GUIDE.md docs/instructor/ta-guide.md
    echo "  ✅ TA_GUIDE.md → docs/instructor/ta-guide.md"
fi

echo ""

# Phase 6: Clean up scripts/ (keep only user-facing)
echo "🧹 Phase 6: Cleaning scripts/ directory..."

# Remove old scripts that were moved (only if they don't exist)
if [ -f "scripts/activate-tinytorch" ]; then
    rm scripts/activate-tinytorch
    echo "  ✅ Removed old activate-tinytorch"
fi

# Keep: scripts/tito (CLI entry point)
if [ -f "scripts/tito" ]; then
    echo "  ✅ Kept scripts/tito (CLI entry)"
fi

echo ""

# Phase 7: Create README files for new directories
echo "📄 Phase 7: Creating README files..."

cat > tools/README.md << 'EOF'
# Development Tools

This directory contains tools for TinyTorch maintainers and contributors.

## Structure

- **`dev/`** - Development environment setup and utilities
- **`build/`** - Build scripts for generating notebooks and metadata
- **`maintenance/`** - Maintenance and cleanup scripts

## For Students

Students don't need anything in this directory. Use the main setup scripts in the project root.

## For Developers

See `docs/development/DEVELOPER_SETUP.md` for complete developer documentation.
EOF

cat > tools/dev/README.md << 'EOF'
# Development Environment Tools

Tools for setting up and maintaining the development environment.

## Scripts

- `setup.sh` - Set up development environment (was `setup-dev.sh`)

## Usage

```bash
# From project root
./tools/dev/setup.sh
```
EOF

cat > tools/build/README.md << 'EOF'
# Build Tools

Scripts for generating student-facing materials from source.

## Scripts

- `generate_notebooks.py` - Generate Jupyter notebooks from source modules
- `generate_metadata.py` - Generate module metadata

## Usage

```bash
# From project root
python tools/build/generate_notebooks.py
python tools/build/generate_metadata.py
```
EOF

cat > tools/maintenance/README.md << 'EOF'
# Maintenance Tools

Scripts for repository maintenance and cleanup.

## Scripts

- `cleanup_history.sh` - Clean up repository history
- `restructure-project.sh` - This restructuring script

## Usage

```bash
# From project root
./tools/maintenance/cleanup_history.sh
```
EOF

echo "  ✅ README files created"
echo ""

# Phase 8: Update references in key files
echo "🔗 Phase 8: Updating file references..."

# Update docs/_static/demos/scripts paths
if [ -f "docs/_static/demos/scripts/generate.sh" ]; then
    # Update shebang and make executable
    chmod +x docs/_static/demos/scripts/generate.sh
    chmod +x docs/_static/demos/scripts/optimize.sh
    chmod +x docs/_static/demos/scripts/validate.sh
    echo "  ✅ Made GIF scripts executable"
fi

# Make tools scripts executable
if [ -f "tools/dev/setup.sh" ]; then
    chmod +x tools/dev/setup.sh
    echo "  ✅ Made tools/dev/setup.sh executable"
fi

if [ -f "tools/maintenance/cleanup_history.sh" ]; then
    chmod +x tools/maintenance/cleanup_history.sh
    echo "  ✅ Made tools/maintenance/cleanup_history.sh executable"
fi

echo ""

# Summary
echo "✅ Restructure Complete!"
echo "======================"
echo ""
echo "📁 New Structure:"
echo "  ├── tools/              # Developer tools"
echo "  │   ├── dev/           # Development utilities"
echo "  │   ├── build/         # Build scripts"
echo "  │   └── maintenance/   # Maintenance scripts"
echo "  ├── docs/              # All documentation + website"
echo "  │   ├── _static/demos/scripts/  # GIF generation"
echo "  │   ├── development/   # Developer guides"
echo "  │   └── instructor/    # Instructor guides"
echo "  └── scripts/           # User-facing only"
echo "      └── tito          # CLI entry"
echo ""
echo "📦 Backup saved at: $BACKUP_DIR"
echo ""
echo "🔍 Next Steps:"
echo "  1. Test website build: cd docs && ./build.sh"
echo "  2. Test TITO commands: tito --help"
echo "  3. Update documentation references"
echo "  4. Commit changes: git add -A && git commit -m 'refactor: professional project structure'"
echo ""
