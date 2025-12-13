# MLSysBook Scripts Directory

This directory contains all automation scripts, tools, and utilities for the Machine Learning Systems book project. The scripts are organized into logical categories for easy discovery and maintenance.

## 📁 Directory Structure

```
tools/scripts/
├── build/           # Build and development scripts
├── content/         # Content management and editing tools
├── maintenance/     # System maintenance and updates
├── testing/         # Test scripts and validation
├── utilities/       # General utility scripts
├── docs/            # Documentation for scripts
├── genai/           # AI and generation tools
├── cross_refs/      # Cross-reference management
├── publish/  # Publishing and deployment
└── ai_menu/         # AI menu and interface tools
```

## 🔨 Build Scripts (`build/`)

Scripts for building, cleaning, and development workflows:

- **`clean.sh`** - Comprehensive cleanup script (build artifacts, caches, temp files)
- **`standardize_sources.sh`** - Standardize source file formatting
- **`generate_stats.py`** - Generate statistics about the Quarto project

### Usage Examples
```bash
# Clean all build artifacts
./build/clean.sh

# Deep clean including caches and virtual environments
./build/clean.sh --deep

# Generate project statistics
python build/generate_stats.py
```

## 📝 Content Management (`content/`)

Tools for managing, editing, and validating book content:

- **`improve_figure_captions.py`** - Enhance figure captions using AI
- **`manage_section_ids.py`** - Manage section IDs and cross-references
- **`find_unreferenced_labels.py`** - Find unused labels and references
- **`find_duplicate_labels.py`** - Detect duplicate labels
- **`extract_headers.py`** - Extract headers from content files
- **`find_acronyms.py`** - Find and manage acronyms
- **`find_fig_references.py`** - Analyze figure references
- **`fix_bibliography.py`** - Fix bibliography formatting
- **`sync_bibliographies.py`** - Synchronize bibliography files
- **`clean_callout_titles.py`** - Clean callout title formatting
- **`collapse_blank_lines.py`** - Remove excessive blank lines

### Usage Examples
```bash
# Improve figure captions
python content/improve_figure_captions.py

# Find unreferenced labels
python content/find_unreferenced_labels.py

# Manage section IDs
python content/manage_section_ids.py
```

## 🔧 Maintenance Scripts (`maintenance/`)

System maintenance, updates, and changelog management:

- **`generate_release_content.py`** - Generate changelog entries and release notes
- **`fix_changelog.py`** - Fix changelog formatting issues
- **`update_texlive_packages.py`** - Update LaTeX package dependencies
- **`cleanup_old_runs.sh`** - Clean up old build runs

### Usage Examples
```bash
# Generate changelog entry or release notes
python maintenance/generate_release_content.py

# Update LaTeX packages
python maintenance/update_texlive_packages.py
```

## 🧪 Testing Scripts (`testing/`)

Test scripts and validation tools:

- **`run_tests.py`** - Run comprehensive test suite
- **`test_section_ids.py`** - Test section ID management

### Usage Examples
```bash
# Run all tests
python testing/run_tests.py

# Test section ID system
python testing/test_section_ids.py
```

## 🛠️ Utilities (`utilities/`)

General-purpose utility scripts:

- **`check_ascii.py`** - Check for non-ASCII characters
- **`check_images.py`** - Validate image files and references
- **`check_sources.py`** - Comprehensive source file validation
- **`fix_titles.py`** - Fix title formatting
- **`count_footnotes.sh`** - Count footnotes
- **`analyze_footnotes.sh`** - Detailed footnote analysis

### Usage Examples
```bash
# Check for non-ASCII characters
python utilities/check_ascii.py

# Validate images
python utilities/check_images.py

# Check source files
python utilities/check_sources.py
```

## 📖 Documentation (`docs/`)

Documentation for scripts and systems:

- **`README.md`** - General scripts documentation
- **`SECTION_ID_SYSTEM.md`** - Section ID management system guide
- **`FIGURE_CAPTIONS.md`** - Figure caption enhancement guide

## 🤖 Specialized Tools

### AI and Generation (`genai/`)
Tools for AI-powered content generation and enhancement.

### Cross-References (`cross_refs/`)
Advanced cross-reference management and validation tools.

### Publishing (`publish/`)
Scripts for publishing and deployment workflows. **Note**: The main publishing workflow is now handled by `./binder publish`.

### AI Menu (`ai_menu/`)
AI-powered menu and interface tools.

## 🚀 Quick Start

### First Time Setup
```bash
# Make all scripts executable
find tools/scripts -name "*.sh" -exec chmod +x {} \;

# Install Python dependencies (if needed)
pip install -r tools/dependencies/requirements.txt
```

### Common Workflows

#### Before Working on Content
```bash
# Clean workspace
./build/clean.sh

# Check project health
python utilities/check_sources.py
```

#### Content Editing Session
```bash
# Improve figures
python content/improve_figure_captions.py

# Find issues
python content/find_unreferenced_labels.py
python content/find_duplicate_labels.py

# Clean up formatting
python content/collapse_blank_lines.py
```

#### Before Publishing
```bash
# Full validation
python testing/run_tests.py
python utilities/check_images.py
python utilities/ascii_checker.py

# Update changelog
python maintenance/update_changelog.py

# Final cleanup
./build/clean.sh

# Publish (using binder)
./binder publish
```

## 📋 Script Categories Summary

| Category | Purpose | Count | Key Scripts |
|----------|---------|-------|-------------|
| **build** | Development & building | 3 | `clean.sh`, `generate_stats.py` |
| **content** | Content management | 11 | `manage_section_ids.py`, `improve_figure_captions.py` |
| **maintenance** | System maintenance | 4 | `generate_release_content.py`, `update_texlive_packages.py` |
| **testing** | Testing & validation | 2 | `run_tests.py`, `test_section_ids.py` |
| **utilities** | General utilities | 6 | `check_sources.py`, `check_ascii.py` |
| **docs** | Documentation | 3 | Various `.md` files |

## 🔍 Finding the Right Script

### By Purpose
- **Need to clean up?** → `build/clean.sh`
- **Content has issues?** → `utilities/check_sources.py`
- **Figures need improvement?** → `content/improve_figure_captions.py`
- **Want project stats?** → `build/generate_stats.py`
- **Need to test changes?** → `testing/run_tests.py`

### By File Type
- **`.sh` scripts** - Shell scripts (mostly in `build/` and `utilities/`)
- **`.py` scripts** - Python scripts (distributed across categories)
- **`.md` files** - Documentation (in `docs/`)

## 🤝 Contributing New Scripts

When adding new scripts:

1. **Choose the right category** based on the script's primary purpose
2. **Follow naming conventions** - descriptive, lowercase with underscores
3. **Add documentation** - Include usage examples and descriptions
4. **Update this README** - Add the script to the appropriate section
5. **Make executable** - `chmod +x` for shell scripts
6. **Test thoroughly** - Ensure scripts work in different environments

## 📞 Support

For issues with specific scripts:
1. Check the script's docstring or comments
2. Look for documentation in the `docs/` directory
3. Run scripts with `--help` flag if available
4. Review this README for context and examples
