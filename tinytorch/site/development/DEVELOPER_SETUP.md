# TinyTorch Developer Setup Guide

**Audience**: Maintainers, contributors, and developers working on TinyTorch itself

**Last Updated**: November 27, 2025

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# Run development setup
./setup-dev.sh

# Activate environment
source .venv/bin/activate

# Verify installation
tito system health
```

---

## Core Development Tools

### Required Tools

These are **required** for TinyTorch development:

```bash
# Python 3.9+
python3 --version

# Virtual environment (included in Python)
python3 -m venv --help

# Git
git --version
```

### Recommended Tools

Highly recommended for productive development:

```bash
# Code formatting
pip install black isort

# Testing
pip install pytest pytest-cov

# Jupyter (for module development)
pip install jupyter jupyterlab

# Type checking
pip install mypy
```

---

## Optional Tools (by Use Case)

### 📹 Demo GIF Generation (Maintainers Only)

**When you need this**: Updating website carousel GIFs when TITO commands change

**Install VHS:**

```bash
# macOS
brew install vhs

# Linux
go install github.com/charmbracelet/vhs@latest

# Verify
vhs --version
```

**Usage:**

```bash
# Generate all carousel GIFs
./scripts/generate-demo-gifs.sh

# Or individual GIFs
vhs site/_static/demos/tapes/01-zero-to-ready.tape

# Optimize file sizes
./scripts/optimize-gifs.sh

# Validate
./scripts/validate-gifs.sh
```

**Documentation**: See `site/_static/demos/GIF_PRODUCTION_GUIDE.md`

**Note**: Students never need VHS. This is purely for marketing material generation.

---

### 📚 Documentation Building

**When you need this**: Building the Jupyter Book website locally

```bash
# Install Jupyter Book
pip install jupyter-book

# Build website
cd site
./build.sh

# Preview
cd _build/html
python -m http.server 8000
open http://localhost:8000
```

---

### 🎨 CLI Development

**When you need this**: Working on TITO commands and Rich UI

```bash
# Rich for terminal UI
pip install rich

# Click for CLI framework (already in requirements.txt)
pip install click

# Test CLI commands
tito --help
tito module --help
tito milestones --help
```

---

## Development Workflow

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .

# Verify
tito --version
```

### 2. Making Changes

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes to code
# Edit files in tito/, tinytorch/, tests/, etc.

# Run tests
pytest tests/

# Format code
black .
isort .
```

### 3. Testing Changes

```bash
# Test TITO commands
tito system health
tito module status
tito milestones list

# Run specific tests
pytest tests/test_specific.py -v

# Run all tests
pytest tests/ -v --cov=tinytorch
```

### 4. Documentation

```bash
# Update relevant docs
# - README.md for user-facing changes
# - docs/ for detailed documentation
# - site/ for website content

# Build docs locally
cd site && ./build.sh
```

### 5. Committing

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new TITO command for xyz"

# Push to your fork
git push origin feature/your-feature

# Create PR on GitHub
```

---

## Project Structure

```
TinyTorch/
├── tito/                    # TITO CLI commands
│   ├── commands/           # Individual command implementations
│   └── core/               # Core utilities
├── tinytorch/              # TinyTorch package (exported code)
│   └── core/               # Core ML components
├── src/                    # Source modules (student workspace)
│   ├── 01_tensor/
│   ├── 02_activations/
│   └── ...
├── tests/                  # Test suite
│   ├── test_*.py          # Unit tests
│   └── */                 # Module-specific tests
├── modules/                # Generated student notebooks
├── site/                   # Jupyter Book website
│   └── _static/demos/     # Demo GIFs (VHS tapes)
├── scripts/                # Automation scripts
├── docs/                   # Documentation
│   └── development/       # Developer docs (this file)
└── milestones/            # Historical milestone scripts
```

---

## Common Development Tasks

### Adding a New TITO Command

1. Create command file: `tito/commands/your_command.py`
2. Inherit from `BaseCommand`
3. Implement `name`, `description`, `add_arguments()`, `run()`
4. Register in `tito/commands/__init__.py`
5. Test with `tito your-command --help`
6. Add tests in `tests/`
7. Update documentation

### Creating Demo GIFs

```bash
# 1. Update tape file with new commands
vim site/_static/demos/tapes/02-build-test-ship.tape

# 2. Regenerate GIF
vhs site/_static/demos/tapes/02-build-test-ship.tape

# 3. Optimize
./scripts/optimize-gifs.sh

# 4. Validate
./scripts/validate-gifs.sh

# 5. Commit updated GIF
git add site/_static/demos/*.gif
git commit -m "docs: update demo GIFs with new commands"
```

### Updating Module Structure

1. Edit source: `src/XX_module/XX_module.py`
2. Run export: `tito src export XX_module`
3. Verify notebook: Check `modules/XX_module/`
4. Test integration: `pytest tests/XX_module/`
5. Update docs: `src/XX_module/README.md`

---

## Troubleshooting

### VHS Not Found

```bash
# Install VHS
brew install vhs  # macOS

# Verify
which vhs
vhs --version
```

### Permission Denied on Scripts

```bash
# Make scripts executable
chmod +x scripts/*.sh
chmod +x setup-dev.sh
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Verify
python -c "import tinytorch; print(tinytorch.__version__)"
```

### Tests Failing

```bash
# Clean environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run tests with verbose output
pytest tests/ -v -s
```

---

## Environment Variables

```bash
# Optional: Set for development
export TINYTORCH_DEV=1              # Enable dev features
export TINYTORCH_DEBUG=1            # Verbose logging
export TINYTORCH_TEST_MODE=1        # Skip slow operations in tests
```

---

## Git Workflow

### Branch Naming

```
feature/add-new-command       # New features
fix/bug-in-export            # Bug fixes
docs/update-readme           # Documentation
refactor/cleanup-tests       # Code refactoring
perf/optimize-loading        # Performance improvements
```

### Commit Messages

Follow conventional commits:

```
feat: add new milestone command
fix: resolve export bug in tensor module
docs: update developer setup guide
test: add integration tests for autograd
refactor: simplify CLI argument parsing
perf: optimize GIF generation script
```

---

## Release Checklist

When preparing a release:

- [ ] All tests pass: `pytest tests/`
- [ ] Documentation updated: `site/`, `README.md`, `CHANGELOG.md`
- [ ] Demo GIFs current: Check TITO commands match
- [ ] Version bumped: `setup.py`, `__init__.py`
- [ ] Git tag created: `git tag v1.0.0`
- [ ] Release notes written
- [ ] PyPI package updated (if applicable)

---

## Getting Help

**For Development Questions:**
- Check existing issues: https://github.com/mlsysbook/TinyTorch/issues
- Review documentation: `docs/` directory
- Ask in discussions: GitHub Discussions

**For CLI Development:**
- See: `docs/development/CLI_TEST_PLAN.md`
- See: `docs/development/CLI_VISUAL_DESIGN.md`

**For GIF Production:**
- See: `site/_static/demos/GIF_PRODUCTION_GUIDE.md`
- See: `site/_static/demos/QUICK_START.md`

---

## Contributing

See `CONTRIBUTING.md` for:
- Code style guidelines
- Testing requirements
- PR submission process
- Code review expectations

---

**Remember**: Students never need to install VHS or other dev tools. They just need Python, the TinyTorch environment, and Jupyter. All dev tooling is optional and for maintainers only.

