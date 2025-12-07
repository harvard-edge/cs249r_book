# Virtual Environment Setup Guide

This repository contains two separate Python projects with independent virtual environments:

## Repository Structure

```
MLSysBook/
├── .venv/                    # Book tooling virtual environment
├── pyproject.toml           # Book dependencies (Quarto, AI tools, etc.)
├── book/                    # Textbook content
├── tinytorch/
│   ├── .venv/              # TinyTorch virtual environment
│   ├── pyproject.toml      # TinyTorch dependencies (numpy, pytest, etc.)
│   └── tinytorch/          # Framework code
└── README.md
```

## Why Two Virtual Environments?

**TinyTorch** and the **Book** have different purposes and dependencies:

- **Book**: Needs Quarto, Pandoc, AI libraries, document processing tools
- **TinyTorch**: Minimal ML framework (numpy, pytest, jupyter)

Keeping them separate:
- ✅ Prevents dependency conflicts
- ✅ Lighter environments for focused work
- ✅ Students can use TinyTorch without book tooling
- ✅ Clear separation of concerns

## Setup Instructions

### For TinyTorch Development

```bash
# Navigate to TinyTorch directory
cd tinytorch

# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Install TinyTorch with dev dependencies
pip install -e ".[dev,visualization]"

# Run tests
pytest tests/
```

### For Book Development

```bash
# Navigate to repository root
cd /Users/VJ/GitHub/MLSysBook

# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Install book tools with all extras
pip install -e ".[dev,ai,build]"

# Build the book
binder build
```

### When You Need Both

If you need TinyTorch available in the book environment (e.g., for building book examples):

```bash
# Activate book's venv
cd /Users/VJ/GitHub/MLSysBook
source .venv/bin/activate

# Install TinyTorch into book's venv
pip install -e tinytorch/
```

## Checking Active Environment

```bash
# Show which Python you're using
which python3

# Show active venv
echo $VIRTUAL_ENV

# Expected outputs:
# Book venv:      /Users/VJ/GitHub/MLSysBook/.venv/bin/python3
# TinyTorch venv: /Users/VJ/GitHub/MLSysBook/tinytorch/.venv/bin/python3
```

## Quick Reference

| Task | Directory | Command |
|------|-----------|---------|
| TinyTorch tests | `tinytorch/` | `source .venv/bin/activate && pytest` |
| TinyTorch development | `tinytorch/` | `source .venv/bin/activate` |
| Book building | Root | `source .venv/bin/activate && binder build` |
| AI tools | Root | `source .venv/bin/activate` |

## .gitignore

Both virtual environments are already ignored:

```gitignore
.venv/
tinytorch/.venv/
```

## Troubleshooting

**Wrong venv activated?**
```bash
deactivate  # Exit current venv
cd <correct-directory>
source .venv/bin/activate
```

**Import errors?**
```bash
# Make sure you're in the right venv
pip list | grep tinytorch  # Should show if installed
pip install -e .  # Reinstall in editable mode
```

**Dependencies out of sync?**
```bash
pip install -e ".[dev]" --upgrade
```
