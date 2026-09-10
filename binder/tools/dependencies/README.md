# Python Dependencies

This directory contains Python dependencies organized in a clean, modular structure.

## Structure

```
requirements/
├── base.txt          # Core dependencies (pandas, requests, etc.)
├── production.txt    # Production-specific (includes base.txt)
└── development.txt   # Full ML stack (includes production.txt)
```

## Files

### `requirements/base.txt` (Core Dependencies)
- Essential packages needed by all environments
- Jupyter, pandas, requests, PyYAML, etc.
- Lightweight foundation (~200MB)

### `requirements/production.txt` (Container/Build)
- Includes `base.txt` + production-specific packages
- OpenAI API, Groq, pre-commit, Ghostscript, etc.
- Excludes heavy ML libraries
- Optimized for containers (~500MB)

### `requirements/development.txt` (Full ML Stack)
- Includes `production.txt` + ML dependencies
- PyTorch, sentence-transformers, FAISS, scikit-learn
- Required for cross-reference generation
- Complete development environment (~4GB)

## Convenience Files

### `requirements.txt` (Default - Full)
```bash
pip install -r requirements.txt
```
References `requirements/development.txt` for complete functionality

### `requirements-build.txt` (Container Optimized)
```bash
pip install -r requirements-build.txt
```
References `requirements/production.txt` for lightweight builds

### `requirements-dev.txt` (Development Alias)
```bash
pip install -r requirements-dev.txt
```
Alias for `requirements/development.txt`

## Usage Patterns

### Container Builds (GitHub Actions)
```yaml
RUN pip install -r tools/dependencies/requirements-build.txt
```
**Result**: Fast builds, small containers (~500MB Python deps)

### Local Development
```bash
# For full development (recommended)
pip install -r tools/dependencies/requirements.txt

# For container-optimized builds only
pip install -r tools/dependencies/requirements-build.txt
```

### Cross-Reference Generation
```bash
# Use full dependencies for ML tools
pip install -r tools/dependencies/requirements.txt
python tools/scripts/cross_refs/cross_refs.py
```

## Migration Notes

- **`requirements.txt`**: Full dependencies (unchanged for backward compatibility)
- **`requirements-build.txt`**: New minimal set for containers (~500MB vs ~4GB)
- **Containers should switch** to `requirements-build.txt` for 3-4GB savings
- **Local development**: Continue using `requirements.txt`
