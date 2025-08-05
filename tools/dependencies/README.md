# Python Dependencies

This directory contains Python dependencies split for different use cases.

## Files

### `requirements.txt` (Default - Full Dependencies)
**Use for**: General development, complete functionality
```bash
pip install -r requirements.txt
```
- Complete dependency set including ML tools
- Includes PyTorch, sentence-transformers, FAISS
- ~4GB total with all dependencies

### `requirements-build.txt` (Container/Build Only)
**Use for**: Container builds, CI/CD, production deployments
```bash
pip install -r requirements-build.txt
```
- Minimal dependencies for Quarto book builds
- Excludes heavy ML libraries (PyTorch, etc.)
- ~500MB total vs ~4GB with ML dependencies
- Optimized for fast container builds

### `requirements-dev.txt` (Development - Same as requirements.txt)
**Use for**: Alternative name for full dependencies
```bash
pip install -r requirements-dev.txt
```
- Identical to `requirements.txt`
- Provided for clarity in development contexts

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