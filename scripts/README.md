# Scripts Directory

This directory contains various Python scripts used for book maintenance and processing.

## Python Dependencies

All Python dependencies are managed through the root-level `requirements.txt` file. This ensures consistent package versions across all scripts and the GitHub Actions workflow.

### Adding New Dependencies

When adding new Python scripts that require external packages:

1. Add the required packages to `requirements.txt` at the project root
2. Include version constraints where appropriate (e.g., `>=1.0.0`)
3. Add comments to group related packages
4. Test locally with: `pip install -r requirements.txt`

### Current Dependencies

The current dependencies include:

- **Quarto/Jupyter**: `jupyterlab-quarto`, `jupyter`
- **NLP**: `nltk` (with stopwords and punkt data)
- **AI Integration**: `openai`, `gradio`
- **Document Processing**: `pybtex`, `pypandoc`, `pyyaml`
- **Image Processing**: `Pillow`
- **Validation**: `jsonschema`
- **Utilities**: `absl-py`

### Subdirectory Requirements Files

Some subdirectories have their own `requirements.txt` files for specific workflows:

- `scripts/genai/requirements.txt` - AI-specific dependencies
- `scripts/quarto_publish/requirements.txt` - Publishing dependencies

These are kept for reference but the main workflow uses the root `requirements.txt`.

### GitHub Actions Integration

The GitHub Actions workflow automatically:

1. Caches Python packages for faster builds
2. Installs all dependencies from `requirements.txt`
3. Downloads required NLTK data
4. Reports cache status in build summaries

Cache is invalidated when `requirements.txt` changes, ensuring dependencies stay up-to-date.

## Pre-commit Setup

The project uses pre-commit hooks for code quality checks. The hooks run automatically on commit and include:

- **Spell checking** with codespell
- **YAML validation** for `_quarto.yml`
- **Markdown formatting** and linting
- **Bibliography formatting** with bibtex-tidy
- **Custom Python scripts** for section ID management and unreferenced label detection

### Setup Instructions

1. **Install pre-commit** (included in requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```

2. **Install the git hooks**:
   ```bash
   pre-commit install
   ```

3. **Run manually** (optional):
   ```bash
   # Run on all files
   pre-commit run --all-files
   
   # Run on specific files
   pre-commit run --files path/to/file.qmd
   ```

### Troubleshooting

- **NLTK data issues**: The hooks automatically download required NLTK data, but if you encounter issues, you can manually run:
  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('punkt')
  ```

- **Python environment**: The hooks use isolated Python environments with the specified dependencies, so they should work regardless of your local Python setup. 