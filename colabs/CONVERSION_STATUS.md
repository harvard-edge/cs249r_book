# Python to Notebook Conversion - Status

## Completed

### ✅ Directory Structure
```
colabs/
├── src/                          # Python source files
│   ├── ch01_ai_triangle.py       # ✅ Created
│   └── utils/
│       └── ai_triangle_sim.py    # ✅ Extracted utility class
│
├── notebooks/
│   └── ch01_ai_triangle.ipynb    # ✅ Already exists (finalized)
│
└── docs/
    └── PYTHON_TO_NOTEBOOK_WORKFLOW.md  # ✅ Complete guide
```

### ✅ Files Created
1. **`src/ch01_ai_triangle.py`** - Python source using percent format (`# %%`)
   - Contains all markdown and code cells
   - Properly formatted for Jupytext conversion
   - Ready for version control

2. **`src/utils/ai_triangle_sim.py`** - Reusable simulator class
   - Extracted from colab for maintainability
   - Documented with docstrings
   - Can be imported by future colabs

3. **`PYTHON_TO_NOTEBOOK_WORKFLOW.md`** - Complete workflow documentation
   - Explains percent format
   - Shows Jupytext usage
   - Includes manual conversion script
   - Best practices and automation ideas

### ✅ Documentation Updates
- Updated `colabs/README.md` with:
  - New directory structure section
  - Workflow for authors
  - Links to conversion documentation
  - Progress checklist

## Current State

**Source of Truth**: `colabs/src/ch01_ai_triangle.py` (Python file)

**Student Distribution**: `colabs/notebooks/ch01_ai_triangle.ipynb` (already finalized)

**Next Conversion**: When updating the colab, edit `.py` file and regenerate `.ipynb`

## How to Use (For Future Updates)

### Option 1: Using Jupytext (Recommended)
```bash
# Install once
pip install jupytext

# Convert single file
jupytext --to notebook colabs/src/ch01_ai_triangle.py \
  --output colabs/notebooks/ch01_ai_triangle.ipynb

# Convert all chapter colabs
jupytext --to notebook colabs/src/ch*.py \
  --output-dir colabs/notebooks/
```

### Option 2: Manual Editing
1. Edit `colabs/notebooks/ch01_ai_triangle.ipynb` directly in Jupyter
2. When ready to extract source:
   ```bash
   jupytext --to py:percent colabs/notebooks/ch01_ai_triangle.ipynb \
     --output colabs/src/ch01_ai_triangle.py
   ```

### Option 3: Paired Notebooks (Best for Development)
```bash
# Pair files (changes sync automatically)
jupytext --set-formats ipynb,py:percent colabs/notebooks/ch01_ai_triangle.ipynb

# Now edits to either .ipynb or .py sync to both!
```

## Test Plan (Next Steps)

### 1. Validate Conversion
```bash
# Convert Python → Notebook
jupytext --to notebook colabs/src/ch01_ai_triangle.py \
  --output /tmp/test_ch01.ipynb

# Compare with original
diff colabs/notebooks/ch01_ai_triangle.ipynb /tmp/test_ch01.ipynb
```

Expected: Minimal differences (metadata only)

### 2. Test in Google Colab
1. Upload `/tmp/test_ch01.ipynb` to Google Colab
2. Run all cells
3. Verify:
   - All imports work
   - Simulator runs correctly
   - Visualizations display
   - Interactive cells allow parameter changes

### 3. Validate Workflow
```bash
# Make small edit to Python source
# Convert to notebook
# Test in Colab
# Commit both files
```

## Future Automation Ideas

### Pre-commit Hook
Auto-convert modified Python colabs to notebooks on commit:
```bash
# .git/hooks/pre-commit
jupytext --to notebook colabs/src/ch*.py --output-dir colabs/notebooks/
git add colabs/notebooks/*.ipynb
```

### CI/CD Pipeline
GitHub Actions workflow to auto-convert on push:
```yaml
# .github/workflows/convert-colabs.yml
- run: pip install jupytext
- run: jupytext --to notebook colabs/src/ch*.py --output-dir colabs/notebooks/
- run: git commit -am "chore: auto-convert colabs"
```

### VSCode Integration
Paired notebooks using Jupytext extension:
- Edit Python or notebook, both stay in sync
- Better git diffs from Python format
- Full Jupyter functionality when needed

## Status: Ready for Testing

- ✅ Python source created and formatted correctly
- ✅ Utility extracted for reusability
- ✅ Workflow documented
- ✅ README updated
- ⏳ Conversion tested (ready for user)
- ⏳ Google Colab validation (ready for user)

**Recommendation**: Test the conversion workflow by:
1. Installing Jupytext: `pip install jupytext`
2. Converting: `jupytext --to notebook colabs/src/ch01_ai_triangle.py`
3. Uploading to Google Colab and running all cells
4. If successful, adopt this workflow for all future colabs
