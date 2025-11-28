# Python to Notebook Conversion Workflow

## Overview

This document describes the workflow for maintaining Lens colabs as Python source files (`.py`) and converting them to Jupyter notebooks (`.ipynb`) for student distribution.

## Why Python Source Files?

**Benefits:**
- **Better git diffs**: Line-by-line changes are clear in `.py` format
- **Code quality**: Easier to run linters, formatters, and static analysis
- **Refactoring**: Extract utilities and reusable components cleanly
- **Version control**: Merge conflicts are easier to resolve
- **Testing**: Can import and test functions directly

**Trade-off:**
- Students receive `.ipynb` files (standard Jupyter/Colab format)
- Conversion step required before distribution

## Directory Structure

```
colabs/
├── src/                          # Python source files (version controlled)
│   ├── ch01_ai_triangle.py       # Chapter 1 colab source
│   ├── ch02_deployment.py        # Chapter 2 colab source
│   ├── ...
│   └── utils/                    # Reusable utilities
│       ├── ai_triangle_sim.py    # AI Triangle simulator class
│       └── visualization.py      # Common plotting functions
│
├── notebooks/                    # Generated Jupyter notebooks (for students)
│   ├── ch01_ai_triangle.ipynb    # Generated from src/ch01_ai_triangle.py
│   ├── ch02_deployment.ipynb     # Generated from src/ch02_deployment.py
│   └── ...
│
└── docs/                         # Documentation
    ├── PYTHON_TO_NOTEBOOK_WORKFLOW.md  # This file
    └── ...
```

## Python Source Format

### Percent Format (Jupytext-Compatible)

Use **percent format** (`# %%`) to define cells in Python files:

```python
# %% [markdown]
# # Colab Title
#
# This is a markdown cell with **bold** and *italic* text.

# %%
import numpy as np
print("This is a code cell")

# %% [markdown]
# Another markdown cell
```

### Cell Types

**Markdown cells:**
```python
# %% [markdown]
# Your markdown content here
# Use standard markdown syntax
```

**Code cells:**
```python
# %%
# Your Python code here
x = 42
```

### Special Directives

**Matplotlib inline (Colab/Jupyter specific):**
```python
# %matplotlib inline
```
This is commented in `.py` files but will work when converted to `.ipynb`.

## Conversion Tools

### Option 1: Jupytext (Recommended)

**Install:**
```bash
pip install jupytext
```

**Convert single file:**
```bash
jupytext --to notebook colabs/src/ch01_ai_triangle.py \
  --output colabs/notebooks/ch01_ai_triangle.ipynb
```

**Convert all files:**
```bash
jupytext --to notebook colabs/src/ch*.py \
  --output-dir colabs/notebooks/
```

**Set kernel metadata:**
```bash
jupytext --to notebook --set-kernel python3 colabs/src/ch01_ai_triangle.py
```

### Option 2: Manual Script (Custom Control)

Create `tools/scripts/convert_colabs.py`:

```python
#!/usr/bin/env python3
"""Convert Python source files to Jupyter notebooks"""

import json
import re
from pathlib import Path


def parse_py_to_cells(py_content):
    """Parse percent-format Python to notebook cells"""
    cells = []
    current_cell = None

    for line in py_content.split('\n'):
        if line.startswith('# %% [markdown]'):
            if current_cell:
                cells.append(current_cell)
            current_cell = {'cell_type': 'markdown', 'source': []}
        elif line.startswith('# %%'):
            if current_cell:
                cells.append(current_cell)
            current_cell = {'cell_type': 'code', 'source': [], 'outputs': []}
        else:
            if current_cell:
                if current_cell['cell_type'] == 'markdown':
                    # Remove leading "# " from markdown lines
                    clean_line = line[2:] if line.startswith('# ') else line
                    current_cell['source'].append(clean_line + '\n')
                else:
                    current_cell['source'].append(line + '\n')

    if current_cell:
        cells.append(current_cell)

    return cells


def create_notebook(cells):
    """Create Jupyter notebook JSON structure"""
    return {
        'cells': [
            {
                'cell_type': cell['cell_type'],
                'metadata': {},
                'source': cell['source'],
                **(
                    {'execution_count': None, 'outputs': []}
                    if cell['cell_type'] == 'code' else {}
                )
            }
            for cell in cells
        ],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.9.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }


def convert_py_to_notebook(src_path, dest_path):
    """Convert Python source to Jupyter notebook"""
    with open(src_path, 'r') as f:
        py_content = f.read()

    cells = parse_py_to_cells(py_content)
    notebook = create_notebook(cells)

    with open(dest_path, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"✓ Converted {src_path} → {dest_path}")


if __name__ == '__main__':
    src_dir = Path('colabs/src')
    dest_dir = Path('colabs/notebooks')

    for py_file in src_dir.glob('ch*.py'):
        nb_file = dest_dir / py_file.with_suffix('.ipynb').name
        convert_py_to_notebook(py_file, nb_file)
```

## Workflow for Content Updates

### 1. Edit Python Source
```bash
# Edit the source file
code colabs/src/ch01_ai_triangle.py
```

### 2. Test Locally (Optional)
```bash
# Run the Python file directly to test logic
python3 colabs/src/ch01_ai_triangle.py

# Or use Jupyter to test the notebook
jupytext --to notebook --execute colabs/src/ch01_ai_triangle.py
```

### 3. Convert to Notebook
```bash
# Convert single file
jupytext --to notebook colabs/src/ch01_ai_triangle.py \
  --output colabs/notebooks/ch01_ai_triangle.ipynb

# Or convert all
jupytext --to notebook colabs/src/ch*.py --output-dir colabs/notebooks/
```

### 4. Version Control
```bash
# Only commit the .py source files
git add colabs/src/ch01_ai_triangle.py

# Optionally commit generated notebooks (for student access)
git add colabs/notebooks/ch01_ai_triangle.ipynb

# Commit
git commit -m "feat: add AI Triangle interactive colab"
```

## Best Practices

### 1. Keep Utilities Separate
Extract reusable code to `colabs/src/utils/`:

```python
# In colab source: colabs/src/ch01_ai_triangle.py
from utils.ai_triangle_sim import AITriangleSimulator

# Students won't see utils/ - it's packaged differently
```

### 2. Clear Cell Boundaries
Use clear comments and spacing:

```python
# %% [markdown]
# ## Section Title
#
# Description of what we're doing.

# %%
# Code implementing the concept
x = compute_something()
x.plot()

# %% [markdown]
# Explanation of results
```

### 3. Test Before Converting
Run Python file directly to catch syntax errors:

```bash
python3 colabs/src/ch01_ai_triangle.py
```

### 4. Use Descriptive Cell Comments
```python
# %% [markdown]
# ### Your Turn: Open-Ended Exploration
#
# Try different configurations...

# %%
# Student experimentation cell
my_system = AITriangleSimulator(...)
```

## Automation (Future)

### Pre-commit Hook
Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Auto-convert modified Python colabs to notebooks

changed_files=$(git diff --cached --name-only | grep 'colabs/src/ch.*\.py')

for src_file in $changed_files; do
    nb_file="colabs/notebooks/$(basename $src_file .py).ipynb"
    jupytext --to notebook "$src_file" --output "$nb_file"
    git add "$nb_file"
done
```

### CI/CD Pipeline
```yaml
# .github/workflows/convert-colabs.yml
name: Convert Colabs

on:
  push:
    paths:
      - 'colabs/src/*.py'

jobs:
  convert:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install jupytext
      - run: jupytext --to notebook colabs/src/ch*.py --output-dir colabs/notebooks/
      - uses: stefanzweifel/git-auto-commit-action@v4
        with:
          commit_message: "chore: auto-convert colabs to notebooks"
```

## FAQ

**Q: Should notebooks be version controlled?**
A: Yes, commit both `.py` (source of truth) and `.ipynb` (student distribution). Git will track meaningful changes in `.py`, notebooks are for convenience.

**Q: What about outputs in notebooks?**
A: Clear outputs before committing. Students run fresh notebooks.

**Q: Can students edit notebooks directly?**
A: Yes! Students work with `.ipynb` files. Our workflow is for textbook authors only.

**Q: How do we handle imports from `utils/`?**
A: For distribution, either:
1. Inline the utility code in generated notebooks
2. Distribute utilities as a package (`pip install lens-mlsys`)
3. Include utility cells at top of notebook

## Current Status

- ✅ Python source format established (percent format)
- ✅ Directory structure created (`src/`, `notebooks/`, `utils/`)
- ✅ First colab converted: `ch01_ai_triangle.py` → `ch01_ai_triangle.ipynb`
- ✅ Utility extracted: `AITriangleSimulator` → `utils/ai_triangle_sim.py`
- ⏳ Conversion script (manual for now, Jupytext recommended)
- ⏳ Pre-commit hook (future automation)
- ⏳ CI/CD pipeline (future automation)

## Next Steps

1. Install Jupytext: `pip install jupytext`
2. Test conversion: Convert ch01 and upload to Google Colab
3. Refine format based on student testing
4. Document package distribution strategy (Lens toolkit)
5. Create remaining chapter colabs in `.py` format
