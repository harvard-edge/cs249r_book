# TinyShakespeare Dataset

A curated character-level language modeling dataset drawn from William Shakespeare plays.

## Overview

TinyShakespeare is the classic benchmark for autoregressive language models, popularized by Andrej Karpathy in `char-rnn` and `nanoGPT`. It serves as the primary dataset for Milestone 05 Part 1: training TinyGPT on character-level sequence prediction.

## Contents

* `tinyshakespeare_sample.txt`: Bundled offline text corpus (~21 KB, 801 lines, ~21,000 characters). Ships directly inside the repository for instant offline training.
* Full corpus (~1.1 MB, ~1 million characters): Downloaded automatically on demand via `DatasetManager().get_tinyshakespeare()`.

## Loading in Python

```python
from pathlib import Path

# Load bundled offline sample directly
sample_path = Path("tinytorch/datasets/tinyshakespeare/tinyshakespeare_sample.txt")
with open(sample_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded {len(text):,} characters of Shakespeare.")
```

Via DatasetManager:

```python
from milestones.data_manager import DatasetManager

dm = DatasetManager()

# Load offline sample (zero download)
sample_text = dm.get_tinyshakespeare(sample_only=True)

# Or download and cache the full 1.1 MB corpus
full_text = dm.get_tinyshakespeare(sample_only=False)
```

## Milestone Mapping

* **Milestone 05 Part 1 (`01_tinygpt_shakespeare.py`):** Trains a 119K-parameter Pre-LN TinyGPT model to learn Shakespearean dialogue cadence and character vocabulary.
