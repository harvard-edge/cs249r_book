# TinyVerse Datasets: Suite Datasheet

This datasheet follows the framework proposed by Gebru et al. in *Datasheets for Datasets* (Communications of the ACM, 2021). It covers the unified collection of educational micro-datasets bundled with TinyTorch, collectively known as **TinyVerse**.

---

## 1. Motivation

### For what purpose was the dataset created?
The TinyVerse suite was created to provide self-contained, offline-first, micro-scale datasets for deep learning and machine learning systems education. Modern ML courses frequently encounter classroom friction caused by network timeouts, massive multi-gigabyte downloads, third-party hosting rate limits, and slow multi-hour training cycles on commodity student hardware (such as laptops without discrete GPUs).

TinyVerse provides curated datasets that:
1. Ship directly inside the Git repository with a combined storage footprint under 500 KB.
2. Require zero internet connectivity, external API keys, or manual unpacking steps.
3. Allow complete end-to-end training and convergence within 30 to 60 seconds on single-core CPU hardware.
4. Support the complete pedagogical progression from single-layer perceptrons up through decoder-only transformers and MLPerf systems optimization.

### Who created the dataset and on behalf of which entity?
Created and curated by the TinyTorch project team and contributors for the Systems Approach to Machine Learning curriculum.

---

## 2. Composition

The TinyVerse suite comprises four distinct modalities and task domains:

| Dataset | Modality | Tasks & Milestones | Samples / Size | Storage Footprint |
|:---|:---|:---|---:|---:|
| **TinyDigits** | Vision (8x8 grayscale images) | MLP revival (M03), LeNet CNN (M04), MLPerf INT8 quantization (M06) | 1,000 train + 200 test | ~310 KB (`.pkl`) |
| **TinyPy** | Source Code (Python AST) | TinyGPT code completion, syntax verification (M05 Part 3) | 102 verified snippets, 1,738 lines | ~73 KB (`.txt`) |
| **TinyShakespeare** | Literature & Drama | TinyGPT autoregressive character modeling (M05 Part 1) | Curated excerpt (~3,800 tokens) | ~21 KB (`.txt`) |
| **TinyTalks** | Conversational Q&A | Generative Q&A, Overfitting Detective, Memorization diagnostics (M05 Part 4) | 431 total Q&A pairs (81 TinyTorch, 350 general) | ~53 KB (`.txt`) |

### Are relationships between data instances explicit?
- **TinyDigits**: Clean stratified split across ten digit classes (0 to 9), with fixed random seed 42 to ensure identical train and test splits across runs.
- **TinyPy**: Functions and classes categorized across math, sorting, data structures, and deep learning operations, separated by standard section headers.
- **TinyShakespeare**: Continuous chronological excerpt preserving character dialogue structure (SPEAKER: text).
- **TinyTalks**: Distinct question-and-answer pairs formatted with strict `Q:` and `A:` line markers, accompanied by disjoint train and test splits to test generalization versus memorization.

---

## 3. Collection Process

### What mechanisms or procedures were used to collect the data?
- **TinyDigits**: Derived from the classic scikit-learn optical recognition of handwritten digits dataset (originally collected by C. Kaynak and E. Alpaydin at Bogazici University, 1998). Filtered and serialized into compact numpy arrays.
- **TinyPy**: Hand-authored, idiomatic Python functions representing core algorithms, numeric methods, and deep learning primitives. Every snippet was compiled and validated using Python's standard `ast.parse` compiler to ensure 100% syntactic correctness before inclusion.
- **TinyShakespeare**: Extracted from William Shakespeare plays (First Folio collection, public domain). Curated to preserve dramatic rhythm while fitting within minimal memory context windows.
- **TinyTalks**: Hand-curated educational question-and-answer pairs covering computer systems, hardware architectures, neural networks, and TinyTorch internal module mechanics.

---

## 4. Preprocessing, Cleaning, and Labeling

### Was any preprocessing done?
- **TinyDigits**: Pixel values normalized to float ranges, balanced 100 samples per class in training set, 20 samples per class in test set.
- **TinyPy**: Strict 4-space indentation enforced, complete type annotations added, docstrings included, trailing whitespace stripped, and syntax-verified via an automated AST compiler pass.
- **TinyShakespeare**: Cleaned of archaic printing artifacts, standardized capitalization, character vocabulary constrained to standard ASCII.
- **TinyTalks**: Punctuation normalized, duplicate entries removed, line length bounded, questions formatted with explicit question marks.

---

## 5. Uses

### What are the primary intended uses?
- Classroom lab exercises, automated autograding pipelines, and interactive student experimentation.
- Instant feedback loops during local development without waiting on multi-hour cloud GPU jobs.
- Clear demonstration of architectural trade-offs:
  - Linear separability limits on XOR and digits.
  - Spatial feature extraction with convolutions.
  - Autoregressive causal attention, tokenization, and temperature sampling.
  - Pareto-optimal efficiency frontiers in systems optimization (quantization, pruning, KV-caching).

### What uses are out of scope?
TinyVerse datasets are deliberately miniaturized for education. They are **not** intended for:
- Pretraining production foundation models.
- Evaluating open-domain coding benchmarks (e.g., HumanEval, SWE-bench).
- High-stakes real-world computer vision deployment.

---

## 6. Distribution and Maintenance

### How is the dataset distributed?
Bundled directly within the `tinytorch/datasets/` directory of the official TinyTorch repository:
- Git clone provides immediate access to all four datasets.
- Available programmatically through `from milestones.data_manager import DatasetManager`.
- Also mirrorable on Hugging Face Hub under the `harvard-edge` organization.

### How is the dataset licensed?
- **TinyDigits**: Open educational release (scikit-learn / BSD-compatible).
- **TinyPy**: MIT License (included in `datasets/tinypy/LICENSE`).
- **TinyShakespeare**: Public Domain (included in `datasets/tinyshakespeare/LICENSE`).
- **TinyTalks**: MIT License (included in `datasets/tinytalks/LICENSE`).
- **Dataset Suite**: MIT License.

### Who maintains the dataset?
Maintained by the TinyTorch maintainers and the Edge Computing Lab. Issues, suggestions, and snippet contributions can be submitted via GitHub issues and pull requests.
