# Datasheet for TinyPy Dataset

*Following the Datasheets for Datasets framework established by Gebru et al. (2018).*

## Motivation

### For what purpose was the dataset created?
TinyPy was created to provide an educational, lightweight, and syntactically validated code dataset tailored for training autoregressive language models and code completion systems. Within the TinyTorch deep learning curriculum, students study transformers and generative language models. Standard benchmark code corpora (such as The Stack, CodeSearchNet, or GitHub dumps) span gigabytes of heterogeneous files that require hours or days to tokenize and train. TinyPy solves this educational bottleneck by providing a self-contained, high-quality ~67 KB corpus of idiomatic Python that trains in minutes on commodity laptop CPUs while exposing students to authentic code completion challenges: strict syntax, indentation preservation, algorithmic reasoning, and type annotations.

### Who created the dataset and on behalf of which entity?
TinyPy was created by the TinyTorch Contributors as part of the TinyTorch open-source educational deep learning framework and textbook project.

### Who funded the creation of the dataset?
The creation of TinyPy was funded through open-source educational contributions to the TinyTorch project without commercial or grant sponsorship.

## Composition

### What do the instances that comprise the dataset represent?
Each instance in TinyPy represents a complete, self-contained Python function or class definition. Every snippet implements a well-defined mathematical operation, algorithmic procedure, deep learning primitive, or data manipulation utility.

### How many instances are there in total?
The dataset contains 102 top-level code instances structured into four categories:
- Math and Number Theory: 30 snippets
- Classic Algorithms and Data Structures: 20 snippets (including 7 complete object-oriented classes)
- TinyTorch and Deep Learning Primitives: 26 snippets (including 5 complete classes)
- Utility and String / List Operations: 26 snippets

Across these 102 snippets, there are 12 classes and 148 individual functions and methods totaling 2,379 lines of code.

### Does the dataset contain all possible instances or is it a sample?
TinyPy is a curated educational sample. It is intentionally designed to be concise, representative, and pedagogically rich rather than exhaustive.

### What data does each instance consist of?
Each instance consists of valid Python source code featuring:
- A descriptive function or class identifier.
- Explicit type annotations on all parameters and return signatures using modern Python typing.
- A comprehensive docstring detailing purpose, arguments, return values, and raised exceptions.
- Four-space indentation and clean formatting.
- Explicit return statements and boundary condition handling.

Example instance:
```python
def relu(x: float) -> float:
    """Compute Rectified Linear Unit activation on a scalar.

    Args:
        x: Input scalar value.

    Returns:
        Activated output zero or positive.
    """
    return x if x > 0.0 else 0.0
```

### Is there a label or target associated with each instance?
In the context of autoregressive language modeling, the dataset does not require external categorical labels. Instead, standard self-supervised next-token prediction uses the token sequence shifted by one position as target labels. For prefix-conditioned completion tasks, the function signature and docstring serve as the input prefix, and the function body serves as the target completion.

### Is any information missing from individual instances?
No. Every function and class is syntactically complete and fully self-contained. The dataset deliberately omits external third-party dependencies to ensure universal portability across any Python environment.

### Are relationships between individual instances made explicit?
Instances are organized into four explicit thematic categories via comment block headers. Within Category 3, certain classes (such as `Sequential` and `Linear`) demonstrate modular relationships typical of neural network frameworks, but each snippet is independently parseable.

### Are there recommended data splits?
For educational experimentation within TinyTorch, we recommend an 85/15 train/validation split:
- Training set: First 85% of lines (~2,022 lines, ~57 KB)
- Validation set: Remaining 15% of lines (~357 lines, ~10 KB)

Because the dataset is shipped as a single contiguous plaintext corpus, instructors and students may also employ cross-validation or category-based held-out splits (for example, training on Categories 1, 2, and 4 while testing zero-shot transfer on Category 3).

### Are there any errors, sources of noise, or redundancies in the dataset?
- Errors: Zero. Every snippet is validated with `ast.parse` during automated verification.
- Noise: No obfuscated code, synthetic artifacts, minified lines, or syntax errors exist.
- Redundancies: Certain shared patterns (such as standard loops, conditionals, and docstring headers) recur intentionally to provide sufficient statistical support for autoregressive transformers to learn Python idioms.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources?
TinyPy is completely self-contained. It relies solely on the Python standard library (`math`, `typing`).

### Does the dataset contain confidential data?
No. All snippets were authored specifically for this educational project or represent standard public algorithmic formulations.

### Does the dataset contain data that might be offensive, insulting, or harmful?
No. The dataset consists purely of technical mathematical, algorithmic, and machine learning source code.

## Collection Process

### How was the data associated with each instance acquired?
The dataset was authored directly by the TinyTorch development team. Each snippet was programmed from first principles according to standard software engineering best practices, modern Python typing conventions, and clear pedagogical requirements.

### What mechanisms or procedures were used to collect the data?
Snippets are stored as modular data objects in the generator script `create_tinypy.py`. The generator compiles and serializes the snippets into the plaintext corpus `tinypy_sample.txt` while verifying syntax through Python's `ast` engine.

### Who was involved in the data collection process and how were they compensated?
The TinyTorch core contributors and curriculum architects created the dataset as part of educational materials development.

### Over what timeframe was the data collected?
The dataset was designed and finalized in 2025 as part of the TinyTorch code completion milestone.

### Were any ethical review processes conducted?
Because TinyPy consists entirely of newly authored mathematical, algorithmic, and educational machine learning code containing no personal data or web scrapes, formal institutional review board approval was not required.

## Preprocessing / Cleaning / Labeling

### Was any preprocessing, cleaning, or labeling of the data done?
Every snippet underwent automated linting and formatting:
- Strict four-space indentation.
- Uniform docstring structure (Description, Args, Returns, Raises).
- Syntactic validation using `ast.parse`.
- Vocabulary auditing to verify character set consistency.

### Was the raw data saved in addition to the preprocessed data?
The generative source of truth is maintained in `create_tinypy.py`, allowing exact regeneration of `tinypy_sample.txt` at any time.

### Is the software used to preprocess and generate the data available?
Yes. The complete generator and verification script is included directly alongside the dataset in `create_tinypy.py`.

## Uses

### Has the dataset been used for any tasks already?
Yes. TinyPy is used within TinyTorch to train miniature autoregressive transformers (TinyGPT), verify character and subword tokenizers, and benchmark syntax-validity completion rates.

### What tasks could the dataset be used for?
- Character-level autoregressive code generation.
- Byte-pair encoding (BPE) vocabulary construction.
- Function body completion from signatures and docstrings.
- Syntax correctness evaluation of small language models.
- Curriculum learning in compiler design and AST analysis.

### Are there tasks for which the dataset should not be used?
TinyPy is an educational dataset (~67 KB) designed for rapid learning and conceptual clarity. It is not intended for training production-grade programming copilots, evaluating general software engineering capability, or benchmarking large-scale multi-file codebases.

## Distribution

### How will the dataset be distributed?
TinyPy is distributed directly within the TinyTorch git repository under `tinytorch/datasets/tinypy/`. No network access or external downloads are required.

### When will the dataset be distributed?
The dataset is distributed starting with TinyTorch milestone releases in 2025.

### Under what license is the dataset distributed?
TinyPy is licensed under the permissive MIT License, permitting academic, educational, and commercial reuse with attribution.

### Have any third parties imposed IP-based or other restrictions?
No third-party intellectual property claims or restrictions apply.

## Maintenance

### Who will be supporting, hosting, and maintaining the dataset?
The dataset is maintained by the TinyTorch Contributors via the official TinyTorch repository.

### How can the curators be contacted?
Questions, bug reports, and pull requests can be submitted via the TinyTorch issue tracker on GitHub.

### Will the dataset be updated?
Minor updates to snippet variety, docstring clarity, or typing annotations may occur across TinyTorch version releases. The deterministic generator guarantees reproducibility for any tagged release.
