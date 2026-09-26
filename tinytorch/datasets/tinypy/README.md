# TinyPy: Educational Python Code Dataset for TinyTorch

**A curated, syntactically verified Python corpus designed for learning code completion and autoregressive language modeling.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Size: ~67KB](https://img.shields.io/badge/Size-~67KB-green.svg)]()
[![Syntax: 100% Valid AST](https://img.shields.io/badge/Syntax-100%25%20AST%20Valid-brightgreen.svg)]()
[![Snippets: 102](https://img.shields.io/badge/Snippets-102-orange.svg)]()

## Overview

**TinyPy** is an educational, syntactically verified Python code dataset crafted specifically for TinyTorch language modeling and code completion experiments.

While traditional language modeling curricula often rely on literary texts such as Shakespeare, code completion has emerged as the defining practical generative AI workload in modern software engineering. TinyPy provides students with a lightweight, clean, and self-contained corpus of idiomatic Python that trains in minutes on standard consumer hardware.

Every single function and class in TinyPy is verified with Python's abstract syntax tree parser (`ast.parse`). Every snippet includes explicit type annotations, complete docstrings, clean four-space indentation, and clean return statements.

## Why Code Completion as an Educational Workload?

Code completion offers unique pedagogical advantages over unstructured natural language:

1. **Formal Grammatical Structure:** Programming languages enforce strict syntactic constraints. A language model must learn exact indentation levels, balanced brackets, colon placement, and keyword sequences.
2. **Deterministic Evaluation:** Student models can be evaluated not only on perplexity or loss, but also on syntactic validity by passing generated completions directly to `ast.parse`.
3. **Semantic Coherence:** Variable scoping, parameter reuse, and mathematical formulas test whether attention heads capture long-range functional dependencies across tokens.
4. **Immediate Practical Relevance:** Students experience firsthand how modern tools like Copilot, Codex, and Claude operate at the token level.

## Dataset Statistics

<table>
  <thead>
    <tr>
      <th width="45%">Property</th>
      <th width="55%">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Total Snippets</b></td>
      <td>102 curated code snippets</td>
    </tr>
    <tr>
      <td><b>Total Classes</b></td>
      <td>12 complete object-oriented classes</td>
    </tr>
    <tr>
      <td><b>Total Functions (including methods)</b></td>
      <td>148 functions with full type signatures</td>
    </tr>
    <tr>
      <td><b>Total Lines of Code</b></td>
      <td>2,379 lines</td>
    </tr>
    <tr>
      <td><b>File Size</b></td>
      <td>67,460 bytes (~67.5 KB)</td>
    </tr>
    <tr>
      <td><b>Character Vocabulary</b></td>
      <td>84 unique characters</td>
    </tr>
    <tr>
      <td><b>Whitespace Tokens</b></td>
      <td>8,069 tokens</td>
    </tr>
    <tr>
      <td><b>Unique Words / Identifiers</b></td>
      <td>2,113 unique vocabulary items</td>
    </tr>
    <tr>
      <td><b>Abstract Syntax Tree (AST) Variety</b></td>
      <td>64 distinct AST node types</td>
    </tr>
    <tr>
      <td><b>Syntactic Validity</b></td>
      <td>100% pass rate via ast.parse</td>
    </tr>
  </tbody>
</table>

## Content Categories

TinyPy spans four foundational domains of computer science and machine learning:

### Category 1: Math and Number Theory (30 snippets)
Arithmetic operations, discrete math, statistics, and numeric manipulation:
- Basic operations: `add`, `subtract`, `multiply`, `divide`
- Properties and parity: `is_even`, `is_odd`, `is_prime`, `is_perfect_square`, `sign`, `absolute_value`
- Combinatorics and series: `factorial`, `fibonacci`, `combinations_count`, `permutations_count`, `digital_root`
- Divisibility and powers: `gcd`, `lcm`, `power`
- Bounds and thresholds: `clamp`, `clip`
- Summary statistics: `mean`, `variance`, `standard_deviation`, `median`, `harmonic_mean`, `geometric_mean`, `sum_of_squares`
- Geometry conversions: `degrees_to_radians`, `radians_to_degrees`, `hypotenuse`

### Category 2: Classic Algorithms and Data Structures (20 snippets)
Standard algorithmic patterns and data structures:
- Search: `linear_search`, `binary_search`
- Sorting: `bubble_sort`, `selection_sort`, `insertion_sort`, `merge_sort`, `quicksort`, `counting_sort`
- Linear structures: `Stack` (LIFO), `Queue` (FIFO), `Node`, `LinkedList`
- Trees and priority: `TreeNode`, `BinarySearchTree` (with insertion and traversal), `PriorityQueue`
- Graph algorithms: `breadth_first_search`, `depth_first_search`, `dijkstra_shortest_path`, `has_cycle_directed`, `topological_sort`

### Category 3: TinyTorch and Deep Learning Primitives (26 snippets)
Core building blocks of the TinyTorch framework:
- Activation functions: `relu`, `sigmoid`, `tanh`, `gelu`, `leaky_relu`, `softmax`, `log_softmax`
- Activation derivatives: `relu_backward`, `sigmoid_backward`, `tanh_backward`
- Loss objectives: `mse_loss`, `mse_loss_backward`, `cross_entropy_loss`, `bce_loss`
- Forward and backward layers: `linear_forward`, `linear_backward`, `dropout_forward`, `clip_grad_norm`, `layer_norm`, `batch_norm1d_inference`
- Encodings: `one_hot_encode`
- Neural modules and autograd: `Tensor` (scalar autograd engine), `Linear` layer, `Sequential` container
- Optimizers: `SGD` (with momentum), `Adam` (with first and second moment tracking)

### Category 4: Utility and String / List Operations (26 snippets)
Common data wrangling and string processing routines:
- List operations: `reverse_list`, `flatten`, `chunk_list`, `filter_positive`, `find_max`, `find_min`, `unique_elements`, `zip_lists`, `sliding_window`, `pad_sequence`
- Vector metrics: `normalize_vector`, `dot_product`, `euclidean_distance`, `cosine_similarity`, `manhattan_distance`
- Matrix operations: `matrix_transpose`, `matrix_multiply`
- String processing: `count_occurrences`, `caesar_cipher`, `is_palindrome`, `tokenize_whitespace`, `char_ngrams`, `run_length_encode`, `run_length_decode`, `word_frequencies`, `levenshtein_distance`

## How Students Use TinyPy

### 1. Autoregressive Language Modeling (TinyGPT)
TinyPy serves as a replacement or companion to Shakespeare when training decoder-only transformers. Students train character-level or subword models to predict the next token:

```python
# Load TinyPy corpus
with open("tinytorch/datasets/tinypy/tinypy_sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
```

### 2. Prefix-Conditioned Function Completion
Students provide a prompt consisting of a function signature and docstring, then let the model complete the implementation body:

```python
prompt = '''def relu(x: float) -> float:
    """Compute Rectified Linear Unit activation."""
'''
# Model generates:
#     return x if x > 0.0 else 0.0
```

### 3. Syntax Verification Metric
Because generated outputs are Python code, students evaluate generative fidelity by parsing sample completions:

```python
import ast

def evaluate_syntax_validity(generated_samples: list[str]) -> float:
    valid_count = 0
    for sample in generated_samples:
        try:
            ast.parse(sample)
            valid_count += 1
        except SyntaxError:
            pass
    return valid_count / len(generated_samples)
```

## Generator Script

The dataset is generated deterministically by `create_tinypy.py`.

### Verify Syntax and Print Statistics
To verify every snippet and print the structural report:
```bash
python create_tinypy.py --verify
```

### Output Corpus to a Custom Destination
To write the dataset to a specific path:
```bash
python create_tinypy.py --output tinypy_sample.txt
```

## Design Principles

- **Zero External Dependencies:** Only standard library Python modules (`math`, `typing`, `ast`, `argparse`, `pathlib`) are used.
- **Pedagogical Clarity:** Every algorithm follows standard textbook implementations with intuitive variable naming.
- **Fast Training Loop:** At ~67 KB, a 4-layer transformer can complete several epochs in under two minutes on a standard CPU.
- **Reproducible:** Generation is completely deterministic and versioned alongside TinyTorch.

## License

TinyPy is released under the [MIT License](LICENSE).
