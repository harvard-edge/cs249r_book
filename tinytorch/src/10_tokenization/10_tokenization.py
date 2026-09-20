# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
r"""
# Module 10: Tokenization - Converting Text to Numbers

Welcome to Module 10! You're about to build tokenization — the critical bridge that converts unstructured human text into discrete numerical sequences that machine learning models and neural networks can process.

## 🔗 Prerequisites & Progress
**You've Built**: Complete training pipeline with neural networks, optimizers, data loaders, and 2D spatial convolutions (`Tensor`, `Autograd`, `Linear`, `Conv2d`, `Trainer`)
**You'll Build**: Text tokenization engines — `Tokenizer` (interface contract), `CharTokenizer`, and `BPETokenizer` (Byte Pair Encoding)
**You'll Enable**: Subword representation learning that powers language modeling in Transformers

<div align="center">
  <img src="tokenization_blueprint.svg" alt="TinyTorch Execution Datapath: Module 10 Tokenization Highlighted" width="550px">
</div>

### Architectural Roadmap

| Stage | Subsystem | Primitives & Capabilities | Status |
| :--- | :--- | :--- | :--- |
| **Modules 01–09** | Foundation & Vision | `Tensor`, `Autograd`, `Linear`, `Conv2d`, `BatchNorm2d`, `Trainer` | Completed |
| **Module 10** | **Subword Tokenization** | `Tokenizer`, `CharTokenizer`, `BPETokenizer`, BPE pair merges | **Active Subsystem** |
| **Modules 11–13** | Language & Attention | `Embedding`, `SelfAttention`, `TransformerBlock` | Downstream Consumers |

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement character-based tokenization for deterministic baseline text processing
2. Build a BPE (Byte Pair Encoding) tokenizer that iteratively merges frequent subword pairs
3. Understand vocabulary data structures, ordered merge replay, and unknown token handling
4. Quantify the fundamental systems trade-off between vocabulary size ($V$) and sequence length ($T$)

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/10_tokenization/tokenization.ipynb`
**Building Side:** Code exports to `tinytorch.core.tokenization`

```python
# Final package structure:
from tinytorch.core.tokenization import Tokenizer, CharTokenizer, BPETokenizer
```

**Why this matters:**
- **Learning:** Complete tokenization system in one focused module for deep mechanical understanding
- **Production:** Proper modular structure like Hugging Face's `tokenizers`, isolating text preprocessing from tensor compute
- **Consistency:** Unified `encode()` and `decode()` contract across all tokenization strategies
- **Integration:** Directly feeds token IDs into the embedding tables in Module 11
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

**Prerequisites**: Module 10 is self-contained — it operates directly on strings, lists, and integer mappings!

**External Dependencies**:
- `numpy` (for numerical arrays and statistical analysis)
- `collections.Counter` (for frequency counting of adjacent token pairs)

**TinyTorch Dependencies**:
- Module 01 (`Tensor`): Downstream integration — converting token ID lists into tensor buffers for model ingestion

### Ingestion & Transformation Pipeline

| Stage | Data Representation | Type & Shape | Systems Operation |
| :--- | :--- | :--- | :--- |
| **Raw Input** | Natural Language Text | `str` (arbitrary length) | UTF-8 byte stream ingestion |
| **Module 10: Tokenization** | Discrete Token IDs | `list[int]` of length $T$ | BPE subword segmentation & vocab ID mapping |
| **Module 11: Embeddings** | Dense Activation Vectors | `Tensor(B, T, D)` | Row lookup in weight matrix $\mathbf{W} \in \mathbb{R}^{V \times D}$ |
| **Module 12: Attention** | Contextual Representations | `Tensor(B, T, D)` | Scaled dot-product attention ($\mathcal{O}(T^2)$ compute) |

Students completing this module establish the text ingestion bridge that powers all natural language processing in TinyTorch.
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp core.tokenization
#| export

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# %% [markdown]
r"""
## 💡 Introduction: Why Tokenization?

Neural networks operate on continuous floating-point tensors, but human communication is rooted in discrete, variable-length text. Tokenization is the foundational systems bridge that converts unstructured strings into bounded sequences of discrete integer IDs suitable for downstream tensor operations.

### The Text-to-Numbers Challenge

Consider the sentence: `"Hello, world!"` — how does a neural network ingest and compute over these characters?

<div align="center">
  <img src="tokenization_pipeline.svg" alt="TinyTorch Tokenization Pipeline: Training vs Encoding" width="650px">
</div>

### Comparing Text and Numerical Representations

| Dimension | Raw Text Domain | Neural Network Domain | MLSys Adaptation Strategy |
| :--- | :--- | :--- | :--- |
| **Data Primitive** | UTF-8 byte stream / characters | Floating-point tensors ($\mathbb{R}^D$) | Tokenizer maps string slice $\to$ integer ID $\to$ dense vector |
| **Vocabulary Bound** | Infinite (new words, typos, slang) | Fixed memory table of size $V$ | Subword BPE splits rare words into known subword units |
| **Sequence Topology** | Variable character length | Fixed or batched sequence length $T$ | Padding and truncation ensure static tensor shapes $(B, T)$ |
| **Computational Cost** | $\mathcal{O}(1)$ string indexing | Quadratic in sequence length: $\mathcal{O}(T^2)$ | Shorter subword sequences reduce attention compute by $>3\times$ |

### The Four-Step Ingestion Pipeline

1. **Token Segmentation**: Partition text into discrete semantic or subword pieces (characters, subwords, words).
2. **Vocabulary Lookup**: Map each token string to a unique integer index $t_i \in \{0, 1, \dots, V - 1\}$.
3. **Out-of-Vocabulary (OOV) Handling**: Map unseen characters or corrupted inputs to a reserved `<UNK>` token (ID $0$).
4. **Reconstructive Decoding**: Invert the vocabulary map to recover human-readable text for generation and inspection.
"""

# %% [markdown]
r"""
## 📐 Foundations: Tokenization Strategies

Different tokenization paradigms navigate the fundamental trade-off between **vocabulary size ($V$)** and **sequence length ($T$)**.

### Character-Level Tokenization

Each observed character receives an individual vocabulary index. Given training corpus $\mathcal{C} = \{\text{"hello"}, \text{"world"}\}$:

$$\Sigma = \{\text{'d'}, \text{'e'}, \text{'h'}, \text{'l'}, \text{'o'}, \text{'r'}, \text{'w'}\}, \qquad \mathcal{V} = [\text{<UNK>}] \cup \operatorname{sorted}(\Sigma)$$

The bidirectional mappings operate deterministically:

$$\operatorname{encode}(c) = \begin{cases} \operatorname{index}(c) & \text{if } c \in \mathcal{V} \\ 0 & (\text{<UNK>}) \end{cases}, \qquad \operatorname{decode}(i) = \mathcal{V}[i]$$

$$\text{"hello"} \quad \xrightarrow{\text{encode}} \quad [3, 2, 4, 4, 5] \quad \xrightarrow{\text{decode}} \quad \text{"hello"}$$

### Strategy Comparison: Character vs Subword vs Word

| Metric / Property | Character-Level | Subword (BPE) | Word-Level |
| :--- | :--- | :--- | :--- |
| **Vocabulary Size ($V$)** | Small ($\approx 100 - 256$) | Balanced ($\approx 10{,}000 - 50{,}000$) | Enormous ($>100{,}000$) |
| **Sequence Length ($T$) for 1k words** | Very Long ($\approx 5{,}000$ tokens) | Compact ($\approx 1{,}300$ tokens) | Minimal ($1{,}000$ tokens) |
| **Out-of-Vocabulary (OOV) Risk** | Minimal (covers alphabet) | Zero with byte fallback | High (plurals, names trigger `<UNK>`) |
| **Embedding Table Memory ($D=4096$)** | $\approx 1.6\text{ MB}$ (cache-resident) | $\approx 160\text{ MB} - 800\text{ MB}$ | $>1.6\text{ GB}$ (spills to DRAM) |
| **Attention Matrix ($T \times T$)** | $25\text{M}$ entries per head | $\approx 1.7\text{M}$ entries per head | $1\text{M}$ entries per head |
| **Relative Attention Compute** | **$14.8\times$ baseline** | **$1.7\times$ baseline** | **$1.0\times$ (fastest attention)** |

BPE achieves the sweet spot: by spending a modest memory budget on a subword vocabulary, it shrinks sequence length by $\approx 3.8\times$, slashing quadratic attention compute while retaining universal coverage.
"""

# %% [markdown]
r"""
## 🏗️ Implementation: Building Tokenization Systems

Let's construct our tokenization architecture step by step, enforcing consistent interfaces and verifying numerical integrity at every stage.

### Tokenization System Architecture

| Component | Class / Method | Systems Responsibility | State & Data Structures |
| :--- | :--- | :--- | :--- |
| **Base Contract** | `Tokenizer` | Defines abstract `encode` / `decode` API | `TOK_UNKNOWN = '<UNK>'`, `TOK_EOW = ' '` |
| **Character Tokenizer** | `CharTokenizer` | Alphabet-level encoding & decoding | `vocab: list[str]`, `char_to_id: dict`, `id_to_char: dict` |
| **Subword BPE** | `BPETokenizer` | Greedy frequency merge training & replay | `merges: list[tuple]`, `token_to_id: dict`, `id_to_token: dict` |
| **Batch Pipeline** | `tokenize_dataset` | Pads & truncates variable-length text | Uniform tensor buffer shapes $(B, T)$ |

### Base Tokenizer Interface

All tokenizers implement a common abstract contract, guaranteeing that downstream components (data loaders, embedding layers, evaluators) interact with identical method signatures:

| Operation | Method Signature | Expected Input | Return Value | Invariant Contract |
| :--- | :--- | :--- | :--- | :--- |
| **Encoding** | `encode(text: str)` | Raw UTF-8 string | `list[int]` | Every element $t \in [0, V-1]$; unseen $\to 0$ |
| **Decoding** | `decode(tokens: list[int])` | Sequence of token IDs | `str` | Known tokens round-trip deterministically |

This ensures consistent behavior across different tokenization strategies.
"""

# %% nbgrader={"grade": false, "grade_id": "base-tokenizer", "solution": true}
#| export
class Tokenizer:
    """
    Base tokenizer class providing the interface for all tokenizers.

    This defines the contract that all tokenizers must follow:
    - encode(): text → list of token IDs
    - decode(): list of token IDs → text
    """

    # Predefined symbolic tokens for common use cases
    TOK_UNKNOWN = '<UNK>'   # UNKNOWN
    # Words come from str.split(), so a space cannot collide with word content.
    # Diagrams write this boundary as </w> to make the otherwise invisible suffix clear.
    TOK_EOW = ' '           # END OF WORD

    def encode(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.

        TODO: Define the interface; the real encoders live in the subclasses

        APPROACH:
        1. This base method only states the contract, so raise NotImplementedError
           with a message that points the caller at CharTokenizer or BPETokenizer
        2. Each subclass overrides encode() to return a list of integer token IDs

        EXAMPLE:
        >>> Tokenizer().encode("abc")
        NotImplementedError: encode() not implemented in base Tokenizer class ...
        >>> CharTokenizer(['a', 'b', 'c']).encode("abc")
        [1, 2, 3]
        """
        ### BEGIN SOLUTION role="scaffold"
        raise NotImplementedError(
            f"encode() not implemented in base Tokenizer class\n"
            f"  ❌ Called encode() on abstract base class {self.__class__.__name__}\n"
            f"  💡 Tokenizer is an interface - use a concrete implementation like CharTokenizer or BPETokenizer\n"
            f"  🔧 Example: tokenizer = CharTokenizer(['a', 'b', 'c']); tokenizer.encode('abc')"
        )
        ### END SOLUTION

    def decode(self, tokens: List[int]) -> str:
        """
        Convert list of token IDs back to text.

        TODO: Define the interface; the real decoders live in the subclasses

        APPROACH:
        1. This base method only states the contract, so raise NotImplementedError
           with a message that points the caller at CharTokenizer or BPETokenizer
        2. Each subclass overrides decode() to return the reconstructed text

        EXAMPLE:
        >>> Tokenizer().decode([1, 2, 3])
        NotImplementedError: decode() not implemented in base Tokenizer class ...
        >>> CharTokenizer(['a', 'b', 'c']).decode([1, 2, 3])
        "abc"
        """
        ### BEGIN SOLUTION role="scaffold"
        raise NotImplementedError(
            f"decode() not implemented in base Tokenizer class\n"
            f"  ❌ Called decode() on abstract base class {self.__class__.__name__}\n"
            f"  💡 Tokenizer is an interface - use a concrete implementation like CharTokenizer or BPETokenizer\n"
            f"  🔧 Example: tokenizer = CharTokenizer(['a', 'b', 'c']); tokenizer.decode([1, 2, 3])"
        )
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Base Tokenizer Interface

This test validates our base tokenizer defines the correct interface for all implementations.

**What we're testing**: Abstract interface definition with NotImplementedError
**Why it matters**: Ensures consistent API across all tokenizer types
**Expected**: Base class raises NotImplementedError for both encode and decode
"""

# %% nbgrader={"grade": true, "grade_id": "test-base-tokenizer", "locked": true, "points": 5}
def test_unit_base_tokenizer():
    """🧪 Test base tokenizer interface."""
    print("🧪 Unit Test: Base Tokenizer Interface...")

    # Test that base class defines the interface
    tokenizer = Tokenizer()

    # Should raise NotImplementedError for both methods
    try:
        tokenizer.encode("test")
        assert False, "encode() should raise NotImplementedError"
    except NotImplementedError:
        pass

    try:
        tokenizer.decode([1, 2, 3])
        assert False, "decode() should raise NotImplementedError"
    except NotImplementedError:
        pass

    print("✅ Base tokenizer interface works correctly!")

if __name__ == "__main__":
    test_unit_base_tokenizer()

# %% [markdown]
r"""
## 🏗️ Character-Level Tokenizer

The character-level tokenizer decomposes text into individual constituent characters. It guarantees deterministic coverage of all symbols present in the training alphabet, but produces sequences proportional to raw character length.

### Step-by-Step Character Tokenization

Given corpus $\mathcal{C} = [\text{"hello"}, \text{"world"}]$:

1. **Alphabet Extraction & Sorting**:
   $$\Sigma = \operatorname{sorted}(\operatorname{unique}(\mathcal{C})) = [\text{' '}, \text{'d'}, \text{'e'}, \text{'h'}, \text{'l'}, \text{'o'}, \text{'r'}, \text{'w'}]$$

2. **Vocabulary Construction**:
   Prepend special token `<UNK>` at index $0$ so unseen symbols map cleanly to an out-of-vocabulary ID:

| Token Symbol | Vocabulary ID | Byte Value | Systems Role |
| :--- | :--- | :--- | :--- |
| `<UNK>` | `0` | — | Fallback out-of-vocabulary indicator |
| `' '` (space) | `1` | `0x20` | Explicit inter-word delimiter |
| `'d'` | `2` | `0x64` | Base alphabet character |
| `'e'` | `3` | `0x65` | Base alphabet character |
| `'h'` | `4` | `0x68` | Base alphabet character |
| `'l'` | `5` | `0x6C` | Base alphabet character |
| `'o'` | `6` | `0x6F` | Base alphabet character |
| `'r'` | `7` | `0x72` | Base alphabet character |
| `'w'` | `8` | `0x77` | Base alphabet character |

3. **Encoding Pipeline**:
   $$\mathbf{x} = \text{"hello"} \quad \xrightarrow{\text{lookup}} \quad [4, 3, 5, 5, 6]$$

4. **Decoding Reconstruction**:
   $$[4, 3, 5, 5, 6] \quad \xrightarrow{\text{reverse lookup}} \quad \text{"hello"}$$
"""

# %% nbgrader={"grade": false, "grade_id": "char-tokenizer", "solution": true}
#| export
class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer that treats each character as a separate token.

    This is the simplest tokenization approach - every character in the
    vocabulary gets its own unique ID.
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        """
        Initialize character tokenizer.

        TODO: Set up vocabulary mappings

        APPROACH:
        1. Store vocabulary list
        2. Create char→id and id→char mappings
        3. Handle special tokens (unknown character)

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.vocab_size
        4  # 3 chars + 1 unknown token
        """
        ### BEGIN SOLUTION role="scaffold"
        if vocab is None:
            vocab = []

        # Add special unknown token
        self.vocab = [Tokenizer.TOK_UNKNOWN] + vocab
        self.vocab_size = len(self.vocab)

        # Create bidirectional mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # Store unknown token ID
        self.unk_id = 0
        ### END SOLUTION

    def build_vocab(self, corpus: List[str]) -> None:
        """
        Build vocabulary from a corpus of text.

        TODO: Extract unique characters and build vocabulary

        APPROACH:
        1. Collect all unique characters from corpus
        2. Sort for consistent ordering
        3. Rebuild mappings with new vocabulary

        HINTS:
        - Use set() to find unique characters
        - Join all texts then convert to set
        - Don't forget the <UNK> token
        """
        ### BEGIN SOLUTION role="scaffold"
        # Collect all unique characters
        all_chars = set()
        for text in corpus:
            all_chars.update(text)

        # Sort for consistent ordering
        unique_chars = sorted(all_chars)

        # Rebuild vocabulary with <UNK> token first
        self.vocab = [Tokenizer.TOK_UNKNOWN] + unique_chars
        self.vocab_size = len(self.vocab)

        # Rebuild mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        ### END SOLUTION

    def encode(self, text: str) -> List[int]:
        """
        Encode text to list of character IDs.

        TODO: Convert each character to its vocabulary ID

        APPROACH:
        1. Iterate through each character in text
        2. Look up character ID in vocabulary
        3. Use unknown token ID for unseen characters

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['h', 'e', 'l', 'o'])
        >>> tokenizer.encode("hello")
        [1, 2, 3, 3, 4]  # maps to h,e,l,l,o
        """
        ### BEGIN SOLUTION role="scaffold"
        tokens = []
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_id))
        return tokens
        ### END SOLUTION

    def decode(self, tokens: List[int]) -> str:
        """
        Decode list of token IDs back to text.

        TODO: Convert each token ID back to its character

        APPROACH:
        1. Look up each token ID in vocabulary
        2. Join characters into string
        3. Handle invalid token IDs gracefully

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['h', 'e', 'l', 'o'])
        >>> tokenizer.decode([1, 2, 3, 3, 4])
        "hello"
        """
        ### BEGIN SOLUTION role="scaffold"
        chars = []
        for token_id in tokens:
            # Use unknown token for invalid IDs
            char = self.id_to_char.get(token_id, Tokenizer.TOK_UNKNOWN)
            chars.append(char)
        return ''.join(chars)
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Character Tokenizer

This test validates our character tokenizer works correctly with vocabulary building, encoding, and decoding.

**What we're testing**: Character-level tokenization with vocabulary management
**Why it matters**: Foundation for text processing with known-character coverage
**Expected**: Correct encoding/decoding, unknown character handling, vocabulary building
"""

# %% nbgrader={"grade": true, "grade_id": "test-char-tokenizer", "locked": true, "points": 15}
def test_unit_char_tokenizer():
    """🧪 Test character tokenizer implementation."""
    print("🧪 Unit Test: Character Tokenizer...")

    # Test basic functionality
    vocab = ['h', 'e', 'l', 'o', ' ', 'w', 'r', 'd']
    tokenizer = CharTokenizer(vocab)

    # Test vocabulary setup
    assert tokenizer.vocab_size == 9  # 8 chars + UNK
    assert tokenizer.vocab[0] == Tokenizer.TOK_UNKNOWN
    assert 'h' in tokenizer.char_to_id

    # Test encoding
    text = "hello"
    tokens = tokenizer.encode(text)
    expected = [1, 2, 3, 3, 4]  # h,e,l,l,o (based on actual vocab order)
    assert tokens == expected, f"Expected {expected}, got {tokens}"

    # Test decoding
    decoded = tokenizer.decode(tokens)
    assert decoded == text, f"Expected '{text}', got '{decoded}'"

    # Test unknown character handling
    tokens_with_unk = tokenizer.encode("hello!")
    assert tokens_with_unk[-1] == 0  # '!' should map to <UNK>

    # Test vocabulary building
    # Expected behaviour: Invoking build_vocab overwrites the vocabulary passed
    # in __init__ above, so after this the 'w' should no longer be a token
    corpus = ["hola mundo", "test text"]
    tokenizer.build_vocab(corpus)
    assert 'w' not in tokenizer.char_to_id
    assert 't' in tokenizer.char_to_id
    assert 'x' in tokenizer.char_to_id

    print("✅ Character tokenizer works correctly!")

if __name__ == "__main__":
    test_unit_char_tokenizer()

# %% [markdown]
"""
Character tokenization provides a simple, robust foundation for text processing. The key insight is that with a small vocabulary (typically <100 characters), we can represent any text drawn from the characters the corpus contained; only a character the corpus never showed falls back to `<UNK>`, which the test above exercised with `'!'`.

**Trade-offs**:
- **Pro**: Out-of-vocabulary is rare (a character has to be unseen, not a word), and any language works once its characters are in the corpus
- **Con**: Long sequences (1 char = 1 token), limited semantic understanding
- **Use case**: When robustness is more important than efficiency
"""

# %% [markdown]
r"""
## 🏗️ Byte Pair Encoding (BPE) Tokenizer

Byte Pair Encoding (BPE) is the industry-standard subword tokenization algorithm utilized by modern Large Language Models, including GPT-2, GPT-4, Llama 3, and Mistral. BPE dynamically constructs a subword vocabulary by iteratively identifying and fusing the most frequently co-occurring adjacent token pairs.

<div align="center">
  <img src="bpe_merge_progression.svg" alt="BPE Merge Progression: From Characters to Subwords" width="650px">
</div>

In our implementation, `</w>` denotes the end-of-word boundary suffix stored internally as `Tokenizer.TOK_EOW = ' '`. Because raw text is pre-segmented on whitespace, this boundary marker cannot collide with literal text. The base alphabet initializes with both interior and word-final variants of every seen character, guaranteeing that unseen word positions still retain recognized character representations.

<div align="center">
  <img src="bpe_merge_collapse.svg" alt="BPE Training Pipeline: Corpus Frequencies, Pair Counting, and Vocab Construction" width="650px">
</div>

### The BPE Training Progression

Consider the training corpus $\mathcal{C} = [\text{"hello"}, \text{"hello"}, \text{"help"}]$, containing $N_{\text{hello}} = 2$ and $N_{\text{help}} = 1$:

#### Step 1: Base Alphabet Initialization
Every word is decomposed into its individual character tokens, fusing the end-of-word delimiter to the final character:

$$\text{"hello"} \implies [\text{'h'}, \text{'e'}, \text{'l'}, \text{'l'}, \text{'o</w>'}], \qquad \text{"help"} \implies [\text{'h'}, \text{'e'}, \text{'l'}, \text{'p</w'}]$$

$$\mathcal{V}_{\text{base}} = [\text{<UNK>}, \text{'e'}, \text{'h'}, \text{'l'}, \text{'o</w>'}, \text{'p</w>'}]$$

#### Step 2: Frequency-Weighted Pair Counting
For each adjacent pair $(t_i, t_{i+1})$, accumulate occurrences weighted by the word frequency:

$$\operatorname{freq}(p) = \sum_{w \in \mathcal{C}} \operatorname{count}(p, w) \times \operatorname{freq}(w)$$

| Candidate Pair | Source Words | Frequency Calculation | Weighted Count | Priority Status |
| :--- | :--- | :--- | :--- | :--- |
| `('h', 'e')` | `"hello"`, `"help"` | $1 \times 2 + 1 \times 1$ | **3** | **Tied Rank #1 (Selected by first-seen)** |
| `('e', 'l')` | `"hello"`, `"help"` | $1 \times 2 + 1 \times 1$ | **3** | Tied Rank #1 |
| `('l', 'l')` | `"hello"` | $1 \times 2$ | **2** | Rank #3 |
| `('l', 'o</w>')`| `"hello"` | $1 \times 2$ | **2** | Rank #3 |
| `('l', 'p</w>')`| `"help"` | $1 \times 1$ | **1** | Rank #5 |

#### Step 3: Greedy Merge Selection & In-Place Replacement
The winning pair `('h', 'e')` is merged into a single atomic token `'he'`:

$$\operatorname{merge}((\text{'h'}, \text{'e'}) \to \text{'he'}): \quad \begin{cases} [\text{'h'}, \text{'e'}, \text{'l'}, \text{'l'}, \text{'o</w>'}] & \implies [\mathbf{\text{'he'}}, \text{'l'}, \text{'l'}, \text{'o</w'}] \\ [\text{'h'}, \text{'e'}, \text{'l'}, \text{'p</w>'}] & \implies [\mathbf{\text{'he'}}, \text{'l'}, \text{'p</w'}] \end{cases}$$

$$\mathcal{V} \leftarrow \mathcal{V} \cup [\mathbf{\text{'he'}}]$$

#### Step 4: Iterative Re-Count and Merge History

| Iteration | Most Frequent Pair | Merged Token | Post-Merge Representations | Vocab Growth |
| :--- | :--- | :--- | :--- | :--- |
| **Start** | — | — | `['h', 'e', 'l', 'l', 'o</w>']`, `['h', 'e', 'l', 'p</w>']` | $|\mathcal{V}| = 6$ |
| **Iter 1** | `('h', 'e')` | `'he'` | `['he', 'l', 'l', 'o</w>']`, `['he', 'l', 'p</w>']` | $|\mathcal{V}| = 7$ |
| **Iter 2** | `('he', 'l')` | `'hel'` | `['hel', 'l', 'o</w>']`, `['hel', 'p</w>']` | $|\mathcal{V}| = 8$ |
| **Iter 3** | `('hel', 'l')` | `'hell'` | `['hell', 'o</w>']`, `['hel', 'p</w>']` | $|\mathcal{V}| = 9$ |
| **Iter 4** | `('hell', 'o</w>')`| `'hello</w>'` | `['hello</w>']`, `['hel', 'p</w>']` | $|\mathcal{V}| = 10$ |
| **Iter 5** | `('hel', 'p</w>')` | `'help</w>'` | `['hello</w>']`, `['help</w>']` | $|\mathcal{V}| = 11$ |

**Systems Takeaway**: On this two-word corpus, frequent whole words collapse into single atomic tokens ($5\text{ tokens} \to 1\text{ token}$, an exact **$5\times$ compression**). On large-scale pretraining corpora, BPE spends its vocabulary budget discovering common morphological stems, prefixes, and roots!
"""

# %% [markdown]
r"""
### Counting Byte Pairs

The core inner loop of BPE tallies every adjacent token pair $(t_i, t_{i+1})$ across all tokenized words, scaling each observation by the containing word's empirical corpus frequency:

$$\operatorname{freq}(t_i, t_{i+1}) = \sum_{w \in \mathcal{D}} \operatorname{freq}(w) \cdot \sum_{j=0}^{|w|-2} \mathbb{I}\Big(w_j = t_i \;\land\; w_{j+1} = t_{i+1}\Big)$$

| Word $w$ | Corpus Count | Token Sequence | Extracted Adjacent Pairs | Pair Contributions |
| :--- | :--- | :--- | :--- | :--- |
| `"hello"` | $2$ | `['h', 'e', 'l', 'l', 'o</w>']` | `('h','e'), ('e','l'), ('l','l'), ('l','o</w>')` | $+2$ to each pair |
| `"help"` | $1$ | `['h', 'e', 'l', 'p</w>']` | `('h','e'), ('e','l'), ('l','p</w>')` | $+1$ to each pair |
| **Totals** | $3\text{ words}$ | — | — | **`('h','e')`: 3, `('e','l')`: 3, `('l','l')`: 2, `('l','o</w>')`: 2, `('l','p</w>')`: 1** |
"""

# %% nbgrader={"grade": false, "grade_id": "bpe-count-pairs", "solution": true}
#| export
def _count_byte_pairs(word_tokens: Dict[str, List[str]], word_freq: Counter) -> Counter:
    """
    Count frequency of all adjacent token pairs across all words.

    Each pair's count is weighted by how often its containing word appears
    in the corpus, so frequent words contribute more to pair statistics.

    TODO: Count all adjacent pairs weighted by word frequency

    APPROACH:
    1. Iterate through each word and its frequency
    2. Get all adjacent pairs from the word's tokens
    3. Add the word's frequency to each pair's count
    4. Return the Counter of pair frequencies

    EXAMPLE:
    >>> word_tokens = {"hello": ['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW]}
    >>> word_freq = Counter({"hello": 3})
    >>> counts = _count_byte_pairs(word_tokens, word_freq)
    >>> counts[('h', 'e')]
    3

    HINT: For each word, walk i from 0 to len(tokens) - 2 and add freq to the count
    of (tokens[i], tokens[i + 1])
    """
    ### BEGIN SOLUTION
    pair_counts = Counter()

    for word, freq in word_freq.items():
        tokens = word_tokens[word]
        # Count adjacent pairs
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += freq

    return pair_counts
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Count Byte Pairs

**What we're testing**: Frequency-weighted pair counting across multiple words
**Why it matters**: The most frequent pair determines which merge to perform next
**Expected**: Pairs appearing in frequent words get higher counts
"""

# %% nbgrader={"grade": true, "grade_id": "test-bpe-count-pairs", "locked": true, "points": 5}
def test_unit_count_byte_pairs():
    """🧪 Test byte pair counting with frequency weighting."""
    print("🧪 Unit Test: Count Byte Pairs...")

    # Two words: "hello" appears 3 times, "help" appears 1 time
    word_tokens = {
        "hello": ['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW],
        "help": ['h', 'e', 'l', 'p'+Tokenizer.TOK_EOW]
    }
    word_freq = Counter({"hello": 3, "help": 1})

    counts = _count_byte_pairs(word_tokens, word_freq)

    # ('h','e') appears in both words: 3 + 1 = 4
    assert counts[('h', 'e')] == 4, f"Expected 4, got {counts[('h', 'e')]}"

    # ('e','l') appears in both words: 3 + 1 = 4
    assert counts[('e', 'l')] == 4, f"Expected 4, got {counts[('e', 'l')]}"

    # ('l','l') appears only in "hello" (freq 3)
    assert counts[('l', 'l')] == 3, f"Expected 3, got {counts[('l', 'l')]}"

    # ('l','p'+Tokenizer.TOK_EOW) appears only in "help" (freq 1)
    assert counts[('l', 'p'+Tokenizer.TOK_EOW)] == 1, f"Expected 1, got {counts[('l', 'p'+Tokenizer.TOK_EOW)]}"

    # Empty case
    empty_counts = _count_byte_pairs({}, Counter())
    assert len(empty_counts) == 0

    print("✅ Byte pair counting works correctly!")

if __name__ == "__main__":
    test_unit_count_byte_pairs()

# %% [markdown]
r"""
### Merging a Byte Pair

Once the highest-frequency pair $p = (t_A, t_B)$ is identified, it must be atomically merged into a combined token $t_{AB} = t_A \circ t_B$ across all active word token sequences.

A single-pass linear scan $\mathcal{O}(L)$ inspects adjacent tokens, advancing by $2$ positions upon a successful match and $1$ position otherwise:

| Scan Index $i$ | Inspected Tokens | Match Condition | Emitted Output | Pointer Advance |
| :--- | :--- | :--- | :--- | :--- |
| **$i = 0$** | `tokens[0]='h'`, `tokens[1]='e'` | **Match:** `('h', 'e') == ('h', 'e')` | Append `'he'` | $i \leftarrow i + 2$ |
| **$i = 2$** | `tokens[2]='l'`, `tokens[3]='l'` | No Match: `('l', 'l') != ('h', 'e')` | Append `'l'` | $i \leftarrow i + 1$ |
| **$i = 3$** | `tokens[3]='l'`, `tokens[4]='o</w>'`| No Match: `('l', 'o</w>') != ('h', 'e')`| Append `'l'` | $i \leftarrow i + 1$ |
| **$i = 4$** | `tokens[4]='o</w>'` (end of word) | Boundary reached ($i = L - 1$) | Append `'o</w>'`| $i \leftarrow i + 1$ |

$$\text{"hello"} : \quad [\text{'h'}, \text{'e'}, \text{'l'}, \text{'l'}, \text{'o</w>'}] \quad \xrightarrow{\operatorname{merge}((\text{'h'}, \text{'e'}) \to \text{'he'})} \quad [\mathbf{\text{'he'}}, \text{'l'}, \text{'l'}, \text{'o</w'}]$$
"""

# %% nbgrader={"grade": false, "grade_id": "bpe-merge-pair", "solution": true}
#| export
def _merge_pair(word_tokens: Dict[str, List[str]], pair: Tuple[str, str]) -> str:
    """
    Merge one pair everywhere it occurs in all word token lists.

    The caller decides which pair (during training, the most frequent one;
    during encoding, each learned merge in order). This function scans every
    word's tokens and replaces adjacent occurrences of the pair with a single
    concatenated token. Modifies word_tokens in place and returns the new
    merged token string.

    TODO: Merge the given pair in all word token sequences

    APPROACH:
    1. For each word in word_tokens, scan through its token list
    2. When two adjacent tokens match the pair, replace with concatenation
    3. Otherwise keep the token as-is
    4. Update word_tokens in place, return the merged token string

    EXAMPLE:
    >>> word_tokens = {"hello": ['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW]}
    >>> merged = _merge_pair(word_tokens, ('h', 'e'))
    >>> word_tokens["hello"]
    ['he', 'l', 'l', 'o'+Tokenizer.TOK_EOW]
    >>> merged
    'he'

    HINTS:
    - Use a while loop with index i to scan each word's tokens
    - When pair matches at position i, append pair[0]+pair[1] and skip 2
    - Otherwise append tokens[i] and advance by 1
    """
    ### BEGIN SOLUTION
    merged_token = pair[0] + pair[1]

    for word in word_tokens:
        tokens = word_tokens[word]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and
                tokens[i] == pair[0] and
                tokens[i + 1] == pair[1]):
                # Merge pair
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        word_tokens[word] = new_tokens

    return merged_token
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Merge Pair

**What we're testing**: In-place merging of a specific pair across all word token lists
**Why it matters**: This is the core operation that builds the BPE vocabulary
**Expected**: Adjacent pair occurrences replaced by concatenated token, non-matching tokens preserved
"""

# %% nbgrader={"grade": true, "grade_id": "test-bpe-merge-pair", "locked": true, "points": 10}
def test_unit_merge_pair():
    """🧪 Test byte pair merging across word token lists."""
    print("🧪 Unit Test: Merge Pair...")

    # Set up word tokens
    word_tokens = {
        "hello": ['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW],
        "help": ['h', 'e', 'l', 'p'+Tokenizer.TOK_EOW]
    }

    # Merge ('h', 'e') → 'he'
    merged = _merge_pair(word_tokens, ('h', 'e'))
    assert merged == 'he', f"Expected 'he', got '{merged}'"
    assert word_tokens["hello"] == ['he', 'l', 'l', 'o'+Tokenizer.TOK_EOW], \
        f"Expected ['he', 'l', 'l', 'o'+TOK_EOW], got {word_tokens['hello']}"
    assert word_tokens["help"] == ['he', 'l', 'p'+Tokenizer.TOK_EOW], \
        f"Expected ['he', 'l', 'p'+TOK_EOW], got {word_tokens['help']}"

    # Now merge ('l', 'l') → 'll' (only affects "hello")
    merged2 = _merge_pair(word_tokens, ('l', 'l'))
    assert merged2 == 'll', f"Expected 'll', got '{merged2}'"
    assert word_tokens["hello"] == ['he', 'll', 'o'+Tokenizer.TOK_EOW], \
        f"Expected ['he', 'll', 'o'+TOK_EOW], got {word_tokens['hello']}"
    # "help" unchanged (no 'l','l' pair)
    assert word_tokens["help"] == ['he', 'l', 'p'+Tokenizer.TOK_EOW], \
        f"help should be unchanged, got {word_tokens['help']}"

    # Edge case: pair not present
    word_tokens_empty = {"ab": ['a', 'b'+Tokenizer.TOK_EOW]}
    _merge_pair(word_tokens_empty, ('x', 'y'))
    assert word_tokens_empty["ab"] == ['a', 'b'+Tokenizer.TOK_EOW], "No-match merge should leave tokens unchanged"

    print("✅ Byte pair merging works correctly!")

if __name__ == "__main__":
    test_unit_merge_pair()

# %% [markdown]
r"""
### BPETokenizer: Assembling the Pieces

You have implemented the dual operational primitives of BPE:
1. `_count_byte_pairs`: Identifies the globally dominant adjacent pair $\operatorname{argmax}_{p} \operatorname{freq}(p)$.
2. `_merge_pair`: Atomically fuses that pair in-place across all tokenized words.

The complete `BPETokenizer` composes these primitives into an end-to-end tokenizer pipeline:

| Pipeline Stage | Method | Systems Operation | Algorithmic Role |
| :--- | :--- | :--- | :--- |
| **Training** | `train(corpus)` | Greedy merge loop | Alternates pair counting and merging until $|\mathcal{V}| = V_{\text{target}}$ |
| **Encoding** | `encode(text)` | Sequential merge replay | Decomposes text to characters, then applies merges in exact learned order |
| **Decoding** | `decode(tokens)` | Array lookup & join | Maps integer IDs to subword strings, resolving space boundaries |

#### The Strict Replay Ordering Invariant

> [!IMPORTANT]
> Merges must be evaluated during `encode()` in the **exact chronological sequence** they were discovered during `train()`. Later subword units are built on top of earlier ones (e.g., `'h' + 'e' \to 'he'`, followed by `'he' + 'l' \to 'hel'`). Applying merges out of order disrupts the subword DAG and produces invalid, uncompressed tokenizations!
"""

# %% nbgrader={"grade": false, "grade_id": "bpe-tokenizer", "solution": true}
#| export
class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer that learns subword units.

    BPE works by:
    1. Starting with character-level vocabulary
    2. Finding most frequent character pairs
    3. Merging frequent pairs into single tokens
    4. Repeating until desired vocabulary size
    """

    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.

        TODO: Set up basic tokenizer state

        APPROACH:
        1. Store target vocabulary size
        2. Initialize empty vocabulary and merge rules
        3. Set up mappings for encoding/decoding

        EXAMPLE:
        >>> tokenizer = BPETokenizer(vocab_size=1000)
        >>> tokenizer.target_vocab_size
        1000
        >>> tokenizer.vocab_size
        0  # No learned vocabulary until train() runs

        HINT: Initialize vocab and merges as empty lists, mappings as empty dicts
        """
        ### BEGIN SOLUTION role="scaffold"
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        self.target_vocab_size = vocab_size
        self.vocab = []
        self.merges = []  # List of (pair, new_token) merges
        self.token_to_id = {}
        self.id_to_token = {}
        ### END SOLUTION

    @property
    def vocab_size(self) -> int:
        """Actual vocabulary size, suitable for allocating an embedding table."""
        return len(self.vocab)

    def _get_word_tokens(self, word: str) -> List[str]:
        """
        Convert word to list of characters with end-of-word marker.

        TODO: Tokenize word into character sequence

        APPROACH:
        1. Split word into characters
        2. Add a space boundary to the last character (shown as </w> in diagrams)
        3. Return list of tokens

        EXAMPLE:
        >>> tokenizer._get_word_tokens("hello")
        ['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW]

        HINT: Use list() to split word into characters, then modify the last element
        """
        ### BEGIN SOLUTION role="scaffold"
        if not word:
            return []

        tokens = list(word)
        tokens[-1] += Tokenizer.TOK_EOW  # Mark end of word
        return tokens
        ### END SOLUTION

    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """
        Get all adjacent pairs from word tokens.

        TODO: Extract all consecutive character pairs

        APPROACH:
        1. Iterate through adjacent tokens
        2. Create pairs of consecutive tokens
        3. Return set of unique pairs

        EXAMPLE:
        >>> tokenizer._get_pairs(['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW])
        {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o'+Tokenizer.TOK_EOW)}

        HINT: Loop from 0 to len(word_tokens)-1 and create tuple pairs
        """
        ### BEGIN SOLUTION role="scaffold"
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs
        ### END SOLUTION

    def train(self, corpus: List[str], vocab_size: int = None) -> None:
        """
        Train BPE on corpus to learn merge rules.

        This is the composition function: it initializes character vocabulary,
        then runs a greedy merge loop using _count_byte_pairs() to find the
        best pair and _merge_pair() to apply it.

        The corpus is a list of texts. Each text is split on whitespace, exactly
        as encode() does, so the symbols the trainer merges (with </w> marking
        each word's last character) are the symbols encode() will later look up.
        Merges never straddle a word boundary. Every observed character gets
        both an internal and word-final form. This minimum alphabet is retained
        even when it exceeds target_vocab_size; vocab_size always reports the
        actual count. Training can also stop below the target when no pairs remain.

        TODO: Implement BPE training using the greedy merge loop

        APPROACH:
        1. Build initial character vocabulary from corpus words
        2. Loop: count pairs, find best, merge it, add to vocab
        3. Stop when vocab reaches target size or no pairs remain
        4. Build final mappings

        EXAMPLE:
        >>> corpus = ["hello", "hello", "help"]
        >>> tokenizer = BPETokenizer(vocab_size=20)
        >>> tokenizer.train(corpus)
        >>> len(tokenizer.vocab) <= 20
        True

        HINTS:
        - Use _get_word_tokens() for initial character tokenization
        - Use _count_byte_pairs(word_tokens, word_freq) to find pair frequencies
        - Use _merge_pair(word_tokens, best_pair) to apply the merge
        - Don't forget to call _build_mappings() at the end
        """
        ### BEGIN SOLUTION role="scaffold"
        if vocab_size is not None:
            if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
                raise ValueError("vocab_size must be a positive integer")
            self.target_vocab_size = vocab_size

        # Count word frequencies and initialize character vocabulary.
        # Split each text on whitespace exactly as encode() does, so that the
        # symbols the trainer merges (with </w> on each word's last character)
        # are the symbols encode() will later look up.
        word_freq = Counter(word for text in corpus for word in text.split())
        vocab = set()
        word_tokens = {}

        for word in word_freq:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            # Every seen character must work both inside and at the end of a word.
            # Keeping both forms lets unseen words reuse the known alphabet.
            vocab.update(word)
            vocab.update(char + Tokenizer.TOK_EOW for char in word)

        self.vocab = sorted(vocab)
        if Tokenizer.TOK_UNKNOWN not in vocab:
            self.vocab = [Tokenizer.TOK_UNKNOWN] + self.vocab

        # Greedy merge loop: count pairs, merge best, repeat
        self.merges = []

        while len(self.vocab) < self.target_vocab_size:
            pair_counts = _count_byte_pairs(word_tokens, word_freq)
            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            merged_token = _merge_pair(word_tokens, best_pair)
            if merged_token not in self.vocab:
                self.vocab.append(merged_token)
            self.merges.append(best_pair)

        self._build_mappings()
        ### END SOLUTION

    def _build_mappings(self):
        """Build token-to-ID and ID-to-token mappings."""
        ### BEGIN SOLUTION role="scaffold"
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        ### END SOLUTION

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """
        Apply learned merge rules to token sequence.

        TODO: Apply BPE merges to token list

        APPROACH:
        1. Start with character-level tokens
        2. Apply each merge rule in order
        3. Continue until no more merges possible

        EXAMPLE:
        >>> # After training, merges might be [('h','e'), ('l','l')]
        >>> tokenizer._apply_merges(['h','e','l','l','o'+Tokenizer.TOK_EOW])
        ['he','ll','o'+Tokenizer.TOK_EOW]  # Applied both merges

        HINT: For each merge pair, scan through tokens and replace adjacent pairs
        """
        ### BEGIN SOLUTION role="scaffold"
        if not self.merges:
            return tokens

        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == merge_pair[0] and
                    tokens[i + 1] == merge_pair[1]):
                    # Apply merge
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens
        ### END SOLUTION

    def encode(self, text: str) -> List[int]:
        """
        Encode text using BPE.

        TODO: Apply BPE encoding to text

        APPROACH:
        1. Split text into words
        2. Convert each word to character tokens
        3. Apply BPE merges
        4. Convert to token IDs

        EXAMPLE:
        >>> tokenizer.encode("hello world")
        [12, 45, 78]  # Token IDs after BPE merging

        HINTS:
        - Use text.split() for simple word splitting
        - Use _get_word_tokens() to get character-level tokens for each word
        - Use _apply_merges() to apply learned merge rules
        - Use token_to_id dictionary with 0 (UNK) as default
        """
        ### BEGIN SOLUTION role="scaffold"
        if not self.vocab:
            return []

        # Simple word splitting (could be more sophisticated)
        words = text.split()
        all_tokens = []

        for word in words:
            # Get character-level tokens
            word_tokens = self._get_word_tokens(word)

            # Apply BPE merges
            merged_tokens = self._apply_merges(word_tokens)

            all_tokens.extend(merged_tokens)

        # Convert to IDs
        token_ids = []
        for token in all_tokens:
            token_ids.append(self.token_to_id.get(token, 0))  # 0 = <UNK>

        return token_ids
        ### END SOLUTION

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        TODO: Convert token IDs back to readable text

        APPROACH:
        1. Convert IDs to tokens
        2. Join tokens together
        3. Clean up word boundaries and markers

        EXAMPLE:
        >>> tokenizer.decode([12, 45, 78])
        "hello world"  # Reconstructed text

        HINTS:
        - Use id_to_token dictionary with Tokenizer.TOK_UNKNOWN as default
        - Join all tokens into single string with ''.join()
        - Word-final tokens already contain a space boundary
        - Normalize whitespace without replacing any literal marker text
        """
        ### BEGIN SOLUTION role="scaffold"
        if not self.id_to_token:
            return ""

        # Convert IDs to tokens
        token_strings = []
        for token_id in tokens:
            token = self.id_to_token.get(token_id, Tokenizer.TOK_UNKNOWN)
            token_strings.append(token)

        # Join and clean up
        text = ''.join(token_strings)

        # Boundaries already are spaces, so literal text such as </w> stays intact.
        # Clean up extra spaces
        text = ' '.join(text.split())

        return text
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: BPE Tokenizer

This test validates our BPE tokenizer learns merge rules and correctly encodes/decodes text.

**What we're testing**: BPE training, merge rule application, encoding and decoding
**Why it matters**: BPE is the standard tokenization for modern language models
**Expected**: Vocabulary building, proper merging, reasonable round-trip on training data
"""

# %% nbgrader={"grade": true, "grade_id": "test-bpe-tokenizer", "locked": true, "points": 20}
def test_unit_bpe_tokenizer():
    """🧪 Test BPE tokenizer implementation."""
    print("🧪 Unit Test: BPE Tokenizer...")

    # Test basic functionality with simple corpus
    corpus = ["hello", "world", "hello", "hell"]  # "hell" and "hello" share prefix
    tokenizer = BPETokenizer(vocab_size=20)
    tokenizer.train(corpus)

    # Check that vocabulary was built
    assert len(tokenizer.vocab) > 0
    assert Tokenizer.TOK_UNKNOWN in tokenizer.vocab

    # Test helper functions
    word_tokens = tokenizer._get_word_tokens("test")
    assert word_tokens[-1].endswith(Tokenizer.TOK_EOW), "Should have end-of-word marker"

    pairs = tokenizer._get_pairs(['h', 'e', 'l', 'l', 'o'+Tokenizer.TOK_EOW])
    assert ('h', 'e') in pairs
    assert ('l', 'l') in pairs

    # Test encoding/decoding
    text = "hello"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)

    # Test round-trip on training data should work well
    for word in corpus:
        tokens = tokenizer.encode(word)
        decoded = tokenizer.decode(tokens)
        assert decoded == word, "Known words must round-trip exactly"

    print("✅ BPE tokenizer works correctly!")

if __name__ == "__main__":
    test_unit_bpe_tokenizer()

# %% [markdown]
r"""
### Subword Compression Dynamics

BPE establishes a continuous continuum between character-level flexibility and word-level brevity. By extracting high-frequency morphological roots from the corpus, BPE compresses redundant character sequences while retaining universal fallback mechanisms for unseen compound words.

#### Step-by-Step Subword Compression Trace

Trace the progressive compression of `"tokenization"` ($12$ characters):

$$\mathbf{x} = \text{"tokenization"} \implies [\text{'t'}, \text{'o'}, \text{'k'}, \text{'e'}, \text{'n'}, \text{'i'}, \text{'z'}, \text{'a'}, \text{'t'}, \text{'i'}, \text{'o'}, \text{'n</w>'}] \quad (L = 12)$$

| Step | Candidate Merge | Active Token Sequence | Length $L$ | Cumulative Compression |
| :--- | :--- | :--- | :--- | :--- |
| **Initial** | — | `['t', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n</w>']` | $12$ | $1.0\times$ (baseline) |
| **Step 1** | `('t', 'o') \to 'to'` | `['to', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n</w>']` | $11$ | $1.09\times$ |
| **Step 2** | `('i', 'o') \to 'io'` | `['to', 'k', 'e', 'n', 'io', 'z', 'a', 't', 'io', 'n</w>']` | $10$ | $1.20\times$ |
| **Step 3** | `('io', 'n</w>') \to 'ion</w>'` | `['to', 'k', 'e', 'n', 'io', 'z', 'a', 't', 'ion</w>']` | $9$ | $1.33\times$ |
| **Step 4** | `('to', 'k') \to 'tok'` | `['tok', 'e', 'n', 'io', 'z', 'a', 't', 'ion</w>']` | $8$ | $1.50\times$ |
| **Step 5** | $\dots \text{ (subword merges)}$ | `['token', 'iz', 'ation</w>']` | $3$ | $4.00\times$ |
| **Final** | `('iz', 'ation</w>') \to 'ization</w>'` | `['token', 'ization</w>']` | **2** | **$6.00\times$ sequence reduction** |

#### Subword Engineering Principles

| Principle | Algorithmic Mechanism | Systems Benefit |
| :--- | :--- | :--- |
| **Data-Driven Vocabulary** | Extracted from empirical co-occurrence statistics | No hand-crafted phonetic or grammatical rules required |
| **Subword Decomposition** | Unseen compounds split into familiar morphemes | Eliminates catastrophic `<UNK>` information loss |
| **Sequence Compression** | Common words collapse into $1$ token | Reduces downstream self-attention memory footprint by up to $(6)^2 = 36\times$ |
| **Prefix & Suffix Discovery** | Naturally isolates common affixes (`un-`, `-ing`, `-tion`) | Shares semantic representations across inflected word forms |
"""

# %% [markdown]
r"""
## 🔧 Integration: Bringing It Together

Tokenization serves as the front-end CPU ingestion phase of modern deep learning workflows. Before tensors reach the GPU for embedding lookup and matrix multiplication, input texts must undergo tokenization, length normalization, and batch tensor assembly.

### End-to-End NLP Execution Datapath

| Processing Stage | Input Format | Output Format | Systems Implementation |
| :--- | :--- | :--- | :--- |
| **1. Text Normalization** | Raw input stream | Normalized UTF-8 string | Strips excess whitespace; handles Unicode canonicalization |
| **2. Subword Tokenization** | Cleaned text string | `list[int]` token IDs | `BPETokenizer.encode()` applies ordered merge replay |
| **3. Batch Padding & Truncation**| Variable `list[list[int]]` | Uniform array $(B, T)$ | `tokenize_dataset()` enforces static dimension ceilings |
| **4. Tensor Ingestion** | NumPy integer matrix | `Tensor(B, T)` | TinyTorch `Tensor` loaded to compute device |
| **5. Embedding Lookup** | Token IDs $(B, T)$ | Dense Activations $(B, T, D)$ | Module 11 `Embedding` layer table lookup |

### Integration Components Built

- **`create_tokenizer(strategy, vocab_size, corpus)`**: Factory pattern for instantiating and training `CharTokenizer` or `BPETokenizer`.
- **`tokenize_dataset(texts, tokenizer, max_length)`**: Batch processing engine enforcing uniform sequence bounds for downstream matrix operations.
- **`analyze_tokenization(texts, tokenizer)`**: Profiling diagnostic reporting compression ratio, sequence length distributions, and vocabulary coverage.
"""

# %% nbgrader={"grade": false, "grade_id": "tokenization-utils", "solution": true}
#| export
def create_tokenizer(strategy: str = "char", vocab_size: int = 1000, corpus: List[str] = None) -> Tokenizer:
    """
    Factory function to create and train tokenizers.

    TODO: Create appropriate tokenizer based on strategy

    APPROACH:
    1. Check strategy type
    2. Create appropriate tokenizer class
    3. Train on corpus if provided
    4. Return configured tokenizer

    EXAMPLE:
    >>> corpus = ["hello world", "test text"]
    >>> tokenizer = create_tokenizer("char", corpus=corpus)
    >>> tokens = tokenizer.encode("hello")
    """
    ### BEGIN SOLUTION role="scaffold"
    if strategy == "char":
        tokenizer = CharTokenizer()
        if corpus:
            tokenizer.build_vocab(corpus)
    elif strategy == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        if corpus:
            tokenizer.train(corpus, vocab_size)
    else:
        raise ValueError(
            f"Unknown tokenization strategy: '{strategy}'\n"
            f"  ❌ Strategy '{strategy}' is not recognized\n"
            f"  💡 TinyTorch supports 'char' (character-level) and 'bpe' (byte-pair encoding) strategies\n"
            f"  🔧 Use: create_tokenizer('char', corpus=texts) or create_tokenizer('bpe', vocab_size=1000, corpus=texts)"
        )

    return tokenizer
    ### END SOLUTION

def tokenize_dataset(texts: List[str], tokenizer: Tokenizer, max_length: int = None) -> List[List[int]]:
    """
    Tokenize a dataset with optional length limits.

    TODO: Tokenize all texts with consistent preprocessing

    APPROACH:
    1. Encode each text with the tokenizer
    2. Apply max_length truncation if specified
    3. Return list of tokenized sequences

    EXAMPLE:
    >>> texts = ["hello world", "tokenize this"]
    >>> tokenizer = CharTokenizer(['h','e','l','o',' ','w','r','d','t','k','n','i','z','s'])
    >>> tokenized = tokenize_dataset(texts, tokenizer, max_length=10)
    >>> all(len(seq) <= 10 for seq in tokenized)
    True

    HINTS:
    - Handle empty texts gracefully (empty list is fine)
    - Truncate from the end if too long: tokens[:max_length]
    """
    ### BEGIN SOLUTION role="scaffold"
    if max_length is not None and (not isinstance(max_length, int) or max_length < 0):
        raise ValueError("max_length must be a nonnegative integer or None")
    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)

        # Apply length limit
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(tokens)

    return tokenized
    ### END SOLUTION

def analyze_tokenization(texts: List[str], tokenizer: Tokenizer) -> Dict[str, float]:
    """
    Analyze tokenization statistics.

    TODO: Compute useful statistics about tokenization

    APPROACH:
    1. Tokenize all texts
    2. Compute sequence length statistics
    3. Calculate compression ratio
    4. Return analysis dictionary

    EXAMPLE:
    >>> texts = ["hello", "world"]
    >>> tokenizer = CharTokenizer(['h','e','l','o','w','r','d'])
    >>> stats = analyze_tokenization(texts, tokenizer)
    >>> 'vocab_size' in stats and 'avg_sequence_length' in stats
    True

    HINTS:
    - Use np.mean() for average sequence length
    - Compression ratio = total_characters / total_tokens
    - Return dict with vocab_size, avg_sequence_length, max_sequence_length, etc.
    """
    ### BEGIN SOLUTION role="scaffold"
    # Tokenize once, then derive every statistic from the result
    tokenized = [tokenizer.encode(text) for text in texts]
    all_tokens = [token for tokens in tokenized for token in tokens]
    total_chars = sum(len(text) for text in texts)
    tokenized_lengths = [len(tokens) for tokens in tokenized]

    stats = {
        'vocab_size': tokenizer.vocab_size,
        'avg_sequence_length': float(np.mean(tokenized_lengths)) if tokenized_lengths else 0.0,
        'max_sequence_length': max(tokenized_lengths) if tokenized_lengths else 0,
        'total_tokens': len(all_tokens),
        'compression_ratio': total_chars / len(all_tokens) if all_tokens else 0,
        'unique_tokens': len(set(all_tokens))
    }

    return stats
    ### END SOLUTION


# %% [markdown]
"""
### 🧪 Unit Test: Tokenization Utilities

This test validates our utility functions for tokenizer creation, dataset processing, and analysis.

**What we're testing**: Factory pattern, batch tokenization, and analysis statistics
**Why it matters**: Essential for building NLP pipelines with consistent preprocessing
**Expected**: Correct tokenizer creation, length limits respected, meaningful statistics
"""

# %% nbgrader={"grade": true, "grade_id": "test-tokenization-utils", "locked": true, "points": 10}
def test_unit_tokenization_utils():
    """🧪 Test tokenization utility functions."""
    print("🧪 Unit Test: Tokenization Utils...")

    # Test tokenizer factory
    corpus = ["hello world", "test text", "more examples"]

    char_tokenizer = create_tokenizer("char", corpus=corpus)
    assert isinstance(char_tokenizer, CharTokenizer)
    assert char_tokenizer.vocab_size > 0

    bpe_tokenizer = create_tokenizer("bpe", vocab_size=50, corpus=corpus)
    assert isinstance(bpe_tokenizer, BPETokenizer)

    # Test dataset tokenization
    texts = ["hello", "world", "test"]
    tokenized = tokenize_dataset(texts, char_tokenizer, max_length=10)
    assert len(tokenized) == len(texts)
    assert all(len(seq) <= 10 for seq in tokenized)

    # Test analysis
    stats = analyze_tokenization(texts, char_tokenizer)
    assert 'vocab_size' in stats
    assert 'avg_sequence_length' in stats
    assert 'compression_ratio' in stats
    assert stats['total_tokens'] > 0

    print("✅ Tokenization utils work correctly!")

if __name__ == "__main__":
    test_unit_tokenization_utils()

# %% [markdown]
"""
## 📊 Systems Analysis: Tokenization Trade-offs

Let's understand the key systems concepts in tokenization: **vocabulary size vs sequence length trade-offs** and **memory implications**.

This analysis reveals why different tokenization strategies make different choices for different use cases.
"""

# %%
def analyze_tokenization_strategies():
    """📊 Compare different tokenization strategies on various texts."""
    print("📊 Analyzing Tokenization Strategies...")
    print("=" * 60)

    # Create test corpus with different text types
    corpus = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming artificial intelligence",
        "Tokenization is fundamental to natural language processing",
        "Subword units balance vocabulary size and sequence length"
    ]

    # Test different strategies
    strategies = [
        ("Character", create_tokenizer("char", corpus=corpus)),
        ("BPE-100", create_tokenizer("bpe", vocab_size=100, corpus=corpus)),
        ("BPE-500", create_tokenizer("bpe", vocab_size=500, corpus=corpus))
    ]

    print(f"{'Strategy':<12} {'Vocab':<8} {'Avg Len':<8} {'Compression':<12} {'Unique':<10}")
    print("-" * 60)

    for name, tokenizer in strategies:
        stats = analyze_tokenization(corpus, tokenizer)

        print(f"{name:<12} {stats['vocab_size']:<8} "
              f"{stats['avg_sequence_length']:<8.1f} "
              f"{stats['compression_ratio']:<12.2f} "
              f"{stats['unique_tokens']:<10}")

    print("\n💡 KEY INSIGHTS:")
    print("   1. Character tokenization: Small vocab, long sequences, covers seen characters")
    print("   2. BPE: Larger vocab trades off with shorter sequences")
    print("   3. Higher compression ratio = more characters per token = efficiency")

    print("\n🚀 REAL-WORLD IMPLICATIONS:")
    print("   - GPT-2/3 use ~50K BPE tokens; GPT-4 uses ~100K")
    print("   - Character models need more compute (longer sequences)")
    print("   - Embedding table size scales with vocabulary size")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_tokenization_strategies()

# %% [markdown]
"""
### Memory Profiling: Actual Tokenizer Memory Usage

Let's measure the real memory footprint of different tokenization strategies. This is crucial for understanding resource requirements in production systems.
"""

# %%
def analyze_tokenization_memory():
    """📊 Measure actual memory usage of different tokenizers."""
    import tracemalloc

    KB_TO_BYTES = 1024

    print("📊 Analyzing Tokenization Memory Usage...")
    print("=" * 70)

    # Create test corpora of varying sizes
    corpus_small = ["hello world"] * 100
    corpus_medium = ["the quick brown fox jumps over the lazy dog"] * 1000
    corpus_large = ["machine learning processes natural language text"] * 5000

    results = []

    for corpus_name, corpus in [("Small (100)", corpus_small),
                                  ("Medium (1K)", corpus_medium),
                                  ("Large (5K)", corpus_large)]:
        # Character tokenizer memory
        tracemalloc.start()
        char_tok = CharTokenizer()
        char_tok.build_vocab(corpus)
        char_current, char_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # BPE tokenizer memory
        tracemalloc.start()
        bpe_tok = BPETokenizer(vocab_size=1000)
        bpe_tok.train(corpus, vocab_size=1000)
        bpe_current, bpe_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            'corpus': corpus_name,
            'char_kb': char_peak / KB_TO_BYTES,
            'bpe_kb': bpe_peak / KB_TO_BYTES,
            'char_vocab': char_tok.vocab_size,
            'bpe_vocab': len(bpe_tok.vocab)
        })

    # Display results
    print(f"{'Corpus':<15} {'Char Mem (KB)':<15} {'BPE Mem (KB)':<15} {'Char Vocab':<12} {'BPE Vocab':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r['corpus']:<15} {r['char_kb']:<15.1f} {r['bpe_kb']:<15.1f} "
              f"{r['char_vocab']:<12} {r['bpe_vocab']:<12}")

    print("\n💡 Key Insights:")
    print("- Character tokenizer: Minimal memory (small vocab ~100 tokens)")
    print("- BPE tokenizer: More memory (larger vocab + merge rules storage)")
    print("- Stored tokenizer memory scales with vocabulary and merge rules; training also stores corpus words")
    print("- BPE merge rules add overhead (list of tuples)")
    print("\n🚀 Production: Use memory-mapped vocabularies for 50K+ token models")

if __name__ == "__main__":
    analyze_tokenization_memory()

# %% [markdown]
"""
### Performance Benchmarking: Encoding/Decoding Speed

Speed matters in production! Let's measure how fast different tokenizers can process text.
This helps understand computational bottlenecks in NLP pipelines.
"""

# %%
def benchmark_tokenization_speed():
    """📊 Measure encoding/decoding speed for different strategies."""
    import time

    print("📊 Benchmarking Tokenization Speed...")
    print("=" * 70)

    # Prepare test data (1000 texts, varying lengths)
    test_texts = [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "machine learning is transforming artificial intelligence",
        "tokenization enables natural language processing in neural networks"
    ] * 250  # 1000 total texts

    # Build tokenizers on training corpus
    corpus = test_texts[:100]
    tokenizers = [
        ("Character", create_tokenizer("char", corpus=corpus)),
        ("BPE-500", create_tokenizer("bpe", vocab_size=500, corpus=corpus)),
        ("BPE-2000", create_tokenizer("bpe", vocab_size=2000, corpus=corpus))
    ]

    print(f"{'Strategy':<12} {'Encode (ms)':<15} {'Decode (ms)':<15} {'Total Tokens':<15}")
    print("-" * 70)

    for name, tokenizer in tokenizers:
        # Benchmark encoding
        start = time.perf_counter()
        all_tokens = [tokenizer.encode(text) for text in test_texts]
        encode_time = (time.perf_counter() - start) * 1000

        # Benchmark decoding
        start = time.perf_counter()
        decoded = [tokenizer.decode(tokens) for tokens in all_tokens]
        decode_time = (time.perf_counter() - start) * 1000

        total_tokens = sum(len(t) for t in all_tokens)

        print(f"{name:<12} {encode_time:<15.1f} {decode_time:<15.1f} {total_tokens:<15}")

    print("\n💡 Key Insights:")
    print("- Character tokenization: Fastest (simple dict lookup, O(n) complexity)")
    print("- BPE tokenization: Slower (requires merge rule application)")
    print("- Larger BPE vocab: Fewer final tokens but more merge operations")
    print("- Decoding is typically faster than encoding")
    print("\n🚀 Production: Use Rust-based tokenizers (Hugging Face tokenizers library)")
    print("   Compiled tokenizers can be 10-100× faster than pure Python!")

if __name__ == "__main__":
    benchmark_tokenization_speed()

# %% [markdown]
"""
### Scaling Analysis: How BPE Training Time Grows

Understanding algorithmic complexity helps us predict performance on larger datasets.
Let's measure how BPE training time scales with corpus size.
"""

# %%
def analyze_bpe_scaling():
    """📊 Analyze how BPE training scales with corpus size."""
    import string
    import time

    KB_TO_BYTES = 1024

    print("📊 Analyzing BPE Training Scaling...")
    print("=" * 70)

    # Seeded locally so the table is reproducible without the package shipping a
    # module-level generator.
    rng = np.random.default_rng(7)

    # Generate random text helper
    def generate_random_text(length=10):
        return ''.join(rng.choice(list(string.ascii_lowercase + ' '), size=length))

    corpus_sizes = [100, 500, 1000, 2500]

    print(f"{'Corpus Size':<15} {'Training Time (ms)':<20} {'Vocab Size':<15} {'Memory (KB)':<15}")
    print("-" * 70)

    for size in corpus_sizes:
        # Generate corpus
        corpus = [generate_random_text(length=15) for _ in range(size)]

        # Measure training time and memory
        import tracemalloc
        tracemalloc.start()

        start = time.perf_counter()
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(corpus, vocab_size=500)
        train_time = (time.perf_counter() - start) * 1000

        memory_kb = tracemalloc.get_traced_memory()[1] / KB_TO_BYTES
        tracemalloc.stop()

        print(f"{size:<15} {train_time:<20.1f} {len(tokenizer.vocab):<15} {memory_kb:<15.1f}")

    print("\n💡 Key Insights:")
    print("- BPE training cost is about (number of merges) x (corpus size)")
    print("- Each merge iteration rescans every word to count all pairs")
    print("- Training memory includes vocabulary, merge rules, and unique corpus words")
    print("- Large corpora (millions of docs) need optimized implementations")
    print("\n🚀 Production strategies:")
    print("   - Sample representative subset for training (~1M sentences)")
    print("   - Use incremental training with checkpointing")
    print("   - Cache pair frequency counts between iterations")

if __name__ == "__main__":
    analyze_bpe_scaling()

# %% [markdown]
r"""
### Performance Analysis: Vocabulary Size vs Sequence Length

The selection of vocabulary size $V$ governs a fundamental systems tension between **CPU memory allocation** (embedding parameter footprint) and **GPU compute complexity** (self-attention sequence scaling):

| Strategy | Target Vocab Size $V$ | Sequence Length $T$ (1k words) | Embedding Table ($D=4096$) | Attention FLOPs ($\mathcal{O}(T^2)$) | Primary Systems Ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Character-Level** | $\approx 100 - 256$ | $\approx 5{,}000$ tokens | $\mathbf{1.6\text{ MB}}$ (L3 cache resident) | **$25.0\text{M MACs}$** ($14.8\times$) | Compute & KV-Cache Bound |
| **BPE-Compact** | $\approx 8{,}000$ | $\approx 1{,}600$ tokens | $\mathbf{131.1\text{ MB}}$ | **$2.56\text{M MACs}$** ($1.5\times$) | Balanced Edge Footprint |
| **BPE-Standard (GPT-2/3)**| $\approx 50{,}257$ | $\approx 1{,}300$ tokens | $\mathbf{823.4\text{ MB}}$ | **$1.69\text{M MACs}$** ($1.0\times$) | Balanced Server Footprint |
| **BPE-Extended (GPT-4/Llama 3)**| $\approx 100{,}000 - 128{,}256$ | $\approx 1{,}150$ tokens | $\mathbf{1.64\text{ GB}} - \mathbf{2.10\text{ GB}}$ | **$1.32\text{M MACs}$** ($0.78\times$) | High VRAM Allocation |
| **Word-Level** | $>200{,}000$ | $\approx 1{,}000$ tokens | $\mathbf{>3.28\text{ GB}}$ | **$1.00\text{M MACs}$** ($0.59\times$) | OOV Spillage & Parameter Blowup |

#### Mathematical Systems Trade-Off

The total inference memory footprint divides between static parameter weights and dynamic activation buffers:

$$M_{\text{embed}} = V \times D \times 4\text{ bytes}, \qquad M_{\text{attn}} = 2 \times B \times H \times T^2 \times 4\text{ bytes}$$

Because attention scales **quadratically** with sequence length $T$ while embedding memory scales **linearly** with vocabulary size $V$, modern LLMs intentionally scale vocabulary up to $\approx 100\text{K} - 128\text{K}$ tokens. This shortens sequence lengths by $15\% - 25\%$, providing compound latency savings throughout deep multi-layer transformer blocks.

#### Industry Benchmark Tokenizers

| Model Family | Tokenizer Framework | Vocabulary Size $V$ | Average Compression | Multilingual Support |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-2 / GPT-3** | Byte-level BPE (`tiktoken`) | $50{,}257$ | $\approx 3.7\text{ chars/token}$ | English-dominated |
| **GPT-4 / ChatGPT** | Byte-level BPE (`cl100k_base`) | $100{,}277$ | $\approx 4.2\text{ chars/token}$ | Enhanced code & multilingual |
| **Llama 3 (Meta)** | Byte-level BPE (`tiktoken`) | $128{,}256$ | $\approx 4.4\text{ chars/token}$ | Optimized code & multi-token words |
| **BERT (Google)** | WordPiece | $30{,}522$ | $\approx 4.1\text{ chars/token}$ | Fixed English vocabulary |
| **T5 (Google)** | SentencePiece Unigram | $32{,}128$ | $\approx 3.9\text{ chars/token}$ | Universal 100+ language support |
"""

# %% [markdown]
"""
## 🧪 Module Integration Test

Let's test our complete tokenization system to ensure everything works together.
"""

# %% nbgrader={"grade": true, "grade_id": "test-module", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire tokenization module.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_base_tokenizer()
    test_unit_char_tokenizer()
    test_unit_count_byte_pairs()
    test_unit_merge_pair()
    test_unit_bpe_tokenizer()
    test_unit_tokenization_utils()

    print("\nRunning integration scenarios...")

    # Test realistic tokenization workflow
    print("🧪 Integration Test: Complete tokenization pipeline...")

    # Create training corpus
    training_corpus = [
        "Natural language processing",
        "Machine learning models",
        "Neural networks learn",
        "Tokenization enables text processing",
        "Embeddings represent meaning"
    ]

    # Train different tokenizers
    char_tokenizer = create_tokenizer("char", corpus=training_corpus)
    bpe_tokenizer = create_tokenizer("bpe", vocab_size=200, corpus=training_corpus)

    # Test on new text
    test_text = "Neural language models"

    # Test character tokenization
    char_tokens = char_tokenizer.encode(test_text)
    char_decoded = char_tokenizer.decode(char_tokens)
    assert char_decoded == test_text, "Character round-trip failed"

    # BPE subword splits preserve text when its characters are known
    bpe_tokens = bpe_tokenizer.encode(test_text)
    bpe_decoded = bpe_tokenizer.decode(bpe_tokens)
    assert bpe_decoded == test_text, "BPE round-trip failed"

    # Test dataset processing
    test_dataset = ["hello world", "tokenize this", "neural networks"]
    char_dataset = tokenize_dataset(test_dataset, char_tokenizer, max_length=20)
    bpe_dataset = tokenize_dataset(test_dataset, bpe_tokenizer, max_length=10)

    assert len(char_dataset) == len(test_dataset)
    assert len(bpe_dataset) == len(test_dataset)
    assert all(len(seq) <= 20 for seq in char_dataset)
    assert all(len(seq) <= 10 for seq in bpe_dataset)

    # Test analysis functions
    char_stats = analyze_tokenization(test_dataset, char_tokenizer)
    bpe_stats = analyze_tokenization(test_dataset, bpe_tokenizer)

    assert char_stats['vocab_size'] > 0
    assert bpe_stats['vocab_size'] > 0
    assert char_stats['compression_ratio'] < bpe_stats['compression_ratio']  # BPE should compress better

    print("✅ End-to-end tokenization pipeline works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 10")

# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

Answer these questions to deepen your systems understanding of tokenization, memory footprint, and transformer computational complexity:

### Question 1: Vocabulary Size, Dictionary Overhead & Embedding VRAM Scaling
You implemented tokenizers spanning from character-level ($V \approx 100$) to BPE subwords ($V \approx 50{,}000$).

**1. Tokenizer RAM Overhead in Python**:
In a Python runtime, each vocabulary entry is stored in a hash table (`dict`) mapping a string token to an integer ID:
- Base string object overhead in CPython: $\approx 50\text{ bytes} + \text{string length}$.
- Integer ID object: $\approx 28\text{ bytes}$.
- Hash table bucket entry (pointer pair + hash code): $\approx 16\text{ bytes}$ with $\approx 2/3$ load factor.
- Total per-entry memory: $\approx 100\text{ bytes}$.

$$\text{Memory}_{\text{char}} \approx 100 \times 100\text{ B} \approx 10\text{ KB}$$
$$\text{Memory}_{\text{BPE}} \approx 50{,}000 \times 100\text{ B} \approx 5.0\text{ MB}$$

**2. Downstream Embedding Table Scaling**:
While a $5\text{ MB}$ tokenizer dictionary easily fits into host CPU RAM, the downstream neural network embedding table $W_{\text{embed}} \in \mathbb{R}^{V \times D}$ and output unembedding head must reside in GPU VRAM (assuming hidden dimension $D = 4{,}096$ and `float32` precision):

$$\text{Embedding Memory} = V \times D \times 4\text{ bytes}$$

| Tokenizer Type | Vocab Size $V$ | Dict RAM | Embedding VRAM ($D=4{,}096$) | Unembedding GEMM FLOPs |
| :--- | :--- | :--- | :--- | :--- |
| **Character** | $100$ | $10\text{ KB}$ | $1.64\text{ MB}$ | $8.19 \times 10^5$ |
| **Small BPE (GPT-2)** | $50{,}257$ | $5.1\text{ MB}$ | $823.4\text{ MB}$ | $4.12 \times 10^8$ |
| **Large BPE (Llama 3)** | $128{,}256$ | $13.5\text{ MB}$ | $2{,}101.2\text{ MB}$ ($2.1\text{ GB}$) | $1.05 \times 10^9$ |

**Systems Takeaway**:
Scaling vocabulary size from $100 \to 128{,}000$ increases embedding table memory by over **$1{,}280\times$**! In models with untied weights, this $2.1\text{ GB}$ cost is paid twice (input embedding + final classification layer), consuming over $4.2\text{ GB}$ of GPU memory before a single transformer block is instantiated.

---

### Question 2: Sequence Length Compression & Quadratic Attention Scaling
For the input phrase `"machine learning"` (16 raw characters):
- **Character Tokenizer**: Yields 16 tokens ($T_{\text{char}} = 16$).
- **BPE Tokenizer**: Yields 3 tokens (`["machine", " learn", "ing"]`, $T_{\text{bpe}} = 3$), achieving a **$5.33\times$ compression factor**.

**Context Budget Utilization**:
If your LLM context window is fixed at $T_{\text{max}} = 512$ tokens:
- A character-level model fits at most $512 / 16 \approx 32$ phrases ($\approx 512$ characters, or roughly 80 words).
- A BPE subword model fits $512 / 3 \approx 170$ phrases ($\approx 2{,}730$ characters, or roughly 450 words).

**Quadratic Attention Impact**:
Transformer self-attention compute scales quadratically with sequence length $\mathcal{O}(T^2)$:
$$\text{Attention FLOPs} = 4 \cdot B \cdot H \cdot T^2 \cdot D_{\text{head}}$$

For a document containing $1{,}000$ characters:
- Character tokenization: $T = 1{,}000 \implies T^2 = 1{,}000{,}000$
- BPE tokenization ($4.5\times$ compression): $T \approx 222 \implies T^2 = 49{,}284$

$$\text{Attention Compute Ratio} = \frac{1{,}000^2}{222^2} \approx \mathbf{20.3\times\text{ faster compute!}}$$

**Systems Takeaway**:
Subword tokenization is not just an NLP convenience; it is a **systems-level computational prerequisite** for long-context transformers. Compressing sequence length by $4.5\times$ cuts self-attention matrix multiplications by over **$20\times$** and slashes KV-cache memory during autoregressive generation by $4.5\times$.

---

### Question 3: Out-of-Vocabulary Robustness, Byte Fallback & Multilingual Fairness
Why did modern LLMs abandon pure word-level vocabularies in favor of Byte-Level BPE?

1. **The Word-Level `<UNK>` Catastrophe**:
   In word tokenizers, any out-of-vocabulary word (slang, typos, code identifiers like `calculate_gradient_norm`) maps to `<UNK>`. The model loses all semantic signal, rendering technical documentation and programming languages unlearnable.

2. **Byte-Level Fallback (Zero Unknown Tokens)**:
   By initializing BPE with all 256 individual raw byte values (`0x00` through `0xFF`), any arbitrary UTF-8 string is guaranteed to be representable. Even unknown emojis or rare Unicode scripts decompose into byte sequences without dropping data.

3. **The Multilingual Tokenizer Tax**:
   Because BPE merges are learned from corpus frequency distributions, high-resource languages (English) learn long, highly compressed subwords ($4\text{ to }5\text{ characters/token}$). In contrast, low-resource scripts (Hindi, Thai, Arabic) or programming code with uncommon indentations often decompose into individual bytes ($1\text{ to }3\text{ tokens per character}$).
   - **Cost Penalty**: A non-English speaker transmitting the same semantic message may consume $3\times$ to $5\times$ more tokens.
   - **Latency Penalty**: Autoregressive decoding generates one token per forward pass; generating $3\times$ more tokens takes $3\times$ longer wall-clock time.
   - **Modern Mitigation**: Modern models like Llama 3 expanded vocabulary to $128\text{K}$ to ensure equitable byte-pair merges across diverse world languages.

---

### Question 4: Production Serving Throughput, Rust Engines & KV-Cache Caching
Consider a production deployment serving $1{,}000{,}000$ API requests per day, with an average prompt length of $500$ tokens:
- **Total Daily Tokens**: $1{,}000{,}000 \times 500 = 500{,}000{,}000\text{ tokens/day}$.

**Throughput Comparison**:
- **Pure Python Tokenizer** ($0.1\text{ ms/token} \implies 10{,}000\text{ tokens/sec}$):
  $$\text{Daily CPU Time} = \frac{500{,}000{,}000 \text{ tokens}}{10{,}000 \text{ tokens/s}} = 50{,}000\text{ seconds} \approx \mathbf{13.89\text{ CPU hours}}$$
- **Fast Rust Tokenizer** (`tiktoken`, Hugging Face `tokenizers` @ $0.002\text{ ms/token} \implies 500{,}000\text{ tokens/sec}$):
  $$\text{Daily CPU Time} = \frac{500{,}000{,}000 \text{ tokens}}{500{,}000 \text{ tokens/s}} = 1{,}000\text{ seconds} \approx \mathbf{16.67\text{ minutes}}$$

**Production Architecture Techniques**:
- **Compiled Multi-threaded Pre-tokenization**: Fast tokenizers split raw text using SIMD-accelerated regular expressions and execute BPE merges in parallel across CPU cores using Rust's `rayon`.
- **System Prompt Prefix Caching**: In conversational agents, system prompts (e.g. 1,500 tokens of tool definitions and instructions) are identical across queries. Production servers tokenize the prompt once and cache both the token IDs and the pre-computed Key-Value (KV) attention tensors in GPU memory, completely bypassing tokenization and initial transformer prefill.
- **Zero-Copy Memory Mapping (`mmap`)**: Vocabulary files and merge ranks are serialized into binary blobs (e.g. trie or perfect hash structures) and loaded via `mmap`, sharing a single read-only physical memory buffer across dozens of worker processes.
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Text Becomes Tokens

**What you built:** Tokenizers that convert text into numerical sequences that neural networks can process.

**Why it matters:** Neural networks can't read text - they need numbers! Your tokenizer bridges
this gap, converting words into token IDs that can be embedded and processed. Every language
model from GPT to Claude uses tokenization as the first step in understanding text.

Your tokenization system is ready for NLP applications.
"""

# %%
def demo_tokenization():
    """🎯 See text become tokens."""
    print("🎯 AHA MOMENT: Text Becomes Tokens")
    print("=" * 45)

    # Create and train a character tokenizer on sample corpus
    corpus = ["hello world", "hello there"]
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(corpus)

    # Encode and decode a test phrase
    text = "hello"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"Input text:       '{text}'")
    print(f"Token IDs:        {tokens}")
    print(f"Decoded back:     '{decoded}'")
    print(f"Match:            {decoded == text}")

    # Show how BPE compresses better
    print("\n--- Comparing Tokenization Strategies ---")
    test_text = "hello world"
    char_tokens = tokenizer.encode(test_text)

    bpe_tokenizer = create_tokenizer("bpe", vocab_size=50, corpus=corpus)
    bpe_tokens = bpe_tokenizer.encode(test_text)

    print(f"Character tokenizer: {len(char_tokens)} tokens")
    print(f"BPE tokenizer:       {len(bpe_tokens)} tokens")

    print("\n✨ Text becomes tokens - language models start here!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_tokenization()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Tokenization

Congratulations! You've built a complete tokenization system for converting text to numerical representations!

### Key Accomplishments
- **Built a character-level tokenizer** with coverage of seen characters and simple implementation
- **Implemented BPE tokenizer** that learns efficient subword representations from data
- **Created vocabulary management** with encoding/decoding and unknown token handling
- **Discovered the vocabulary size vs sequence length trade-off** through systems analysis
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory scaling**: Embedding table size = vocab_size x embed_dim (can be 100+ MB)
- **Sequence length trade-offs**: BPE compresses text, reducing compute by 3-4x
- **Training complexity**: BPE training costs about (merges x corpus size), since every merge rescans the corpus
- **Production patterns**: Rust tokenizers are 10-100x faster than pure Python

### Ready for Next Steps
Your tokenization implementation enables text processing for language models.
Export with: `tito module complete 10`

**Next**: Module 11 will add learnable embeddings that convert your token IDs into rich vector representations!
"""
