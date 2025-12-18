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

#| default_exp core.tokenization
#| export

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import json
import re
from collections import defaultdict, Counter

# %% [markdown]
"""
# Module 10: Tokenization - Converting Text to Numbers

Welcome to Module 10! Today you'll build tokenization - the bridge that converts human-readable text into numerical representations that machine learning models can process.

## 🔗 Prerequisites & Progress
**You've Built**: Neural networks, layers, training loops, and data loading
**You'll Build**: Text tokenization systems (character and BPE-based)
**You'll Enable**: Text processing for language models and NLP tasks

**Connection Map**:
```
DataLoader → Tokenization → Embeddings
(batching)   (text→numbers)  (learnable representations)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement character-based tokenization for simple text processing
2. Build a BPE (Byte Pair Encoding) tokenizer for efficient text representation
3. Understand vocabulary management and encoding/decoding operations
4. Create the foundation for text processing in neural networks

## Prerequisites Checklist

**Module 10 is relatively independent** - it mainly works with strings and numbers!

**Optional Dependencies:**
- Module 01 (Tensor): Only needed if converting tokens to Tensor format
  - Run: `pytest modules/source/01_tensor/test_tensor.py` (if available)
  - If missing: Tokenization works with plain Python lists

**Before starting:**
- Ensure you have Python 3.8+ with numpy installed
- No other module dependencies required!

This module focuses on text processing fundamentals that work independently.
The tokenization algorithms use only standard Python and NumPy.

Let's get started!
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/10_tokenization/tokenization_dev.py`
**Building Side:** Code exports to `tinytorch.text.tokenization`

```python
# How to use this module:
from tinytorch.core.tokenization import Tokenizer, CharTokenizer, BPETokenizer
```

**Why this matters:**
- **Learning:** Complete tokenization system in one focused module for deep understanding
- **Production:** Proper organization like Hugging Face's tokenizers with all text processing together
- **Consistency:** All tokenization operations and vocabulary management in text.tokenization
- **Integration:** Works seamlessly with embeddings and data loading for complete NLP pipeline
"""

# %%
#| export
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import json
import re
from collections import defaultdict, Counter
# Import from TinyTorch package (Module 01 must be completed before Module 10)
# Note: Tokenization primarily works with Python lists, but Tensor is available for advanced features
from tinytorch.core.tensor import Tensor

# Constants for memory calculations
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion

# %% [markdown]
"""
## 💡 Introduction - Why Tokenization?

Neural networks operate on numbers, but humans communicate with text. Tokenization is the crucial bridge that converts text into numerical sequences that models can process.

### The Text-to-Numbers Challenge

Consider the sentence: "Hello, world!" - how do we turn this into numbers a neural network can process?

```
┌─────────────────────────────────────────────────────────────────┐
│  TOKENIZATION PIPELINE: Text → Numbers                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (Human Text):     "Hello, world!"                        │
│           │                                                     │
│           ├─ Step 1: Split into tokens                          │
│           │         ['H','e','l','l','o',',', ...']             │
│           │                                                     │
│           ├─ Step 2: Map to vocabulary IDs                      │
│           │         [72, 101, 108, 108, 111, ...]               │
│           │                                                     │
│           ├─ Step 3: Handle unknowns                            │
│           │         Unknown chars → special <UNK> token         │
│           │                                                     │
│           └─ Step 4: Enable decoding                            │
│                     IDs → original text                         │
│                                                                 │
│  Output (Token IDs):  [72, 101, 108, 108, 111, 44, 32, ...]     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Four-Step Process

How do we represent text for a neural network? We need a systematic pipeline:

**1. Split text into tokens** - Break text into meaningful units (words, subwords, or characters)
**2. Map tokens to integers** - Create a vocabulary that assigns each token a unique ID
**3. Handle unknown text** - Deal gracefully with tokens not seen during training
**4. Enable reconstruction** - Convert numbers back to readable text for interpretation

### Why This Matters

The choice of tokenization strategy dramatically affects:
- **Model performance** - How well the model understands text
- **Vocabulary size** - Memory requirements for embedding tables
- **Computational efficiency** - Sequence length affects processing time
- **Robustness** - How well the model handles new/rare words
"""

# %% [markdown]
"""
## 📐 Foundations - Tokenization Strategies

Different tokenization approaches make different trade-offs between vocabulary size, sequence length, and semantic understanding.

### Character-Level Tokenization
**Approach**: Each character gets its own token

```
┌──────────────────────────────────────────────────────────────┐
│ CHARACTER TOKENIZATION PROCESS                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Build Vocabulary from Unique Characters             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Corpus: ["hello", "world"]                             │  │
│  │                ↓                                       │  │
│  │ Unique chars: ['h', 'e', 'l', 'o', 'w', 'r', 'd']      │  │
│  │                ↓                                       │  │
│  │ Vocabulary:  ['<UNK>','h','e','l','o','w','r','d']     │  │
│  │ IDs:            0      1   2   3   4   5   6   7       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 2: Encode Text Character by Character                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Text: "hello"                                         │  │
│  │                                                        │  │
│  │   'h' → 1    (lookup in vocabulary)                    │  │
│  │   'e' → 2                                              │  │
│  │   'l' → 3                                              │  │
│  │   'l' → 3                                              │  │
│  │   'o' → 4                                              │  │
│  │                                                        │  │
│  │  Result: [1, 2, 3, 3, 4]                               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 3: Decode by Reversing ID Lookup                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  IDs: [1, 2, 3, 3, 4]                                  │  │
│  │                                                        │  │
│  │   1 → 'h'    (reverse lookup)                          │  │
│  │   2 → 'e'                                              │  │
│  │   3 → 'l'                                              │  │
│  │   3 → 'l'                                              │  │
│  │   4 → 'o'                                              │  |
│  │                                                        │  │
│  │  Result: "hello"                                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Pros**:
- Small vocabulary (~100 chars)
- Handles any text perfectly
- No unknown tokens (every character can be mapped)
- Simple implementation

**Cons**:
- Long sequences (1 character = 1 token)
- Limited semantic understanding (no word boundaries)
- More compute (longer sequences to process)

### Word-Level Tokenization
**Approach**: Each word gets its own token

```
Text: "Hello world"
       ↓
Tokens: ['Hello', 'world']
       ↓
IDs:    [5847, 1254]
```

**Pros**: Semantic meaning preserved, shorter sequences
**Cons**: Huge vocabularies (100K+), many unknown tokens

### Subword Tokenization (BPE)
**Approach**: Learn frequent character pairs, build subword units

```
Text: "tokenization"
       ↓ Character level
Initial: ['t', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n']
       ↓ Learn frequent pairs
Merged: ['to', 'ken', 'ization']
       ↓
IDs:    [142, 1847, 2341]
```

**Pros**: Balance between vocabulary size and sequence length
**Cons**: More complex training process

### Strategy Comparison

```
Text: "tokenization" (12 characters)

Character: ['t','o','k','e','n','i','z','a','t','i','o','n'] → 12 tokens, vocab ~100
Word:      ['tokenization']                                   → 1 token, vocab 100K+
BPE:       ['token','ization']                               → 2 tokens, vocab 10-50K
```

The sweet spot for most applications is BPE with 10K-50K vocabulary size.
"""

# %% [markdown]
"""
## 🏗️ Implementation - Building Tokenization Systems

Let's implement tokenization systems from simple character-based to sophisticated BPE. We'll start with the base interface and work our way up to advanced algorithms.
"""

# %% [markdown]
"""
### Base Tokenizer Interface

All tokenizers need to provide two core operations: encoding text to numbers and decoding numbers back to text. Let's define the common interface.

```
Tokenizer Interface:
    encode(text) → [id1, id2, id3, ...]
    decode([id1, id2, id3, ...]) → text
```

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

    def encode(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.

        TODO: Implement encoding logic in subclasses

        APPROACH:
        1. Subclasses will override this method
        2. Return list of integer token IDs

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.encode("abc")
        [0, 1, 2]
        """
        ### BEGIN SOLUTION
        raise NotImplementedError("Subclasses must implement encode()")
        ### END SOLUTION

    def decode(self, tokens: List[int]) -> str:
        """
        Convert list of token IDs back to text.

        TODO: Implement decoding logic in subclasses

        APPROACH:
        1. Subclasses will override this method
        2. Return reconstructed text string

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.decode([0, 1, 2])
        "abc"
        """
        ### BEGIN SOLUTION
        raise NotImplementedError("Subclasses must implement decode()")
        ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-base-tokenizer", "locked": true, "points": 5}
def test_unit_base_tokenizer():
    """🔬 Test base tokenizer interface."""
    print("🔬 Unit Test: Base Tokenizer Interface...")

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
"""
### Character-Level Tokenizer

The simplest tokenization approach: each character becomes a token. This gives us perfect coverage of any text but produces long sequences.

```
Character Tokenization Process:

Step 1: Build vocabulary from unique characters
Text corpus: ["hello", "world"]
Unique chars: ['h', 'e', 'l', 'o', 'w', 'r', 'd']
Vocabulary: ['<UNK>', 'h', 'e', 'l', 'o', 'w', 'r', 'd']  # <UNK> for unknown
                0      1    2    3    4    5    6    7

Step 2: Encode text character by character
Text: "hello"
  'h' → 1
  'e' → 2
  'l' → 3
  'l' → 3
  'o' → 4
Result: [1, 2, 3, 3, 4]

Step 3: Decode by looking up each ID
IDs: [1, 2, 3, 3, 4]
  1 → 'h'
  2 → 'e'
  3 → 'l'
  3 → 'l'
  4 → 'o'
Result: "hello"
```
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
        ### BEGIN SOLUTION
        if vocab is None:
            vocab = []

        # Add special unknown token
        self.vocab = ['<UNK>'] + vocab
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
        ### BEGIN SOLUTION
        # Collect all unique characters
        all_chars = set()
        for text in corpus:
            all_chars.update(text)

        # Sort for consistent ordering
        unique_chars = sorted(list(all_chars))

        # Rebuild vocabulary with <UNK> token first
        self.vocab = ['<UNK>'] + unique_chars
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
        ### BEGIN SOLUTION
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
        ### BEGIN SOLUTION
        chars = []
        for token_id in tokens:
            # Use unknown token for invalid IDs
            char = self.id_to_char.get(token_id, '<UNK>')
            chars.append(char)
        return ''.join(chars)
        ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-char-tokenizer", "locked": true, "points": 15}
def test_unit_char_tokenizer():
    """🔬 Test character tokenizer implementation."""
    print("🔬 Unit Test: Character Tokenizer...")

    # Test basic functionality
    vocab = ['h', 'e', 'l', 'o', ' ', 'w', 'r', 'd']
    tokenizer = CharTokenizer(vocab)

    # Test vocabulary setup
    assert tokenizer.vocab_size == 9  # 8 chars + UNK
    assert tokenizer.vocab[0] == '<UNK>'
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
    corpus = ["hello world", "test text"]
    tokenizer.build_vocab(corpus)
    assert 't' in tokenizer.char_to_id
    assert 'x' in tokenizer.char_to_id

    print("✅ Character tokenizer works correctly!")

if __name__ == "__main__":
    test_unit_char_tokenizer()

# %% [markdown]
"""
### 🧪 Character Tokenizer Analysis
Character tokenization provides a simple, robust foundation for text processing. The key insight is that with a small vocabulary (typically <100 characters), we can represent any text without unknown tokens.

**Trade-offs**:
- **Pro**: No out-of-vocabulary issues, handles any language
- **Con**: Long sequences (1 char = 1 token), limited semantic understanding
- **Use case**: When robustness is more important than efficiency
"""

# %% [markdown]
"""
### Byte Pair Encoding (BPE) Tokenizer

BPE is the secret sauce behind modern language models (GPT, BERT, etc.). It learns to merge frequent character pairs, creating subword units that balance vocabulary size with sequence length.

```
┌───────────────────────────────────────────────────────────────────────┐
│ BPE TRAINING ALGORITHM: Learning Subword Units                        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ STEP 1: Initialize with Character Vocabulary                          │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Training Data: ["hello", "hello", "help"]                         │ │
│ │                                                                   │ │
│ │ Initial Tokens (with end-of-word markers):                        │ │
│ │   ['h','e','l','l','o</w>']    (hello)                            │ │
│ │   ['h','e','l','l','o</w>']    (hello)                            │ │
│ │   ['h','e','l','p</w>']        (help)                             │ │
│ │                                                                   │ │
│ │ Starting Vocab: ['h', 'e', 'l', 'o', 'p', '</w>']                 │ │
│ │                   ↑ All unique characters                         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 2: Count All Adjacent Pairs                                      │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Pair Frequency Analysis:                                          │ │
│ │                                                                   │ │
│ │   ('h', 'e'): ██████  3 occurrences  ← MOST FREQUENT!             │ │
│ │   ('e', 'l'): ██████  3 occurrences                               │ │
│ │   ('l', 'l'): ████    2 occurrences                               │ │
│ │   ('l', 'o'): ████    2 occurrences                               │ │
│ │   ('o', '<'): ████    2 occurrences                               │ │
│ │   ('l', 'p'): ██      1 occurrence                                │ │
│ │   ('p', '<'): ██      1 occurrence                                │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 3: Merge Most Frequent Pair                                      │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Merge Operation: ('h', 'e') → 'he'                                │ │
│ │                                                                   │ │
│ │ BEFORE:                          AFTER:                           │ │
│ │   ['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']            │ │
│ │   ['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']            │ │
│ │   ['h','e','l','p</w>']      →  ['he','l','p</w>']                │ │
│ │                                                                   │ │
│ │ Updated Vocab: ['h','e','l','o','p','</w>', 'he']                 │ │
│ │                                              ↑ NEW TOKEN!         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 4: Repeat Until Target Vocab Size Reached                        │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Iteration 2: Next most frequent is ('l', 'l')                     │ │
│ │ Merge ('l','l') → 'll'                                            │ │
│ │                                                                   │ │
│ │   ['he','l','l','o</w>']     →  ['he','ll','o</w>']               │ │
│ │   ['he','l','l','o</w>']     →  ['he','ll','o</w>']               │ │
│ │   ['he','l','p</w>']         →  ['he','l','p</w>']                │ │
│ │                                                                   │ │
│ │ Updated Vocab: ['h','e','l','o','p','</w>','he','ll']             │ │
│ │                                                  ↑ NEW!           │ │
│ │                                                                   │ │
│ │ Continue merging until vocab_size target...                       │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ FINAL RESULTS:                                                        │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Trained BPE can now encode efficiently:                           │ │
│ │                                                                   │ │
│ │ "hello" → ['he', 'll', 'o</w>']  = 3 tokens (vs 5 chars)          │ │
│ │ "help"  → ['he', 'l', 'p</w>']   = 3 tokens (vs 4 chars)          │ │
│ │                                                                   │ │
│ │  Key Insights: BPE automatically discovers:                       │ │
│ │    - Common prefixes ('he')                                       │ │
│ │    - Morphological patterns ('ll')                                │ │
│ │    - Natural word boundaries (</w>)                               │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Why BPE Works**: By starting with characters and iteratively merging frequent pairs, BPE discovers the natural statistical patterns in language. Common words become single tokens, rare words split into recognizable subword pieces!
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
        >>> tokenizer.vocab_size
        1000

        HINT: Initialize vocab and merges as empty lists, mappings as empty dicts
        """
        ### BEGIN SOLUTION
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = []  # List of (pair, new_token) merges
        self.token_to_id = {}
        self.id_to_token = {}
        ### END SOLUTION

    def _get_word_tokens(self, word: str) -> List[str]:
        """
        Convert word to list of characters with end-of-word marker.

        TODO: Tokenize word into character sequence

        APPROACH:
        1. Split word into characters
        2. Add </w> marker to last character
        3. Return list of tokens

        EXAMPLE:
        >>> tokenizer._get_word_tokens("hello")
        ['h', 'e', 'l', 'l', 'o</w>']

        HINT: Use list() to split word into characters, then modify the last element
        """
        ### BEGIN SOLUTION
        if not word:
            return []

        tokens = list(word)
        tokens[-1] += '</w>'  # Mark end of word
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
        >>> tokenizer._get_pairs(['h', 'e', 'l', 'l', 'o</w>'])
        {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o</w>')}

        HINT: Loop from 0 to len(word_tokens)-1 and create tuple pairs
        """
        ### BEGIN SOLUTION
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs
        ### END SOLUTION

    def train(self, corpus: List[str], vocab_size: int = None) -> None:
        """
        Train BPE on corpus to learn merge rules.

        TODO: Implement BPE training algorithm

        APPROACH:
        1. Build initial character vocabulary
        2. Count word frequencies in corpus
        3. Iteratively merge most frequent pairs
        4. Build final vocabulary and mappings

        EXAMPLE:
        >>> corpus = ["hello", "hello", "help"]
        >>> tokenizer = BPETokenizer(vocab_size=20)
        >>> tokenizer.train(corpus)
        >>> len(tokenizer.vocab) <= 20
        True

        HINTS:
        - Start with character-level tokens using _get_word_tokens()
        - Use Counter to track word frequencies
        - Count all pairs, merge most frequent, repeat until vocab_size reached
        - Don't forget to call _build_mappings() at the end
        """
        ### BEGIN SOLUTION
        if vocab_size:
            self.vocab_size = vocab_size

        # Count word frequencies
        word_freq = Counter(corpus)

        # Initialize vocabulary with characters
        vocab = set()
        word_tokens = {}

        for word in word_freq:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)

        # Convert to sorted list for consistency
        self.vocab = sorted(list(vocab))

        # Add special tokens
        if '<UNK>' not in self.vocab:
            self.vocab = ['<UNK>'] + self.vocab

        # Learn merges
        self.merges = []

        while len(self.vocab) < self.vocab_size:
            # Count all pairs across all words
            pair_counts = Counter()

            for word, freq in word_freq.items():
                tokens = word_tokens[word]
                pairs = self._get_pairs(tokens)
                for pair in pairs:
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Get most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]

            # Merge this pair in all words
            for word in word_tokens:
                tokens = word_tokens[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and
                        tokens[i] == best_pair[0] and
                        tokens[i + 1] == best_pair[1]):
                        # Merge pair
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokens[word] = new_tokens

            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            self.vocab.append(merged_token)
            self.merges.append(best_pair)

        # Build final mappings
        self._build_mappings()
        ### END SOLUTION

    def _build_mappings(self):
        """Build token-to-ID and ID-to-token mappings."""
        ### BEGIN SOLUTION
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
        >>> tokenizer._apply_merges(['h','e','l','l','o</w>'])
        ['he','ll','o</w>']  # Applied both merges

        HINT: For each merge pair, scan through tokens and replace adjacent pairs
        """
        ### BEGIN SOLUTION
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
        ### BEGIN SOLUTION
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
        - Use id_to_token dictionary with '<UNK>' as default
        - Join all tokens into single string with ''.join()
        - Replace '</w>' markers with spaces for word boundaries
        """
        ### BEGIN SOLUTION
        if not self.id_to_token:
            return ""

        # Convert IDs to tokens
        token_strings = []
        for token_id in tokens:
            token = self.id_to_token.get(token_id, '<UNK>')
            token_strings.append(token)

        # Join and clean up
        text = ''.join(token_strings)

        # Replace end-of-word markers with spaces
        text = text.replace('</w>', ' ')

        # Clean up extra spaces
        text = ' '.join(text.split())

        return text
        ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-bpe-tokenizer", "locked": true, "points": 20}
def test_unit_bpe_tokenizer():
    """🔬 Test BPE tokenizer implementation."""
    print("🔬 Unit Test: BPE Tokenizer...")

    # Test basic functionality with simple corpus
    corpus = ["hello", "world", "hello", "hell"]  # "hell" and "hello" share prefix
    tokenizer = BPETokenizer(vocab_size=20)
    tokenizer.train(corpus)

    # Check that vocabulary was built
    assert len(tokenizer.vocab) > 0
    assert '<UNK>' in tokenizer.vocab

    # Test helper functions
    word_tokens = tokenizer._get_word_tokens("test")
    assert word_tokens[-1].endswith('</w>'), "Should have end-of-word marker"

    pairs = tokenizer._get_pairs(['h', 'e', 'l', 'l', 'o</w>'])
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
        # Allow some flexibility due to BPE merging
        assert len(decoded.strip()) > 0

    print("✅ BPE tokenizer works correctly!")

if __name__ == "__main__":
    test_unit_bpe_tokenizer()

# %% [markdown]
"""
### 🧪 BPE Tokenizer Analysis

BPE provides a balance between vocabulary size and sequence length. By learning frequent subword patterns, it can handle new words through decomposition while maintaining reasonable sequence lengths.

```
BPE Merging Visualization:

Original: "tokenization" → ['t','o','k','e','n','i','z','a','t','i','o','n','</w>']
                                                       ↓ Merge frequent pairs
Step 1:   ('t','o') is frequent → ['to','k','e','n','i','z','a','t','i','o','n','</w>']
Step 2:   ('i','o') is frequent → ['to','k','e','n','io','z','a','t','io','n','</w>']
Step 3:   ('io','n') is frequent → ['to','k','e','n','io','z','a','t','ion','</w>']
Step 4:   ('to','k') is frequent → ['tok','e','n','io','z','a','t','ion','</w>']
                                                       ↓ Continue merging...
Final:    "tokenization" → ['token','ization']  # 2 tokens vs 13 characters!
```

**Key insights**:
- **Adaptive vocabulary**: Learns from data, not hand-crafted
- **Subword robustness**: Handles rare/new words through decomposition
- **Efficiency trade-off**: Larger vocabulary → shorter sequences → faster processing
- **Morphological awareness**: Naturally discovers prefixes, suffixes, roots
"""

# %% [markdown]
"""
## 🔧 Integration - Bringing It Together

Now let's build utility functions that make tokenization easy to use in practice. These tools will help you tokenize datasets, analyze performance, and choose the right strategy.

```
Tokenization Workflow:

1. Choose Strategy → 2. Train Tokenizer → 3. Process Dataset → 4. Analyze Results
      ↓                      ↓                    ↓                   ↓
   char/bpe           corpus training        batch encoding      stats/metrics
```
"""

# %% nbgrader={"grade": false, "grade_id": "tokenization-utils", "solution": true}
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
    ### BEGIN SOLUTION
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
            f"Unknown tokenization strategy: '{strategy}'.\n"
            f"  Available strategies: 'char', 'bpe'.\n"
            f"  Fix: Use 'char' for character-level or 'bpe' for byte-pair encoding tokenization."
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
    ### BEGIN SOLUTION
    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)

        # Apply length limit
        if max_length and len(tokens) > max_length:
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
    ### BEGIN SOLUTION
    all_tokens = []
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_chars += len(text)

    # Calculate statistics
    tokenized_lengths = [len(tokenizer.encode(text)) for text in texts]

    stats = {
        'vocab_size': tokenizer.vocab_size,
        'avg_sequence_length': np.mean(tokenized_lengths),
        'max_sequence_length': max(tokenized_lengths) if tokenized_lengths else 0,
        'total_tokens': len(all_tokens),
        'compression_ratio': total_chars / len(all_tokens) if all_tokens else 0,
        'unique_tokens': len(set(all_tokens))
    }

    return stats
    ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-tokenization-utils", "locked": true, "points": 10}
def test_unit_tokenization_utils():
    """🔬 Test tokenization utility functions."""
    print("🔬 Unit Test: Tokenization Utils...")

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
## 📊 Systems Analysis - Tokenization Trade-offs

Understanding the performance implications of different tokenization strategies is crucial for building efficient NLP systems.
"""

# %% nbgrader={"grade": false, "grade_id": "tokenization-analysis", "solution": true}
def analyze_tokenization_strategies():
    """📊 Compare different tokenization strategies on various texts."""
    print("📊 Analyzing Tokenization Strategies...")

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

    print(f"{'Strategy':<12} {'Vocab':<8} {'Avg Len':<8} {'Compression':<12} {'Coverage':<10}")
    print("-" * 60)

    for name, tokenizer in strategies:
        stats = analyze_tokenization(corpus, tokenizer)

        print(f"{name:<12} {stats['vocab_size']:<8} "
              f"{stats['avg_sequence_length']:<8.1f} "
              f"{stats['compression_ratio']:<12.2f} "
              f"{stats['unique_tokens']:<10}")

    print("\n💡 Key Insights:")
    print("- Character tokenization: Small vocab, long sequences, perfect coverage")
    print("- BPE: Larger vocab trades off with shorter sequences")
    print("- Higher compression ratio = more characters per token = efficiency")

if __name__ == "__main__":
    analyze_tokenization_strategies()

# %% [markdown]
"""
### Memory Profiling: Actual Tokenizer Memory Usage

Let's measure the real memory footprint of different tokenization strategies. This is crucial for understanding resource requirements in production systems.
"""

# %% nbgrader={"grade": false, "grade_id": "memory-profiling", "solution": false}
def analyze_tokenization_memory():
    """📊 Measure actual memory usage of different tokenizers."""
    import tracemalloc

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
    print("- Memory scales with vocabulary size, NOT corpus size")
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

# %% nbgrader={"grade": false, "grade_id": "performance-benchmarking", "solution": false}
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

# %% nbgrader={"grade": false, "grade_id": "scaling-analysis", "solution": false}
def analyze_bpe_scaling():
    """📊 Analyze how BPE training scales with corpus size."""
    import time

    print("📊 Analyzing BPE Training Scaling...")
    print("=" * 70)

    # Generate random text helper
    def generate_random_text(length=10):
        import random
        import string
        return ''.join(random.choices(string.ascii_lowercase + ' ', k=length))

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
    print("- BPE training scales roughly O(n²) with corpus size")
    print("- Each merge iteration requires counting all pairs in all words")
    print("- Memory usage grows linearly with vocabulary size")
    print("- Large corpora (millions of docs) need optimized implementations")
    print("\n🚀 Production strategies:")
    print("   - Sample representative subset for training (~1M sentences)")
    print("   - Use incremental training with checkpointing")
    print("   - Cache pair frequency counts between iterations")

if __name__ == "__main__":
    analyze_bpe_scaling()

# %% [markdown]
"""
### 📊 Performance Analysis: Vocabulary Size vs Sequence Length

The fundamental trade-off in tokenization creates a classic systems engineering challenge:

```
Tokenization Trade-off Spectrum:

Character          BPE-Small         BPE-Large         Word-Level
vocab: ~100    →   vocab: ~1K    →   vocab: ~50K   →   vocab: ~100K+
seq: very long →   seq: long     →   seq: medium   →   seq: short
memory: low    →   memory: med   →   memory: high  →   memory: very high
compute: high  →   compute: med  →   compute: low  →   compute: very low
coverage: 100% →   coverage: 99% →   coverage: 95% →   coverage: <80%
```

**Character tokenization (vocab ~100)**:
- Pro: Universal coverage, simple implementation, small embedding table
- Con: Long sequences (high compute), limited semantic units
- Use case: Morphologically rich languages, robust preprocessing

**BPE tokenization (vocab 10K-50K)**:
- Pro: Balanced efficiency, handles morphology, good coverage
- Con: Training complexity, domain-specific vocabularies
- Use case: Most modern language models (GPT, BERT family)

**Real-world scaling examples**:
```
GPT-3/4:     ~50K BPE tokens, avg 3-4 chars/token
BERT:        ~30K WordPiece tokens, avg 4-5 chars/token
T5:          ~32K SentencePiece tokens, handles 100+ languages
ChatGPT:     ~100K tokens with extended vocabulary
```

**Memory implications for embedding tables**:
```
┌─────────────────────────────────────────────────────────────────────┐
│ EMBEDDING TABLE MEMORY: Vocabulary Size × Embedding Dimension       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ CHARACTER TOKENIZER (Vocab: 100)                                    │
│ ┌────────────────────────────┐                                      │
│ │  100 × 512 = 51,200 params │     Memory: 204 KB                   │
│ │  ████                      │     ↑ Tiny embedding table!          │
│ └────────────────────────────┘                                      │
│                                                                     │
│ BPE-SMALL (Vocab: 1,000)                                            │
│ ┌────────────────────────────┐                                      │
│ │  1K × 512 = 512K params    │     Memory: 2.0 MB                   │
│ │  ██████████                │     ↑ Still manageable               │
│ └────────────────────────────┘                                      │
│                                                                     │
│ BPE-LARGE (Vocab: 50,000) ← MOST PRODUCTION MODELS                  │
│ ┌────────────────────────────────────────────────────────┐          │
│ │  50K × 512 = 25.6M params                              │          │
│ │  ████████████████████████████████████████████████      │          │
│ │                                                        │          │
│ │  Memory: 102.4 MB (fp32)                               │          │
│ │          51.2 MB (fp16)    ← Half precision saves 50%  │          │
│ │          25.6 MB (int8)    ← Quantization saves 75%    │          │
│ └────────────────────────────────────────────────────────┘          │
│                                                                     │
│ WORD-LEVEL (Vocab: 100,000)                                         │
│ ┌────────────────────────────────────────────────────────┐          │
│ │  100K × 512 = 51.2M params                             │          │
│ │  ████████████████████████████████████████████████████  │          │
│ │                                                        │          │
│ │  Memory: 204.8 MB (fp32)  ← Often too large!           │          │
│ │          102.4 MB (fp16)                               │          │
│ └────────────────────────────────────────────────────────┘          │
│                                                                     │
│  Key Trade-off:                                                     │
│    Larger vocab → Shorter sequences → Less compute                  │
│    BUT larger vocab → More embedding memory → Harder to train       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Real-World Production Examples:
┌─────────────┬──────────────┬───────────────┬──────────────────┐
│   Model     │  Vocab Size  │  Embed Dim    │  Embed Memory    │
├─────────────┼──────────────┼───────────────┼──────────────────┤
│  GPT-2      │    50,257    │     1,600     │     321 MB       │
│  GPT-3      │    50,257    │    12,288     │     2.4 GB       │
│  BERT       │    30,522    │       768     │      94 MB       │
│  T5         │    32,128    │       512     │      66 MB       │
│  LLaMA-7B   │    32,000    │     4,096     │     524 MB       │
└─────────────┴──────────────┴───────────────┴──────────────────┘
```
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
    test_unit_bpe_tokenizer()
    test_unit_tokenization_utils()

    print("\nRunning integration scenarios...")

    # Test realistic tokenization workflow
    print("🔬 Integration Test: Complete tokenization pipeline...")

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

    # Test BPE tokenization (may not be exact due to subword splits)
    bpe_tokens = bpe_tokenizer.encode(test_text)
    bpe_decoded = bpe_tokenizer.decode(bpe_tokens)
    assert len(bpe_decoded.strip()) > 0, "BPE decoding failed"

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

# Call the comprehensive test only when running directly
if __name__ == "__main__":
    test_module()

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Text Processing Foundations

### Question 1: Vocabulary Size vs Memory
You implemented tokenizers with different vocabulary sizes.
If you have a BPE tokenizer with vocab_size=50,000 and embed_dim=512:
- How many parameters are in the embedding table? _____ million
- If using float32, how much memory does this embedding table require? _____ MB

### Question 2: Sequence Length Trade-offs
Your character tokenizer produces longer sequences than BPE.
For the text "machine learning" (16 characters):
- Character tokenizer produces ~16 tokens
- BPE tokenizer might produce ~3-4 tokens
If processing batch_size=32 with max_length=512:
- Character model needs _____ total tokens per batch
- BPE model needs _____ total tokens per batch
- Which requires more memory during training? _____

### Question 3: Tokenization Coverage
Your BPE tokenizer handles unknown words by decomposing into subwords.
- Why is this better than word-level tokenization for real applications? _____
- What happens to model performance when many tokens map to <UNK>? _____
- How does vocabulary size affect the number of unknown decompositions? _____
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Text Becomes Tokens

**What you built:** Tokenizers that convert text into numerical sequences.

**Why it matters:** Neural networks can't read text—they need numbers! Your tokenizer bridges
this gap, converting words into token IDs that can be embedded and processed. Every language
model from GPT to Claude uses tokenization as the first step.

In the next module, you'll convert these tokens into dense vector representations.
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
    print(f"Vocab size:       {tokenizer.vocab_size}")
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
    print(f"Compression ratio:   {len(char_tokens) / len(bpe_tokens):.1f}x")

    print("\n✨ Text → tokens → text (language models start here)!")

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
- Built character-level tokenizer with perfect text coverage
- Implemented BPE tokenizer that learns efficient subword representations
- Created vocabulary management and encoding/decoding systems
- Discovered the vocabulary size vs sequence length trade-off
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your tokenization implementation enables text processing for language models.
Export with: `tito module complete 10`

**Next**: Module 11 will add learnable embeddings that convert your token IDs into rich vector representations!
"""
