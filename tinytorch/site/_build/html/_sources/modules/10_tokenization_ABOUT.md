# Module 10: Tokenization

:::{admonition} Module Info
:class: note

**ARCHITECTURE TIER** | Difficulty: ●●○○ | Time: 4-6 hours | Prerequisites: 01-08

**Prerequisites: Foundation tier (Modules 01-08)** means you should have completed:
- Tensor operations (Module 01)
- Basic neural network components (Modules 02-04)
- Training fundamentals (Modules 05-07)

Tokenization is relatively independent and works primarily with strings and numbers. If you can manipulate Python strings and dictionaries, you're ready.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F10_tokenization%2F10_tokenization.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/10_tokenization/10_tokenization.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/10_tokenization.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Neural networks operate on numbers, but humans communicate with text. Tokenization is the crucial bridge that converts text into numerical sequences that models can process. Every language model from GPT to BERT starts with tokenization, transforming raw strings like "Hello, world!" into sequences of integers that neural networks can consume.

In this module, you'll build two tokenization systems from scratch: a simple character-level tokenizer that treats each character as a token, and a sophisticated Byte Pair Encoding (BPE) tokenizer that learns efficient subword representations. You'll discover the fundamental trade-off in text processing: vocabulary size versus sequence length. Small vocabularies produce long sequences; large vocabularies produce short sequences but require huge embedding tables.

By the end, you'll understand why GPT uses 50,000 tokens, how tokenizers handle unknown words, and the memory implications of vocabulary choices in production systems.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** character-level tokenization for robust text coverage and BPE tokenization for efficient subword representation
- **Understand** the vocabulary size versus sequence length trade-off and its impact on memory and computation
- **Master** encoding and decoding operations that convert between text and numerical token IDs
- **Connect** your implementation to production tokenizers used in GPT, BERT, and modern language models
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Tokenization System
flowchart LR
    subgraph "Your Tokenization System"
        A["Base Interface<br/>encode/decode"]
        B["CharTokenizer<br/>character-level"]
        C["BPETokenizer<br/>subword units"]
        D["Vocabulary<br/>management"]
        E["Utilities<br/>dataset processing"]
    end

    A --> B
    A --> C
    B --> D
    C --> D
    D --> E

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
    style E fill:#e2d5f1
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `Tokenizer` base class | Interface contract: encode/decode |
| 2 | `CharTokenizer` | Character-level vocabulary, perfect coverage |
| 3 | `BPETokenizer` | Byte Pair Encoding, learning merges |
| 4 | Vocabulary building | Unique character extraction, frequency analysis |
| 5 | Utility functions | Dataset processing, analysis tools |

**The pattern you'll enable:**
```python
# Converting text to numbers for neural networks
tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.train(corpus)
token_ids = tokenizer.encode("Hello world")  # [142, 1847, 2341]
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- GPU-accelerated tokenization (production tokenizers use Rust/C++)
- Advanced segmentation algorithms (SentencePiece, Unigram models)
- Language-specific preprocessing (that's Module 11: Embeddings)
- Tokenizer serialization and loading (PyTorch handles this with `save_pretrained()`)

**You are building the conceptual foundation.** Production optimizations come later.

## API Reference

This section provides a quick reference for the tokenization classes you'll build. Think of it as your cheat sheet while implementing and debugging.

### Base Tokenizer Interface

```python
Tokenizer()
```
- Abstract base class defining the tokenizer contract
- All tokenizers must implement `encode()` and `decode()`

### CharTokenizer

```python
CharTokenizer(vocab: Optional[List[str]] = None)
```
- Character-level tokenizer treating each character as a token
- `vocab`: Optional list of characters to include in vocabulary

| Method | Signature | Description |
|--------|-----------|-------------|
| `build_vocab` | `build_vocab(corpus: List[str]) -> None` | Extract unique characters from corpus |
| `encode` | `encode(text: str) -> List[int]` | Convert text to character IDs |
| `decode` | `decode(tokens: List[int]) -> str` | Convert character IDs back to text |

**Properties:**
- `vocab`: List of characters in vocabulary
- `vocab_size`: Total number of unique characters + special tokens
- `char_to_id`: Mapping from characters to IDs
- `id_to_char`: Mapping from IDs to characters
- `unk_id`: ID for unknown characters (always 0)

### BPETokenizer

```python
BPETokenizer(vocab_size: int = 1000)
```
- Byte Pair Encoding tokenizer learning subword units
- `vocab_size`: Target vocabulary size after training

| Method | Signature | Description |
|--------|-----------|-------------|
| `train` | `train(corpus: List[str], vocab_size: int = None) -> None` | Learn BPE merges from corpus |
| `encode` | `encode(text: str) -> List[int]` | Convert text to subword token IDs |
| `decode` | `decode(tokens: List[int]) -> str` | Convert token IDs back to text |

**Helper Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `_get_word_tokens` | `_get_word_tokens(word: str) -> List[str]` | Convert word to character list with end-of-word marker |
| `_get_pairs` | `_get_pairs(word_tokens: List[str]) -> Set[Tuple[str, str]]` | Extract all adjacent character pairs |
| `_apply_merges` | `_apply_merges(tokens: List[str]) -> List[str]` | Apply learned merge rules to token sequence |
| `_build_mappings` | `_build_mappings() -> None` | Build token-to-ID and ID-to-token dictionaries |

**Properties:**
- `vocab`: List of tokens (characters + learned merges)
- `vocab_size`: Total vocabulary size
- `merges`: List of learned merge rules (pair tuples)
- `token_to_id`: Mapping from tokens to IDs
- `id_to_token`: Mapping from IDs to tokens

### Utility Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `create_tokenizer` | `create_tokenizer(strategy: str, vocab_size: int, corpus: List[str]) -> Tokenizer` | Factory for creating tokenizers |
| `tokenize_dataset` | `tokenize_dataset(texts: List[str], tokenizer: Tokenizer, max_length: int) -> List[List[int]]` | Batch tokenization with length limits |
| `analyze_tokenization` | `analyze_tokenization(texts: List[str], tokenizer: Tokenizer) -> Dict[str, float]` | Compute statistics and metrics |

## Core Concepts

This section covers the fundamental ideas you need to understand tokenization deeply. These concepts apply to every NLP system, from simple chatbots to large language models.

### Text to Numbers

Neural networks process numbers, not text. When you pass the string "Hello" to a model, it must first become a sequence of integers. This transformation happens in four steps: split text into tokens (units of meaning), build a vocabulary mapping each unique token to an integer ID, encode text by looking up each token's ID, and enable decoding to reconstruct the original text from IDs.

The simplest approach treats each character as a token. Consider the word "hello": split into characters `['h', 'e', 'l', 'l', 'o']`, build a vocabulary with IDs `{'h': 1, 'e': 2, 'l': 3, 'o': 4}`, encode to `[1, 2, 3, 3, 4]`, and decode back by reversing the lookup. This implementation is beautifully simple:

```python
def encode(self, text: str) -> List[int]:
    """Encode text to list of character IDs."""
    tokens = []
    for char in text:
        tokens.append(self.char_to_id.get(char, self.unk_id))
    return tokens
```

The elegance is in the simplicity: iterate through each character, look up its ID in the vocabulary dictionary, and use the unknown token ID for unseen characters. This gives perfect coverage: any text can be encoded without errors, though the sequences can be long.

### Vocabulary Building

Before encoding text, you need a vocabulary: the complete set of tokens your tokenizer recognizes. For character-level tokenization, this means extracting all unique characters from a training corpus.

Here's how the vocabulary building process works:

```python
def build_vocab(self, corpus: List[str]) -> None:
    """Build vocabulary from a corpus of text."""
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
```

The special `<UNK>` token at position 0 handles characters not in the vocabulary. When encoding text with unknown characters, they all map to ID 0. This graceful degradation prevents crashes while signaling that information was lost.

Character vocabularies are tiny (typically 50-200 tokens depending on language), which means small embedding tables. A 100-character vocabulary with 512-dimensional embeddings requires only 51,200 parameters, about 200 KB of memory. This is dramatically smaller than word-level vocabularies with 100,000+ entries.

### Byte Pair Encoding (BPE)

Character tokenization has a fatal flaw for neural networks: sequences are too long. A 50-word sentence might produce 250 character tokens. Processing 250 tokens through self-attention layers is computationally expensive, and the model must learn to compose characters into words from scratch.

BPE solves this by learning subword units. The algorithm is elegant: start with a character-level vocabulary, count all adjacent character pairs in the corpus, merge the most frequent pair into a new token, and repeat until reaching the target vocabulary size.

Consider training BPE on the corpus `["hello", "hello", "help"]`. Each word starts with end-of-word markers: `['h','e','l','l','o</w>']`, `['h','e','l','l','o</w>']`, `['h','e','l','p</w>']`. Count all pairs: `('h','e')` appears 3 times, `('e','l')` appears 3 times, `('l','l')` appears 2 times. The most frequent is `('h','e')`, so merge it:

```text
# Merge operation: ('h', 'e') → 'he'
# Before:
['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']
['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']
['h','e','l','p</w>']      →  ['he','l','p</w>']
```

The vocabulary grows from `['h','e','l','o','p','</w>']` to `['h','e','l','o','p','</w>','he']`. Continue merging: next most frequent is `('l','l')`, so merge to get `'ll'`. The vocabulary becomes `['h','e','l','o','p','</w>','he','ll']`. After sufficient merges, "hello" encodes as `['he','ll','o</w>']` (3 tokens instead of 5 characters).

Here's how the training loop works:

```python
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
```

This iterative merging automatically discovers linguistic patterns: common prefixes ("un", "re"), suffixes ("ing", "ed"), and frequent words become single tokens. The algorithm requires no linguistic knowledge, learning purely from statistics.

### Special Tokens

Production tokenizers include special tokens beyond `<UNK>`. Common ones include `<PAD>` for padding sequences to equal length, `<BOS>` (beginning of sequence) and `<EOS>` (end of sequence) for marking boundaries, and `<SEP>` for separating multiple text segments. GPT-style models often use `<|endoftext|>` to mark document boundaries.

The choice of special tokens affects the embedding table size. If you reserve 10 special tokens and have a 50,000 token vocabulary, your embedding table has 50,010 rows. Each special token needs learned parameters just like regular tokens.

### Encoding and Decoding

Encoding converts text to token IDs; decoding reverses the process. For BPE, encoding requires applying learned merge rules in order:

```python
def encode(self, text: str) -> List[int]:
    """Encode text using BPE."""
    # Split text into words
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
```

Decoding is simpler: look up each ID, join the tokens, and clean up markers:

```python
def decode(self, tokens: List[int]) -> str:
    """Decode token IDs back to text."""
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
```

The round-trip text → IDs → text should be lossless for known vocabulary. Unknown tokens degrade gracefully, mapping to `<UNK>` in both directions.

### Computational Complexity

Character tokenization is fast: encoding is O(n) where n is the string length (one dictionary lookup per character), and decoding is also O(n) (one reverse lookup per ID). The operations are embarrassingly parallel since each character processes independently.

BPE is slower due to merge rule application. Training BPE scales approximately O(n² × m) where n is corpus size and m is the number of merges. Each merge iteration requires counting all pairs across the entire corpus, then updating token sequences. For a 10,000-word corpus learning 5,000 merges, this can take seconds to minutes depending on implementation.

Encoding with trained BPE is O(n × m) where n is text length and m is the number of merge rules. Each merge rule must scan the token sequence looking for applicable pairs. Production tokenizers optimize this with trie data structures and caching, achieving near-linear time.

| Operation | Character | BPE Training | BPE Encoding |
|-----------|-----------|--------------|--------------|
| **Complexity** | O(n) | O(n² × m) | O(n × m) |
| **Typical Speed** | 1-5 ms/1K chars | 100-1000 ms/10K corpus | 5-20 ms/1K chars |
| **Bottleneck** | Dictionary lookup | Pair frequency counting | Merge rule application |

### Vocabulary Size Versus Sequence Length

The fundamental trade-off in tokenization creates a spectrum of choices. Small vocabularies (100-500 tokens) produce long sequences because each token represents little information (individual characters or very common subwords). Large vocabularies (50,000+ tokens) produce short sequences because each token represents more information (whole words or meaningful subword units).

Memory and computation scale oppositely:

**Embedding table memory** = vocabulary size × embedding dimension × bytes per parameter
**Sequence processing cost** = sequence length² × embedding dimension (for attention)

A character tokenizer with vocabulary 100 and embedding dimension 512 needs 100 × 512 × 4 = 204 KB for embeddings. But a 50-word sentence produces roughly 250 character tokens, requiring 250² = 62,500 attention computations per layer.

A BPE tokenizer with vocabulary 50,000 and embedding dimension 512 needs 50,000 × 512 × 4 = 102 MB for embeddings. But that same 50-word sentence might produce only 75 BPE tokens, requiring 75² = 5,625 attention computations per layer.

The attention cost savings (62,500 vs 5,625) dwarf the embedding memory cost (204 KB vs 102 MB) for models with multiple layers. This is why production language models use large vocabularies: the embedding table fits easily in memory, while shorter sequences dramatically reduce training and inference time.

Modern language models balance these factors:

| Model | Vocabulary | Strategy | Sequence Length (typical) |
|-------|-----------|----------|---------------------------|
| **GPT-2/3** | 50,257 | BPE | ~50-200 tokens per sentence |
| **BERT** | 30,522 | WordPiece | ~40-150 tokens per sentence |
| **T5** | 32,128 | SentencePiece | ~40-180 tokens per sentence |
| **Character** | ~100 | Character | ~250-1000 tokens per sentence |

## Production Context

### Your Implementation vs. Production Tokenizers

Your TinyTorch tokenizers demonstrate the core algorithms, but production tokenizers optimize for speed and scale. The conceptual differences are minimal: the same BPE algorithm, the same vocabulary mappings, the same encode/decode operations. The implementation differences are dramatic.

| Feature | Your Implementation | Hugging Face Tokenizers |
|---------|---------------------|-------------------------|
| **Language** | Pure Python | Rust (compiled to native code) |
| **Speed** | 1-10 ms/sentence | 0.01-0.1 ms/sentence (100× faster) |
| **Parallelization** | Single-threaded | Multi-threaded with Rayon |
| **Vocabulary storage** | Python dict | Trie data structure |
| **Special features** | Basic encode/decode | Padding, truncation, attention masks |
| **Pretrained models** | Train from scratch | Load from Hugging Face Hub |

### Code Comparison

The following comparison shows equivalent tokenization in TinyTorch and Hugging Face. Notice how the high-level API mirrors production tools, making your learning transferable.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.core.tokenization import BPETokenizer

# Train tokenizer on corpus
corpus = ["hello world", "machine learning"]
tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.train(corpus)

# Encode text
text = "hello machine"
token_ids = tokenizer.encode(text)

# Decode back to text
decoded = tokenizer.decode(token_ids)
```
````

````{tab-item} ⚡ Hugging Face
```python
from tokenizers import Tokenizer, models, trainers

# Train tokenizer on corpus (same algorithm!)
corpus = ["hello world", "machine learning"]
tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=1000)
tokenizer.train_from_iterator(corpus, trainer)

# Encode text
text = "hello machine"
output = tokenizer.encode(text)
token_ids = output.ids

# Decode back to text
decoded = tokenizer.decode(token_ids)
```
````
`````

Let's walk through each section to understand the comparison:

- **Lines 1-3 (Imports)**: TinyTorch exposes `BPETokenizer` directly from the tokenization module. Hugging Face uses a more modular design with separate `models` and `trainers` for flexibility across algorithms (BPE, WordPiece, Unigram).

- **Lines 5-8 (Training)**: Both train on the same corpus using the same BPE algorithm. TinyTorch uses a simpler API with `train()` method. Hugging Face separates model definition from training for composability, but the underlying algorithm is identical.

- **Lines 10-12 (Encoding)**: TinyTorch returns a list of integers directly. Hugging Face returns an `Encoding` object with additional metadata (attention masks, offsets, etc.), and you extract the IDs with `.ids` attribute. Same numerical result.

- **Lines 14-15 (Decoding)**: Both use `decode()` with the token ID list. Output is identical. The core operation is the same: look up each ID in the vocabulary and join the tokens.

```{tip} What's Identical

The BPE algorithm, merge rule learning, vocabulary structure, and encode/decode logic. When you debug tokenization issues in production, you'll understand exactly what's happening because you built the same system.
```

### Why Tokenization Matters at Scale

To appreciate why tokenization choices matter, consider the scale of modern systems:

- **GPT-3 training**: Processing 300 billion tokens required careful vocabulary selection. Using character tokenization would have increased sequence lengths by 3-4×, multiplying training time by 9-16× (quadratic attention cost).

- **Embedding table memory**: A 50,000 token vocabulary with 12,288-dimensional embeddings (GPT-3 size) requires 50,000 × 12,288 × 4 bytes = **2.4 GB** just for the embedding layer. This is ~0.14% of GPT-3's 175 billion total parameters, a reasonable fraction.

- **Real-time inference**: Chatbots must tokenize user input in milliseconds. Python tokenizers take 5-20 ms per sentence; Rust tokenizers take 0.05-0.2 ms. At 1 million requests per day, this saves ~5 hours of compute time daily.

## Check Your Understanding

Test yourself with these systems thinking questions. They're designed to build intuition for the performance characteristics you'll encounter in production NLP systems.

**Q1: Vocabulary Memory Calculation**

You train a BPE tokenizer with `vocab_size=30,000` for a production model. If using 768-dimensional embeddings with float32 precision, how much memory does the embedding table require?

```{admonition} Answer
:class: dropdown

30,000 × 768 × 4 bytes = **92,160,000 bytes ≈ 92.16 MB**

Breakdown:
- Vocabulary size: 30,000 tokens
- Embedding dimension: 768 (BERT-base size)
- Float32: 4 bytes per parameter
- Total parameters: 30,000 × 768 = 23,040,000
- Memory: 23.04M × 4 = 92.16 MB

This is why vocabulary size matters! Doubling to 60K vocab would double embedding memory to ~184 MB.
```

**Q2: Sequence Length Trade-offs**

A sentence contains 200 characters. With character tokenization it produces 200 tokens. With BPE it produces 50 tokens (4:1 compression). If processing batch size 32 with attention:

- How many attention computations for character tokenization per batch?
- How many for BPE tokenization per batch?

```{admonition} Answer
:class: dropdown

**Character tokenization:**
- Sequence length: 200 tokens
- Attention per sequence: 200² = 40,000 operations
- Batch size: 32
- Total: 32 × 40,000 = **1,280,000 attention operations**

**BPE tokenization:**
- Sequence length: 50 tokens (200 chars ÷ 4)
- Attention per sequence: 50² = 2,500 operations
- Batch size: 32
- Total: 32 × 2,500 = **80,000 attention operations**

BPE is **16× faster** for attention! This is why modern models use subword tokenization despite larger embedding tables.
```

**Q3: Unknown Token Handling**

Your BPE tokenizer encounters the word "supercalifragilistic" (not in training corpus). Character tokenizer maps it to 22 known tokens. BPE tokenizer decomposes it into subwords like `['super', 'cal', 'ifr', 'ag', 'il', 'istic']` (6 tokens). Which is better?

```{admonition} Answer
:class: dropdown

**BPE is better for production:**

- **Efficiency**: 6 tokens vs 22 tokens = 3.7× shorter sequence
- **Semantics**: Subwords like "super" and "istic" carry meaning; individual characters don't
- **Generalization**: Model learns that "super" prefix modifies meaning (superman, supermarket)
- **Memory**: 6² = 36 attention computations vs 22² = 484 (13× faster)

**Character tokenization advantages:**
- **Perfect coverage**: Never maps to `<UNK>`, always recovers original text
- **Simplicity**: No complex merge rules or training

For rare/unknown words, BPE's subword decomposition provides better semantic understanding and efficiency, which is why GPT, BERT, and T5 all use variants of subword tokenization.
```

**Q4: Compression Ratio Analysis**

You analyze two tokenizers on a 10,000 character corpus:
- Character tokenizer: 10,000 tokens
- BPE tokenizer: 2,500 tokens

What's the compression ratio, and what does it tell you about efficiency?

```{admonition} Answer
:class: dropdown

**Compression ratio: 10,000 ÷ 2,500 = 4.0**

This means each BPE token represents an average of 4 characters.

**Efficiency implications:**
- **Sequence processing**: 4× shorter sequences = 16× faster attention (quadratic scaling)
- **Context window**: With max length 512, character tokenizer handles 512 chars (~100 words); BPE handles 2,048 chars (~400 words)
- **Information density**: Each BPE token carries more semantic information (subword vs character)

**Trade-off**: BPE vocabulary is ~100× larger (10K tokens vs 100), increasing embedding memory from ~200 KB to ~20 MB. This trade-off heavily favors BPE for models with multiple transformer layers where attention cost dominates.
```

**Q5: Training Corpus Size Impact**

Training BPE on 1,000 words takes 100ms. How long will 10,000 words take? What about 100,000 words?

```{admonition} Answer
:class: dropdown

BPE training scales approximately **O(n²)** where n is corpus size (due to repeated pair counting across the corpus).

- **1,000 words**: 100 ms (baseline)
- **10,000 words**: ~10,000 ms = 10 seconds (100× longer, due to 10² scaling)
- **100,000 words**: ~1,000,000 ms = 1,000 seconds ≈ **16.7 minutes** (10,000× longer)

**Production strategies to handle this:**
- Sample representative subset (~50K-100K sentences is usually sufficient)
- Use incremental/online BPE that doesn't recount all pairs each iteration
- Parallelize pair counting across corpus chunks
- Cache frequent pair statistics
- Use optimized implementations (Rust, C++) that are 100-1000× faster

Note: Encoding with trained BPE is much faster (linear time), only training is slow.
```

## Further Reading

For students who want to understand the academic foundations and production implementations of tokenization:

### Seminal Papers

- **Neural Machine Translation of Rare Words with Subword Units** - Sennrich et al. (2016). The original BPE paper that introduced subword tokenization for neural machine translation. Shows how BPE handles rare words through decomposition and achieves better translation quality. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

- **SentencePiece: A simple and language independent approach to subword tokenization** - Kudo & Richardson (2018). Extends BPE with language-agnostic tokenization that handles raw text without pre-tokenization. Used in T5, ALBERT, and many multilingual models. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)

- **BERT: Pre-training of Deep Bidirectional Transformers** - Devlin et al. (2018). While primarily about transformers, introduces WordPiece tokenization used in BERT family models. Section 4.1 discusses vocabulary and tokenization choices. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

### Additional Resources

- **Library**: [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) - Production Rust implementation with Python bindings. Explore the source to see optimized BPE.
- **Tutorial**: "Byte Pair Encoding Tokenization" - [Hugging Face Course](https://huggingface.co/learn/nlp-course/chapter6/5) - Interactive tutorial showing BPE in action with visualizations
- **Textbook**: "Speech and Language Processing" by Jurafsky & Martin - Chapter 2 covers tokenization, including Unicode handling and language-specific issues

## What's Next

```{seealso} Coming Up: Module 11 - Embeddings

Convert your token IDs into learnable dense vector representations. You'll implement embedding tables that transform discrete tokens into continuous vectors, enabling neural networks to capture semantic relationships in text.
```

**Preview - How Your Tokenizer Gets Used in Future Modules:**

| Module | What It Does | Your Tokenization In Action |
|--------|--------------|----------------------------|
| **11: Embeddings** | Learnable lookup tables | `embedding = Embedding(vocab_size=1000, dim=128)` |
| **12: Attention** | Sequence-to-sequence processing | Token sequences attend to each other |
| **13: Transformers** | Complete language models | Full pipeline: tokenize → embed → attend → predict |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/10_tokenization/10_tokenization.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/10_tokenization/10_tokenization.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
