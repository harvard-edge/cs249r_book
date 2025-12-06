---
title: "Tokenization - Text to Numerical Sequences"
description: "Build character-level and BPE tokenizers that convert text into token sequences for language models"
difficulty: "⭐⭐"
time_estimate: "4-5 hours"
prerequisites: ["Tensor"]
next_steps: ["Embeddings"]
learning_objectives:
  - "Implement character-level tokenization with vocabulary management and special token handling"
  - "Build BPE (Byte Pair Encoding) tokenizer that learns optimal subword units from corpus statistics"
  - "Understand vocabulary size vs sequence length trade-offs affecting model parameters and computation"
  - "Design efficient text processing pipelines with encoding, decoding, and serialization"
  - "Analyze tokenization throughput and compression ratios for production NLP systems"
---

# 10. Tokenization - Text to Numerical Sequences

**ARCHITECTURE TIER** | Difficulty: ⭐⭐ (2/4) | Time: 4-5 hours

## Overview

Build tokenization systems that convert raw text into numerical sequences for language models. This module implements character-level and Byte Pair Encoding (BPE) tokenizers that balance vocabulary size, sequence length, and computational efficiency—the fundamental trade-off shaping every modern NLP system from GPT-4 to Google Translate. You'll understand why vocabulary size directly affects model parameters while sequence length impacts transformer computation, and how BPE optimally balances both extremes.

## Learning Objectives

By the end of this module, you will be able to:

- **Implement character-level tokenization with vocabulary management**: Build tokenizers with bidirectional token-to-ID mappings, special token handling (PAD, UNK, BOS, EOS), and graceful unknown character handling for robust multilingual support
- **Build BPE (Byte Pair Encoding) tokenizer**: Implement the iterative merge algorithm that learns optimal subword units by counting character pair frequencies—the same approach powering GPT, BERT, and modern transformers
- **Understand vocabulary size vs sequence length trade-offs**: Analyze how vocabulary choices affect model parameters (embedding matrix size = vocab_size × embed_dim) and computation (transformer attention is O(n²) in sequence length)
- **Design efficient text processing pipelines**: Create production-ready tokenizers with encoding/decoding, vocabulary serialization for deployment, and proper special token management for batching
- **Analyze tokenization throughput and compression ratios**: Measure tokens/second performance, compare character vs BPE on sequence length reduction, and understand scaling to billions of tokens in production systems

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement character-level tokenizer with vocabulary building and encode/decode operations, then build BPE algorithm that iteratively merges frequent character pairs to learn optimal subword units
2. **Use**: Tokenize Shakespeare and modern text datasets, compare character vs BPE on sequence length reduction, measure tokenization throughput on large corpora, and test subword decomposition on rare/unknown words
3. **Reflect**: Why does vocabulary size directly control model parameters (embedding matrix rows)? How does sequence length affect transformer computation (O(n²) attention)? What's the optimal balance for mobile deployment vs cloud serving? How do tokenization choices impact multilingual model design?

```{admonition} Systems Reality Check
:class: tip

**Production Context**: GPT-4 uses a 100K-token vocabulary trained on trillions of tokens. Every token in the vocabulary adds a row to the embedding matrix—at 12,288 dimensions, that's 1.2B parameters just for embeddings. Meanwhile, transformers have O(n²) attention complexity, so reducing sequence length from 1000 to 300 tokens cuts computation by 11x. This vocabulary size vs sequence length trade-off shapes every design decision in modern NLP: GPT-3 doubled vocabulary from GPT-2 (50K→100K) specifically to handle code and reduce sequence lengths for long documents.

**Performance Note**: Google Translate processes billions of sentences daily through tokenization pipelines. Tokenization throughput (measured in tokens/second) is critical for serving at scale—character-level achieves ~1M tokens/sec (simple lookup) while BPE achieves ~100K tokens/sec (iterative merge application). Production systems cache tokenization results and batch aggressively to amortize preprocessing costs. At OpenAI's scale ($700/million tokens), every tokenization optimization directly impacts economics.
```

## Implementation Guide

### Base Tokenizer Interface

All tokenizers share a common interface: encode text to token IDs and decode IDs back to text. This abstraction enables consistent usage across different tokenization strategies.

```python
class Tokenizer:
    """Base tokenizer interface defining the contract for all tokenizers.

    All tokenization strategies (character, BPE, WordPiece) must implement:
    - encode(text) → List[int]: Convert text to token IDs
    - decode(token_ids) → str: Convert token IDs back to text
    """

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs."""
        raise NotImplementedError("Subclasses must implement encode()")

    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        raise NotImplementedError("Subclasses must implement decode()")
```

**Design Pattern**: Abstract base class enforces consistent API across tokenization strategies, enabling drop-in replacement for performance testing (character vs BPE benchmarks).

### Character-Level Tokenizer

The simplest tokenization approach: each character becomes a token. Provides perfect coverage of any text with a tiny vocabulary (~100 characters), but produces long sequences.

```python
class CharTokenizer(Tokenizer):
    """Character-level tokenizer treating each character as a separate token.

    Trade-offs:
    - Small vocabulary (typically 100-500 characters)
    - Long sequences (1 character = 1 token)
    - Perfect coverage (no unknown tokens if vocab includes all Unicode)
    - Simple implementation (direct character-to-ID mapping)

    Example:
        "hello" → ['h','e','l','l','o'] → [8, 5, 12, 12, 15] (5 tokens)
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        """Initialize with optional vocabulary.

        Args:
            vocab: List of characters to include in vocabulary.
                   If None, vocabulary is built later via build_vocab().
        """
        if vocab is None:
            vocab = []

        # Reserve ID 0 for unknown token (robust handling of unseen characters)
        self.vocab = ['<UNK>'] + vocab
        self.vocab_size = len(self.vocab)

        # Bidirectional mappings for efficient encode/decode
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # Cache unknown token ID for fast lookup
        self.unk_id = 0

    def build_vocab(self, corpus: List[str]) -> None:
        """Build vocabulary from text corpus.

        Args:
            corpus: List of text strings to extract characters from.

        Process:
            1. Collect all unique characters across entire corpus
            2. Sort alphabetically for consistent ordering across runs
            3. Rebuild char↔ID mappings with <UNK> token at position 0
        """
        # Extract all unique characters
        all_chars = set()
        for text in corpus:
            all_chars.update(text)

        # Sort for reproducibility (important for model deployment)
        unique_chars = sorted(list(all_chars))

        # Rebuild vocabulary with special token first
        self.vocab = ['<UNK>'] + unique_chars
        self.vocab_size = len(self.vocab)

        # Rebuild bidirectional mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """Convert text to list of character IDs.

        Args:
            text: String to tokenize.

        Returns:
            List of integer token IDs, one per character.
            Unknown characters map to ID 0 (<UNK>).

        Example:
            >>> tokenizer.encode("hello")
            [8, 5, 12, 12, 15]  # Depends on vocabulary ordering
        """
        tokens = []
        for char in text:
            # Use .get() with unk_id default for graceful unknown handling
            tokens.append(self.char_to_id.get(char, self.unk_id))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text.

        Args:
            tokens: List of integer token IDs.

        Returns:
            Reconstructed text string.
            Invalid IDs map to '<UNK>' character.
        """
        chars = []
        for token_id in tokens:
            char = self.id_to_char.get(token_id, '<UNK>')
            chars.append(char)
        return ''.join(chars)
```

**Key Implementation Details:**

- **Special Token Reservation**: `<UNK>` token must occupy ID 0 consistently across vocabularies for model compatibility
- **Bidirectional Mappings**: Both `char_to_id` (encoding) and `id_to_char` (decoding) enable O(1) lookup performance
- **Unknown Character Handling**: Graceful degradation prevents crashes on unseen characters (critical for multilingual models encountering rare Unicode)
- **Vocabulary Consistency**: Sorted character ordering ensures reproducible vocabularies across training runs (important for model deployment)

### BPE (Byte Pair Encoding) Tokenizer

The algorithm powering GPT and modern transformers: iteratively merge frequent character pairs to discover optimal subword units. Balances vocabulary size (model parameters) with sequence length (computational cost).

```python
class BPETokenizer(Tokenizer):
    """Byte Pair Encoding tokenizer for subword tokenization.

    Algorithm:
        1. Initialize: Start with character-level vocabulary
        2. Count: Find all adjacent character pair frequencies in corpus
        3. Merge: Replace most frequent pair with new merged token
        4. Repeat: Continue until vocabulary reaches target size

    Trade-offs:
        - Larger vocabulary (typically 10K-50K tokens)
        - Shorter sequences (2-4x compression vs character-level)
        - Subword decomposition handles rare/unknown words gracefully
        - Training complexity (requires corpus statistics)

    Example:
        Training: "hello" appears 1000x, "hell" appears 500x
        Learns: 'h'+'e' → 'he' (freq pair), 'l'+'l' → 'll' (freq pair)
        Result: "hello" → ['he', 'll', 'o</w>'] (3 tokens vs 5 characters)
    """

    def __init__(self, vocab_size: int = 1000):
        """Initialize BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size (includes special tokens +
                       characters + learned merges). Typical: 10K-50K.
        """
        self.vocab_size = vocab_size
        self.vocab = []              # Final vocabulary tokens
        self.merges = []             # Learned merge rules: [(pair, merged_token), ...]
        self.token_to_id = {}        # Token string → integer ID
        self.id_to_token = {}        # Integer ID → token string

    def _get_word_tokens(self, word: str) -> List[str]:
        """Convert word to character tokens with end-of-word marker.

        Args:
            word: String to tokenize at character level.

        Returns:
            List of character tokens with '</w>' suffix on last character.
            End-of-word marker enables learning of word boundaries.

        Example:
            >>> _get_word_tokens("hello")
            ['h', 'e', 'l', 'l', 'o</w>']
        """
        if not word:
            return []

        tokens = list(word)
        tokens[-1] += '</w>'  # Mark word boundaries for BPE
        return tokens

    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """Extract all adjacent character pairs from token sequence.

        Args:
            word_tokens: List of token strings.

        Returns:
            Set of unique adjacent pairs (useful for frequency counting).

        Example:
            >>> _get_pairs(['h', 'e', 'l', 'l', 'o</w>'])
            {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o</w>')}
        """
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs

    def train(self, corpus: List[str], vocab_size: int = None) -> None:
        """Train BPE on corpus to learn merge rules.

        Args:
            corpus: List of text strings (typically words or sentences).
            vocab_size: Override target vocabulary size if provided.

        Training Process:
            1. Count word frequencies in corpus
            2. Initialize with character-level tokens (all unique characters)
            3. Iteratively:
                a. Count all adjacent pair frequencies across all words
                b. Merge most frequent pair into new token
                c. Update word representations with merged token
                d. Add merged token to vocabulary
            4. Stop when vocabulary reaches target size
            5. Build final token↔ID mappings
        """
        if vocab_size:
            self.vocab_size = vocab_size

        # Count word frequencies (training on token statistics, not raw text)
        word_freq = Counter(corpus)

        # Initialize vocabulary and word token representations
        vocab = set()
        word_tokens = {}

        for word in word_freq:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)  # Collect all unique character tokens

        # Convert to sorted list for reproducibility
        self.vocab = sorted(list(vocab))

        # Add special unknown token
        if '<UNK>' not in self.vocab:
            self.vocab = ['<UNK>'] + self.vocab

        # Learn merge rules iteratively
        self.merges = []

        while len(self.vocab) < self.vocab_size:
            # Count all adjacent pairs across all words (weighted by frequency)
            pair_counts = Counter()

            for word, freq in word_freq.items():
                tokens = word_tokens[word]
                pairs = self._get_pairs(tokens)
                for pair in pairs:
                    pair_counts[pair] += freq  # Weight by word frequency

            if not pair_counts:
                break  # No more pairs to merge

            # Select most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]

            # Apply merge to all word representations
            for word in word_tokens:
                tokens = word_tokens[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    # Check if current position matches merge pair
                    if (i < len(tokens) - 1 and
                        tokens[i] == best_pair[0] and
                        tokens[i + 1] == best_pair[1]):
                        # Merge pair into single token
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

        # Build final token↔ID mappings for efficient encode/decode
        self._build_mappings()

    def _build_mappings(self):
        """Build bidirectional token↔ID mappings from vocabulary."""
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """Apply learned merge rules to token sequence.

        Args:
            tokens: List of character-level tokens.

        Returns:
            List of tokens after applying all learned merges.

        Process:
            Apply each merge rule in the order learned during training.
            Early merges have priority over later merges.
        """
        if not self.merges:
            return tokens

        # Apply each merge rule sequentially
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

    def encode(self, text: str) -> List[int]:
        """Encode text using learned BPE merges.

        Args:
            text: String to tokenize.

        Returns:
            List of integer token IDs after applying BPE merges.

        Process:
            1. Split text into words (simple whitespace split)
            2. Convert each word to character-level tokens
            3. Apply learned BPE merges to create subword units
            4. Convert subword tokens to integer IDs
        """
        if not self.vocab:
            return []

        # Simple word splitting (production systems use more sophisticated approaches)
        words = text.split()
        all_tokens = []

        for word in words:
            # Start with character-level tokens
            word_tokens = self._get_word_tokens(word)

            # Apply BPE merges
            merged_tokens = self._apply_merges(word_tokens)

            all_tokens.extend(merged_tokens)

        # Convert tokens to IDs (unknown tokens map to ID 0)
        token_ids = []
        for token in all_tokens:
            token_ids.append(self.token_to_id.get(token, 0))

        return token_ids

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: List of integer token IDs.

        Returns:
            Reconstructed text string.

        Process:
            1. Convert IDs to token strings
            2. Join tokens together
            3. Remove end-of-word markers and restore spaces
        """
        if not self.id_to_token:
            return ""

        # Convert IDs to token strings
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

**BPE Algorithm Insights:**

- **Training Phase**: Learn merge rules from corpus statistics by iteratively merging most frequent adjacent pairs
- **Inference Phase**: Apply learned merges in order to segment new text into optimal subword units
- **Frequency-Based Learning**: Common patterns ("ing", "ed", "tion") become single tokens, reducing sequence length
- **Graceful Degradation**: Unseen words decompose into known subwords (e.g., "unhappiness" → ["un", "happi", "ness"])
- **Word Boundary Awareness**: End-of-word markers (`</w>`) enable learning of prefix vs suffix patterns

### Tokenization Utilities

Production-ready utilities for tokenizer creation, dataset processing, and performance analysis.

```python
def create_tokenizer(strategy: str = "char",
                    vocab_size: int = 1000,
                    corpus: List[str] = None) -> Tokenizer:
    """Factory function to create and train tokenizers.

    Args:
        strategy: Tokenization approach ("char" or "bpe").
        vocab_size: Target vocabulary size (for BPE).
        corpus: Training corpus for vocabulary building.

    Returns:
        Trained tokenizer instance.

    Example:
        >>> corpus = ["hello world", "machine learning"]
        >>> tokenizer = create_tokenizer("bpe", vocab_size=500, corpus=corpus)
        >>> tokens = tokenizer.encode("hello")
    """
    if strategy == "char":
        tokenizer = CharTokenizer()
        if corpus:
            tokenizer.build_vocab(corpus)
    elif strategy == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        if corpus:
            tokenizer.train(corpus, vocab_size)
    else:
        raise ValueError(f"Unknown tokenization strategy: {strategy}")

    return tokenizer

def analyze_tokenization(texts: List[str],
                        tokenizer: Tokenizer) -> Dict[str, float]:
    """Analyze tokenization statistics for performance evaluation.

    Args:
        texts: List of text strings to analyze.
        tokenizer: Trained tokenizer instance.

    Returns:
        Dictionary containing:
        - vocab_size: Number of unique tokens in vocabulary
        - avg_sequence_length: Mean tokens per text
        - max_sequence_length: Longest tokenized sequence
        - total_tokens: Total tokens across all texts
        - compression_ratio: Average characters per token (higher = better)
        - unique_tokens: Number of distinct tokens used

    Use Cases:
        - Compare character vs BPE on sequence length reduction
        - Measure compression efficiency (chars/token ratio)
        - Identify vocabulary utilization (unique_tokens / vocab_size)
    """
    all_tokens = []
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_chars += len(text)

    tokenized_lengths = [len(tokenizer.encode(text)) for text in texts]

    stats = {
        'vocab_size': (tokenizer.vocab_size
                      if hasattr(tokenizer, 'vocab_size')
                      else len(tokenizer.vocab)),
        'avg_sequence_length': np.mean(tokenized_lengths),
        'max_sequence_length': max(tokenized_lengths) if tokenized_lengths else 0,
        'total_tokens': len(all_tokens),
        'compression_ratio': total_chars / len(all_tokens) if all_tokens else 0,
        'unique_tokens': len(set(all_tokens))
    }

    return stats
```

**Analysis Metrics Explained:**

- **Compression Ratio**: Characters per token (higher = more efficient). BPE typically achieves 3-5x vs character-level at 1.0x
- **Vocabulary Utilization**: unique_tokens / vocab_size indicates whether vocabulary is appropriately sized
- **Sequence Length**: Directly impacts transformer computation (O(n²) attention complexity)

## Getting Started

### Prerequisites

Ensure you understand tensor operations from Module 01:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify tensor module
tito test tensor
```

**Why This Prerequisite Matters:**

- Tokenization produces integer tensors (sequences of token IDs)
- Embedding layers (Module 11) use token IDs to index into weight matrices
- Understanding tensor shapes is critical for batching variable-length sequences

### Development Workflow

1. **Open the development file**: `modules/10_tokenization/tokenization_dev.ipynb`
2. **Implement base Tokenizer interface**: Define encode() and decode() methods as abstract interface
3. **Build CharTokenizer**: Implement vocabulary building, character-to-ID mappings, encode/decode with unknown token handling
4. **Implement BPE algorithm**:
   - Character pair counting with frequency statistics
   - Iterative merge logic (find most frequent pair, merge across corpus)
   - Vocabulary construction from learned merges
   - Merge application during encoding
5. **Create utility functions**: Tokenizer factory, dataset processing, performance analysis
6. **Test on real data**:
   - Compare character vs BPE on sequence length reduction
   - Measure compression ratios (characters per token)
   - Test unknown word handling via subword decomposition
   - Analyze vocabulary utilization
7. **Optimize for performance**: Measure tokenization throughput (tokens/second), profile merge application, test on large corpora
8. **Export and verify**: `tito module complete 10 && tito test tokenization`

**Development Tips:**

- Start with small corpus (100 words, vocab_size=200) to debug BPE algorithm
- Print learned merge rules to understand what patterns BPE discovers
- Visualize sequence length vs vocabulary size trade-off with multiple BPE configurations
- Test on rare/misspelled words to verify subword decomposition works
- Profile with different vocabulary sizes to find optimal performance point

## Testing

### Comprehensive Test Suite

Run the full test suite to verify tokenization functionality:

```bash
# TinyTorch CLI (recommended)
tito test tokenization

# Direct pytest execution
python -m pytest tests/ -k tokenization -v
```

### Test Coverage Areas

- **Base tokenizer interface**: Abstract class enforces encode/decode contract
- **Character tokenizer correctness**: Vocabulary building from corpus, encode/decode round-trip accuracy, unknown character handling with `<UNK>` token
- **BPE merge learning**: Pair frequency counting, merge application correctness, vocabulary size convergence, merge order preservation
- **Vocabulary management**: Token-to-ID mapping consistency, bidirectional lookup correctness, special token ID reservation
- **Edge case handling**: Empty strings, single characters, Unicode characters, whitespace-only text, very long sequences
- **Round-trip accuracy**: Encode→decode produces original text for all vocabulary characters
- **Performance benchmarks**: Tokenization throughput (tokens/second), vocabulary size vs encode time scaling, batch processing efficiency

### Inline Testing & Validation

The module includes comprehensive inline tests with progress tracking:

```python
# Example inline test output
🔬 Unit Test: Base Tokenizer Interface...
✅ encode() raises NotImplementedError correctly
✅ decode() raises NotImplementedError correctly
📈 Progress: Base Tokenizer Interface ✓

🔬 Unit Test: Character Tokenizer...
✅ Vocabulary built with 89 unique characters
✅ Encode/decode round-trip: "hello" → [8,5,12,12,15] → "hello"
✅ Unknown character maps to <UNK> token (ID 0)
✅ Vocabulary building from corpus works correctly
📈 Progress: Character Tokenizer ✓

🔬 Unit Test: BPE Tokenizer...
✅ Character-level initialization successful
✅ Pair extraction: ['h','e','l','l','o</w>'] → {('h','e'), ('l','l'), ...}
✅ Training learned 195 merge rules from corpus
✅ Vocabulary size reached target (200 tokens)
✅ Sequence length reduced 3.2x vs character-level
✅ Unknown words decompose into subwords gracefully
📈 Progress: BPE Tokenizer ✓

🔬 Unit Test: Tokenization Utils...
✅ Tokenizer factory creates correct instances
✅ Dataset processing handles variable lengths
✅ Analysis computes compression ratios correctly
📈 Progress: Tokenization Utils ✓

📊 Analyzing Tokenization Strategies...
Strategy      Vocab    Avg Len  Compression   Coverage
------------------------------------------------------------
Character     89       43.2     1.00          89
BPE-100       100      28.5     1.52          87
BPE-500       500      13.8     3.14          245

💡 Key Insights:
- Character: Small vocab, long sequences, perfect coverage
- BPE: Larger vocab, shorter sequences, better compression
- Higher compression ratio = more characters per token = efficiency

🎉 ALL TESTS PASSED! Module ready for export.
```

### Manual Testing Examples

```python
from tokenization_dev import CharTokenizer, BPETokenizer, create_tokenizer, analyze_tokenization

# Test character-level tokenization
char_tokenizer = CharTokenizer()
corpus = ["hello world", "machine learning is awesome"]
char_tokenizer.build_vocab(corpus)

text = "hello"
char_ids = char_tokenizer.encode(text)
char_decoded = char_tokenizer.decode(char_ids)
print(f"Character: '{text}' → {char_ids} → '{char_decoded}'")
# Output: Character: 'hello' → [8, 5, 12, 12, 15] → 'hello'

# Test BPE tokenization
bpe_tokenizer = BPETokenizer(vocab_size=500)
bpe_tokenizer.train(corpus)

bpe_ids = bpe_tokenizer.encode(text)
bpe_decoded = bpe_tokenizer.decode(bpe_ids)
print(f"BPE: '{text}' → {bpe_ids} → '{bpe_decoded}'")
# Output: BPE: 'hello' → [142, 201] → 'hello'  # Fewer tokens!

# Compare sequence lengths
long_text = "The quick brown fox jumps over the lazy dog" * 10
char_len = len(char_tokenizer.encode(long_text))
bpe_len = len(bpe_tokenizer.encode(long_text))
print(f"Sequence length reduction: {char_len / bpe_len:.1f}x")
# Output: Sequence length reduction: 3.2x

# Analyze tokenization statistics
test_corpus = [
    "Neural networks learn patterns",
    "Transformers use attention mechanisms",
    "Tokenization enables text processing"
]

char_stats = analyze_tokenization(test_corpus, char_tokenizer)
bpe_stats = analyze_tokenization(test_corpus, bpe_tokenizer)

print(f"Character - Vocab: {char_stats['vocab_size']}, "
      f"Avg Length: {char_stats['avg_sequence_length']:.1f}, "
      f"Compression: {char_stats['compression_ratio']:.2f}")
# Output: Character - Vocab: 89, Avg Length: 42.3, Compression: 1.00

print(f"BPE - Vocab: {bpe_stats['vocab_size']}, "
      f"Avg Length: {bpe_stats['avg_sequence_length']:.1f}, "
      f"Compression: {bpe_stats['compression_ratio']:.2f}")
# Output: BPE - Vocab: 500, Avg Length: 13.5, Compression: 3.13
```

## Systems Thinking Questions

### Real-World Applications

**OpenAI GPT Series:**
- **GPT-2**: 50,257 BPE tokens trained on 8M web pages (WebText corpus); vocabulary size chosen to balance 38M embedding parameters (50K × 768 dim) with sequence length for 1024-token context
- **GPT-3**: Increased to 100K vocabulary to handle code (indentation, operators) and reduce sequence lengths for long documents; embedding matrix alone: 1.2B parameters (100K × 12,288 dim)
- **GPT-4**: Advanced tiktoken library with 100K+ tokens, optimized for tokenization throughput at scale ($700/million tokens means every millisecond counts)
- **Question**: Why did OpenAI double vocabulary size from GPT-2→GPT-3? Consider the trade-off: 2x more embedding parameters vs sequence length reduction for code/long documents. What breaks if vocabulary is too small? Too large?

**Google Multilingual Models:**
- **SentencePiece**: Used in BERT, T5, PaLM for 100+ languages without language-specific preprocessing; unified tokenization enables shared vocabulary across languages
- **Vocabulary Sharing**: Multilingual models use single vocabulary for all languages (e.g., mT5: 250K SentencePiece tokens cover 101 languages); trade-off between per-language coverage and total vocabulary size
- **Production Scaling**: Google Translate processes billions of sentences daily; tokenization throughput and vocabulary lookup latency are critical for serving at scale
- **Question**: English needs ~30K tokens for 99% coverage; Chinese ideographic characters need 50K+. Should a multilingual model use one shared vocabulary or separate vocabularies per language? Consider: shared vocabulary enables zero-shot transfer but reduces per-language coverage.

**Code Models (GitHub Copilot, AlphaCode):**
- **Specialized Vocabularies**: Code tokenizers handle programming language syntax (indentation, operators, keywords) and natural language (comments, docstrings); balance code-specific tokens vs natural language
- **Identifier Handling**: Variable names like `getUserProfile` vs `get_user_profile` require different tokenization strategies (camelCase splitting, underscore boundaries)
- **Trade-off**: Larger vocabulary for code-specific tokens reduces sequence length but increases embedding matrix size; rare identifier fragments still need subword decomposition
- **Question**: Should a code tokenizer treat `getUserProfile` as 1 token, 3 tokens (`get`, `User`, `Profile`), or 15 character tokens? Consider: single token = short sequence but huge vocabulary; character-level = long sequences but handles any identifier.

**Production NLP Pipelines:**
- **Google Translate**: Billions of sentences daily require high-throughput tokenization (character: ~1M tokens/sec, BPE: ~100K tokens/sec); vocabulary size affects both model memory and inference speed
- **OpenAI API**: Tokenization cost is significant at $700/million tokens; every optimization (caching, batch processing, vocabulary size tuning) directly impacts economics
- **Mobile Deployment**: Edge models (on-device speech recognition, keyboards) use smaller vocabularies (5K-10K) to fit memory constraints, trading sequence length for model size
- **Question**: If your tokenizer processes 10K tokens/second but your model serves 100K requests/second (each 50 tokens), how do you scale? Consider: pre-tokenize and cache? Batch aggressively? Optimize vocabulary?

### Tokenization Foundations

**Vocabulary Size vs Model Parameters:**
- **Embedding Matrix Scaling**: Embedding parameters = vocab_size × embed_dim
  - GPT-2: 50K vocab × 768 dim = 38.4M parameters (just embeddings!)
  - GPT-3: 100K vocab × 12,288 dim = 1.23B parameters (just embeddings!)
  - BERT-base: 30K vocab × 768 dim = 23M parameters
- **Training Impact**: Larger vocabulary means more parameters to train; embedding gradients scale with vocabulary size (affects memory and optimizer state size)
- **Deployment Constraints**: Embedding matrix must fit in memory during inference; on-device models use smaller vocabularies (5K-10K) to meet memory budgets
- **Question**: If you increase vocabulary from 10K to 100K (10x), how does this affect: (1) Model size? (2) Training memory (gradients + optimizer states)? (3) Inference latency (vocabulary lookup)?

**Sequence Length vs Computation:**
- **Transformer Attention Complexity**: O(n²) where n = sequence length; doubling sequence length quadruples attention computation
- **BPE Compression**: Reduces "unhappiness" (11 chars) to ["un", "happi", "ness"] (3 tokens) → 13.4x less attention computation (11² vs 3²)
- **Batch Processing**: Sequences padded to max length in batch; character-level (1000 tokens) requires 11x more computation than BPE-level (300 tokens) even if actual content is shorter
- **Memory Scaling**: Attention matrices scale as (batch_size × n²); character-level consumes far more GPU memory than BPE
- **Question**: Given text "machine learning" (16 chars), compare computation: (1) Character tokenizer → 16 tokens → 16² = 256 attention ops; (2) BPE → 3 tokens → 3² = 9 attention ops. What's the computational savings ratio? How does this scale to 1000-token documents?

**Rare Word Handling:**
- **Word-Level Failure**: Word tokenizers map unknown words to `<UNK>` token → complete information loss (can't distinguish "antidisestablishmentarianism" from "supercalifragilisticexpialidocious")
- **BPE Graceful Degradation**: Decomposes unknown words into known subwords: "unhappiness" → ["un", "happi", "ness"] preserves semantic information even if full word never seen during training
- **Morphological Generalization**: BPE learns prefixes ("un-", "pre-", "anti-") and suffixes ("-ing", "-ed", "-ness") as tokens, enabling compositional understanding
- **Question**: How does BPE handle "antidisestablishmentarianism" (28 chars) even if never seen during training? Trace the decomposition: which subwords would be discovered? How does this enable the model to understand the word's meaning?

**Tokenization as Compression:**
- **Frequent Pattern Learning**: BPE learns common patterns become single tokens: "ing" → 1 token, "ed" → 1 token, "tion" → 1 token (similar to dictionary-based compression like LZW)
- **Information Theory Connection**: Optimal encoding assigns short codes to frequent symbols (Huffman coding); BPE is essentially dictionary-based compression optimized for language statistics
- **Compression Ratio**: Character-level = 1.0 chars/token (by definition); BPE typically achieves 3-5 chars/token depending on vocabulary size and language
- **Question**: BPE and gzip both learn frequent patterns and replace with short codes. What's the key difference? Hint: BPE operates at subword granularity (preserves linguistic units), gzip operates at byte level (ignores linguistic structure).

### Performance Characteristics

**Tokenization Throughput:**
- **Character-Level Speed**: ~1M tokens/second (simple array lookup: char → ID via hash map)
- **BPE Speed**: ~100K tokens/second (iterative merge application: must scan for applicable merge rules)
- **Production Caching**: Systems cache tokenization results to amortize preprocessing cost (especially for repeated queries or batch processing)
- **Bottleneck Analysis**: If tokenization takes 10ms and model inference takes 100ms (single request), tokenization is 9% overhead; but for batch_size=1000, tokenization becomes 100ms (10ms × 1000 requests) while model inference might be 200ms due to batching efficiency → tokenization is now 33% overhead!
- **Question**: Your tokenizer processes 10K tokens/sec. Model serves 100K requests/sec, each request has 50 tokens. Total tokenization throughput needed: 5M tokens/sec. What do you do? Consider: (1) Parallelize tokenization across CPUs? (2) Cache frequent queries? (3) Switch to character tokenizer (10x faster)? (4) Optimize BPE implementation?

**Memory vs Compute Trade-offs:**
- **Large Vocabulary**: More memory (embedding matrix) but faster tokenization (fewer merge applications) and shorter sequences (less attention computation)
- **Small Vocabulary**: Less memory (smaller embedding matrix) but slower tokenization (more merge rules to apply) and longer sequences (more attention computation)
- **Optimal Vocabulary Size**: Depends on deployment constraints—edge devices (mobile, IoT) prioritize memory (use smaller vocab, accept longer sequences); cloud serving prioritizes throughput (use larger vocab, reduce sequence length)
- **Embedding Matrix Memory**: GPT-3's 100K vocabulary × 12,288 dim × 2 bytes (fp16) = 2.5GB just for embeddings; quantization to int8 reduces to 1.25GB
- **Question**: For edge deployment (mobile device with 2GB RAM budget), should you prioritize: (1) Smaller vocabulary (5K tokens, saves 400MB embedding memory) accepting longer sequences? (2) Larger vocabulary (50K tokens, uses 2GB embeddings) for shorter sequences? Consider: attention computation scales quadratically with sequence length.

**Batching and Padding:**
- **Padding Waste**: Variable-length sequences padded to max length in batch; wasted computation on padding tokens (don't contribute to loss but consume attention operations)
- **Character-Level Penalty**: Longer sequences require more padding—if batch contains [10, 50, 500] character-level tokens, all padded to 500 → 490 + 450 + 0 = 940 wasted tokens (65% waste)
- **BPE Advantage**: Shorter sequences reduce padding waste—same batch as [3, 15, 150] BPE tokens, padded to 150 → 147 + 135 + 0 = 282 wasted tokens (still 63% waste, but absolute numbers smaller)
- **Dynamic Batching**: Group similar-length sequences to reduce padding waste (collate_fn in DataLoader)
- **Question**: Batch of sequences with lengths [10, 50, 500] tokens. (1) Character-level: Total computation = 3 × 500² = 750K attention operations. (2) BPE reduces to [3, 15, 150]: Total = 3 × 150² = 67.5K operations (11x reduction). But what if you sort and batch by length: [[10, 50], [500]] → Char: 2×50² + 1×500² = 255K; BPE: 2×15² + 1×150² = 23K. How much does batching strategy matter?

**Multilingual Considerations:**
- **Shared Vocabulary**: Enables zero-shot cross-lingual transfer (model trained on English can handle French without fine-tuning) but reduces per-language coverage
- **Language-Specific Vocabulary Size**: English: 26 letters → 30K tokens for 99% coverage; Chinese: 50K+ characters → need 60K tokens for equivalent coverage; Arabic: morphologically rich → needs more subword decomposition
- **Vocabulary Allocation**: Multilingual model with 100K shared vocabulary must allocate tokens across languages; high-resource languages (English) get better coverage than low-resource languages (Swahili)
- **Question**: Should a multilingual model use: (1) One shared vocabulary (100K tokens across all languages, enables transfer but dilutes per-language coverage)? (2) Separate vocabularies per language (30K English + 60K Chinese = 90K total, better coverage but no cross-lingual transfer)? Consider: shared embedding space enables "cat" (English) to align with "chat" (French) via training.

## Ready to Build?

You're about to implement the tokenization systems that power every modern language model—from GPT-4 processing trillions of tokens to Google Translate serving billions of requests daily. Tokenization is the critical bridge between human language (text) and neural networks (numbers), and the design decisions you make have profound effects on model size, computational cost, and generalization ability.

By building these systems from scratch, you'll understand the fundamental trade-off shaping modern NLP: **vocabulary size vs sequence length**. Larger vocabularies mean more model parameters (embedding matrix size = vocab_size × embed_dim) but shorter sequences (less computation, especially in transformers with O(n²) attention). Smaller vocabularies mean fewer parameters but longer sequences requiring more computation. You'll see why BPE emerged as the dominant approach—balancing both extremes optimally through learned subword decomposition—and why every major language model (GPT, BERT, T5, LLaMA) uses some form of subword tokenization.

This module connects directly to Module 11 (Embeddings): your token IDs will index into embedding matrices, converting discrete tokens into continuous vectors. Understanding tokenization deeply—not just as a black-box API but as a system with measurable performance characteristics and design trade-offs—will make you a better ML systems engineer. You'll appreciate why GPT-3 doubled vocabulary size from GPT-2 (50K→100K to handle code and long documents), why mobile models use tiny 5K vocabularies (memory constraints), and why production systems aggressively cache tokenization results (throughput optimization).

Take your time, experiment with different vocabulary sizes (100, 1000, 10000), and measure everything: sequence length reduction, compression ratios, tokenization throughput. This is where text becomes numbers, where linguistics meets systems engineering, and where you'll develop the intuition needed to make smart trade-offs in production NLP systems.

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/10_tokenization/tokenization_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/10_tokenization/tokenization_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/10_tokenization/tokenization_dev.ipynb
:class-header: bg-light

Browse the Jupyter notebook and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/09_spatial.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/11_embeddings.html" title="next page">Next Module →</a>
</div>
