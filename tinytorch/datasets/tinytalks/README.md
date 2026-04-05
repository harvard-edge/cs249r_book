# TinyTalks: A Conversational Q&A Dataset for Educational Transformers

**A carefully curated question-answering dataset designed for learning transformer architectures**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Size: ~50KB](https://img.shields.io/badge/Size-~50KB-blue.svg)]()
[![Version: 1.0.0](https://img.shields.io/badge/Version-1.0.0-green.svg)]()

---

## 📖 Overview

**TinyTalks** is a lightweight, pedagogically-designed conversational dataset for training transformer models in educational settings. Unlike large-scale datasets that require hours of training, TinyTalks enables students to see their first transformer learn meaningful patterns in **under 5 minutes**.

### Why TinyTalks?

✅ **Fast Training** - Trains in 3-5 minutes on a laptop
✅ **Verifiable Learning** - Clear success metrics (correct vs. incorrect answers)
✅ **Progressive Difficulty** - 5 levels from greetings to reasoning
✅ **Educational Focus** - Designed for "aha!" moments, not benchmarks
✅ **Zero Dependencies** - Ships with TinyTorch, no downloads needed
✅ **Reproducible** - Deterministic generation, versioned releases

---

## 📊 Dataset Statistics

<table>
  <thead>
    <tr>
      <th width="40%">Property</th>
      <th width="60%">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Total Q&A Pairs</b></td>
      <td>350</td>
    </tr>
    <tr>
      <td><b>File Size</b></td>
      <td>~40 KB</td>
    </tr>
    <tr>
      <td><b>Vocabulary Size</b></td>
      <td>~1,500 unique tokens (character-level)</td>
    </tr>
    <tr>
      <td><b>Avg Question Length</b></td>
      <td>8 words</td>
    </tr>
    <tr>
      <td><b>Avg Answer Length</b></td>
      <td>10 words</td>
    </tr>
    <tr>
      <td><b>Training Split</b></td>
      <td>245 pairs (70%)</td>
    </tr>
    <tr>
      <td><b>Validation Split</b></td>
      <td>53 pairs (15%)</td>
    </tr>
    <tr>
      <td><b>Test Split</b></td>
      <td>52 pairs (15%)</td>
    </tr>
  </tbody>
</table>

---

## 🎯 Content Structure

TinyTalks is organized into **5 progressive difficulty levels**:

### **Level 1: Greetings & Identity (50 pairs)**
Basic conversational patterns and self-identification.

```
Q: Hello!
A: Hi there! How can I help you today?

Q: What is your name?
A: I am TinyBot, a simple AI assistant.
```

**Learning Goal:** Model learns conversation structure and identity.

---

### **Level 2: Simple Facts (100 pairs)**
Factual knowledge about the world (colors, animals, objects).

```
Q: What color is the sky?
A: The sky is blue during the day.

Q: How many legs does a dog have?
A: A dog has four legs.
```

**Learning Goal:** Model learns to retrieve and state facts.

---

### **Level 3: Basic Math (50 pairs)**
Simple arithmetic operations (single-digit addition, subtraction).

```
Q: What is 2 plus 3?
A: 2 plus 3 equals 5.

Q: What is 7 minus 4?
A: 7 minus 4 equals 3.
```

**Learning Goal:** Model learns basic numerical reasoning.

---

### **Level 4: Common Sense Reasoning (100 pairs)**
Purpose, usage, and logical relationships.

```
Q: What do you use an umbrella for?
A: You use an umbrella to stay dry in the rain.

Q: Where do fish live?
A: Fish live in water.
```

**Learning Goal:** Model learns causal and functional relationships.

---

### **Level 5: Multi-turn Context (50 pairs)**
Simple context tracking across exchanges.

```
Q: I like pizza.
A: Pizza is delicious! What toppings do you like?

Q: I like pepperoni.
A: Pepperoni is a popular choice!
```

**Learning Goal:** Model begins to track conversational context.

---

## 🚀 Quick Start

### Loading the Dataset

```python
# Load full dataset
with open('datasets/tinytalks/tinytalks_v1.txt', 'r') as f:
    text = f.read()

# Or use pre-split versions
with open('datasets/tinytalks/splits/train.txt', 'r') as f:
    train_text = f.read()
```

### Training a Transformer

```python
# See milestones/05_2017_transformer/tinybot_demo.py for full example
from tinytorch.models.transformer import GPT
from tinytorch.text.tokenization import CharTokenizer

# Initialize model
tokenizer = CharTokenizer()
tokenizer.fit(train_text)

model = GPT(
    vocab_size=len(tokenizer),
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    max_seq_len=64
)

# Train for 5 minutes → See meaningful results!
```

### Expected Performance

After training for **10-20 epochs** (~3-5 minutes):
- ✅ Correctly answers Level 1-2 questions (~80% accuracy)
- ✅ Maintains grammatical structure
- ✅ Generates coherent (if not always correct) responses
- ⚠️ Level 3-5 show partial understanding

This demonstrates the transformer has **learned patterns**, not just memorized.

---

## 📐 Dataset Format

**Simple, human-readable text format:**

```
Q: [Question text]
A: [Answer text]

Q: [Next question]
A: [Next answer]
```

**Rationale:**
- Character-level tokenization (no special tokenizers needed)
- Easy to inspect and validate
- Works with any text processing pipeline
- Human-readable for debugging

**Delimiter:** Empty line separates Q&A pairs.

---

## 🔬 Dataset Creation Methodology

### Generation Process

1. **Manual Curation** - All Q&A pairs hand-written by TinyTorch maintainers
2. **Diversity Sampling** - Systematic coverage of topics within each level
3. **Quality Control** - Each pair reviewed for grammar, factual accuracy, appropriateness
4. **Balance Verification** - Ensured even distribution across levels
5. **Reproducibility** - Generation script (`scripts/generate_tinytalks.py`) produces identical output

### Quality Assurance

- ✅ Grammar check (automated + manual review)
- ✅ Factual accuracy verification
- ✅ No offensive or biased content
- ✅ No personally identifiable information
- ✅ Balanced topic distribution
- ✅ Appropriate for all ages

### Validation Script

```bash
python datasets/tinytalks/scripts/validate_dataset.py
```

Checks:
- Format consistency
- No duplicate pairs
- Balanced splits
- Character encoding (UTF-8)
- Line endings (Unix)

---

## 📊 Dataset Statistics

Run `scripts/stats.py` to generate:

```bash
python datasets/tinytalks/scripts/stats.py
```

Output:
- Total pairs per level
- Vocabulary statistics
- Length distributions
- Split sizes
- Character frequency

---

## 🎓 Educational Use Cases

### Primary Use: Module 13 (Transformers)

TinyTalks is designed as the **canonical dataset** for TinyTorch's Transformer milestone:

- **milestones/05_2017_transformer/tinybot_demo.py** - Main training demo
- Students see their first transformer learn in < 5 minutes
- Clear success metric: Can it answer questions?
- "Wow, I built this!" moment

### Secondary Uses

1. **Tokenization** (Module 10) - Character vs. BPE comparison
2. **Embeddings** (Module 11) - Visualize learned embeddings
3. **Attention** (Module 12) - Inspect attention patterns on Q&A
4. **Debugging** - Small enough to trace gradients manually
5. **Experimentation** - Test architecture changes quickly

---

## ⚖️ License

**Creative Commons Attribution 4.0 International (CC BY 4.0)**

You are free to:
- ✅ Share — copy and redistribute in any format
- ✅ Adapt — remix, transform, and build upon the material
- ✅ Commercial use allowed

Under these terms:
- **Attribution** — Cite TinyTalks (see below)
- **No additional restrictions**

See [LICENSE](LICENSE) for full text.

---

## 📚 Citation

If you use TinyTalks in your work, please cite:

```bibtex
@dataset{tinytalks2025,
  title={TinyTalks: A Conversational Q\&A Dataset for Educational Transformers},
  author={TinyTorch Contributors},
  year={2025},
  publisher={GitHub},
  url={https://github.com/harvard-edge/cs249r_book/tree/main/tinytorch/datasets/tinytalks},
  version={1.0.0}
}
```

**Text citation:**
TinyTorch Contributors. (2025). TinyTalks: A Conversational Q&A Dataset for Educational Transformers (Version 1.0.0). https://github.com/harvard-edge/cs249r_book/tree/main/tinytorch/datasets/tinytalks

---

## 🔄 Versioning

**Version 1.0.0** (Current)
- Initial release: 350 Q&A pairs across 5 levels
- Character-level format
- 70/15/15 train/val/test split

**Planned:**
- v1.1 - Add 100 more Level 4-5 pairs for better reasoning
- v2.0 - Multi-language support (Spanish, French)
- v3.0 - Expanded to 1,000 pairs with more complex reasoning

See [CHANGELOG.md](CHANGELOG.md) for detailed history.

---

## 🤝 Contributing

We welcome contributions! Ways to help:

1. **Report Issues** - Found a factual error or typo? Open an issue.
2. **Suggest Q&A Pairs** - Submit ideas for new questions via PR.
3. **Translations** - Help translate TinyTalks to other languages.
4. **Validation** - Test on different models and report results.

**Guidelines:**
- Follow existing format and style
- Ensure factual accuracy
- Keep language simple and clear
- No offensive or biased content
- Appropriate for all ages (G-rated)

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/harvard-edge/cs249r_book/issues)
- **Discussions:** [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions)
- **Email:** [info@mlsysbook.ai](mailto:info@mlsysbook.ai) (for sensitive issues)

---

## 🙏 Acknowledgments

**Inspired by:**
- bAbI Dataset (Facebook AI Research) - Reasoning tasks
- SQuAD - Question answering format
- TinyStories - Simplicity philosophy
- TinyTorch Community - Feedback and testing

**Created for:**
- Students learning transformer architectures
- Educators teaching NLP
- Researchers prototyping small models
- Developers testing implementations

---

## 📖 Additional Documentation

- **[DATASHEET.md](DATASHEET.md)** - Comprehensive dataset metadata (Gebru et al. format)
- **[examples/demo_usage.py](examples/demo_usage.py)** - Complete usage examples
- **[scripts/README.md](scripts/README.md)** - Scripts documentation

---

## 🌟 Why "TinyTalks"?

The name embodies our philosophy:

- **Tiny** - Small enough to train in minutes, not hours
- **Talks** - Conversational, accessible, human-like
- **Educational** - Designed for learning, not leaderboards

Just like TinyTorch makes deep learning accessible, TinyTalks makes conversational AI **immediate and tangible**.

---

*Built with ❤️ by the TinyTorch community*

*"The best way to understand transformers is to see them learn."*
