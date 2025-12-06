# TinyTorch Datasets

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Ship-with-Repo Datasets for Fast Learning</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Small datasets for instant iteration + standard benchmarks for validation</p>
</div>

**Purpose**: Understand TinyTorch's dataset strategy and where to find each dataset used in milestones.

## Design Philosophy

TinyTorch uses a two-tier dataset approach:

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 2rem 0;">

<div style="background: #e3f2fd; border: 1px solid #2196f3; padding: 1.5rem; border-radius: 0.5rem;">
<h3 style="margin: 0 0 1rem 0; color: #1976d2;">Shipped Datasets</h3>
<p style="margin: 0 0 1rem 0;"><strong>~350 KB total - Ships with repository</strong></p>
<ul style="margin: 0; font-size: 0.9rem;">
<li>Small enough to fit in Git (~1K samples each)</li>
<li>Fast training (seconds to minutes)</li>
<li>Instant gratification for learners</li>
<li>Works offline - no download needed</li>
<li>Perfect for rapid iteration</li>
</ul>
</div>

<div style="background: #f3e5f5; border: 1px solid #9c27b0; padding: 1.5rem; border-radius: 0.5rem;">
<h3 style="margin: 0 0 1rem 0; color: #7b1fa2;">Downloaded Datasets</h3>
<p style="margin: 0 0 1rem 0;"><strong>~180 MB - Auto-downloaded when needed</strong></p>
<ul style="margin: 0; font-size: 0.9rem;">
<li>Standard ML benchmarks (MNIST, CIFAR-10)</li>
<li>Larger scale (~60K samples)</li>
<li>Used for validation and scaling</li>
<li>Downloaded automatically by milestones</li>
<li>Cached locally for reuse</li>
</ul>
</div>

</div>

**Philosophy**: Following Andrej Karpathy's "~1K samples" approach—small datasets for learning, full benchmarks for validation.

---

## Shipped Datasets (Included with TinyTorch)

### TinyDigits - Handwritten Digit Recognition

<div style="background: #fff5f5; border-left: 4px solid #e74c3c; padding: 1.5rem; margin: 1.5rem 0;">

**Location**: `datasets/tinydigits/`  
**Size**: ~310 KB  
**Used by**: Milestones 03 & 04 (MLP and CNN examples)

**Contents:**
- 1,000 training samples
- 200 test samples
- 8×8 grayscale images (downsampled from MNIST)
- 10 classes (digits 0-9)

**Format**: Python pickle file with NumPy arrays

**Why 8×8?**
- Fast iteration: Trains in seconds
- Memory-friendly: Small enough to debug
- Conceptually complete: Same challenges as 28×28 MNIST
- Git-friendly: Only 310 KB vs 10 MB for full MNIST

**Usage in milestones:**
```python
# Automatically loaded by milestones
from datasets.tinydigits import load_tinydigits
X_train, y_train, X_test, y_test = load_tinydigits()
# X_train shape: (1000, 8, 8)
# y_train shape: (1000,)
```

</div>

### TinyTalks - Conversational Q&A Dataset

<div style="background: #f0fff4; border-left: 4px solid #22c55e; padding: 1.5rem; margin: 1.5rem 0;">

**Location**: `datasets/tinytalks/`  
**Size**: ~40 KB  
**Used by**: Milestone 05 (Transformer/GPT text generation)

**Contents:**
- 350 Q&A pairs across 5 difficulty levels
- Character-level text data
- Topics: General knowledge, math, science, reasoning
- Balanced difficulty distribution

**Format**: Plain text files with Q: / A: format

**Why conversational format?**
- Engaging: Questions feel natural
- Varied: Different answer lengths and complexity
- Educational: Difficulty levels scaffold learning
- Practical: Mirrors real chatbot use cases

**Example:**
```
Q: What is the capital of France?
A: Paris

Q: If a train travels 120 km in 2 hours, what is its average speed?
A: 60 km/h
```

**Usage in milestones:**
```python
# Automatically loaded by transformer milestones
from datasets.tinytalks import load_tinytalks
dataset = load_tinytalks()
# Returns list of (question, answer) pairs
```

See detailed documentation: `datasets/tinytalks/README.md`

</div>

---

## Downloaded Datasets (Auto-Downloaded On-Demand)

These standard benchmarks download automatically when you run relevant milestone scripts:

### MNIST - Handwritten Digit Classification

<div style="background: #fffbeb; border-left: 4px solid #f59e0b; padding: 1.5rem; margin: 1.5rem 0;">

**Downloads to**: `milestones/datasets/mnist/`  
**Size**: ~10 MB (compressed)  
**Used by**: `milestones/03_1986_mlp/02_rumelhart_mnist.py`

**Contents:**
- 60,000 training samples
- 10,000 test samples
- 28×28 grayscale images
- 10 classes (digits 0-9)

**Auto-download**: When you run the MNIST milestone script, it automatically:
1. Checks if data exists locally
2. Downloads if needed (~10 MB)
3. Caches for future runs
4. Loads data using your TinyTorch DataLoader

**Purpose**: Validate that your framework achieves production-level results (95%+ accuracy target)

**Milestone goal**: Implement backpropagation and achieve 95%+ accuracy—matching 1986 Rumelhart's breakthrough.

</div>

### CIFAR-10 - Natural Image Classification

<div style="background: #fdf2f8; border-left: 4px solid #ec4899; padding: 1.5rem; margin: 1.5rem 0;">

**Downloads to**: `milestones/datasets/cifar-10/`  
**Size**: ~170 MB (compressed)  
**Used by**: `milestones/04_1998_cnn/02_lecun_cifar10.py`

**Contents:**
- 50,000 training samples
- 10,000 test samples
- 32×32 RGB images
- 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)

**Auto-download**: Milestone script handles everything:
1. Downloads from official source
2. Verifies integrity
3. Caches locally
4. Preprocesses for your framework

**Purpose**: Prove your CNN implementation works on real natural images (75%+ accuracy target)

**Milestone goal**: Build LeNet-style CNN achieving 75%+ accuracy—demonstrating spatial intelligence.

</div>

---

## Dataset Selection Rationale

### Why These Specific Datasets?

**TinyDigits (not full MNIST):**
- 100× faster training iterations
- Ships with repo (no download)
- Same conceptual challenges
- Perfect for learning and debugging

**TinyTalks (custom dataset):**
- Designed for educational progression
- Scaffolded difficulty levels
- Character-level tokenization friendly
- Engaging conversational format

**MNIST (when scaling up):**
- Industry standard benchmark
- Validates your implementation
- Comparable to published results
- 95%+ accuracy is achievable milestone

**CIFAR-10 (for CNN validation):**
- Natural images (harder than digits)
- RGB channels (multi-dimensional)
- Standard CNN benchmark
- 75%+ with basic CNN proves it works

---

## Accessing Datasets

### For Students

**You don't need to manually download anything!**

```bash
# Just run milestone scripts
cd milestones/03_1986_mlp
python 01_rumelhart_tinydigits.py  # Uses shipped TinyDigits

python 02_rumelhart_mnist.py       # Auto-downloads MNIST if needed
```

The milestones handle all data loading automatically.

### For Developers/Researchers

**Direct dataset access:**

```python
# Shipped datasets (always available)
from datasets.tinydigits import load_tinydigits
X_train, y_train, X_test, y_test = load_tinydigits()

from datasets.tinytalks import load_tinytalks
conversations = load_tinytalks()

# Downloaded datasets (through milestones)
# See milestones/data_manager.py for download utilities
```

---

## Dataset Sizes Summary

| Dataset | Size | Samples | Ships With Repo | Purpose |
|---------|------|---------|-----------------|---------|
| TinyDigits | 310 KB | 1,200 | Yes | Fast MLP/CNN iteration |
| TinyTalks | 40 KB | 350 pairs | Yes | Transformer learning |
| MNIST | 10 MB | 70,000 | Downloads | MLP validation |
| CIFAR-10 | 170 MB | 60,000 | Downloads | CNN validation |

**Total shipped**: ~350 KB  
**Total with benchmarks**: ~180 MB

---

## Why Ship-with-Repo Matters

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; margin: 1.5rem 0;">

**Traditional ML courses:**
- "Download MNIST (10 MB)"
- "Download CIFAR-10 (170 MB)"
- Wait for downloads before starting
- Large files in Git (bad practice)

**TinyTorch approach:**
- Clone repo → Immediately start learning
- Train first model in under 1 minute
- Full benchmarks download only when scaling
- Git repo stays small and fast

**Educational benefit**: Students see working models within minutes, not hours.

</div>

---

## Frequently Asked Questions

**Q: Why not use full MNIST from the start?**  
A: TinyDigits trains 100× faster, enabling rapid iteration during learning. MNIST validates your complete implementation later.

**Q: Can I use my own datasets?**  
A: Absolutely! TinyTorch is a real framework—add your data loading code just like PyTorch.

**Q: Why ship datasets in Git?**  
A: 350 KB is negligible (smaller than many images), and it enables offline learning with instant iteration.

**Q: Where does CIFAR-10 download from?**  
A: Official sources via `milestones/data_manager.py`, with integrity verification.

**Q: Can I skip the large downloads?**  
A: Yes! You can work through most milestones using only shipped datasets. Downloaded datasets are for validation milestones.

---

## Related Documentation

- [Milestones Guide](chapters/milestones.md) - See how each dataset is used in historical achievements
- [Student Workflow](student-workflow.md) - Learn the development cycle
- [Quick Start](quickstart-guide.md) - Start building in 15 minutes

**Dataset implementation details**: See `datasets/tinydigits/README.md` and `datasets/tinytalks/README.md` for technical specifications.
