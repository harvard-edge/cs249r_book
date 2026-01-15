# TinyDigits Dataset

A curated subset of the sklearn digits dataset for rapid ML prototyping and educational demonstrations.

Following Karpathy's "~1000 samples" philosophy for educational datasets.

## Contents

- **Training**: 1000 samples (100 per digit, 0-9)
- **Test**: 200 samples (20 per digit, balanced)
- **Format**: 8×8 grayscale images, float32 normalized [0, 1]
- **Size**: ~310 KB total (vs 10 MB MNIST, 50× smaller)

## Files

```
datasets/tinydigits/
├── train.pkl  # {'images': (1000, 8, 8), 'labels': (1000,)}
└── test.pkl   # {'images': (200, 8, 8), 'labels': (200,)}
```

## Usage

```python
import pickle

# Load training data
with open('datasets/tinydigits/train.pkl', 'rb') as f:
    data = pickle.load(f)
    train_images = data['images']  # (1000, 8, 8)
    train_labels = data['labels']  # (1000,)

# Load test data
with open('datasets/tinydigits/test.pkl', 'rb') as f:
    data = pickle.load(f)
    test_images = data['images']   # (200, 8, 8)
    test_labels = data['labels']   # (200,)
```

## Purpose

**Educational Infrastructure**: Designed for teaching ML systems with real data at edge-device scale.

Following Andrej Karpathy's philosophy: "~1000 samples is the sweet spot for educational datasets."

- **Decent accuracy**: Achieves ~80% test accuracy on MLPs (vs <20% with 150 samples)
- **Fast training**: <10 sec on CPU, instant feedback loop
- **Balanced classes**: Perfect 100 samples per digit (0-9)
- **Offline-capable**: Ships with repo, no downloads needed
- **USB-friendly**: 310 KB fits on any device, even RasPi0
- **Real learning curve**: Model improves visibly across epochs

## Curation Process

Created from the sklearn digits dataset (8×8 downsampled MNIST):

1. **Balanced Sampling**: 100 training samples per digit class (1000 total)
2. **Test Split**: 20 samples per digit (200 total) from remaining examples
3. **Random Seeding**: Reproducible selection (seed=42)
4. **Normalization**: Pixels normalized to [0, 1] range
5. **Shuffled**: Training and test sets randomly shuffled for fair evaluation

The sklearn digits dataset itself is derived from the UCI ML hand-written digits datasets.

## Why TinyDigits vs Full MNIST?

| Metric | MNIST | TinyDigits | Benefit |
|--------|-------|------------|---------|
| Samples | 60,000 | 1,000 | 60× fewer samples |
| File size | 10 MB | 310 KB | 32× smaller |
| Train time | 5-10 min | <10 sec | 30-60× faster |
| Test accuracy (MLP) | ~92% | ~80% | Close enough for learning |
| Download | Network required | Ships with repo | Always available |
| Resolution | 28×28 (784 pixels) | 8×8 (64 pixels) | Faster forward pass |
| Edge deployment | Challenging | Perfect | Works on RasPi0 |

## Educational Progression

TinyDigits serves as the first step in a scaffolded learning path:

1. **TinyDigits (8×8)** ← Start here: Learn MLP/CNN basics with instant feedback
2. **Full MNIST (28×28)** ← Graduate to: Standard benchmark, longer training
3. **CIFAR-10 (32×32 RGB)** ← Advanced: Color images, real-world complexity

## Citation

TinyDigits is curated from the sklearn digits dataset for educational use in TinyTorch.

**Original Source**:
- sklearn.datasets.load_digits()
- Derived from UCI ML hand-written digits datasets
- License: BSD 3-Clause (sklearn)

**TinyTorch Curation**:
```bibtex
@misc{tinydigits2025,
  title={TinyDigits: Curated Educational Dataset for ML Systems Learning},
  author={TinyTorch Project},
  year={2025},
  note={Balanced subset of sklearn digits optimized for edge deployment}
}
```

## Generation

To regenerate this dataset from the original sklearn data:

```bash
python3 datasets/tinydigits/create_tinydigits.py
```

This ensures reproducibility and allows customization for specific educational needs.

## License

See [LICENSE](LICENSE) for details. TinyDigits inherits the BSD 3-Clause license from sklearn.
