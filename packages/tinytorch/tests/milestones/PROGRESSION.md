# From Perceptron to Transformer: How Neural Networks Evolved

This document traces the key innovations in neural network history, showing how each breakthrough solved a specific problem left by its predecessor. We're not just listing milestones—we're showing how they connect.

```
1958: PERCEPTRON
┌─────────────────┐
│   Input (x)     │
│       ↓         │
│   w·x + b       │  ← Single neuron, linear decision boundary
│       ↓         │
│   Output (y)    │
└─────────────────┘
Problem: Can't learn XOR (non-linear patterns)
         ↓

1986: BACKPROPAGATION (XOR)
┌─────────────────┐
│   Input (x)     │
│       ↓         │
│   Linear(2,4)   │  ← Hidden layer
│       ↓         │
│   Tanh()        │  ← Non-linearity (KEY!)
│       ↓         │
│   Linear(4,1)   │
│       ↓         │
│   Sigmoid()     │
│       ↓         │
│   Output (y)    │
└─────────────────┘
Problem: Only tested on toy problems
         ↓

1989: MLP ON REAL DATA
┌─────────────────┐
│  Image (8×8)    │
│       ↓         │
│   Flatten()     │  ← Loses spatial structure!
│       ↓         │
│  Linear(64,128) │
│       ↓         │
│   ReLU()        │
│       ↓         │
│  Linear(128,64) │
│       ↓         │
│   ReLU()        │
│       ↓         │
│  Linear(64,10)  │  ← 10 classes
│       ↓         │
│   Softmax       │
└─────────────────┘
Problem: Flattening destroys spatial relationships
         ↓

1998: CONVOLUTIONAL NETWORKS
┌─────────────────┐
│  Image (1,8,8)  │  ← Preserves 2D structure!
│       ↓         │
│ Conv2d(1,8,3×3) │  ← Spatial filters
│       ↓         │
│   ReLU()        │
│       ↓         │
│ MaxPool2d(2×2)  │  ← Spatial downsampling
│       ↓         │
│   Flatten()     │
│       ↓         │
│  Linear(72,10)  │
└─────────────────┘
Problem: Images have spatial structure, sequences have temporal structure
         ↓

2017: TRANSFORMER (ATTENTION)
┌─────────────────────┐
│  Sequence [1,2,3,4] │
│         ↓           │
│  Embedding(10,16)   │  ← Token → vector
│         ↓           │
│  PositionalEnc(16)  │  ← Add position info
│         ↓           │
│  MultiHeadAttn(2)   │  ← Attend to all positions
│         ↓           │
│  Linear(16,10)      │  ← Predict tokens
│         ↓           │
│  Output [1,2,3,4]   │
└─────────────────────┘
```

## How Each Innovation Builds on the Last

### 1. Perceptron → XOR: Adding Non-linearity

```python
# Perceptron (fails on XOR)
y = w·x + b

# MLP (solves XOR)
h = tanh(W1·x + b1)  # Hidden layer learns features
y = σ(W2·h + b2)     # Output layer combines features
```

Here's the thing: without non-linearity, stacking layers doesn't help. Two linear layers collapse into one:

```
Layer 1: y = W1·x + b1
Layer 2: z = W2·y + b2 = W2·(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1 + b2)
Result: Still just a linear function!
```

The activation function (tanh, sigmoid, ReLU) is what makes depth meaningful.

### 2. XOR → MLP: Scaling to Real Data

```python
# XOR: 4 samples, 2 features
X = [[0,0], [0,1], [1,0], [1,1]]

# Digits: 1000 samples, 64 features (8×8 images)
X = load_tiny_digits()  # Real-world complexity
```

Solving XOR was a proof of concept. But to be useful, neural networks needed to handle:
- Real datasets (not 4 hand-crafted samples)
- High-dimensional inputs (images have 64+ pixels, not 2 features)
- Multiple classes (10 digits, not binary)

That's what the MLP milestone demonstrates.

### 3. MLP → CNN: Preserving Spatial Structure

```python
# MLP: Flatten destroys structure
image = [[1,2,3],
         [4,5,6],
         [7,8,9]]
flat = [1,2,3,4,5,6,7,8,9]  # Lost neighborhood info!

# CNN: Preserve structure
conv = Conv2d(1, 8, kernel_size=3)
features = conv(image)  # Learns spatial patterns
```

When you flatten an image into a vector, you lose neighborhood information. Pixel (1,1) is spatially close to (0,1), (1,0), (2,1), (1,2)—but after flattening, the network doesn't know that.

Convolution fixes this by scanning a small filter across the image, preserving local structure. As a bonus, you get massive parameter savings:

```
MLP:     64 inputs × 128 hidden = 8,192 parameters
CNN:     3×3 kernel × 8 filters = 72 parameters (113× fewer!)
```

### 4. CNN → Transformer: From Spatial to Sequential

```python
# CNN: Spatial relationships (2D)
image[i,j] relates to image[i±1, j±1]

# Transformer: Temporal relationships (1D)
sequence[t] relates to sequence[0...T]
```

CNNs work great for images because spatial relationships are local—edges, corners, textures. But sequences (text, time series) have different structure. The first word in a sentence can affect the meaning of the 100th word.

Attention solves this by letting every position look at every other position:


```python
# For each position, compute attention to all positions
Q = query(x)    # What am I looking for?
K = key(x)      # What do I contain?
V = value(x)    # What should I output?

attention = softmax(Q @ K.T / √d)  # Where to look
output = attention @ V              # What to copy
```

## The Common Thread: Gradient Flow

Every innovation requires **proper backpropagation**:

```python
# Forward pass
y = f(x, θ)

# Backward pass (compute ∂L/∂θ)
loss.backward()

# Update
θ = θ - lr * ∂L/∂θ
```

### Gradient Flow Examples:

**Perceptron**:
```
∂L/∂w = ∂L/∂y · ∂y/∂w = (y - target) · x
```

**MLP (chain rule)**:
```
∂L/∂W1 = ∂L/∂y · ∂y/∂h · ∂h/∂W1
         └─────┴─────┴──────┘
         Chain through layers
```

**CNN (convolution)**:
```
∂L/∂kernel = ∂L/∂output · ∂output/∂kernel
            = ∂L/∂output ⊗ input  (convolution!)
```

**Transformer (attention)**:
```
∂L/∂Q = ∂L/∂attn · ∂attn/∂scores · ∂scores/∂Q
        └────────┴────────────┴──────────┘
        Through softmax and matmul
```

## Milestone Verification: What We Test

### 1. **Perceptron** (1958)
- ✅ Forward pass runs with your `Tensor`, `Linear`, and `Sigmoid`
- ✅ Outputs have reasonable shapes and probabilities
- ✅ No training infrastructure is required yet

### 2. **XOR Crisis** (1969)
- ✅ A single-layer network cannot solve XOR
- ✅ Accuracy stays bounded by the linear limitation
- ✅ Students see the exact failure that motivated deeper networks

### 3. **MLP Revival** (1986)
- ✅ Hidden-layer MLP solves XOR
- ✅ TinyDigits training reaches the milestone accuracy target
- ✅ Loss, autograd, optimizer, dataloader, and trainer compose end-to-end

### 4. **CNN Revolution** (1998)
- ✅ LeNet-style CNN trains on TinyDigits
- ✅ Convolution and pooling layers compose with the training stack
- ✅ Optional CIFAR-10 scale-up exercises the same CNN path on larger images

### 5. **Transformer Era** (2017)
- ✅ Attention solves reversal, copy, and mixed sequence tasks
- ✅ Embeddings, positional encoding, multi-head attention, and transformer blocks compose
- ✅ Sequence accuracy meets the milestone targets

### 6. **MLPerf Benchmarks** (2018)
- ✅ Optimization pipeline runs on prior milestone models
- ✅ Benchmarking reports reproducible performance metrics
- ✅ KV caching demonstrates generation speedup

## Fair Comparisons

### MLP vs CNN (Digits)

**Setup** (identical training budget):
```python
# Both models
batch_size = 32
epochs = 25
samples = 1000
updates = 25 × (1000 ÷ 32) = 775 gradient updates
```

**Results**:
| Model | Accuracy | Loss Decrease | Parameters |
|-------|----------|---------------|------------|
| MLP   | 82.0%    | 52.3%         | 10,890     |
| CNN   | 82.0%    | 68.1%         | 1,098      |

**Insights**:
- Same accuracy (8×8 too small for CNN advantage)
- CNN converges faster (better loss reduction)
- CNN uses 10× fewer parameters
- On larger images, CNN dominates

## The Big Picture

```
PERCEPTRON (1958)
    ↓ Add hidden layers + non-linearity
BACKPROPAGATION (1986)
    ↓ Scale to real data + deeper networks
MLP (1989)
    ↓ Preserve spatial structure
CNN (1998)
    ↓ Handle sequential/temporal structure
TRANSFORMER (2017)
    ↓ Scale to billions of parameters
MODERN DEEP LEARNING (2020s)
```

## Key Takeaways

1. **Each innovation solves a specific limitation**
   - Perceptron → XOR: Need non-linearity
   - XOR → MLP: Need to scale
   - MLP → CNN: Need spatial awareness
   - CNN → Transformer: Need long-range dependencies

2. **All require proper gradients**
   - Every new operation needs backward pass
   - Chain rule connects everything
   - Tests verify gradients actually flow

3. **Learning verification is critical**
   - Code running ≠ model learning
   - Must verify: loss ↓, accuracy ↑, gradients flow
   - Fair comparisons require matched training budgets

4. **Building blocks compound**
   - Transformer uses: Linear (1958), ReLU (1989), Embeddings (2013)
   - Each milestone stands on previous work
   - Modern systems combine all these ideas

## What's Next?

The journey continues:
- **Residual connections** (ResNet, 2015)
- **Batch normalization** (2015)
- **Transformers at scale** (GPT, BERT, 2018+)
- **Diffusion models** (2020+)
- **Mixture of Experts** (2023+)

But they all build on these five fundamental milestones! 🚀
