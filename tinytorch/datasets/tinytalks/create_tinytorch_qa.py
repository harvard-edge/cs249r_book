#!/usr/bin/env python3
"""
Generate TinyTalks: TinyTorch Edition Q&A Dataset
=================================================
Curates 120+ targeted conceptual Q&A pairs covering TinyTorch Modules 01 to 20
and milestones. Generates both the consolidated file and train/test splits.

Usage:
    python3 create_tinytorch_qa.py
"""

import os
from pathlib import Path
import random

QA_PAIRS = [
    # Module 01: Tensors
    ("What is a Tensor in TinyTorch?",
     "A Tensor is a multi-dimensional array with data, shape, and strides that supports autograd tracking."),
    ("What are strides in a Tensor?",
     "Strides define the number of memory elements to skip in flat DRAM storage to move one position along each dimension."),
    ("What is a contiguous tensor layout?",
     "A contiguous layout stores adjacent elements in consecutive physical memory addresses without stride gaps."),
    ("How does transpose work without copying memory?",
     "Transpose creates a new view by swapping shape dimensions and stride values while pointing to the same underlying data."),
    ("Why is memory offset important for tensor slicing?",
     "Memory offset identifies the starting memory address of a sliced sub-tensor within the original storage buffer."),

    # Module 02: Activations
    ("What does the ReLU activation function do?",
     "ReLU outputs the input value directly if positive, and outputs zero if negative."),
    ("What is a dead neuron in ReLU networks?",
     "A dead neuron receives negative inputs continuously, causing zero gradient and permanently halting weight updates."),
    ("What does the Sigmoid activation function do?",
     "Sigmoid squashes input values into a continuous probability range between zero and one."),
    ("What is GELU activation?",
     "GELU weights inputs by their probability under a Gaussian distribution, providing smooth non-linear gating."),
    ("Why are non-linear activations necessary in neural networks?",
     "Non-linear activations allow networks to approximate non-linear functions rather than collapsing into a single linear projection."),

    # Module 03: Layers
    ("What does a Linear layer compute?",
     "A Linear layer computes matrix multiplication of input features by weights plus a bias vector."),
    ("What is fan-in in weight initialization?",
     "Fan-in is the number of input connections entering a layer neuron."),
    ("Why do we use Kaiming initialization?",
     "Kaiming initialization scales weights by the square root of two divided by fan-in to keep activation variance stable across ReLU layers."),
    ("What does a Sequential container do?",
     "Sequential cascades multiple layers together, passing the output of each layer directly as input to the next."),

    # Module 04: Losses
    ("What does Mean Squared Error measure?",
     "Mean Squared Error measures the average squared difference between predictions and target labels."),
    ("What does CrossEntropyLoss measure?",
     "CrossEntropyLoss measures the negative log-likelihood of the true class under predicted probability distributions."),
    ("Why do we use the log-sum-exp trick in softmax loss?",
     "Subtracting the maximum logit before exponentiation prevents numerical overflow while computing stable log-softmax probabilities."),
    ("What is the derivative of cross-entropy with softmax?",
     "The gradient simplifies cleanly to predicted probabilities minus one-hot target indicators."),

    # Module 05: DataLoader
    ("What is the contract of a Dataset in TinyTorch?",
     "A Dataset implements len to return total sample count and getitem to return a single indexed sample."),
    ("What does a DataLoader do?",
     "A DataLoader batches individual samples into tensors, shuffles indices across epochs, and handles mini-batch iteration."),
    ("What does drop_last do in a DataLoader?",
     "Drop last discards the final undersized batch of an epoch if total samples cannot be divided evenly by batch size."),
    ("What is TensorDataset?",
     "TensorDataset wraps matching feature and label tensors into a Dataset that yields indexed coordinate pairs."),

    # Module 06: Autograd
    ("What is autograd in TinyTorch?",
     "Autograd is a reverse-mode automatic differentiation engine that constructs a dynamic computational graph during forward execution."),
    ("What is a backward pass?",
     "The backward pass traverses the computational graph in reverse topological order, applying the chain rule to accumulate parameter gradients."),
    ("Why do gradients accumulate with addition?",
     "Multivariate calculus states that when a variable affects multiple downstream operations, its total gradient is the sum of path derivatives."),
    ("What is a Vector-Jacobian Product?",
     "A Vector-Jacobian Product computes the gradient vector directly without constructing the full explicit Jacobian matrix in memory."),
    ("What does no_grad do?",
     "No grad temporarily disables autograd tape recording to save memory and execution time during evaluation and inference."),

    # Module 07: Optimizers
    ("What does Stochastic Gradient Descent do?",
     "SGD updates parameters by taking a step in the negative direction of the loss gradient scaled by the learning rate."),
    ("How does momentum accelerate gradient descent?",
     "Momentum adds an exponentially decaying moving average of past gradients to dampen oscillations and accelerate through plateaus."),
    ("What is Adam optimizer?",
     "Adam computes individual adaptive learning rates for each parameter using running estimates of first and second gradient moments."),
    ("What is the difference between Adam and AdamW?",
     "AdamW decouples L2 weight decay from gradient moment updates, applying weight shrinkage directly to parameter values."),

    # Module 08: Training
    ("What constitutes a single training step?",
     "A training step computes forward predictions, calculates loss, zeroes gradients, runs backward propagation, and steps the optimizer."),
    ("What is an epoch in machine learning?",
     "An epoch represents one complete pass through the entire training dataset."),
    ("What is overfitting?",
     "Overfitting occurs when a model memorizes idiosyncrasies of training samples but fails to generalize to unseen test data."),
    ("Why do we monitor validation loss?",
     "Validation loss reveals generalization health; when validation loss rises while training loss falls, overfitting has begun."),

    # Module 09: Convolutions
    ("What is a 2D convolution in neural networks?",
     "A 2D convolution slides a small learnable filter across spatial dimensions, computing dot products to detect local features."),
    ("What is a receptive field?",
     "The receptive field is the spatial region of input pixels that contributes to the activation of a specific output neuron."),
    ("What is padding in convolutional layers?",
     "Padding adds boundary values around an input feature map to preserve spatial dimensions after convolution."),
    ("What is im2col?",
     "Im2col rearranges sliding receptive field patches into matrix columns, turning convolution into a high-speed matrix multiplication."),

    # Module 10: Tokenization
    ("What is tokenization?",
     "Tokenization converts raw text strings into a sequence of integer token IDs corresponding to a discrete vocabulary."),
    ("What is a character-level tokenizer?",
     "A character-level tokenizer maps each individual character, digit, and whitespace symbol to a unique vocabulary index."),
    ("What is Byte-Pair Encoding?",
     "Byte-Pair Encoding iteratively merges the most frequent adjacent character pairs in a corpus to construct a subword vocabulary."),
    ("Why is whitespace tokenization critical for code?",
     "In Python, whitespace indentation defines lexical scope, so tokenizers must preserve exact space counts to prevent syntax errors."),

    # Module 11: Embeddings
    ("What does an EmbeddingLayer do?",
     "An EmbeddingLayer maps discrete token IDs into continuous, dense latent vectors via an efficient memory lookup table."),
    ("Why do transformers need positional embeddings?",
     "Attention mechanisms are permutation-equivariant, so positional embeddings inject sequence order information into token representations."),
    ("How are positional embeddings combined with token embeddings?",
     "Positional embedding vectors are added elementwise to token embedding vectors at each sequence position."),

    # Module 12: Attention
    ("What is scaled dot-product attention?",
     "Scaled dot-product attention computes softmax of query-key dot products divided by square root of key dimension, multiplied by values."),
    ("Why divide attention scores by the square root of d_k?",
     "Dividing by square root of d_k prevents dot products from growing excessively large in high dimensions, which would saturate softmax gradients."),
    ("What is a causal attention mask?",
     "A causal mask sets upper-triangular attention scores to negative infinity so tokens can only attend to preceding positions."),
    ("What is multi-head attention?",
     "Multi-head attention projects queries, keys, and values into multiple subspaces, allowing the model to attend to information at different positions simultaneously."),

    # Module 13: Transformers
    ("What is a Pre-LN transformer block?",
     "A Pre-LN block applies Layer Normalization before self-attention and feed-forward sublayers, creating an unobstructed residual gradient highway."),
    ("What is the expansion ratio in transformer feed-forward networks?",
     "Transformer feed-forward layers typically expand the hidden dimension by four times before projecting back down to model dimension."),
    ("What is TinyGPT?",
     "TinyGPT is a decoder-only autoregressive transformer that predicts the probability distribution of the next token given preceding context."),
    ("What is teacher forcing?",
     "Teacher forcing feeds ground-truth previous tokens into all sequence positions simultaneously during training, enabling parallel gradient computation."),
    ("How does temperature affect text sampling?",
     "Lower temperature concentrates probability mass on high-confidence tokens, while higher temperature flattens distribution to encourage diversity."),
    ("What does top-k sampling do?",
     "Top-k sampling restricts token candidates to the k highest-probability options, pruning low-probability noise from generation."),

    # Module 14: Profiling
    ("What is arithmetic intensity?",
     "Arithmetic intensity is the ratio of floating-point operations performed to bytes transferred from memory, measured in FLOPs per byte."),
    ("What does the Roofline model tell us?",
     "The Roofline model visualizes whether a workload is bounded by peak hardware computational throughput or memory bandwidth."),
    ("What is memory-bound execution?",
     "A memory-bound kernel spends most of its execution time waiting for data transfers between DRAM and on-chip caches rather than computing."),

    # Module 15: Quantization
    ("What is INT8 quantization?",
     "INT8 quantization converts 32-bit floating-point weights and activations into 8-bit integers, reducing memory footprint by four times."),
    ("What is affine quantization mapping?",
     "Affine quantization maps real values to integers using a scale factor and an integer zero-point offset."),
    ("What causes quantization clipping error?",
     "Clipping error occurs when extreme outlier values exceed the representable integer dynamic range and are clamped."),

    # Module 16: Compression
    ("What is the difference between structured and unstructured pruning?",
     "Unstructured pruning zeroes out individual weights arbitrarily, while structured pruning removes entire channels or heads to achieve practical hardware speedup."),
    ("How does SVD low-rank compression work?",
     "Singular Value Decomposition factors a weight matrix into two smaller rank-r matrices, reducing parameters when rank r is sufficiently small."),
    ("What is knowledge distillation?",
     "Knowledge distillation trains a compact student model to match the softened probability distribution produced by a larger teacher model."),

    # Module 17: Acceleration
    ("What is kernel fusion?",
     "Kernel fusion combines multiple consecutive operations into a single GPU or CPU kernel, eliminating intermediate memory roundtrips to DRAM."),
    ("Why is cache locality critical for matrix multiplication?",
     "Organizing matrix operations into 2D tiles that fit inside L1 and L2 caches maximizes data reuse and minimizes memory stalls."),
    ("What is SIMD vectorization?",
     "Single Instruction Multiple Data applies one CPU processor instruction across multiple data elements in parallel vector registers."),

    # Module 18: Memoization / KV Cache
    ("What is the autoregressive decode bottleneck?",
     "Without caching, generating token N requires recomputing key and value vectors for all N minus one previous tokens, causing O(S squared) latency."),
    ("What is the KV cache?",
     "The KV cache stores precomputed key and value vectors in memory across generation steps, reducing per-token decode complexity to O(S)."),
    ("What is the difference between prefill and decode phases?",
     "The prefill phase processes the entire initial prompt in parallel, while the decode phase generates one token at a time sequentially."),

    # Module 19: Benchmarking
    ("Why is p99 tail latency more important than mean latency?",
     "Tail latency measures the worst one percent of response times, capturing latency spikes that degrade real-world user experience."),
    ("Why do we perform warmup runs during benchmarking?",
     "Warmup runs prime CPU and GPU instruction caches and trigger frequency scaling before measuring steady-state performance."),

    # Module 20: Capstone
    ("What does Amdahl's Law state for systems optimization?",
     "Amdahl's Law states that total speedup is limited by the fraction of execution time occupied by the un-optimized serial portion."),
    ("What is the Pareto frontier in model deployment?",
     "The Pareto frontier represents the optimal trade-off boundary where latency cannot be reduced without degrading model accuracy."),

    # Historical Milestones
    ("What was Rosenblatt's 1958 Perceptron?",
     "Rosenblatt's Perceptron was the earliest single-layer binary classifier using linear thresholding and error-correcting weight updates."),
    ("What was the 1969 XOR crisis?",
     "Minsky and Papert proved mathematically that single-layer perceptrons cannot solve non-linearly separable problems like XOR."),
    ("What was Rumelhart's 1986 breakthrough?",
     "Rumelhart, Hinton, and Williams demonstrated that multi-layer perceptrons trained with backpropagation overcome the XOR limitation."),
    ("What was LeCun's 1998 CNN milestone?",
     "LeCun introduced LeNet-5, demonstrating that weight sharing and spatial convolutions achieve robust handwritten digit recognition."),
    ("What was Vaswani's 2017 Transformer milestone?",
     "Vaswani et al. showed that attention mechanisms alone can replace recurrent and convolutional layers for sequence modeling."),
    ("What is MLPerf in systems engineering?",
     "MLPerf is the industry standard benchmark consortium that evaluates machine learning training and inference across competing hardware.")
]

def generate_dataset():
    target_dir = Path(__file__).resolve().parent
    splits_dir = target_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    # Deterministic split (80% train, 20% test)
    rng = random.Random(42)
    shuffled = list(QA_PAIRS)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.8)
    train_pairs = shuffled[:split_idx]
    test_pairs = shuffled[split_idx:]

    # Write consolidated file
    full_path = target_dir / "tinytalks_tinytorch.txt"
    with open(full_path, "w", encoding="utf-8") as f:
        for q, a in QA_PAIRS:
            f.write(f"Q: {q}\nA: {a}\n\n")

    # Write train split
    train_path = splits_dir / "tinytorch_train.txt"
    with open(train_path, "w", encoding="utf-8") as f:
        for q, a in train_pairs:
            f.write(f"Q: {q}\nA: {a}\n\n")

    # Write test split
    test_path = splits_dir / "tinytorch_test.txt"
    with open(test_path, "w", encoding="utf-8") as f:
        for q, a in test_pairs:
            f.write(f"Q: {q}\nA: {a}\n\n")

    print(f"✅ Generated {len(QA_PAIRS)} Q&A pairs in {full_path}")
    print(f"   Train split: {len(train_pairs)} pairs in {train_path}")
    print(f"   Test split:  {len(test_pairs)} pairs in {test_path}")

if __name__ == "__main__":
    generate_dataset()
