#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    📦 Pre-Built Networks for Optimization                     ║
║                  (Same architectures from Milestones 01-05)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

These are the SAME network architectures you built in earlier milestones:
- Perceptron: Milestone 01 (1957 Rosenblatt)
- DigitMLP: Milestone 03 (1986 Rumelhart)
- SimpleCNN: Milestone 04 (1998 LeCun)
- MinimalTransformer: Milestone 05 (2017 Vaswani)

In Milestone 06 (MLPerf), we focus on OPTIMIZING these networks, not building them.
You've already proven you can build them - now let's make them production-ready!

Usage:
    from networks import DigitMLP, SimpleCNN, MinimalTransformer

    # These use YOUR Tiny🔥Torch implementations under the hood!
    mlp = DigitMLP()       # YOUR Linear, ReLU
    cnn = SimpleCNN()      # YOUR Conv2d, MaxPool2d
    transformer = MinimalTransformer()  # YOUR Attention, Embeddings
"""

import numpy as np


# ============================================================================
# MILESTONE 01: Perceptron (1957 - Rosenblatt)
# ============================================================================

class Perceptron:
    """
    The original Perceptron from Milestone 01.

    A single-layer linear classifier - the foundation of neural networks.
    Architecture: Input → Linear(in_features, num_classes)

    From: Rosenblatt (1957) "The Perceptron: A Probabilistic Model"
    """

    def __init__(self, input_size=64, num_classes=10):
        from tinytorch.core.layers import Linear

        self.fc = Linear(input_size, num_classes)
        self.layers = [self.fc]
        self.name = "Perceptron"

    def forward(self, x):
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.fc.parameters()


# ============================================================================
# MILESTONE 03: Multi-Layer Perceptron (1986 - Rumelhart, Hinton, Williams)
# ============================================================================

class DigitMLP:
    """
    Multi-Layer Perceptron for digit classification from Milestone 03.

    Architecture: Input(64) → Linear(64→32) → ReLU → Linear(32→10)

    From: Rumelhart, Hinton, Williams (1986) "Learning representations
          by back-propagating errors"
    """

    def __init__(self, input_size=64, hidden_size=32, num_classes=10):
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU

        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, num_classes)

        self.layers = [self.fc1, self.fc2]
        self.name = "DigitMLP"

    def forward(self, x):
        # Flatten if needed (handles 8x8 images)
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


# ============================================================================
# MILESTONE 04: Convolutional Neural Network (1998 - LeCun)
# ============================================================================

class SimpleCNN:
    """
    Simple CNN for digit classification from Milestone 04.

    Architecture: Conv(1→4) → ReLU → MaxPool → Conv(4→8) → ReLU → MaxPool → Linear → 10

    From: LeCun et al. (1998) "Gradient-based learning applied to document recognition"
    """

    def __init__(self, num_classes=10):
        from tinytorch.core.spatial import Conv2d, MaxPool2d
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU

        # Convolutional layers
        self.conv1 = Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        # For 8x8 input: after 2 pools of 2x2, we get 2x2 spatial, 8 channels = 32 features
        self.fc = Linear(32, num_classes)

        self.layers = [self.conv1, self.conv2, self.fc]
        self.name = "SimpleCNN"

    def forward(self, x):
        # Expect (batch, channels, height, width)
        # If (batch, height, width), add channel dimension
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten and classify
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


# ============================================================================
# MILESTONE 05: Minimal Transformer (2017 - Vaswani et al.)
# ============================================================================

class MinimalTransformer:
    """
    Minimal Transformer for sequence tasks from Milestone 05.

    Architecture: Embedding → PositionalEncoding → MultiHeadAttention → FFN → Output

    From: Vaswani et al. (2017) "Attention is All You Need"
    """

    def __init__(self, vocab_size=27, embed_dim=32, num_heads=2, seq_len=8):
        from tinytorch.core.embeddings import Embedding, PositionalEncoding
        from tinytorch.core.attention import MultiHeadAttention
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Embedding layers
        self.token_embed = Embedding(vocab_size, embed_dim)
        self.pos_encode = PositionalEncoding(embed_dim, seq_len)

        # Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Feed-forward
        self.ff1 = Linear(embed_dim, embed_dim * 4)
        self.relu = ReLU()
        self.ff2 = Linear(embed_dim * 4, embed_dim)

        # Output projection
        self.output = Linear(embed_dim, vocab_size)

        self.layers = [self.token_embed, self.attention, self.ff1, self.ff2, self.output]
        self.name = "MinimalTransformer"

    def forward(self, x):
        # x: (batch, seq_len) token indices
        # Embed
        x = self.token_embed(x)
        x = self.pos_encode(x)

        # Attention
        x = self.attention(x)

        # Feed-forward
        ff = self.ff1(x)
        ff = self.relu(ff)
        ff = self.ff2(ff)
        x = Tensor(x.data + ff.data, requires_grad=x.requires_grad)  # Residual

        # Output
        logits = self.output(x)
        return logits

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


# ============================================================================
# UTILITY: Get all networks
# ============================================================================

def get_all_networks():
    """Get a dictionary of all milestone networks."""
    return {
        'perceptron': Perceptron,
        'mlp': DigitMLP,
        'cnn': SimpleCNN,
        'transformer': MinimalTransformer,
    }


def get_network(name: str):
    """Get a network by name."""
    networks = get_all_networks()
    if name.lower() not in networks:
        raise ValueError(f"Unknown network: {name}. Available: {list(networks.keys())}")
    return networks[name.lower()]()


# Import Tensor for residual connection
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    Tensor = None


# ============================================================================
# TEST: Verify networks can be instantiated
# ============================================================================

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold cyan]📦 Testing Milestone Networks[/bold cyan]\n")

    table = Table(title="Network Status")
    table.add_column("Network", style="cyan")
    table.add_column("Parameters", style="yellow")
    table.add_column("Status", style="green")

    for name, NetworkClass in get_all_networks().items():
        try:
            network = NetworkClass()
            param_count = sum(p.data.size for p in network.parameters())
            table.add_row(name.upper(), f"{param_count:,}", "✅ OK")
        except Exception as e:
            table.add_row(name.upper(), "-", f"❌ {e}")

    console.print(table)
