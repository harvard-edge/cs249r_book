"""
TinyTorch - Build ML Systems From First Principles

A complete educational ML framework for learning neural network internals
by implementing everything from scratch.

Students progressively build this package module by module.
Imports are optional - only available after completing each module.
"""

__version__ = "0.1.1"

# ============================================================================
# Progressive Imports - Available as students complete modules
# ============================================================================
# Each import is wrapped in try/except so the package works even when
# students haven't completed all modules yet.

# Module 01: Tensor
try:
    from .core.tensor import Tensor
except ImportError:
    Tensor = None

# Module 02: Activations
try:
    from .core.activations import Sigmoid, ReLU, Tanh, GELU, Softmax
except ImportError:
    Sigmoid = ReLU = Tanh = GELU = Softmax = None

# Module 03: Layers
try:
    from .core.layers import Layer, Linear, Dropout
except ImportError:
    Layer = Linear = Dropout = None

# Module 04: Losses
try:
    from .core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
except ImportError:
    MSELoss = CrossEntropyLoss = BinaryCrossEntropyLoss = None

# Module 05: Data Loading
try:
    from .core.dataloader import Dataset, TensorDataset, DataLoader
    from .core.dataloader import RandomHorizontalFlip, RandomCrop, Compose
except ImportError:
    Dataset = TensorDataset = DataLoader = None
    RandomHorizontalFlip = RandomCrop = Compose = None

# Module 06: Autograd - Enable if available
try:
    from .core.autograd import enable_autograd
    enable_autograd()
except ImportError:
    pass

# Module 07: Optimizers
try:
    from .core.optimizers import SGD, Adam, AdamW
except ImportError:
    SGD = Adam = AdamW = None

# Module 08: Training
try:
    from .core.training import Trainer, CosineSchedule, clip_grad_norm
except ImportError:
    Trainer = CosineSchedule = clip_grad_norm = None

# Module 09: Convolutions (CNN)
try:
    from .core.spatial import Conv2d, MaxPool2d, AvgPool2d
except ImportError:
    Conv2d = MaxPool2d = AvgPool2d = None

# Module 10: Tokenization
try:
    from .core.tokenization import Tokenizer, CharTokenizer, BPETokenizer
except ImportError:
    Tokenizer = CharTokenizer = BPETokenizer = None

# Module 11: Embeddings
try:
    from .core.embeddings import Embedding, PositionalEncoding, EmbeddingLayer
except ImportError:
    Embedding = PositionalEncoding = EmbeddingLayer = None

# Module 12: Attention
try:
    from .core.attention import MultiHeadAttention, scaled_dot_product_attention
except ImportError:
    MultiHeadAttention = scaled_dot_product_attention = None

# Module 13: Transformers
try:
    from .core.transformer import LayerNorm, MLP, TransformerBlock, GPT, create_causal_mask
except ImportError:
    LayerNorm = MLP = TransformerBlock = GPT = create_causal_mask = None

# Module 19: Benchmarking
try:
    from .perf import benchmarking
except ImportError:
    benchmarking = None

# Module 20: Olympics (submission infrastructure)
try:
    from . import olympics
except ImportError:
    olympics = None

# ============================================================================
# Public API - All symbols that may be available
# ============================================================================
__all__ = [
    '__version__',
    # Core
    'Tensor',
    'Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax',
    'Layer', 'Linear', 'Dropout',
    'MSELoss', 'CrossEntropyLoss', 'BinaryCrossEntropyLoss',
    'SGD', 'Adam', 'AdamW',
    'Trainer', 'CosineSchedule', 'clip_grad_norm',
    # Data
    'Dataset', 'TensorDataset', 'DataLoader',
    'RandomHorizontalFlip', 'RandomCrop', 'Compose',
    # Spatial
    'Conv2d', 'MaxPool2d', 'AvgPool2d',
    # Text
    'Tokenizer', 'CharTokenizer', 'BPETokenizer',
    'Embedding', 'PositionalEncoding', 'EmbeddingLayer',
    # Attention & Transformers
    'MultiHeadAttention', 'scaled_dot_product_attention',
    'LayerNorm', 'MLP', 'TransformerBlock', 'GPT', 'create_causal_mask',
    # Performance & Competition
    'benchmarking',
    'olympics',
]
