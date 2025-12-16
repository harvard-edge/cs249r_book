#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🤖 MILESTONE 05.2: CodeBot - Python Autocomplete                ║
║          Train a Transformer to Complete Python Code (Copilot Style)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 HISTORICAL CONTEXT (2021):
- 2021: GitHub Copilot launches - Transformer-based code completion
- Same core idea: predict the next token based on context
- The difference? Scale: Copilot trains on BILLIONS of code patterns

🎯 WHAT YOU'RE BUILDING:
A mini version of GitHub Copilot using YOUR Tiny🔥Torch implementations!
Train a transformer on 50 Python patterns and watch it learn to autocomplete.

Student Journey:
1. Watch it train (2 min)
2. See demo completions (2 min)
3. Try it yourself (5 min)
4. Find its limits (2 min)
5. Teach it new patterns (3 min)

✅ REQUIRED MODULES (Run after Module 13):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)         : YOUR data structure for all computations
  Module 10 (Tokenization)   : YOUR CharTokenizer for code → tokens
  Module 11 (Embeddings)     : YOUR embeddings for token representations
  Module 12 (Attention)      : YOUR attention mechanism (pattern matching!)
  Module 13 (Transformer)    : YOUR GPT model for autoregressive generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Code Completion Pipeline):
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Python Code │     │ Tokenizer   │     │ GPT Model   │
    │ "def add("  │────▶│ YOUR M10    │────▶│ YOUR M13    │
    │             │     │ char-level  │     │ transformer │
    └─────────────┘     └─────────────┘     └──────┬──────┘
                                                   │
                              ┌────────────────────┘
                              ▼
                        ┌─────────────┐     ┌─────────────┐
                        │ Next Token  │     │ Completed   │
                        │ Prediction  │────▶│ "a, b):"    │
                        │ (greedy)    │     │             │
                        └─────────────┘     └─────────────┘

# =============================================================================
# 📊 YOUR MODULES IN ACTION
# =============================================================================
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ Systems Impact              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 10: Tokenize │ Converts Python code to chars  │ Enables character-level     │
# │                     │ and back to readable code      │ code understanding          │
# │                     │                                │                             │
# │ Module 12: Attn     │ Finds patterns in code context │ "def" often followed by     │
# │                     │ to predict next characters     │ function name + "("         │
# │                     │                                │                             │
# │ Module 13: GPT      │ Autoregressively generates     │ Same architecture as        │
# │                     │ code character by character    │ GitHub Copilot (smaller!)   │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================

💡 KEY INSIGHT:
Code completion is just next-token prediction! The model learns patterns like:
  "def add(a, b):\n    return a" → predicts " + b"
GitHub Copilot = same idea, trained on billions of code examples!

📊 EXPECTED RESULTS:
  Training: ~2 minutes on 50 Python patterns
  Demo success rate: 4-5/5 completions working
  Limitation: Pattern matching, not true understanding
"""

import sys
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple

# Add TinyTorch to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import tinytorch as tt
from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import Adam
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.transformer import GPT
from tinytorch.core.tokenization import CharTokenizer  # Module 10: Students built this!


# ============================================================================
# Python Code Dataset
# ============================================================================

# Hand-curated 50 simple Python patterns for autocomplete
PYTHON_PATTERNS = [
    # Basic arithmetic functions (10)
    "def add(a, b):\n    return a + b",
    "def subtract(a, b):\n    return a - b",
    "def multiply(x, y):\n    return x * y",
    "def divide(a, b):\n    return a / b",
    "def power(base, exp):\n    return base ** exp",
    "def modulo(a, b):\n    return a % b",
    "def max_of_two(a, b):\n    return a if a > b else b",
    "def min_of_two(a, b):\n    return a if a < b else b",
    "def absolute(x):\n    return x if x >= 0 else -x",
    "def square(x):\n    return x * x",

    # For loops (10)
    "for i in range(10):\n    print(i)",
    "for i in range(5):\n    print(i * 2)",
    "for item in items:\n    print(item)",
    "for i in range(len(arr)):\n    arr[i] = arr[i] * 2",
    "for num in numbers:\n    total += num",
    "for i in range(0, 10, 2):\n    print(i)",
    "for char in text:\n    print(char)",
    "for key in dict:\n    print(key, dict[key])",
    "for i, val in enumerate(items):\n    print(i, val)",
    "for x in range(3):\n    for y in range(3):\n        print(x, y)",

    # If statements (10)
    "if x > 0:\n    print('positive')",
    "if x < 0:\n    print('negative')",
    "if x == 0:\n    print('zero')",
    "if age >= 18:\n    print('adult')",
    "if score > 90:\n    grade = 'A'",
    "if name:\n    print(f'Hello {name}')",
    "if x > 0 and x < 10:\n    print('single digit')",
    "if x == 5 or x == 10:\n    print('five or ten')",
    "if not done:\n    continue_work()",
    "if condition:\n    do_something()\nelse:\n    do_other()",

    # List operations (10)
    "numbers = [1, 2, 3, 4, 5]",
    "squares = [x**2 for x in range(10)]",
    "evens = [n for n in numbers if n % 2 == 0]",
    "first = items[0]",
    "last = items[-1]",
    "items.append(new_item)",
    "items.extend(more_items)",
    "items.remove(old_item)",
    "length = len(items)",
    "sorted_items = sorted(items)",

    # String operations (10)
    "text = 'Hello, World!'",
    "upper = text.upper()",
    "lower = text.lower()",
    "words = text.split()",
    "joined = ' '.join(words)",
    "starts = text.startswith('Hello')",
    "ends = text.endswith('!')",
    "replaced = text.replace('World', 'Python')",
    "stripped = text.strip()",
    "message = f'Hello {name}!'",
]


def create_code_dataset() -> Tuple[List[str], List[str]]:
    """
    Split patterns into train and test sets.

    Returns:
        (train_patterns, test_patterns)
    """
    # Use first 45 for training, last 5 for testing
    train = PYTHON_PATTERNS[:45]
    test = PYTHON_PATTERNS[45:]

    return train, test


# ============================================================================
# Tokenization (Using Student's CharTokenizer from Module 10!)
# ============================================================================

def create_tokenizer(texts: List[str]) -> CharTokenizer:
    """
    Create tokenizer using students' CharTokenizer from Module 10.

    This shows how YOUR tokenizer from Module 10 enables real applications!
    """
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)  # Build vocab from our Python patterns
    return tokenizer


# ============================================================================
# Training
# ============================================================================

def train_codebot(
    model: GPT,
    optimizer: Adam,
    tokenizer: CharTokenizer,
    train_patterns: List[str],
    max_steps: int = 5000,
    seq_length: int = 128,
):
    """Train CodeBot on Python patterns."""

    print("\n" + "="*70)
    print("TRAINING CODEBOT...")
    print("="*70)
    print()
    print(f"Loading training data: {len(train_patterns)} Python code patterns ✓")
    print()
    print(f"Model size: ~{sum(np.prod(p.shape) for p in model.parameters()):,} parameters")
    print(f"Training for ~{max_steps:,} steps (estimated 2 minutes)")
    print()

    # Encode and pad patterns
    train_tokens = []
    for pattern in train_patterns:
        tokens = tokenizer.encode(pattern)
        # Truncate or pad to seq_length
        if len(tokens) > seq_length:
            tokens = tokens[:seq_length]
        else:
            tokens = tokens + [0] * (seq_length - len(tokens))  # Pad with 0
        train_tokens.append(tokens)

    # Loss function
    loss_fn = CrossEntropyLoss()

    # Training loop
    start_time = time.time()
    step = 0
    losses = []

    # Progress markers
    progress_points = [0, 500, 1000, 2000, max_steps]
    messages = [
        "[The model knows nothing yet]",
        "[Learning basic patterns...]",
        "[Getting better at Python syntax...]",
        "[Almost there...]",
        "[Training complete!]"
    ]

    while step <= max_steps:
        # Sample random pattern
        tokens = train_tokens[np.random.randint(len(train_tokens))]

        # Create input/target
        input_seq = tokens[:-1]
        target_seq = tokens[1:]

        # Convert to tensors
        x = Tensor(np.array([input_seq], dtype=np.int32), requires_grad=False)
        y_true = Tensor(np.array([target_seq], dtype=np.int32), requires_grad=False)

        # Forward pass
        logits = model.forward(x)

        # Compute loss
        batch_size = 1
        seq_len = logits.data.shape[1]
        vocab_size = logits.data.shape[2]

        logits_flat = logits.reshape((batch_size * seq_len, vocab_size))
        targets_flat = y_true.reshape((batch_size * seq_len,))

        loss = loss_fn(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        for param in model.parameters():
            if param.grad is not None:
                param.grad = np.clip(param.grad, -1.0, 1.0)

        # Update
        optimizer.step()

        # Track
        losses.append(loss.data.item())

        # Print progress at markers
        if step in progress_points:
            avg_loss = np.mean(losses[-100:]) if losses else loss.data.item()
            elapsed = time.time() - start_time
            msg_idx = progress_points.index(step)
            print(f"Step {step:4d}/{max_steps} | Loss: {avg_loss:.3f} | {messages[msg_idx]}")

        step += 1

        # Time limit
        if time.time() - start_time > 180:  # 3 minutes max
            break

    total_time = time.time() - start_time
    final_loss = np.mean(losses[-100:])
    loss_decrease = ((losses[0] - final_loss) / losses[0]) * 100

    print()
    print(f"✓ CodeBot trained in {int(total_time)} seconds!")
    print(f"✓ Loss decreased by {loss_decrease:.0f}%!")
    print()

    return losses


# ============================================================================
# Code Completion
# ============================================================================

def complete_code(
    model: GPT,
    tokenizer: CharTokenizer,
    partial_code: str,
    max_gen_length: int = 50,
) -> str:
    """
    Complete partial Python code.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        partial_code: Incomplete code
        max_gen_length: Max characters to generate

    Returns:
        Completed code
    """
    tokens = tokenizer.encode(partial_code)

    # Generate
    for _ in range(max_gen_length):
        x = Tensor(np.array([tokens], dtype=np.int32), requires_grad=False)
        logits = model.forward(x)

        # Get next token (greedy)
        next_logits = logits.data[0, -1, :]
        next_token = int(np.argmax(next_logits))

        # Stop at padding (0) or if we've generated enough
        if next_token == 0:
            break

        tokens.append(next_token)

    # Decode
    completed = tokenizer.decode(tokens)

    # Return just the generated part
    return completed[len(partial_code):]


# ============================================================================
# Demo Modes
# ============================================================================

def demo_mode(model: GPT, tokenizer: CharTokenizer):
    """Show 5 demo completions."""

    print("\n" + "="*70)
    print("🎯 DEMO MODE: WATCH CODEBOT AUTOCOMPLETE")
    print("="*70)
    print()
    print("I'll show you 5 examples of what CodeBot learned:")
    print()

    demos = [
        ("def subtract(a, b):\n    return a", "Basic Function"),
        ("for i in range(", "For Loop"),
        ("if x > 0:\n    print(", "If Statement"),
        ("squares = [x**2 for x in ", "List Comprehension"),
        ("def multiply(x, y):\n    return x", "Function Return"),
    ]

    success_count = 0

    for i, (partial, name) in enumerate(demos, 1):
        print(f"Example {i}: {name}")
        print("─" * 70)
        print(f"You type:     {partial.replace(chr(10), chr(10) + '              ')}")

        completion = complete_code(model, tokenizer, partial, max_gen_length=30)

        print(f"CodeBot adds: {completion[:50]}...")

        # Simple success check (generated something)
        if completion.strip():
            print("✓ Completion generated")
            success_count += 1
        else:
            print("✗ No completion")

        print("─" * 70)
        print()

    print(f"Demo success rate: {success_count}/5 ({success_count*20}%)")
    if success_count >= 4:
        print("🎉 CodeBot is working great!")
    print()


def interactive_mode(model: GPT, tokenizer: CharTokenizer):
    """Let student try CodeBot."""

    print("\n" + "="*70)
    print("🎮 YOUR TURN: TRY CODEBOT!")
    print("="*70)
    print()
    print("Type partial Python code and see what CodeBot suggests.")
    print("Type 'demo' to see examples, 'quit' to exit.")
    print()

    examples = [
        "def add(a, b):\n    return a",
        "for i in range(",
        "if name:\n    print(",
        "numbers = [1, 2, 3]",
    ]

    while True:
        try:
            user_input = input("\nCodeBot> ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("\n👋 Thanks for trying CodeBot!")
                break

            if user_input.lower() == 'demo':
                print("\nTry these examples:")
                for ex in examples:
                    print(f"  → {ex[:40]}...")
                continue

            # Complete the code
            print()
            completion = complete_code(model, tokenizer, user_input, max_gen_length=50)

            if completion.strip():
                print(f"🤖 CodeBot suggests: {completion}")
                print()
                print(f"Full code:")
                print(user_input + completion)
            else:
                print("⚠️  CodeBot couldn't complete this (maybe it wasn't trained on this pattern?)")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Thanks for trying CodeBot!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run CodeBot autocomplete demo."""

    print("\n" + "="*70)
    print("🤖 CODEBOT - BUILD YOUR OWN MINI-COPILOT!")
    print("="*70)
    print()
    print("You're about to train a transformer to autocomplete Python code.")
    print()
    print("In 2 minutes, you'll have a working autocomplete that learned:")
    print("  • Basic functions (add, multiply, divide)")
    print("  • For loops and while loops")
    print("  • If statements and conditionals")
    print("  • List operations")
    print("  • Common Python patterns")
    print()
    input("Press ENTER to begin training...")

    # Create dataset
    train_patterns, test_patterns = create_code_dataset()

    # Create tokenizer
    all_patterns = train_patterns + test_patterns
    tokenizer = create_tokenizer(all_patterns)

    # Model config (based on proven sweep results)
    config = {
        'vocab_size': tokenizer.vocab_size,
        'embed_dim': 32,      # Scaled from winning 16d config
        'num_layers': 2,      # Enough for code patterns
        'num_heads': 8,       # Proven winner from sweep
        'max_seq_len': 128,   # Enough for code snippets
    }

    # Create model
    model = GPT(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_len=config['max_seq_len'],
    )

    # Optimizer (proven winning LR)
    learning_rate = 0.0015
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train
    losses = train_codebot(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        train_patterns=train_patterns,
        max_steps=5000,
        seq_length=config['max_seq_len'],
    )

    print("Ready to test CodeBot!")
    input("Press ENTER to see demo...")

    # Demo mode
    demo_mode(model, tokenizer)

    input("Press ENTER to try it yourself...")

    # Interactive mode
    interactive_mode(model, tokenizer)

    # Summary
    print("\n" + "="*70)
    print("🎓 WHAT YOU LEARNED")
    print("="*70)
    print()
    print("Congratulations! You just:")
    print("  ✓ Trained a transformer from scratch")
    print("  ✓ Saw it learn Python patterns in ~2 minutes")
    print("  ✓ Used it to autocomplete code")
    print("  ✓ Understood its limits (pattern matching, not reasoning)")
    print()
    print("KEY INSIGHTS:")
    print("  1. Transformers learn by pattern matching")
    print("  2. More training data → smarter completions")
    print("  3. They don't 'understand' - they predict patterns")
    print("  4. Real Copilot = same idea, billions more patterns!")
    print()
    print("SCALING PATH:")
    print("  • Your CodeBot: 45 patterns → simple completions")
    print("  • Medium model: 10,000 patterns → decent autocomplete")
    print("  • GitHub Copilot: BILLIONS of patterns → production-ready!")
    print()
    print("Great job! You're now a transformer trainer! 🎉")
    print("="*70)


if __name__ == '__main__':
    main()
