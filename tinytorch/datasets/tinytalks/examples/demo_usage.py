"""
TinyTalks Dataset Usage Examples

Demonstrates how to load and use the TinyTalks dataset for training
transformer models.

Usage:
    python examples/demo_usage.py
"""

from pathlib import Path


def example1_load_full_dataset():
    """Example 1: Load the full dataset"""
    print("=" * 60)
    print("Example 1: Loading Full Dataset")
    print("=" * 60)
    
    dataset_path = Path(__file__).parent.parent / "tinytalks_v1.txt"
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"✓ Loaded dataset from: {dataset_path.name}")
    print(f"  Total size: {len(text)} characters")
    print(f"  Total lines: {len(text.splitlines())} lines")
    
    # Show first 300 characters
    print(f"\n  First 300 characters:")
    print(f"  {'-' * 58}")
    print(f"  {text[:300]}...")
    
    return text


def example2_load_train_split():
    """Example 2: Load training split only"""
    print("\n" + "=" * 60)
    print("Example 2: Loading Training Split")
    print("=" * 60)
    
    train_path = Path(__file__).parent.parent / "splits" / "train.txt"
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    print(f"✓ Loaded training split from: {train_path.name}")
    print(f"  Size: {len(train_text)} characters")
    
    return train_text


def example3_parse_qa_pairs():
    """Example 3: Parse Q&A pairs from text"""
    print("\n" + "=" * 60)
    print("Example 3: Parsing Q&A Pairs")
    print("=" * 60)
    
    dataset_path = Path(__file__).parent.parent / "tinytalks_v1.txt"
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Parse Q&A pairs
    qa_pairs = []
    blocks = text.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) == 2:
            q_line = lines[0]
            a_line = lines[1]
            
            if q_line.startswith('Q: ') and a_line.startswith('A: '):
                question = q_line[3:]  # Remove "Q: "
                answer = a_line[3:]    # Remove "A: "
                qa_pairs.append((question, answer))
    
    print(f"✓ Parsed {len(qa_pairs)} Q&A pairs")
    print(f"\n  First 5 pairs:")
    print(f"  {'-' * 58}")
    for i, (q, a) in enumerate(qa_pairs[:5], 1):
        print(f"\n  {i}. Q: {q}")
        print(f"     A: {a}")
    
    return qa_pairs


def example4_character_tokenization():
    """Example 4: Character-level tokenization"""
    print("\n" + "=" * 60)
    print("Example 4: Character-Level Tokenization")
    print("=" * 60)
    
    dataset_path = Path(__file__).parent.parent / "tinytalks_v1.txt"
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Build character vocabulary
    vocab = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    print(f"✓ Built character vocabulary")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Characters: {repr(''.join(vocab[:20]))}")
    
    # Encode a sample
    sample = "Q: Hello! A: Hi there!"
    encoded = [char_to_idx[ch] for ch in sample]
    
    print(f"\n  Sample text: {sample}")
    print(f"  Encoded: {encoded[:20]}...")
    
    # Decode back
    decoded = ''.join([idx_to_char[idx] for idx in encoded])
    print(f"  Decoded: {decoded}")
    
    assert sample == decoded, "Encoding/decoding mismatch!"
    print(f"  ✓ Encoding/decoding verified")
    
    return vocab, char_to_idx, idx_to_char


def example5_prepare_for_transformer():
    """Example 5: Prepare data for transformer training"""
    print("\n" + "=" * 60)
    print("Example 5: Preparing Data for Transformer")
    print("=" * 60)
    
    # Load training data
    train_path = Path(__file__).parent.parent / "splits" / "train.txt"
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    # Build vocabulary
    vocab = sorted(set(train_text))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    
    print(f"✓ Prepared data for training")
    print(f"  Training text size: {len(train_text)} characters")
    print(f"  Vocabulary size: {len(vocab)}")
    
    # Show example sequence creation
    seq_length = 32
    sample_seq = train_text[:seq_length]
    sample_target = train_text[1:seq_length+1]
    
    print(f"\n  Example input sequence (first {seq_length} chars):")
    print(f"    {repr(sample_seq)}")
    print(f"\n  Example target sequence (shifted by 1):")
    print(f"    {repr(sample_target)}")
    
    return train_text, vocab, char_to_idx


def example6_using_with_tinytorch():
    """Example 6: Using with TinyTorch (pseudocode)"""
    print("\n" + "=" * 60)
    print("Example 6: Using with TinyTorch (Pseudocode)")
    print("=" * 60)
    
    print("""
  # Import TinyTorch components
  from tinytorch.models.transformer import GPT
  from tinytorch.text.tokenization import CharTokenizer
  from tinytorch.core.optimizers import Adam
  from tinytorch.core.losses import CrossEntropyLoss
  
  # Load dataset
  with open('datasets/tinytalks/splits/train.txt', 'r') as f:
      train_text = f.read()
  
  # Initialize tokenizer
  tokenizer = CharTokenizer()
  tokenizer.fit(train_text)
  
  # Initialize model
  model = GPT(
      vocab_size=len(tokenizer),
      embed_dim=128,
      num_layers=4,
      num_heads=4,
      max_seq_len=64
  )
  
  # Initialize optimizer and loss
  optimizer = Adam(model.parameters(), lr=0.001)
  criterion = CrossEntropyLoss()
  
  # Training loop (simplified)
  for epoch in range(10):
      # ... create batches from train_text ...
      # ... forward pass ...
      # ... compute loss ...
      # ... backward pass ...
      # ... optimizer step ...
      print(f"Epoch {epoch+1}, Loss: {loss}")
  
  # Generate text
  prompt = "Q: What is your name?"
  response = model.generate(prompt, tokenizer)
  print(response)
  """)
    
    print(f"\n  See milestones/05_2017_transformer/tinybot_demo.py")
    print(f"  for a complete working example!")


def main():
    """Run all examples"""
    print("\n")
    print("*" * 60)
    print("  TinyTalks Dataset - Usage Examples")
    print("*" * 60)
    
    # Run examples
    text = example1_load_full_dataset()
    train_text = example2_load_train_split()
    qa_pairs = example3_parse_qa_pairs()
    vocab, char_to_idx, idx_to_char = example4_character_tokenization()
    train_text, vocab, char_to_idx = example5_prepare_for_transformer()
    example6_using_with_tinytorch()
    
    print("\n" + "=" * 60)
    print("  ✅ All examples completed successfully!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

