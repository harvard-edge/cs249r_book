"""
TinyTalks Dataset Statistics

Generates comprehensive statistics about the TinyTalks dataset including:
- Vocabulary statistics
- Length distributions
- Character frequencies
- Split sizes

Usage:
    python scripts/stats.py
"""

from pathlib import Path
from collections import Counter
import json


def load_qa_pairs(file_path):
    """Load Q&A pairs from a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pairs = []
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) == 2:
            q_line = lines[0]
            a_line = lines[1]
            
            if q_line.startswith('Q: ') and a_line.startswith('A: '):
                question = q_line[3:]  # Remove "Q: "
                answer = a_line[3:]    # Remove "A: "
                pairs.append((question, answer))
    
    return pairs


def compute_vocabulary_stats(pairs):
    """Compute vocabulary statistics"""
    all_text = ' '.join([f"{q} {a}" for q, a in pairs])
    
    # Character-level vocabulary
    char_vocab = set(all_text)
    char_freq = Counter(all_text)
    
    # Word-level vocabulary
    words = all_text.lower().split()
    word_vocab = set(words)
    word_freq = Counter(words)
    
    return {
        'char_vocab_size': len(char_vocab),
        'char_freq': char_freq,
        'word_vocab_size': len(word_vocab),
        'word_freq': word_freq,
        'total_chars': len(all_text),
        'total_words': len(words),
    }


def compute_length_stats(pairs):
    """Compute length statistics"""
    q_lengths = [len(q) for q, a in pairs]
    a_lengths = [len(a) for q, a in pairs]
    
    q_word_lengths = [len(q.split()) for q, a in pairs]
    a_word_lengths = [len(a.split()) for q, a in pairs]
    
    return {
        'question_char_lengths': {
            'min': min(q_lengths),
            'max': max(q_lengths),
            'avg': sum(q_lengths) / len(q_lengths),
        },
        'answer_char_lengths': {
            'min': min(a_lengths),
            'max': max(a_lengths),
            'avg': sum(a_lengths) / len(a_lengths),
        },
        'question_word_lengths': {
            'min': min(q_word_lengths),
            'max': max(q_word_lengths),
            'avg': sum(q_word_lengths) / len(q_word_lengths),
        },
        'answer_word_lengths': {
            'min': min(a_word_lengths),
            'max': max(a_word_lengths),
            'avg': sum(a_word_lengths) / len(a_word_lengths),
        },
    }


def print_stats():
    """Print comprehensive dataset statistics"""
    print("=" * 70)
    print("  TinyTalks Dataset Statistics")
    print("=" * 70)
    
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent
    
    # Load all datasets
    full_pairs = load_qa_pairs(dataset_dir / "tinytalks_v1.txt")
    train_pairs = load_qa_pairs(dataset_dir / "splits" / "train.txt")
    val_pairs = load_qa_pairs(dataset_dir / "splits" / "val.txt")
    test_pairs = load_qa_pairs(dataset_dir / "splits" / "test.txt")
    
    print("\n📊 DATASET SIZES")
    print("-" * 70)
    print(f"  Total Q&A pairs:     {len(full_pairs)}")
    print(f"  Training pairs:      {len(train_pairs)} ({len(train_pairs)/len(full_pairs)*100:.1f}%)")
    print(f"  Validation pairs:    {len(val_pairs)} ({len(val_pairs)/len(full_pairs)*100:.1f}%)")
    print(f"  Test pairs:          {len(test_pairs)} ({len(test_pairs)/len(full_pairs)*100:.1f}%)")
    
    # Vocabulary statistics
    vocab_stats = compute_vocabulary_stats(full_pairs)
    
    print("\n📖 VOCABULARY STATISTICS")
    print("-" * 70)
    print(f"  Character vocabulary size:  {vocab_stats['char_vocab_size']}")
    print(f"  Word vocabulary size:       {vocab_stats['word_vocab_size']}")
    print(f"  Total characters:           {vocab_stats['total_chars']}")
    print(f"  Total words:                {vocab_stats['total_words']}")
    
    # Length statistics
    length_stats = compute_length_stats(full_pairs)
    
    print("\n📏 LENGTH STATISTICS")
    print("-" * 70)
    print("  Question lengths (characters):")
    print(f"    Min: {length_stats['question_char_lengths']['min']}")
    print(f"    Max: {length_stats['question_char_lengths']['max']}")
    print(f"    Avg: {length_stats['question_char_lengths']['avg']:.1f}")
    
    print("\n  Answer lengths (characters):")
    print(f"    Min: {length_stats['answer_char_lengths']['min']}")
    print(f"    Max: {length_stats['answer_char_lengths']['max']}")
    print(f"    Avg: {length_stats['answer_char_lengths']['avg']:.1f}")
    
    print("\n  Question lengths (words):")
    print(f"    Min: {length_stats['question_word_lengths']['min']}")
    print(f"    Max: {length_stats['question_word_lengths']['max']}")
    print(f"    Avg: {length_stats['question_word_lengths']['avg']:.1f}")
    
    print("\n  Answer lengths (words):")
    print(f"    Min: {length_stats['answer_word_lengths']['min']}")
    print(f"    Max: {length_stats['answer_word_lengths']['max']}")
    print(f"    Avg: {length_stats['answer_word_lengths']['avg']:.1f}")
    
    # Top words
    print("\n🔤 TOP 20 MOST COMMON WORDS")
    print("-" * 70)
    for word, count in vocab_stats['word_freq'].most_common(20):
        print(f"  {word:15s} : {count:3d} times")
    
    # Top characters
    print("\n🔡 TOP 20 MOST COMMON CHARACTERS")
    print("-" * 70)
    for char, count in vocab_stats['char_freq'].most_common(20):
        char_display = repr(char) if char in [' ', '\n', '\t'] else char
        print(f"  {char_display:15s} : {count:4d} times")
    
    # File sizes
    print("\n💾 FILE SIZES")
    print("-" * 70)
    files = [
        ("Full dataset", dataset_dir / "tinytalks_v1.txt"),
        ("Training split", dataset_dir / "splits" / "train.txt"),
        ("Validation split", dataset_dir / "splits" / "val.txt"),
        ("Test split", dataset_dir / "splits" / "test.txt"),
    ]
    
    for name, path in files:
        size_bytes = path.stat().st_size
        size_kb = size_bytes / 1024
        print(f"  {name:20s} : {size_kb:6.1f} KB ({size_bytes:,} bytes)")
    
    # Sample Q&A pairs
    print("\n📝 SAMPLE Q&A PAIRS (first 5)")
    print("-" * 70)
    for i, (q, a) in enumerate(full_pairs[:5], 1):
        print(f"\n  {i}. Q: {q}")
        print(f"     A: {a}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_stats()

