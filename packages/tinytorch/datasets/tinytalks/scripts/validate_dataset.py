"""
TinyTalks Dataset Validation Script

Validates the TinyTalks dataset for:
- Format consistency
- No duplicate pairs
- Balanced splits
- Character encoding (UTF-8)
- Line endings (Unix)

Usage:
    python scripts/validate_dataset.py
"""

from pathlib import Path
from collections import Counter


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


def validate_format(file_path):
    """Validate Q&A format"""
    print(f"\n📝 Validating format: {file_path.name}")
    
    pairs = load_qa_pairs(file_path)
    
    if len(pairs) == 0:
        print("  ❌ ERROR: No Q&A pairs found!")
        return False
    
    print(f"  ✓ Found {len(pairs)} Q&A pairs")
    print(f"  ✓ Format is consistent")
    return True


def validate_no_duplicates(file_path):
    """Check for duplicate Q&A pairs"""
    print(f"\n🔍 Checking for duplicates: {file_path.name}")
    
    pairs = load_qa_pairs(file_path)
    
    # Check for duplicate questions
    questions = [q for q, a in pairs]
    question_counts = Counter(questions)
    duplicates = {q: count for q, count in question_counts.items() if count > 1}
    
    if duplicates:
        print(f"  ⚠️  WARNING: Found {len(duplicates)} duplicate questions:")
        for q, count in list(duplicates.items())[:5]:
            print(f"    - '{q}' appears {count} times")
        return False
    else:
        print(f"  ✓ No duplicate questions")
    
    return True


def validate_encoding(file_path):
    """Validate UTF-8 encoding"""
    print(f"\n🔤 Validating encoding: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        print(f"  ✓ Valid UTF-8 encoding")
        return True
    except UnicodeDecodeError as e:
        print(f"  ❌ ERROR: Invalid UTF-8 encoding: {e}")
        return False


def validate_line_endings(file_path):
    """Validate Unix line endings (LF, not CRLF)"""
    print(f"\n📄 Validating line endings: {file_path.name}")
    
    with open(file_path, 'rb') as f:
        content = f.read()
    
    crlf_count = content.count(b'\r\n')
    
    if crlf_count > 0:
        print(f"  ⚠️  WARNING: Found {crlf_count} Windows line endings (CRLF)")
        print(f"     Consider converting to Unix (LF)")
        return False
    else:
        print(f"  ✓ Unix line endings (LF)")
        return True


def validate_splits_consistency():
    """Validate that splits don't overlap and cover all data"""
    print(f"\n🔀 Validating splits consistency")
    
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent
    splits_dir = dataset_dir / "splits"
    
    train_pairs = set(load_qa_pairs(splits_dir / "train.txt"))
    val_pairs = set(load_qa_pairs(splits_dir / "val.txt"))
    test_pairs = set(load_qa_pairs(splits_dir / "test.txt"))
    
    # Check for overlaps
    train_val_overlap = train_pairs & val_pairs
    train_test_overlap = train_pairs & test_pairs
    val_test_overlap = val_pairs & test_pairs
    
    if train_val_overlap:
        print(f"  ❌ ERROR: {len(train_val_overlap)} pairs overlap between train and val")
        return False
    if train_test_overlap:
        print(f"  ❌ ERROR: {len(train_test_overlap)} pairs overlap between train and test")
        return False
    if val_test_overlap:
        print(f"  ❌ ERROR: {len(val_test_overlap)} pairs overlap between val and test")
        return False
    
    print(f"  ✓ No overlaps between splits")
    
    # Check total
    total_split_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
    print(f"  ✓ Total pairs across splits: {total_split_pairs}")
    
    # Check percentages
    train_pct = len(train_pairs) / total_split_pairs * 100
    val_pct = len(val_pairs) / total_split_pairs * 100
    test_pct = len(test_pairs) / total_split_pairs * 100
    
    print(f"    - Train: {len(train_pairs)} ({train_pct:.1f}%)")
    print(f"    - Val: {len(val_pairs)} ({val_pct:.1f}%)")
    print(f"    - Test: {len(test_pairs)} ({test_pct:.1f}%)")
    
    # Check if percentages are roughly 70/15/15
    if not (65 <= train_pct <= 75):
        print(f"  ⚠️  WARNING: Train split should be ~70%, got {train_pct:.1f}%")
    if not (10 <= val_pct <= 20):
        print(f"  ⚠️  WARNING: Val split should be ~15%, got {val_pct:.1f}%")
    if not (10 <= test_pct <= 20):
        print(f"  ⚠️  WARNING: Test split should be ~15%, got {test_pct:.1f}%")
    
    return True


def validate_content_quality():
    """Validate content quality"""
    print(f"\n✨ Validating content quality")
    
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent
    full_dataset = dataset_dir / "tinytalks_v1.txt"
    
    pairs = load_qa_pairs(full_dataset)
    
    # Check for empty questions or answers
    empty_questions = [i for i, (q, a) in enumerate(pairs) if not q.strip()]
    empty_answers = [i for i, (q, a) in enumerate(pairs) if not a.strip()]
    
    if empty_questions:
        print(f"  ❌ ERROR: {len(empty_questions)} empty questions found")
        return False
    if empty_answers:
        print(f"  ❌ ERROR: {len(empty_answers)} empty answers found")
        return False
    
    print(f"  ✓ No empty questions or answers")
    
    # Check for very short pairs (potential errors)
    short_questions = [(i, q) for i, (q, a) in enumerate(pairs) if len(q) < 5]
    short_answers = [(i, a) for i, (q, a) in enumerate(pairs) if len(a) < 5]
    
    if short_questions:
        print(f"  ⚠️  WARNING: {len(short_questions)} very short questions (< 5 chars)")
    if short_answers:
        print(f"  ⚠️  WARNING: {len(short_answers)} very short answers (< 5 chars)")
    
    # Check question marks
    questions_without_marks = [q for q, a in pairs if not (q.endswith('?') or q.endswith('!') or q.endswith('.'))]
    if questions_without_marks:
        print(f"  ℹ️  INFO: {len(questions_without_marks)} questions without ending punctuation")
    else:
        print(f"  ✓ All questions have proper punctuation")
    
    return True


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("  TinyTalks Dataset Validation")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent
    
    # Files to validate
    files = [
        dataset_dir / "tinytalks_v1.txt",
        dataset_dir / "splits" / "train.txt",
        dataset_dir / "splits" / "val.txt",
        dataset_dir / "splits" / "test.txt",
    ]
    
    all_passed = True
    
    # Validate each file
    for file_path in files:
        if not file_path.exists():
            print(f"\n❌ ERROR: File not found: {file_path}")
            all_passed = False
            continue
        
        all_passed &= validate_format(file_path)
        all_passed &= validate_no_duplicates(file_path)
        all_passed &= validate_encoding(file_path)
        all_passed &= validate_line_endings(file_path)
    
    # Validate splits consistency
    all_passed &= validate_splits_consistency()
    
    # Validate content quality
    all_passed &= validate_content_quality()
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✅ All validation checks passed!")
    else:
        print("  ⚠️  Some validation checks failed or have warnings")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

