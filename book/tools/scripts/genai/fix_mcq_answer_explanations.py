#!/usr/bin/env python3
"""
Fix MCQ answer explanations that incorrectly reference the correct option.

This script detects and fixes cases where the answer explanation lists the correct
option as one of the incorrect options. For example:
- "The correct answer is A... Options A, C, and D describe..." (WRONG)
- Should be: "The correct answer is A... Options B, C, and D describe..." (CORRECT)

Addresses issue #1034.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def extract_correct_answer_letter(answer_text: str) -> Optional[str]:
    """Extract the correct answer letter from the answer text."""
    match = re.match(r'The correct answer is ([A-D])', answer_text)
    if match:
        return match.group(1)
    return None


def find_option_references(answer_text: str) -> List[Tuple[str, List[str]]]:
    """
    Find all references to options in the answer text.

    Returns:
        List of tuples (full_match, list_of_letters) for each pattern found
    """
    patterns = [
        # "Options A, B, and C" or "Options A, B, C"
        r'([Oo]ptions ([A-D])(?:,\s*([A-D]))*(?:,?\s*and\s*([A-D])))',
        r'([Oo]ptions ([A-D])(?:,\s*([A-D]))+)',
        # Singular "Option X is incorrect/wrong" patterns
        r'([Oo]ption ([A-D]) is (?:incorrect|wrong))',
    ]

    references = []
    for pattern in patterns:
        for match in re.finditer(pattern, answer_text):
            full_match = match.group(1)
            # Extract all letter groups from the match
            letters = [g for g in match.groups()[1:] if g and g.isalpha()]
            if letters:
                references.append((full_match, letters))

    return references


def get_all_option_letters(num_choices: int) -> List[str]:
    """Get all available option letters based on number of choices."""
    return [chr(65 + i) for i in range(num_choices)]  # A, B, C, D, etc.


def fix_option_reference(
    full_match: str,
    referenced_letters: List[str],
    correct_letter: str,
    all_letters: List[str]
) -> Optional[str]:
    """
    Fix an option reference if it incorrectly includes the correct answer.

    Args:
        full_match: The full matched text (e.g., "Options A, C, and D" or "Option C is incorrect")
        referenced_letters: List of letters referenced (e.g., ['A', 'C', 'D'] or ['C'])
        correct_letter: The correct answer letter (e.g., 'A')
        all_letters: All available option letters (e.g., ['A', 'B', 'C', 'D'])

    Returns:
        Fixed text if correction needed, None otherwise
    """
    # Check if the correct letter is in the referenced letters
    if correct_letter not in referenced_letters:
        return None  # No fix needed

    # Find the missing incorrect letters
    incorrect_letters = [l for l in all_letters if l != correct_letter]
    referenced_incorrect = [l for l in referenced_letters if l != correct_letter]
    missing_letters = [l for l in incorrect_letters if l not in referenced_incorrect]

    if not missing_letters:
        return None  # Can't determine what to replace with

    # Build the corrected letter list
    # Replace the correct letter with the first missing letter
    corrected_letters = referenced_incorrect + missing_letters[:1]
    corrected_letters.sort()  # Keep alphabetical order

    # Check if this is a singular "Option X is incorrect" pattern
    if 'is incorrect' in full_match or 'is wrong' in full_match:
        # Replace with just the first missing letter
        return full_match.replace(correct_letter, missing_letters[0])

    # Reconstruct the text for plural patterns
    if len(corrected_letters) == 1:
        return f"Option {corrected_letters[0]}"
    elif len(corrected_letters) == 2:
        return f"Options {corrected_letters[0]} and {corrected_letters[1]}"
    else:
        # Format as "Options A, B, and C"
        all_but_last = ", ".join(corrected_letters[:-1])
        return f"Options {all_but_last}, and {corrected_letters[-1]}"


def fix_mcq_answer(question: Dict) -> Tuple[bool, str]:
    """
    Fix MCQ answer explanation if it has incorrect option references.

    Returns:
        Tuple of (was_fixed, details_message)
    """
    if question.get('question_type') != 'MCQ':
        return False, ""

    answer_text = question.get('answer', '')
    if not answer_text:
        return False, ""

    # Get the correct answer letter
    correct_letter = extract_correct_answer_letter(answer_text)
    if not correct_letter:
        return False, "Could not extract correct answer letter"

    # Get all available letters
    num_choices = len(question.get('choices', []))
    if num_choices == 0:
        return False, "No choices found"

    all_letters = get_all_option_letters(num_choices)

    # Find option references in the text
    references = find_option_references(answer_text)
    if not references:
        return False, ""

    # Fix each reference if needed
    fixed_text = answer_text
    fixes_made = []

    for full_match, letters in references:
        fixed_match = fix_option_reference(full_match, letters, correct_letter, all_letters)
        if fixed_match:
            fixed_text = fixed_text.replace(full_match, fixed_match, 1)
            fixes_made.append(f"'{full_match}' → '{fixed_match}'")

    if fixes_made:
        question['answer'] = fixed_text
        return True, "; ".join(fixes_made)

    return False, ""


def fix_quiz_file(file_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Fix all MCQ answers in a quiz file.

    Returns:
        Dictionary with statistics: {'fixed': N, 'total_mcqs': M}
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️  Skipping {file_path.name}: JSON parse error at line {e.lineno}")
        return {'fixed': 0, 'total_mcqs': 0}

    stats = {'fixed': 0, 'total_mcqs': 0}
    fixes_details = []

    # Process each section
    for section in data.get('sections', []):
        quiz_data = section.get('quiz_data', {})
        questions = quiz_data.get('questions', [])

        for i, question in enumerate(questions, 1):
            if question.get('question_type') == 'MCQ':
                stats['total_mcqs'] += 1
                was_fixed, details = fix_mcq_answer(question)

                if was_fixed:
                    stats['fixed'] += 1
                    section_title = section.get('section_title', 'Unknown')
                    q_text = question.get('question', '')[:60]
                    fixes_details.append(
                        f"  [{section_title}] Q{i}: {q_text}...\n    {details}"
                    )

    # Write back if not dry run and fixes were made
    if not dry_run and stats['fixed'] > 0:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    if fixes_details:
        print(f"\n{file_path.name}:")
        for detail in fixes_details:
            print(detail)

    return stats


def main():
    """Main function to fix all quiz files."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fix MCQ answer explanations that incorrectly reference the correct option'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without making changes'
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Fix a specific quiz file (default: all quiz files)'
    )

    args = parser.parse_args()

    # Find all quiz JSON files
    if args.file:
        quiz_files = [args.file]
    else:
        quarto_path = Path(__file__).resolve().parent.parent.parent.parent / 'quarto'
        quiz_files = list(quarto_path.glob('contents/**/*_quizzes.json'))
        quiz_files.sort()

    if not quiz_files:
        print("No quiz files found.")
        return

    print(f"{'DRY RUN: ' if args.dry_run else ''}Scanning {len(quiz_files)} quiz files...")
    print("=" * 80)

    total_stats = {'fixed': 0, 'total_mcqs': 0, 'files_with_fixes': 0}

    for quiz_file in quiz_files:
        if not quiz_file.exists():
            print(f"Warning: {quiz_file} not found")
            continue

        stats = fix_quiz_file(quiz_file, dry_run=args.dry_run)
        total_stats['fixed'] += stats['fixed']
        total_stats['total_mcqs'] += stats['total_mcqs']
        if stats['fixed'] > 0:
            total_stats['files_with_fixes'] += 1

    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Total MCQ questions: {total_stats['total_mcqs']}")
    print(f"  Fixed: {total_stats['fixed']}")
    print(f"  Files with fixes: {total_stats['files_with_fixes']}")

    if args.dry_run and total_stats['fixed'] > 0:
        print(f"\nRun without --dry-run to apply these fixes.")
    elif total_stats['fixed'] > 0:
        print(f"\n✅ Fixes applied successfully!")
    else:
        print(f"\n✅ No issues found!")


if __name__ == '__main__':
    main()
