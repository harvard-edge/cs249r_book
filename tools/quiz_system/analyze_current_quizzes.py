#!/usr/bin/env python3
"""
Analyze Current Quiz Distribution
==================================
Quick analysis of existing quizzes to understand the problem.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_all_quizzes():
    """Analyze distribution of question types across all existing quizzes."""
    
    base_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")
    
    overall_stats = {
        "total_questions": 0,
        "question_types": Counter(),
        "mcq_distribution": Counter(),
        "chapters_analyzed": 0,
        "by_chapter": {}
    }
    
    # Find all quiz files
    quiz_files = list(base_path.glob("**/*_quizzes.json"))
    
    print("="*60)
    print("CURRENT QUIZ ANALYSIS")
    print("="*60)
    
    for quiz_file in quiz_files:
        chapter_name = quiz_file.parent.name
        
        with open(quiz_file, 'r') as f:
            data = json.load(f)
        
        chapter_stats = {
            "questions": 0,
            "types": Counter(),
            "mcq_answers": Counter()
        }
        
        for section in data.get("sections", []):
            quiz = section.get("quiz_data", {})
            if quiz.get("quiz_needed", False):
                questions = quiz.get("questions", [])
                
                for q in questions:
                    q_type = q.get("question_type", "")
                    chapter_stats["types"][q_type] += 1
                    overall_stats["question_types"][q_type] += 1
                    chapter_stats["questions"] += 1
                    overall_stats["total_questions"] += 1
                    
                    # Track MCQ distribution
                    if q_type == "MCQ":
                        answer = q.get("answer", "")
                        if "correct answer is A" in answer:
                            choice = "A"
                        elif "correct answer is B" in answer:
                            choice = "B"
                        elif "correct answer is C" in answer:
                            choice = "C"
                        elif "correct answer is D" in answer:
                            choice = "D"
                        else:
                            choice = "?"
                        
                        chapter_stats["mcq_answers"][choice] += 1
                        overall_stats["mcq_distribution"][choice] += 1
        
        overall_stats["by_chapter"][chapter_name] = chapter_stats
        overall_stats["chapters_analyzed"] += 1
        
        # Print chapter summary
        print(f"\n{chapter_name}:")
        print(f"  Total: {chapter_stats['questions']} questions")
        print(f"  Types: {dict(chapter_stats['types'])}")
        if chapter_stats["mcq_answers"]:
            print(f"  MCQ Distribution: {dict(chapter_stats['mcq_answers'])}")
    
    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    
    print(f"\nTotal Questions: {overall_stats['total_questions']}")
    print(f"Chapters Analyzed: {overall_stats['chapters_analyzed']}")
    
    print("\nQuestion Type Distribution:")
    total_q = overall_stats['total_questions']
    for q_type, count in overall_stats['question_types'].most_common():
        percentage = (count / total_q * 100) if total_q > 0 else 0
        print(f"  {q_type}: {count} ({percentage:.1f}%)")
    
    print("\nMCQ Answer Distribution:")
    total_mcq = sum(overall_stats['mcq_distribution'].values())
    for choice in ['A', 'B', 'C', 'D']:
        count = overall_stats['mcq_distribution'].get(choice, 0)
        percentage = (count / total_mcq * 100) if total_mcq > 0 else 0
        expected = 25.0
        deviation = percentage - expected
        print(f"  {choice}: {count} ({percentage:.1f}%, deviation: {deviation:+.1f}%)")
    
    # Calculate chi-square
    if total_mcq > 0:
        expected_per_choice = total_mcq / 4
        chi_square = sum(
            ((overall_stats['mcq_distribution'].get(c, 0) - expected_per_choice) ** 2) / expected_per_choice
            for c in ['A', 'B', 'C', 'D']
        )
        print(f"\nChi-square statistic: {chi_square:.2f}")
        print(f"Balanced threshold: 7.815 (95% confidence)")
        print(f"Status: {'❌ IMBALANCED' if chi_square > 7.815 else '✅ Balanced'}")
    
    # Identify problems
    print("\n" + "="*60)
    print("IDENTIFIED ISSUES")
    print("="*60)
    
    issues = []
    
    # Check question type diversity
    mcq_percentage = (overall_stats['question_types']['MCQ'] / total_q * 100) if total_q > 0 else 0
    if mcq_percentage > 40:
        issues.append(f"Too many MCQ questions ({mcq_percentage:.1f}% vs ideal 30-40%)")
    
    calc_percentage = (overall_stats['question_types'].get('CALC', 0) / total_q * 100) if total_q > 0 else 0
    if calc_percentage < 10:
        issues.append(f"Too few CALC questions ({calc_percentage:.1f}% vs ideal 10-15% for technical chapters)")
    
    short_percentage = (overall_stats['question_types'].get('SHORT', 0) / total_q * 100) if total_q > 0 else 0
    if short_percentage < 20:
        issues.append(f"Too few SHORT questions ({short_percentage:.1f}% vs ideal 20-25%)")
    
    # Check MCQ balance
    if total_mcq > 0 and chi_square > 7.815:
        issues.append(f"MCQ answers are imbalanced (chi-square: {chi_square:.2f})")
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    return overall_stats

if __name__ == "__main__":
    stats = analyze_all_quizzes()