#!/usr/bin/env python3
"""
Apply Quiz Optimizations to Existing System
============================================
This script modifies the existing quiz generation system with optimizations.
"""

import re
import shutil
from pathlib import Path

def apply_optimizations():
    """Apply optimizations to the existing quiz generation script."""
    
    quiz_script = Path("tools/scripts/genai/quizzes.py")
    
    # Backup original
    shutil.copy(quiz_script, "tools/scripts/genai/quizzes_original.py")
    
    with open(quiz_script, 'r') as f:
        content = f.read()
    
    # Find the SYSTEM_PROMPT
    prompt_match = re.search(r'(SYSTEM_PROMPT = f""")(.*?)(""")', content, re.DOTALL)
    
    if not prompt_match:
        print("❌ Could not find SYSTEM_PROMPT")
        return
    
    original_prompt = prompt_match.group(2)
    
    # Add optimizations
    optimized_prompt = original_prompt + """

## CRITICAL OPTIMIZATIONS FOR QUIZ QUALITY

### 1. MCQ ANSWER DISTRIBUTION (MANDATORY)
**SEVERE ISSUE**: Currently 60% of MCQ answers are B. This MUST be fixed.

**REQUIREMENTS**:
- Use this rotation pattern: A, D, C, A, D, C (skip B frequently)
- Track distribution: If you've used B once, avoid it for next 3 MCQs
- Each letter should be correct 25% of the time across all MCQs
- Make all distractors equally plausible

**Example Distribution for 4 MCQs**: A, D, C, A (no B)
**Example Distribution for 8 MCQs**: A, D, C, A, D, C, B, A

### 2. QUESTION TYPE REQUIREMENTS BY CHAPTER

**Technical Chapters** (training, optimizations, hw_acceleration, benchmarking, efficient_ai, ops, data_engineering):
- CALC: 25-30% (MANDATORY - currently <1%!)
- SHORT: 25-30%
- MCQ: 20-25%
- TF: 10-15%
- Others: 10-15%

**CALC Examples**:
- Memory: "7B FP32 model (28GB) → INT8 = ? GB saved"
- Speedup: "70% pruning → theoretical speedup = ?"
- Throughput: "Batch 32, latency 50ms = ? samples/sec"
- Show calculations: "28GB × 0.25 = 7GB (75% reduction)"

**Conceptual Chapters** (responsible_ai, privacy_security, sustainable_ai, ai_for_good, robust_ai):
- SHORT: 30-35%
- TF: 20-25%
- MCQ: 20-25%
- FILL: 10-15%
- Others: 10-15%
- CALC: 0-5%

**Balanced Chapters** (introduction, ml_systems, dl_primer, frameworks, workflow, conclusion):
- SHORT: 25-30%
- MCQ: 20-25%
- TF: 15-20%
- CALC: 10-15%
- Others: 15-20%

### 3. KNOWLEDGE PROGRESSION
- Reference concepts from earlier chapters when applicable
- Use phrases like: "Building on [concept] from Chapter X..."
- Connect related ideas across sections
- 15-20% of questions should reference prior knowledge

### 4. SELF-CHECK PURPOSE
These are learning reinforcement tools, NOT grading instruments:
- Help students identify knowledge gaps
- Provide explanations that teach
- Progress from basic to complex within each section
- Focus on understanding, not memorization

### 5. QUALITY CHECKLIST
Before finalizing each section's quiz:
✓ MCQ answers follow rotation pattern (A,D,C,A,D,C)
✓ CALC questions included for technical content (with real numbers)
✓ Variety of question types (avoid >40% of any single type)
✓ At least one question builds on prior knowledge
✓ Explanations provide learning value, not just answers
"""
    
    # Replace the prompt
    new_content = content.replace(
        prompt_match.group(0),
        prompt_match.group(1) + optimized_prompt + prompt_match.group(3)
    )
    
    # Also add chapter detection logic
    chapter_detection = """
# Chapter type detection for optimization
TECHNICAL_CHAPTERS = ["training", "optimizations", "hw_acceleration", "benchmarking", 
                      "efficient_ai", "ops", "data_engineering"]
CONCEPTUAL_CHAPTERS = ["responsible_ai", "privacy_security", "sustainable_ai", 
                       "ai_for_good", "robust_ai"]

def get_chapter_type(file_path):
    \"\"\"Determine chapter type from file path.\"\"\"
    for chapter in TECHNICAL_CHAPTERS:
        if chapter in str(file_path):
            return "technical"
    for chapter in CONCEPTUAL_CHAPTERS:
        if chapter in str(file_path):
            return "conceptual"
    return "balanced"
"""
    
    # Add before the build_user_prompt function
    build_prompt_match = re.search(r'(def build_user_prompt)', new_content)
    if build_prompt_match:
        new_content = new_content.replace(
            build_prompt_match.group(0),
            chapter_detection + "\n\n" + build_prompt_match.group(0)
        )
    
    # Modify build_user_prompt to include chapter type hint
    user_prompt_addition = """
    
    # Add chapter type hint
    chapter_type = get_chapter_type(chapter_title) if chapter_title else "balanced"
    if chapter_type == "technical":
        prompt_addition = "\\n\\nNOTE: This is a TECHNICAL chapter. Include 25-30% CALC questions with real calculations."
    elif chapter_type == "conceptual":
        prompt_addition = "\\n\\nNOTE: This is a CONCEPTUAL chapter. Focus on SHORT reflection questions (30-35%)."
    else:
        prompt_addition = "\\n\\nNOTE: This is a BALANCED chapter. Mix all question types appropriately."
    
    return prompt + prompt_addition
"""
    
    # Find the end of build_user_prompt function
    build_end_match = re.search(r'(def build_user_prompt.*?)(return.*?)(\n(?=def|\n#|\nclass|\Z))', 
                                new_content, re.DOTALL)
    if build_end_match:
        # Replace the return statement
        new_content = new_content.replace(
            build_end_match.group(2),
            user_prompt_addition
        )
    
    # Write the optimized version
    with open(quiz_script, 'w') as f:
        f.write(new_content)
    
    print("✅ Optimizations applied successfully!")
    print("\nKey improvements:")
    print("1. MCQ answer rotation pattern (A,D,C) to fix 60% B-bias")
    print("2. CALC questions (25-30%) for technical chapters")
    print("3. Question type diversity based on chapter type")
    print("4. Knowledge progression across chapters")
    print("5. Self-check focus for learning reinforcement")
    
    print("\n📝 Original saved as: tools/scripts/genai/quizzes_original.py")
    print("📝 Optimized version: tools/scripts/genai/quizzes.py")
    
    return True

def test_single_chapter():
    """Test generation on a single chapter."""
    
    import subprocess
    
    print("\n" + "="*60)
    print("TESTING OPTIMIZED GENERATION")
    print("="*60)
    
    # Test on optimizations chapter (should have lots of CALC)
    cmd = [
        "python3", "tools/scripts/genai/quizzes.py",
        "--mode", "generate",
        "-f", "quarto/contents/core/optimizations/optimizations.qmd",
        "-o", "test_optimized_quiz.json",
        "--model", "gpt-4o"
    ]
    
    print("\nTesting on Optimizations chapter (technical)...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Generation successful!")
            
            # Analyze the output
            import json
            with open("test_optimized_quiz.json", 'r') as f:
                data = json.load(f)
            
            from collections import Counter
            
            stats = {
                "total": 0,
                "types": Counter(),
                "mcq_dist": Counter()
            }
            
            for section in data.get("sections", []):
                quiz = section.get("quiz_data", {})
                if quiz.get("quiz_needed", False):
                    for q in quiz.get("questions", []):
                        stats["total"] += 1
                        q_type = q.get("question_type", "")
                        stats["types"][q_type] += 1
                        
                        if q_type == "MCQ":
                            answer = q.get("answer", "")
                            for choice in ["A", "B", "C", "D"]:
                                if f"correct answer is {choice}" in answer:
                                    stats["mcq_dist"][choice] += 1
                                    break
            
            print("\n📊 Results:")
            print(f"Total questions: {stats['total']}")
            print(f"Question types: {dict(stats['types'])}")
            print(f"MCQ distribution: {dict(stats['mcq_dist'])}")
            
            # Check improvements
            calc_pct = (stats['types'].get('CALC', 0) / max(stats['total'], 1)) * 100
            print(f"\n✅ CALC percentage: {calc_pct:.1f}% (target: 25-30%)")
            
            if stats['mcq_dist']:
                total_mcq = sum(stats['mcq_dist'].values())
                b_pct = (stats['mcq_dist'].get('B', 0) / total_mcq) * 100
                print(f"✅ MCQ 'B' percentage: {b_pct:.1f}% (was 60%, target: 25%)")
        else:
            print("❌ Generation failed")
            print("Error:", result.stderr[:500])
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Apply optimizations and test."""
    
    print("🚀 Applying optimizations to quiz generation system...")
    
    if apply_optimizations():
        print("\n✅ Optimizations applied!")
        
        # Optionally test
        response = input("\nTest generation on a chapter? (y/n): ")
        if response.lower() == 'y':
            test_single_chapter()
    else:
        print("❌ Failed to apply optimizations")

if __name__ == "__main__":
    main()