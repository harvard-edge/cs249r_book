#!/bin/bash
# Run Quiz Optimization Using Existing Script
# ============================================

echo "======================================================================"
echo "AUTONOMOUS QUIZ OPTIMIZATION"
echo "======================================================================"
echo ""
echo "This script will optimize quiz generation to fix:"
echo "1. MCQ answer bias (currently 60% B answers)"
echo "2. Lack of CALC questions (currently <1%)"
echo "3. Question type imbalance"
echo "4. Build on prior knowledge using knowledge map"
echo ""

# Test chapters for optimization
CHAPTERS=("optimizations" "training" "introduction")

# Create results directory
RESULTS_DIR="experiments/optimized_quizzes"
mkdir -p "$RESULTS_DIR"

# Save original prompt file (we'll modify and restore)
cp tools/scripts/genai/quizzes.py tools/scripts/genai/quizzes_original.py

echo "Testing on ${#CHAPTERS[@]} priority chapters..."
echo ""

for chapter in "${CHAPTERS[@]}"; do
    echo "======================================================================"
    echo "OPTIMIZING: $chapter"
    echo "======================================================================"
    
    # Create a temporary modified version of quizzes.py with optimizations
    python3 - << EOF
import sys
import re

# Read the original script
with open('tools/scripts/genai/quizzes_original.py', 'r') as f:
    content = f.read()

# Find the SYSTEM_PROMPT definition
import_match = re.search(r'SYSTEM_PROMPT = f"""(.*?)"""', content, re.DOTALL)
if import_match:
    original_prompt = import_match.group(1)
    
    # Add optimizations based on chapter type
    chapter = "$chapter"
    
    if chapter in ["optimizations", "training", "hw_acceleration", "benchmarking"]:
        # Technical chapter - needs CALC questions
        optimization = '''

## CRITICAL OPTIMIZATIONS FOR TECHNICAL CHAPTER

**MANDATORY Question Type Distribution**:
- CALC: 25-30% (PRIORITY - currently <1% globally!)
- MCQ: 20-25%
- SHORT: 25-30%
- Others: 20-25%

**CALC Question Examples for {chapter}**:
- "Calculate memory savings when quantizing a 7B FP32 model to INT8"
- "Calculate theoretical speedup from 70% pruning"
- "Calculate compression ratio: 175B teacher to 7B student"

**MCQ ANSWER FIX**: 
- Make A or D the correct answer 60% of the time (currently underused)
- Avoid B as correct answer (currently 60% of all MCQs!)
'''.format(chapter=chapter)
    else:
        # Conceptual chapter
        optimization = '''

## OPTIMIZATIONS FOR CONCEPTUAL CHAPTER

**Question Type Distribution**:
- SHORT: 30-35% (for reflection)
- TF: 20-25%
- MCQ: 20-25%
- Others: 20-25%

**MCQ ANSWER FIX**:
- Rotate correct answers: A, C, D, A, C, D...
- AVOID B as correct answer (currently overused)
'''
    
    # Apply the optimization
    modified_prompt = original_prompt + optimization
    modified_content = content.replace(original_prompt, modified_prompt)
    
    # Write the modified version
    with open('tools/scripts/genai/quizzes.py', 'w') as f:
        f.write(modified_content)
    
    print(f"✅ Applied optimizations for {chapter}")
EOF
    
    # Run the quiz generation with optimized prompt
    echo "🚀 Generating optimized quiz..."
    
    QMD_FILE="quarto/contents/core/$chapter/${chapter}.qmd"
    OUTPUT_FILE="$RESULTS_DIR/${chapter}_optimized.json"
    
    # Use the existing quiz script with GPT-4o
    python3 tools/scripts/genai/quizzes.py --mode generate -f "$QMD_FILE" -o "$OUTPUT_FILE" --model gpt-4o
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully generated optimized quiz for $chapter"
        
        # Analyze the results
        python3 - << EOF
import json
from collections import Counter

with open("$OUTPUT_FILE", 'r') as f:
    data = json.load(f)

stats = {
    "total": 0,
    "types": Counter(),
    "mcq_dist": Counter(),
    "calc": 0
}

for section in data.get("sections", []):
    quiz = section.get("quiz_data", {})
    if quiz.get("quiz_needed", False):
        for q in quiz.get("questions", []):
            stats["total"] += 1
            q_type = q.get("question_type", "")
            stats["types"][q_type] += 1
            
            if q_type == "CALC":
                stats["calc"] += 1
            
            if q_type == "MCQ":
                answer = q.get("answer", "")
                for choice in ["A", "B", "C", "D"]:
                    if f"correct answer is {choice}" in answer:
                        stats["mcq_dist"][choice] += 1
                        break

print(f"📊 Results:")
print(f"   Total questions: {stats['total']}")
print(f"   Types: {dict(stats['types'])}")
print(f"   MCQ distribution: {dict(stats['mcq_dist'])}")
print(f"   CALC questions: {stats['calc']} ({stats['calc']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "")
EOF
        
        # Copy to actual location if successful
        FINAL_FILE="quarto/contents/core/$chapter/${chapter}_quizzes.json"
        cp "$OUTPUT_FILE" "$FINAL_FILE"
        echo "   📁 Saved to: $FINAL_FILE"
    else
        echo "❌ Failed to generate quiz for $chapter"
    fi
    
    echo ""
done

# Restore original script
mv tools/scripts/genai/quizzes_original.py tools/scripts/genai/quizzes.py

echo "======================================================================"
echo "OPTIMIZATION COMPLETE"
echo "======================================================================"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "To apply to all chapters, run:"
echo "  for chapter in introduction ml_systems dl_primer data_engineering \\"
echo "    dnn_architectures frameworks training efficient_ai optimizations \\"
echo "    hw_acceleration benchmarking ops ondevice_learning robust_ai \\"
echo "    privacy_security responsible_ai sustainable_ai ai_for_good \\"
echo "    workflow conclusion; do"
echo "    ./run_optimization.sh \$chapter"
echo "  done"