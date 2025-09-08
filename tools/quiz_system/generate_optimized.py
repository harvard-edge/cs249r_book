#!/usr/bin/env python3
"""
Direct Quiz Generation with Optimizations
==========================================
A simplified version that directly calls OpenAI API to generate optimized quizzes.
"""

import json
import os
import re
from pathlib import Path
from collections import Counter
from datetime import datetime

# You'll need to set this environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_sections_from_qmd(file_path):
    """Extract sections from a QMD file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = []
    lines = content.split('\n')
    
    current_section = None
    for i, line in enumerate(lines):
        if line.startswith('## ') and not line.startswith('###'):
            if current_section:
                sections.append(current_section)
            
            # Extract title and ID
            title_match = re.match(r'^##\s+(.+?)(\s*\{#([\w\-]+)\})?\s*$', line)
            if title_match:
                current_section = {
                    'title': title_match.group(1),
                    'id': f"#{title_match.group(3)}" if title_match.group(3) else f"#sec-{i}",
                    'text': ''
                }
        elif current_section:
            current_section['text'] += line + '\n'
    
    if current_section:
        sections.append(current_section)
    
    # Filter out quiz answers sections
    sections = [s for s in sections if 'quiz answer' not in s['title'].lower()]
    
    return sections

def create_optimized_prompt(chapter_name, section_title, section_text, previous_sections=None):
    """Create an optimized prompt for quiz generation."""
    
    # Determine chapter type
    technical_chapters = ["optimizations", "training", "hw_acceleration", "benchmarking"]
    conceptual_chapters = ["introduction", "responsible_ai", "privacy_security"]
    
    is_technical = chapter_name in technical_chapters
    is_conceptual = chapter_name in conceptual_chapters
    
    system_prompt = """You are an expert educational content creator for a university-level ML Systems textbook.
Generate pedagogically sound self-check quiz questions that help students verify their understanding.

CRITICAL REQUIREMENTS:

1. MCQ ANSWER DISTRIBUTION:
   - Rotate correct answers strictly: A, D, C, A, D, C (avoid B which is overused)
   - Never use the same correct answer position twice in a row
   - Make all distractors equally plausible

2. QUESTION TYPE VARIETY:
   - Use multiple question types per section
   - Avoid over-reliance on any single type

3. SELF-CHECK PURPOSE:
   - Questions should help students identify knowledge gaps
   - Provide learning reinforcement, not assessment
   - Include explanations that teach, not just confirm

Return a JSON object with this structure:
{
    "quiz_needed": true/false,
    "rationale": {...},
    "questions": [...]
}"""
    
    # Add specific requirements based on chapter type
    if is_technical:
        type_requirements = """
TECHNICAL CHAPTER REQUIREMENTS:
- CALC: 25-30% (calculations with real numbers)
- MCQ: 20-25% (conceptual understanding)
- SHORT: 25-30% (explain tradeoffs)
- TF: 10-15% (misconceptions)
- Others: 10-15%

CALC QUESTION EXAMPLES:
- "Calculate memory savings: 7B parameter FP32 model (28GB) to INT8 = ?"
- "Calculate speedup: 70% pruning of 350M parameters = ?"
- "Calculate throughput: batch size 32, latency 50ms = ?"
"""
    elif is_conceptual:
        type_requirements = """
CONCEPTUAL CHAPTER REQUIREMENTS:
- SHORT: 30-35% (reflection and analysis)
- TF: 20-25% (principles and misconceptions)
- MCQ: 20-25% (conceptual understanding)
- FILL: 10-15% (key terms)
- Others: 10-15%
- CALC: 0-5% (only if relevant)
"""
    else:
        type_requirements = """
BALANCED CHAPTER REQUIREMENTS:
- SHORT: 25-30%
- MCQ: 20-25%
- TF: 15-20%
- CALC: 10-15%
- Others: 15-20%
"""
    
    # Build knowledge context if there are previous sections
    knowledge_context = ""
    if previous_sections and len(previous_sections) > 0:
        knowledge_context = """
PREVIOUS CONCEPTS IN THIS CHAPTER:
Students have already learned:
"""
        for i, prev in enumerate(previous_sections[-3:], 1):
            knowledge_context += f"- Section {i}: {prev.get('title', 'Unknown')}\n"
        
        knowledge_context += """
Build on these concepts where appropriate.
"""
    
    user_prompt = f"""Chapter: {chapter_name}
Section: {section_title}

{type_requirements}

{knowledge_context}

Section Content:
{section_text[:2000]}  # Limit to avoid token issues

Generate 3-5 high-quality self-check questions following the requirements above.
Remember: MCQ answers must rotate (A, D, C pattern), avoid B."""
    
    return system_prompt, user_prompt

def call_openai_api(system_prompt, user_prompt):
    """Call OpenAI API directly using curl."""
    import subprocess
    import json
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    data = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 2000
    }
    
    # Use curl to call OpenAI API
    cmd = [
        "curl", "-s",
        "https://api.openai.com/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {OPENAI_API_KEY}",
        "-d", json.dumps(data)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        response = json.loads(result.stdout)
        
        if "choices" in response:
            content = response["choices"][0]["message"]["content"]
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        
        return {"quiz_needed": False, "rationale": "Failed to generate quiz"}
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        return {"quiz_needed": False, "rationale": f"API error: {str(e)}"}

def generate_optimized_quiz(chapter_path):
    """Generate an optimized quiz for a chapter."""
    
    chapter_name = chapter_path.name
    qmd_file = chapter_path / f"{chapter_name}.qmd"
    
    if not qmd_file.exists():
        print(f"❌ File not found: {qmd_file}")
        return None
    
    print(f"\n📚 Generating optimized quiz for: {chapter_name}")
    
    # Extract sections
    sections = extract_sections_from_qmd(qmd_file)
    print(f"   Found {len(sections)} sections")
    
    # Generate quiz for each section
    quiz_sections = []
    previous_sections = []
    
    for i, section in enumerate(sections):
        print(f"   Section {i+1}/{len(sections)}: {section['title'][:40]}...")
        
        # Create optimized prompt
        system_prompt, user_prompt = create_optimized_prompt(
            chapter_name, 
            section['title'], 
            section['text'],
            previous_sections
        )
        
        # Call API
        quiz_data = call_openai_api(system_prompt, user_prompt)
        
        quiz_sections.append({
            "section_id": section['id'],
            "section_title": section['title'],
            "quiz_data": quiz_data
        })
        
        if quiz_data.get('quiz_needed', False):
            questions = quiz_data.get('questions', [])
            print(f"      ✅ Generated {len(questions)} questions")
        else:
            print(f"      ⏭️  No quiz needed")
        
        previous_sections.append(section)
    
    # Create final structure
    quiz_output = {
        "metadata": {
            "source_file": str(qmd_file),
            "timestamp": datetime.now().isoformat(),
            "optimization": "v3",
            "total_sections": len(sections),
            "sections_with_quizzes": sum(1 for s in quiz_sections if s['quiz_data'].get('quiz_needed', False))
        },
        "sections": quiz_sections
    }
    
    return quiz_output

def analyze_quiz(quiz_data):
    """Analyze quiz for quality metrics."""
    
    metrics = {
        "total_questions": 0,
        "question_types": Counter(),
        "mcq_distribution": Counter(),
        "calc_count": 0
    }
    
    for section in quiz_data.get("sections", []):
        quiz = section.get("quiz_data", {})
        if quiz.get("quiz_needed", False):
            questions = quiz.get("questions", [])
            metrics["total_questions"] += len(questions)
            
            for q in questions:
                q_type = q.get("question_type", "")
                metrics["question_types"][q_type] += 1
                
                if q_type == "CALC":
                    metrics["calc_count"] += 1
                
                if q_type == "MCQ":
                    answer = q.get("answer", "")
                    for choice in ["A", "B", "C", "D"]:
                        if f"correct answer is {choice}" in answer or f"answer is {choice}" in answer:
                            metrics["mcq_distribution"][choice] += 1
                            break
    
    # Calculate chi-square for MCQ
    chi_square = 0
    if metrics["mcq_distribution"]:
        total_mcq = sum(metrics["mcq_distribution"].values())
        if total_mcq >= 4:
            expected = total_mcq / 4
            chi_square = sum(
                ((metrics["mcq_distribution"].get(c, 0) - expected) ** 2) / expected
                for c in ["A", "B", "C", "D"]
            )
    
    metrics["mcq_chi_square"] = round(chi_square, 2)
    metrics["mcq_balanced"] = chi_square < 7.815
    
    return metrics

def main():
    """Main function to run optimized quiz generation."""
    
    if not OPENAI_API_KEY:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("\n" + "="*60)
    print("OPTIMIZED QUIZ GENERATION")
    print("="*60)
    
    # Test chapters
    test_chapters = ["optimizations", "introduction", "responsible_ai"]
    
    results = {}
    
    for chapter_name in test_chapters:
        chapter_path = Path(f"/Users/VJ/GitHub/MLSysBook/quarto/contents/core/{chapter_name}")
        
        # Generate optimized quiz
        quiz_data = generate_optimized_quiz(chapter_path)
        
        if quiz_data:
            # Save quiz
            output_file = Path(f"experiments/quiz_optimization/{chapter_name}_optimized.json")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(quiz_data, f, indent=2)
            
            # Analyze results
            metrics = analyze_quiz(quiz_data)
            results[chapter_name] = metrics
            
            print(f"\n📊 Results for {chapter_name}:")
            print(f"   Total questions: {metrics['total_questions']}")
            print(f"   Types: {dict(metrics['question_types'])}")
            print(f"   MCQ distribution: {dict(metrics['mcq_distribution'])}")
            print(f"   MCQ balanced: {'✅' if metrics['mcq_balanced'] else '❌'} (χ²={metrics['mcq_chi_square']})")
            print(f"   CALC questions: {metrics['calc_count']} ({metrics['calc_count']/max(metrics['total_questions'],1)*100:.1f}%)")
            print(f"   Saved to: {output_file}")
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    total_questions = sum(r.get("total_questions", 0) for r in results.values())
    all_mcq = Counter()
    for r in results.values():
        all_mcq.update(r.get("mcq_distribution", {}))
    
    print(f"\nTotal questions generated: {total_questions}")
    print(f"Overall MCQ distribution: {dict(all_mcq)}")
    
    if all_mcq:
        total_mcq = sum(all_mcq.values())
        expected = total_mcq / 4
        overall_chi = sum(
            ((all_mcq.get(c, 0) - expected) ** 2) / expected
            for c in ["A", "B", "C", "D"]
            if expected > 0
        )
        print(f"Overall MCQ balance: {'✅' if overall_chi < 7.815 else '❌'} (χ²={overall_chi:.2f})")
    
    print("\n✅ Generation complete!")

if __name__ == "__main__":
    main()