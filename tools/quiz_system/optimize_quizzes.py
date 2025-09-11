#!/usr/bin/env python3
"""
Direct Quiz Optimization Runner
================================
Runs optimization experiments and measures improvements.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

class QuizOptimizer:
    def __init__(self):
        self.base_path = Path("/Users/VJ/GitHub/MLSysBook")
        self.results_dir = self.base_path / "experiments" / "quiz_optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.iterations = []
        
    def analyze_quiz_file(self, file_path):
        """Analyze a quiz file for quality metrics."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        metrics = {
            "total_questions": 0,
            "question_types": Counter(),
            "mcq_distribution": Counter(),
            "calc_count": 0,
            "sections_with_quiz": 0,
            "total_sections": len(data.get("sections", [])),
            "knowledge_references": 0
        }
        
        for section in data.get("sections", []):
            quiz = section.get("quiz_data", {})
            if quiz.get("quiz_needed", False):
                metrics["sections_with_quiz"] += 1
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
                            if f"correct answer is {choice}" in answer:
                                metrics["mcq_distribution"][choice] += 1
                                break
                    
                    # Check for knowledge references
                    question_text = q.get("question", "").lower()
                    if any(term in question_text for term in ["chapter", "previously", "recall", "as we learned"]):
                        metrics["knowledge_references"] += 1
        
        # Calculate quality score
        metrics["quality_score"] = self.calculate_quality_score(metrics)
        
        return metrics
    
    def calculate_quality_score(self, metrics):
        """Calculate overall quality score (0-100)."""
        score = 0
        
        # 1. Question Type Diversity (20 points)
        num_types = len([t for t in metrics["question_types"] if metrics["question_types"][t] > 0])
        score += min(num_types * 4, 20)  # 5 points per type, max 20
        
        # 2. MCQ Balance (20 points)
        if metrics["mcq_distribution"]:
            total_mcq = sum(metrics["mcq_distribution"].values())
            if total_mcq >= 4:
                chi_square = sum(
                    ((metrics["mcq_distribution"].get(c, 0) - total_mcq/4) ** 2) / (total_mcq/4)
                    for c in ["A", "B", "C", "D"]
                )
                if chi_square < 7.815:  # Well balanced
                    score += 20
                elif chi_square < 15:   # Somewhat balanced
                    score += 15
                elif chi_square < 30:   # Poor balance
                    score += 10
                else:                    # Very poor
                    score += 5
        
        # 3. CALC Questions (20 points)
        calc_percentage = (metrics["calc_count"] / max(metrics["total_questions"], 1)) * 100
        if calc_percentage >= 15:
            score += 20
        elif calc_percentage >= 10:
            score += 15
        elif calc_percentage >= 5:
            score += 10
        elif calc_percentage > 0:
            score += 5
        
        # 4. Coverage (20 points)
        coverage = metrics["sections_with_quiz"] / max(metrics["total_sections"], 1)
        score += coverage * 20
        
        # 5. Knowledge Integration (20 points)
        knowledge_percentage = (metrics["knowledge_references"] / max(metrics["total_questions"], 1)) * 100
        if knowledge_percentage >= 20:
            score += 20
        elif knowledge_percentage >= 10:
            score += 15
        elif knowledge_percentage >= 5:
            score += 10
        elif knowledge_percentage > 0:
            score += 5
        
        return round(score, 1)
    
    def run_iteration(self, iteration_num, chapters, optimization_level):
        """Run one iteration of quiz generation with specific optimizations."""
        
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration_num}: Optimization Level {optimization_level}")
        print(f"{'='*60}")
        
        iteration_results = {
            "iteration": iteration_num,
            "timestamp": datetime.now().isoformat(),
            "optimization_level": optimization_level,
            "chapters": {}
        }
        
        for chapter in chapters:
            print(f"\n📚 Processing: {chapter}")
            
            # Create optimized prompt file for this iteration
            prompt_file = self.create_optimized_prompt(chapter, optimization_level)
            
            # Run quiz generation
            output_file = self.results_dir / f"iter{iteration_num}_{chapter}_quiz.json"
            
            cmd = [
                "python3", "tools/scripts/genai/quizzes.py",
                "--mode", "generate",
                "-f", f"quarto/contents/core/{chapter}/{chapter}.qmd",
                "-o", str(output_file),
                "--model", "gpt-4o"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if output_file.exists():
                    metrics = self.analyze_quiz_file(output_file)
                    iteration_results["chapters"][chapter] = metrics
                    
                    print(f"  ✅ Generated: {metrics['total_questions']} questions")
                    print(f"  📊 Quality Score: {metrics['quality_score']}/100")
                    print(f"  📈 Types: {dict(metrics['question_types'])}")
                    print(f"  🎯 MCQ Dist: {dict(metrics['mcq_distribution'])}")
                    print(f"  🧮 CALC: {metrics['calc_count']} ({metrics['calc_count']/max(metrics['total_questions'],1)*100:.1f}%)")
                else:
                    print(f"  ❌ Generation failed")
                    iteration_results["chapters"][chapter] = {"error": "Generation failed"}
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏱️ Timeout")
                iteration_results["chapters"][chapter] = {"error": "Timeout"}
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                iteration_results["chapters"][chapter] = {"error": str(e)}
        
        # Calculate overall metrics
        iteration_results["overall"] = self.calculate_overall_metrics(iteration_results["chapters"])
        
        self.iterations.append(iteration_results)
        return iteration_results
    
    def create_optimized_prompt(self, chapter, optimization_level):
        """Create an optimized prompt based on optimization level."""
        
        # Read base prompt from quiz script
        with open("tools/scripts/genai/quizzes.py", 'r') as f:
            content = f.read()
        
        # Find and extract SYSTEM_PROMPT
        import re
        prompt_match = re.search(r'SYSTEM_PROMPT = f"""(.*?)"""', content, re.DOTALL)
        if not prompt_match:
            return None
        
        base_prompt = prompt_match.group(1)
        
        # Apply optimizations based on level
        if optimization_level == 1:
            # Basic MCQ balance fix
            optimization = """

## OPTIMIZATION LEVEL 1: MCQ Balance

**MCQ Answer Distribution Requirements**:
- Rotate correct answers: A, D, C, A, D, C (avoid B which is overused)
- Each choice should be correct ~25% of the time
- Make distractors equally plausible
"""
        
        elif optimization_level == 2:
            # Add CALC emphasis
            optimization = """

## OPTIMIZATION LEVEL 2: MCQ Balance + CALC Emphasis

**MCQ Requirements**:
- Use A and D as correct answers 60% of the time (currently underused)
- Minimize B as correct answer (currently 60% of all MCQs)

**CALC Requirements for Technical Chapters**:
- Include 2-3 CALC questions per section
- Examples: memory calculations, speedup metrics, compression ratios
- Show step-by-step solutions
"""
        
        elif optimization_level == 3:
            # Full optimization with knowledge integration
            optimization = """

## OPTIMIZATION LEVEL 3: Full Optimization

**Question Type Distribution**:
- CALC: 20-25% for technical chapters, 5% for conceptual
- MCQ: 20-25% (balanced A:25%, B:25%, C:25%, D:25%)
- SHORT: 25-30% (analysis and tradeoffs)
- TF: 15-20% (misconceptions)
- FILL: 10-15% (key terms)
- ORDER: 5-10% (processes)

**MCQ Balance**:
- Strict rotation: If last MCQ was B, next should be A, C, or D
- Track distribution within section

**Knowledge Integration**:
- Reference concepts from earlier chapters when applicable
- Use phrases like "Building on the gradient descent concept from Chapter 7..."
- Connect related concepts across chapters

**CALC Examples**:
- Quantization: "7B FP32 model (28GB) to INT8 = ? GB saved"
- Pruning: "Model with 350M params, 70% pruned = ? remaining"
- Throughput: "Batch 32, latency 50ms = ? samples/sec"
"""
        
        else:
            optimization = ""
        
        # Create modified script
        modified_prompt = base_prompt + optimization
        modified_content = content.replace(base_prompt, modified_prompt)
        
        # Save modified script
        temp_script = self.results_dir / f"quizzes_opt{optimization_level}.py"
        with open(temp_script, 'w') as f:
            f.write(modified_content)
        
        # Replace the main script temporarily
        import shutil
        shutil.copy("tools/scripts/genai/quizzes.py", "tools/scripts/genai/quizzes_backup.py")
        shutil.copy(temp_script, "tools/scripts/genai/quizzes.py")
        
        return temp_script
    
    def calculate_overall_metrics(self, chapters_data):
        """Calculate overall metrics across all chapters."""
        
        total_questions = 0
        all_mcq_dist = Counter()
        all_types = Counter()
        total_calc = 0
        quality_scores = []
        
        for chapter, data in chapters_data.items():
            if "error" not in data:
                total_questions += data.get("total_questions", 0)
                all_mcq_dist.update(data.get("mcq_distribution", {}))
                all_types.update(data.get("question_types", {}))
                total_calc += data.get("calc_count", 0)
                quality_scores.append(data.get("quality_score", 0))
        
        # Calculate chi-square for MCQ distribution
        chi_square = 0
        if all_mcq_dist:
            total_mcq = sum(all_mcq_dist.values())
            if total_mcq > 0:
                expected = total_mcq / 4
                chi_square = sum(
                    ((all_mcq_dist.get(c, 0) - expected) ** 2) / expected
                    for c in ["A", "B", "C", "D"]
                    if expected > 0
                )
        
        return {
            "total_questions": total_questions,
            "mcq_distribution": dict(all_mcq_dist),
            "mcq_chi_square": round(chi_square, 2),
            "question_types": dict(all_types),
            "calc_percentage": round((total_calc / max(total_questions, 1)) * 100, 1),
            "avg_quality_score": round(sum(quality_scores) / max(len(quality_scores), 1), 1)
        }
    
    def compare_iterations(self):
        """Compare all iterations and identify the best."""
        
        print("\n" + "="*60)
        print("ITERATION COMPARISON")
        print("="*60)
        
        comparison = []
        
        for iteration in self.iterations:
            overall = iteration["overall"]
            comparison.append({
                "iteration": iteration["iteration"],
                "optimization_level": iteration["optimization_level"],
                "avg_quality": overall["avg_quality_score"],
                "mcq_chi_square": overall["mcq_chi_square"],
                "calc_percentage": overall["calc_percentage"],
                "total_questions": overall["total_questions"]
            })
        
        # Print comparison table
        print("\n| Iter | Opt Level | Quality | MCQ χ² | CALC % | Questions |")
        print("|------|-----------|---------|--------|--------|-----------|")
        
        for comp in comparison:
            mcq_status = "✅" if comp["mcq_chi_square"] < 7.815 else "❌"
            print(f"| {comp['iteration']:4} | {comp['optimization_level']:9} | {comp['avg_quality']:7.1f} | {comp['mcq_chi_square']:6.1f}{mcq_status} | {comp['calc_percentage']:6.1f} | {comp['total_questions']:9} |")
        
        # Identify best iteration
        best = max(comparison, key=lambda x: x["avg_quality"])
        
        print(f"\n🏆 Best Iteration: {best['iteration']} (Optimization Level {best['optimization_level']})")
        print(f"   Quality Score: {best['avg_quality']}/100")
        
        return best
    
    def save_results(self):
        """Save all results to a comprehensive report."""
        
        report_path = self.results_dir / "optimization_report.json"
        
        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "iterations": self.iterations,
                "best_configuration": self.compare_iterations()
            }, f, indent=2)
        
        print(f"\n💾 Full report saved to: {report_path}")


def main():
    """Run the optimization experiments."""
    
    print("\n" + "="*70)
    print("QUIZ OPTIMIZATION EXPERIMENT RUNNER")
    print("="*70)
    
    optimizer = QuizOptimizer()
    
    # Test chapters - mix of technical and conceptual
    test_chapters = ["optimizations", "introduction", "responsible_ai"]
    
    print(f"\nTest Chapters: {', '.join(test_chapters)}")
    print("\nRunning 3 iterations with increasing optimization levels...")
    
    # Iteration 1: Basic MCQ balance fix
    optimizer.run_iteration(1, test_chapters, optimization_level=1)
    time.sleep(2)  # Brief pause
    
    # Iteration 2: MCQ + CALC emphasis
    optimizer.run_iteration(2, test_chapters, optimization_level=2)
    time.sleep(2)
    
    # Iteration 3: Full optimization
    optimizer.run_iteration(3, test_chapters, optimization_level=3)
    
    # Compare and identify best
    best = optimizer.compare_iterations()
    
    # Save all results
    optimizer.save_results()
    
    # Restore original script
    import shutil
    if Path("tools/scripts/genai/quizzes_backup.py").exists():
        shutil.copy("tools/scripts/genai/quizzes_backup.py", "tools/scripts/genai/quizzes.py")
        os.remove("tools/scripts/genai/quizzes_backup.py")
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    if best["avg_quality"] >= 75:
        print("\n✅ SUCCESS: Achieved target quality (75+)")
        print("   Recommended: Use Optimization Level", best["optimization_level"])
    else:
        print("\n⚠️  PARTIAL SUCCESS: Quality improved but below target")
        print("   Consider additional prompt refinements")
    
    return best


if __name__ == "__main__":
    main()