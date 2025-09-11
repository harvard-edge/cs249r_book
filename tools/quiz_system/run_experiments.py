#!/usr/bin/env python3
"""
Run AutoQuiz Experiments
========================

This script runs the 5 experiments using the existing quiz generation code
with research-based enhancements and knowledge map integration.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "genai"))
sys.path.insert(0, str(Path(__file__).parent))

# Import existing quiz generator
import quizzes

# Import knowledge map
from knowledge_map import KnowledgeMap, enhance_prompt_with_knowledge_map

class ExperimentRunner:
    """Runs experiments on quiz generation with different strategies."""
    
    def __init__(self, base_path: str = "/Users/VJ/GitHub/MLSysBook"):
        self.base_path = Path(base_path)
        self.results_dir = self.base_path / "experiments" / "quiz_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_map = KnowledgeMap()
        
    def run_experiment_1_baseline(self):
        """Experiment 1: Baseline performance with existing system."""
        
        print("\n" + "="*60)
        print("EXPERIMENT 1: BASELINE PERFORMANCE")
        print("="*60)
        
        test_chapters = ["introduction", "optimizations", "responsible_ai"]
        results = {}
        
        for chapter in test_chapters:
            print(f"\n📚 Testing: {chapter}")
            qmd_file = self.base_path / "quarto" / "contents" / "core" / chapter / f"{chapter}.qmd"
            
            if not qmd_file.exists():
                print(f"  ⚠️  File not found: {qmd_file}")
                continue
            
            # Use existing quiz generation
            output_file = self.results_dir / f"exp1_baseline_{chapter}.json"
            
            import argparse
            args = argparse.Namespace(
                model="gpt-4o",
                output=str(output_file)
            )
            
            start_time = time.time()
            try:
                quizzes.generate_for_file(str(qmd_file), args)
                generation_time = time.time() - start_time
                
                # Analyze results
                with open(output_file, 'r') as f:
                    quiz_data = json.load(f)
                
                metrics = self.analyze_quiz(quiz_data)
                metrics['generation_time'] = generation_time
                
                results[chapter] = metrics
                print(f"  ✅ Baseline metrics:")
                print(f"     - Questions: {metrics['total_questions']}")
                print(f"     - MCQ Balance: {metrics['mcq_chi_square']:.2f}")
                print(f"     - Question Types: {dict(metrics['question_types'])}")
                print(f"     - Generation Time: {generation_time:.1f}s")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                results[chapter] = {"error": str(e)}
        
        # Save experiment results
        self.save_experiment_results("experiment_1_baseline", results)
        return results
    
    def run_experiment_2_blooms_taxonomy(self):
        """Experiment 2: Bloom's Taxonomy Integration."""
        
        print("\n" + "="*60)
        print("EXPERIMENT 2: BLOOM'S TAXONOMY OPTIMIZATION")
        print("="*60)
        
        # Modify the system prompt to include Bloom's guidance
        original_prompt = quizzes.SYSTEM_PROMPT
        
        blooms_addition = """

## BLOOM'S TAXONOMY INTEGRATION

For each section, generate questions at different cognitive levels:
1. REMEMBER (20%): Factual recall - "What is...", "Define...", "List..."
2. UNDERSTAND (30%): Explain concepts - "Explain why...", "Describe how...", "Compare..."
3. APPLY (25%): Use in new situations - "Calculate...", "Demonstrate...", "Solve..."
4. ANALYZE (15%): Break down systems - "What are the tradeoffs...", "Why does X cause Y..."
5. EVALUATE/CREATE (10%): Judge or design - "Design a system...", "Evaluate the best approach..."

For each question, explicitly identify the Bloom's level in your internal reasoning.

Example Questions by Level:
- REMEMBER: "What is the time complexity of gradient descent?"
- UNDERSTAND: "Explain why batch normalization improves training stability"
- APPLY: "Calculate the memory savings from INT8 quantization of a 7B parameter model"
- ANALYZE: "Compare the tradeoffs between model pruning and quantization"
- EVALUATE: "Design a deployment strategy for a 100B parameter model on edge devices"
"""
        
        quizzes.SYSTEM_PROMPT = original_prompt + blooms_addition
        
        results = {}
        test_chapter = "optimizations"  # Good for testing different levels
        
        print(f"\n📚 Testing Bloom's integration on: {test_chapter}")
        
        qmd_file = self.base_path / "quarto" / "contents" / "core" / test_chapter / f"{test_chapter}.qmd"
        output_file = self.results_dir / f"exp2_blooms_{test_chapter}.json"
        
        args = argparse.Namespace(model="gpt-4o", output=str(output_file))
        
        try:
            quizzes.generate_for_file(str(qmd_file), args)
            
            with open(output_file, 'r') as f:
                quiz_data = json.load(f)
            
            # Analyze Bloom's distribution
            blooms_dist = self.analyze_blooms_distribution(quiz_data)
            metrics = self.analyze_quiz(quiz_data)
            metrics['blooms_distribution'] = blooms_dist
            
            results[test_chapter] = metrics
            
            print(f"  ✅ Bloom's Distribution:")
            for level, count in blooms_dist.items():
                print(f"     - {level}: {count}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[test_chapter] = {"error": str(e)}
        
        # Restore original prompt
        quizzes.SYSTEM_PROMPT = original_prompt
        
        self.save_experiment_results("experiment_2_blooms", results)
        return results
    
    def run_experiment_3_knowledge_map(self):
        """Experiment 3: Knowledge Map Integration."""
        
        print("\n" + "="*60)
        print("EXPERIMENT 3: KNOWLEDGE MAP INTEGRATION")
        print("="*60)
        
        # Test on Optimizations chapter which builds on many prerequisites
        test_chapter = "optimizations"
        
        print(f"\n📚 Testing knowledge map on: {test_chapter}")
        
        # Get the original build_user_prompt function
        original_build_prompt = quizzes.build_user_prompt
        
        # Create enhanced version with knowledge map
        def enhanced_build_prompt(section_title, section_text, chapter_number=None, 
                                chapter_title=None, previous_quizzes=None):
            # Get original prompt
            base_prompt = original_build_prompt(section_title, section_text, 
                                              chapter_number, chapter_title, 
                                              previous_quizzes)
            
            # Enhance with knowledge map
            if chapter_title:
                chapter_name = chapter_title.lower().replace(' ', '_')
                enhanced = enhance_prompt_with_knowledge_map(base_prompt, chapter_name, section_title)
                return enhanced
            
            return base_prompt
        
        # Replace the function
        quizzes.build_user_prompt = enhanced_build_prompt
        
        results = {}
        
        qmd_file = self.base_path / "quarto" / "contents" / "core" / test_chapter / f"{test_chapter}.qmd"
        output_file = self.results_dir / f"exp3_knowledge_{test_chapter}.json"
        
        args = argparse.Namespace(model="gpt-4o", output=str(output_file))
        
        try:
            quizzes.generate_for_file(str(qmd_file), args)
            
            with open(output_file, 'r') as f:
                quiz_data = json.load(f)
            
            metrics = self.analyze_quiz(quiz_data)
            
            # Check for prerequisite references
            prereq_refs = self.count_prerequisite_references(quiz_data)
            metrics['prerequisite_references'] = prereq_refs
            
            results[test_chapter] = metrics
            
            print(f"  ✅ Knowledge Integration:")
            print(f"     - Questions with prerequisites: {prereq_refs}")
            print(f"     - CALC questions: {metrics['calc_count']}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[test_chapter] = {"error": str(e)}
        
        # Restore original function
        quizzes.build_user_prompt = original_build_prompt
        
        self.save_experiment_results("experiment_3_knowledge", results)
        return results
    
    def run_experiment_4_calc_emphasis(self):
        """Experiment 4: CALC Question Emphasis for Technical Chapters."""
        
        print("\n" + "="*60)
        print("EXPERIMENT 4: CALC QUESTION EMPHASIS")
        print("="*60)
        
        # Modify prompt for CALC emphasis
        original_prompt = quizzes.SYSTEM_PROMPT
        
        calc_emphasis = """

## ENHANCED CALC QUESTION GENERATION

For the Optimizations chapter, include 2-3 CALC questions per section using these formulas:

### Quantization Calculations:
- Memory savings = (1 - new_bits/original_bits) * 100%
- Model size reduction = original_size * (new_bits/original_bits)
- Example: "A 13B parameter model with FP32 weights (4 bytes/param) is quantized to INT8. Calculate the memory savings and new model size."

### Pruning Calculations:
- Sparsity = pruned_parameters / total_parameters
- Theoretical speedup = 1 / (1 - sparsity) for structured pruning
- Example: "After pruning 70% of weights from a 350M parameter model, calculate the remaining parameters and theoretical speedup."

### Knowledge Distillation:
- Compression ratio = teacher_params / student_params
- Inference speedup = teacher_latency / student_latency
- Example: "A teacher model with 175B parameters and 100ms latency is distilled to a 7B parameter student with 10ms latency. Calculate compression ratio and speedup."

### Performance Metrics:
- Throughput = batch_size / latency_seconds
- Cost per million tokens = (compute_time_hours * hourly_rate) / tokens_millions
- Example: "With batch size 32 and latency 50ms, calculate throughput. At $2/hour GPU cost, what's the cost per million tokens?"

IMPORTANT: Show step-by-step calculations in the answer.
"""
        
        quizzes.SYSTEM_PROMPT = original_prompt + calc_emphasis
        
        results = {}
        test_chapters = ["optimizations", "hw_acceleration"]  # Both great for CALC
        
        for chapter in test_chapters:
            print(f"\n📚 Testing CALC emphasis on: {chapter}")
            
            qmd_file = self.base_path / "quarto" / "contents" / "core" / chapter / f"{chapter}.qmd"
            output_file = self.results_dir / f"exp4_calc_{chapter}.json"
            
            args = argparse.Namespace(model="gpt-4o", output=str(output_file))
            
            try:
                quizzes.generate_for_file(str(qmd_file), args)
                
                with open(output_file, 'r') as f:
                    quiz_data = json.load(f)
                
                metrics = self.analyze_quiz(quiz_data)
                calc_percentage = (metrics['calc_count'] / metrics['total_questions'] * 100) if metrics['total_questions'] > 0 else 0
                
                results[chapter] = metrics
                
                print(f"  ✅ CALC Questions:")
                print(f"     - Count: {metrics['calc_count']}/{metrics['total_questions']}")
                print(f"     - Percentage: {calc_percentage:.1f}%")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                results[chapter] = {"error": str(e)}
        
        # Restore original prompt
        quizzes.SYSTEM_PROMPT = original_prompt
        
        self.save_experiment_results("experiment_4_calc", results)
        return results
    
    def run_experiment_5_model_comparison(self):
        """Experiment 5: Compare different models."""
        
        print("\n" + "="*60)
        print("EXPERIMENT 5: MODEL COMPARISON")
        print("="*60)
        
        models_to_test = ["gpt-4o", "gpt-4o-mini"]
        test_chapter = "introduction"  # Use a simpler chapter for comparison
        
        results = {}
        
        for model in models_to_test:
            print(f"\n🤖 Testing model: {model}")
            
            qmd_file = self.base_path / "quarto" / "contents" / "core" / test_chapter / f"{test_chapter}.qmd"
            output_file = self.results_dir / f"exp5_model_{model.replace('-', '_')}_{test_chapter}.json"
            
            args = argparse.Namespace(model=model, output=str(output_file))
            
            start_time = time.time()
            try:
                quizzes.generate_for_file(str(qmd_file), args)
                generation_time = time.time() - start_time
                
                with open(output_file, 'r') as f:
                    quiz_data = json.load(f)
                
                metrics = self.analyze_quiz(quiz_data)
                metrics['generation_time'] = generation_time
                metrics['model'] = model
                
                results[model] = metrics
                
                print(f"  ✅ Model Performance:")
                print(f"     - Questions: {metrics['total_questions']}")
                print(f"     - Quality Score: {metrics['quality_score']:.1f}/100")
                print(f"     - Generation Time: {generation_time:.1f}s")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                results[model] = {"error": str(e)}
        
        self.save_experiment_results("experiment_5_models", results)
        return results
    
    def analyze_quiz(self, quiz_data: Dict) -> Dict:
        """Analyze quiz for metrics."""
        
        metrics = {
            "total_questions": 0,
            "question_types": Counter(),
            "mcq_distribution": Counter(),
            "mcq_chi_square": 0,
            "calc_count": 0,
            "quality_score": 0
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
                        # Extract answer choice
                        if "The correct answer is A" in answer:
                            metrics["mcq_distribution"]["A"] += 1
                        elif "The correct answer is B" in answer:
                            metrics["mcq_distribution"]["B"] += 1
                        elif "The correct answer is C" in answer:
                            metrics["mcq_distribution"]["C"] += 1
                        elif "The correct answer is D" in answer:
                            metrics["mcq_distribution"]["D"] += 1
        
        # Calculate chi-square for MCQ distribution
        if metrics["mcq_distribution"]:
            total_mcq = sum(metrics["mcq_distribution"].values())
            expected = total_mcq / 4
            chi_square = 0
            for choice in ["A", "B", "C", "D"]:
                observed = metrics["mcq_distribution"].get(choice, 0)
                if expected > 0:
                    chi_square += ((observed - expected) ** 2) / expected
            metrics["mcq_chi_square"] = chi_square
        
        # Calculate quality score (simplified)
        metrics["quality_score"] = self.calculate_quality_score(metrics)
        
        return metrics
    
    def calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score."""
        
        score = 0
        
        # Question diversity (25 points)
        num_types = len([t for t, c in metrics["question_types"].items() if c > 0])
        score += min(num_types * 5, 25)
        
        # MCQ balance (25 points)
        if metrics["mcq_chi_square"] < 7.815:  # Balanced
            score += 25
        elif metrics["mcq_chi_square"] < 15:
            score += 15
        else:
            score += 5
        
        # Has CALC questions (25 points for technical content)
        if metrics["calc_count"] > 0:
            calc_ratio = metrics["calc_count"] / max(metrics["total_questions"], 1)
            score += min(calc_ratio * 100, 25)
        
        # Total questions (25 points)
        if metrics["total_questions"] >= 15:
            score += 25
        elif metrics["total_questions"] >= 10:
            score += 20
        elif metrics["total_questions"] >= 5:
            score += 15
        else:
            score += 10
        
        return score
    
    def analyze_blooms_distribution(self, quiz_data: Dict) -> Counter:
        """Analyze Bloom's taxonomy distribution in questions."""
        
        blooms_keywords = {
            "remember": ["what", "define", "list", "name", "identify"],
            "understand": ["explain", "describe", "summarize", "compare"],
            "apply": ["calculate", "solve", "demonstrate", "implement"],
            "analyze": ["analyze", "differentiate", "examine", "contrast"],
            "evaluate": ["evaluate", "justify", "critique", "recommend"],
            "create": ["design", "develop", "propose", "formulate"]
        }
        
        distribution = Counter()
        
        for section in quiz_data.get("sections", []):
            quiz = section.get("quiz_data", {})
            if quiz.get("quiz_needed", False):
                for q in quiz.get("questions", []):
                    question_text = q.get("question", "").lower()
                    
                    # Simple keyword matching (could be improved with NLP)
                    for level, keywords in blooms_keywords.items():
                        if any(kw in question_text for kw in keywords):
                            distribution[level] += 1
                            break
        
        return distribution
    
    def count_prerequisite_references(self, quiz_data: Dict) -> int:
        """Count questions that reference prerequisite concepts."""
        
        prerequisite_keywords = [
            "previously", "earlier", "chapter", "recall", "remember from",
            "as we learned", "building on", "based on", "from our understanding"
        ]
        
        count = 0
        for section in quiz_data.get("sections", []):
            quiz = section.get("quiz_data", {})
            if quiz.get("quiz_needed", False):
                for q in quiz.get("questions", []):
                    question_text = q.get("question", "").lower()
                    if any(kw in question_text for kw in prerequisite_keywords):
                        count += 1
        
        return count
    
    def save_experiment_results(self, experiment_name: str, results: Dict):
        """Save experiment results to file."""
        
        output_file = self.results_dir / f"{experiment_name}_summary.json"
        
        summary = {
            "experiment": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "analysis": self.generate_analysis(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def generate_analysis(self, results: Dict) -> Dict:
        """Generate analysis of experiment results."""
        
        analysis = {
            "total_questions": sum(r.get("total_questions", 0) for r in results.values() if "error" not in r),
            "avg_quality_score": 0,
            "avg_generation_time": 0,
            "recommendations": []
        }
        
        quality_scores = [r.get("quality_score", 0) for r in results.values() if "error" not in r]
        if quality_scores:
            analysis["avg_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        gen_times = [r.get("generation_time", 0) for r in results.values() if "generation_time" in r]
        if gen_times:
            analysis["avg_generation_time"] = sum(gen_times) / len(gen_times)
        
        # Generate recommendations
        if analysis["avg_quality_score"] < 70:
            analysis["recommendations"].append("Consider enhancing prompts for better quality")
        
        return analysis


def main():
    """Run all experiments."""
    
    parser = argparse.ArgumentParser(description="Run AutoQuiz experiments")
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific experiment (1-5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments"
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.all:
        print("\n" + "="*70)
        print("RUNNING ALL AUTOQUIZ EXPERIMENTS")
        print("="*70)
        
        runner.run_experiment_1_baseline()
        runner.run_experiment_2_blooms_taxonomy()
        runner.run_experiment_3_knowledge_map()
        runner.run_experiment_4_calc_emphasis()
        runner.run_experiment_5_model_comparison()
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*70)
        
    elif args.experiment:
        experiments = {
            1: runner.run_experiment_1_baseline,
            2: runner.run_experiment_2_blooms_taxonomy,
            3: runner.run_experiment_3_knowledge_map,
            4: runner.run_experiment_4_calc_emphasis,
            5: runner.run_experiment_5_model_comparison
        }
        
        experiments[args.experiment]()
    
    else:
        print("Please specify --experiment N or --all")


if __name__ == "__main__":
    main()