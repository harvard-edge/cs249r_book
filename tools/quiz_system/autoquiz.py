#!/usr/bin/env python3
"""
AutoQuiz: Intelligent Quiz Generation System
============================================

Main entry point for the AutoQuiz system that wraps around the existing
quiz generation logic with experimental enhancements and auto-tuning.

This system:
1. Uses the existing quiz generation as the core engine
2. Adds experimental prompt modifications
3. Provides auto-tuning based on quality metrics
4. Supports multiple models including local Ollama
5. Tracks and optimizes MCQ balance and question diversity
"""

import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime

# Add the core module to path
sys.path.insert(0, str(Path(__file__).parent))

# Import core quiz generator
from core import quiz_generator

# Import experiment modules
from experiments.prompt_variations import PromptModifier
from experiments.model_adapters import ModelAdapter
from analysis.quality_tracker import QualityTracker
from experiments.experiment_runner import ExperimentRunner

class AutoQuiz:
    """
    Main AutoQuiz system that orchestrates quiz generation with improvements.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize AutoQuiz with optional configuration."""
        self.config = self._load_config(config_path)
        self.prompt_modifier = PromptModifier()
        self.model_adapter = ModelAdapter()
        self.quality_tracker = QualityTracker()
        self.experiment_runner = ExperimentRunner()
        
        # Auto-tuning parameters
        self.auto_tune_enabled = self.config.get("auto_tune", True)
        self.quality_threshold = self.config.get("quality_threshold", 70)
        self.max_retries = self.config.get("max_retries", 3)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "auto_tune": True,
            "quality_threshold": 70,
            "max_retries": 3,
            "default_model": "gpt-4o",
            "use_local_model": False,
            "experiments_enabled": True,
            "prompt_strategy": "adaptive",
            "mcq_balance_tracking": True,
            "calc_emphasis_for_technical": True
        }
    
    def generate_quiz(self, 
                     qmd_file: str,
                     output_file: Optional[str] = None,
                     experiment_mode: bool = False) -> Dict:
        """
        Generate quiz with auto-tuning and quality optimization.
        
        Args:
            qmd_file: Path to the QMD file
            output_file: Optional output path for quiz JSON
            experiment_mode: Whether to run in experiment mode
            
        Returns:
            Generated quiz data with quality metrics
        """
        
        chapter_name = Path(qmd_file).parent.name
        best_result = None
        best_quality = 0
        
        # Determine strategy based on chapter type
        strategy = self._determine_strategy(chapter_name)
        
        print(f"\n🎯 AutoQuiz Generation for: {chapter_name}")
        print(f"   Strategy: {strategy['name']}")
        print(f"   Model: {strategy['model']}")
        
        # Try generation with retries if auto-tuning is enabled
        attempts = self.max_retries if self.auto_tune_enabled else 1
        
        for attempt in range(attempts):
            print(f"\n   Attempt {attempt + 1}/{attempts}...")
            
            # Modify the quiz generator's behavior
            result = self._generate_with_strategy(
                qmd_file, 
                strategy, 
                attempt,
                output_file
            )
            
            # Analyze quality
            quality_score = self.quality_tracker.analyze(result)
            print(f"   Quality Score: {quality_score:.1f}/100")
            
            if quality_score > best_quality:
                best_quality = quality_score
                best_result = result
            
            # If quality is good enough, stop
            if quality_score >= self.quality_threshold:
                print(f"   ✅ Quality threshold met!")
                break
            
            # Adjust strategy for next attempt
            if self.auto_tune_enabled and attempt < attempts - 1:
                strategy = self._adjust_strategy(strategy, quality_score, result)
                print(f"   📊 Adjusting strategy for next attempt...")
        
        # Save the best result
        if output_file and best_result:
            with open(output_file, 'w') as f:
                json.dump(best_result, f, indent=2)
            print(f"\n✅ Quiz saved to: {output_file}")
            print(f"   Final Quality Score: {best_quality:.1f}/100")
        
        return best_result
    
    def _determine_strategy(self, chapter_name: str) -> Dict:
        """Determine generation strategy based on chapter characteristics."""
        
        # Technical chapters need more CALC questions
        technical_chapters = [
            "benchmarking", "optimizations", "hw_acceleration",
            "efficient_ai", "training", "ops", "data_engineering"
        ]
        
        # Conceptual chapters need more reflection questions
        conceptual_chapters = [
            "responsible_ai", "privacy_security", "sustainable_ai",
            "ai_for_good", "robust_ai", "introduction"
        ]
        
        strategy = {
            "name": "balanced",
            "model": self.config["default_model"],
            "prompt_modifications": [],
            "emphasis": None
        }
        
        if chapter_name in technical_chapters:
            strategy["name"] = "technical"
            strategy["emphasis"] = "CALC"
            strategy["prompt_modifications"].append("calc_emphasis")
        elif chapter_name in conceptual_chapters:
            strategy["name"] = "conceptual"
            strategy["emphasis"] = "SHORT"
            strategy["prompt_modifications"].append("conceptual_focus")
        
        # Add MCQ balance if needed
        if self.config.get("mcq_balance_tracking"):
            current_distribution = self.quality_tracker.get_mcq_distribution()
            if self._needs_rebalancing(current_distribution):
                strategy["prompt_modifications"].append("mcq_balance")
        
        return strategy
    
    def _generate_with_strategy(self, 
                               qmd_file: str,
                               strategy: Dict,
                               attempt: int,
                               output_file: Optional[str]) -> Dict:
        """Generate quiz using the core generator with strategic modifications."""
        
        # Save original prompt
        original_prompt = quiz_generator.SYSTEM_PROMPT
        original_build_prompt = quiz_generator.build_user_prompt
        
        try:
            # Apply prompt modifications
            if strategy["prompt_modifications"]:
                modified_prompt = self.prompt_modifier.modify(
                    original_prompt,
                    strategy["prompt_modifications"],
                    attempt=attempt
                )
                quiz_generator.SYSTEM_PROMPT = modified_prompt
            
            # Override the model call if using local model
            if self.config.get("use_local_model"):
                original_call = quiz_generator.call_openai
                quiz_generator.call_openai = self.model_adapter.call_model
            
            # Create args for the generator
            import argparse
            args = argparse.Namespace(
                model=strategy["model"],
                output=output_file or "temp_quiz.json"
            )
            
            # Run the core generator
            quiz_generator.generate_for_file(qmd_file, args)
            
            # Load and return the generated quiz
            with open(args.output, 'r') as f:
                result = json.load(f)
            
            # Clean up temp file if needed
            if not output_file and Path(args.output).exists():
                Path(args.output).unlink()
            
            return result
            
        finally:
            # Restore original functions
            quiz_generator.SYSTEM_PROMPT = original_prompt
            if self.config.get("use_local_model"):
                quiz_generator.call_openai = original_call
    
    def _needs_rebalancing(self, distribution: Counter) -> bool:
        """Check if MCQ distribution needs rebalancing."""
        if not distribution:
            return False
        
        total = sum(distribution.values())
        if total < 4:
            return False
        
        # Calculate chi-square statistic
        expected = total / 4
        chi_square = sum(
            ((distribution.get(c, 0) - expected) ** 2) / expected
            for c in ["A", "B", "C", "D"]
        )
        
        # Threshold for 95% confidence with df=3
        return chi_square > 7.815
    
    def _adjust_strategy(self, strategy: Dict, quality_score: float, result: Dict) -> Dict:
        """Adjust strategy based on quality analysis."""
        
        # Analyze what went wrong
        issues = self.quality_tracker.identify_issues(result)
        
        # Adjust based on issues
        if "low_question_diversity" in issues:
            strategy["prompt_modifications"].append("diverse_types")
        
        if "mcq_imbalance" in issues:
            strategy["prompt_modifications"].append("mcq_strict_balance")
        
        if "insufficient_calc" in issues and strategy["emphasis"] == "CALC":
            strategy["prompt_modifications"].append("calc_mandatory")
        
        if "too_many_easy" in issues:
            strategy["prompt_modifications"].append("increase_difficulty")
        
        # Try a different model if quality is very low
        if quality_score < 50 and strategy["model"] == "gpt-4o":
            strategy["model"] = "gpt-4-turbo"
            print("   🔄 Switching to GPT-4-turbo for better results")
        
        return strategy
    
    def run_experiments(self, chapters: List[str]):
        """Run experiments to find optimal settings."""
        
        print("\n" + "="*70)
        print("AUTOQUIZ EXPERIMENTATION MODE")
        print("="*70)
        
        results = self.experiment_runner.run_all_experiments(chapters, self)
        
        # Generate recommendations
        recommendations = self.experiment_runner.analyze_results(results)
        
        print("\n" + "="*70)
        print("EXPERIMENT RESULTS")
        print("="*70)
        print(f"\nOptimal Configuration:")
        for key, value in recommendations.items():
            print(f"  {key}: {value}")
        
        # Update configuration based on experiments
        if input("\nApply recommended configuration? (y/n): ").lower() == 'y':
            self.config.update(recommendations)
            self._save_config()
            print("✅ Configuration updated!")
    
    def _save_config(self):
        """Save current configuration."""
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)


def main():
    """Main entry point for AutoQuiz."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AutoQuiz: Intelligent quiz generation with auto-tuning"
    )
    
    parser.add_argument(
        "mode",
        choices=["generate", "experiment", "batch"],
        help="Operation mode"
    )
    
    parser.add_argument(
        "-f", "--file",
        help="QMD file to process (generate mode)"
    )
    
    parser.add_argument(
        "-d", "--directory",
        help="Directory of QMD files (batch mode)"
    )
    
    parser.add_argument(
        "-c", "--chapters",
        nargs="+",
        help="Chapters for experiments"
    )
    
    parser.add_argument(
        "--config",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--no-auto-tune",
        action="store_true",
        help="Disable auto-tuning"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Initialize AutoQuiz
    autoquiz = AutoQuiz(args.config)
    
    if args.no_auto_tune:
        autoquiz.auto_tune_enabled = False
    
    # Execute based on mode
    if args.mode == "generate":
        if not args.file:
            print("Error: --file required for generate mode")
            return
        
        autoquiz.generate_quiz(args.file, args.output)
    
    elif args.mode == "experiment":
        chapters = args.chapters or ["introduction", "benchmarking", "responsible_ai"]
        autoquiz.run_experiments(chapters)
    
    elif args.mode == "batch":
        if not args.directory:
            print("Error: --directory required for batch mode")
            return
        
        # Process all QMD files in directory
        qmd_files = Path(args.directory).glob("**/*.qmd")
        for qmd_file in qmd_files:
            if "quiz" not in str(qmd_file):  # Skip quiz files
                output = qmd_file.parent / f"{qmd_file.stem}_quiz.json"
                autoquiz.generate_quiz(str(qmd_file), str(output))


if __name__ == "__main__":
    main()