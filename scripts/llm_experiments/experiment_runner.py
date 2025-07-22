# Comprehensive LLM Model and Length Optimization Experiment Runner
#
# This is the main orchestration system that runs systematic experiments
# to find the optimal Ollama model and explanation length for cross-references.

import json
import time
import requests
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from test_cases import TEST_CASES, LENGTH_TARGETS
from llm_judge import LLMJudge

class ExperimentRunner:
    """
    Orchestrates comprehensive experiments to optimize LLM model selection
    and explanation length for cross-reference generation.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Models to test (add more as they become available)
        self.test_models = [
            "qwen2.5:7b",
            "qwen2.5:14b", 
            "qwen2.5:32b",
            "llama3.1:8b",
            "llama3.1:70b",
            "mistral:7b",
            "mistral-nemo:12b",
            "gemma2:9b",
            "gemma2:27b",
            "phi3:14b",
            "codellama:13b"
        ]
        
        self.judge = LLMJudge(judge_model="qwen2.5:32b")  # Use powerful model as judge
        self.experiment_results = []
    
    def check_available_models(self) -> List[str]:
        """Check which models are actually available in Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                available = []
                models_data = response.json().get("models", [])
                available_names = [model["name"] for model in models_data]
                
                for model in self.test_models:
                    if model in available_names:
                        available.append(model)
                        
                print(f"✅ Found {len(available)} available models: {', '.join(available)}")
                return available
            else:
                print(f"❌ Failed to get models list: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error checking available models: {e}")
            return []

    def generate_explanation(self, model: str, source_title: str, source_content: str, 
                           target_title: str, target_content: str, 
                           connection_type: str, length_target: Dict) -> Optional[str]:
        """Generate an explanation using a specific model and length target"""
        
        # Truncate content for context efficiency
        source_snippet = source_content[:800] + "..." if len(source_content) > 800 else source_content
        target_snippet = target_content[:800] + "..." if len(target_content) > 800 else target_content
        
        prompt = f"""You are writing cross-reference explanations for a Machine Learning Systems textbook. Create natural, varied explanations that tell students WHY they should follow the connection.

Source Section: "{source_title}"
{source_snippet}

Target Section: "{target_title}"  
{target_snippet}

Write a natural {length_target['min_words']}-{length_target['max_words']} word explanation that completes: "See also: {target_title} - [your explanation]"

Use varied, engaging language. Examples of good explanations:
- "provides essential background on neural network mathematics"
- "shows practical applications of these optimization techniques"  
- "dives deeper into the implementation details"
- "explains why this matters for deployment decisions"
- "contrasts different approaches to model compression"
- "demonstrates real-world uses of edge computing"
- "covers prerequisite concepts for understanding transformers"
- "explores advanced aspects of distributed training"

CRITICAL: Your explanation must be {length_target['min_words']}-{length_target['max_words']} words and provide value beyond the section title.

Write ONLY the explanation phrase (no prefix):"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 80
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                explanation = response.json().get("response", "").strip()
                
                # Clean up the explanation
                prefixes_to_remove = ["Explanation:", "- ", "• ", '"', "'", "contextual:", "foundational:", "practical:", "detailed:", "comparative:"]
                for prefix in prefixes_to_remove:
                    if explanation.lower().startswith(prefix.lower()):
                        explanation = explanation[len(prefix):].strip()
                
                explanation = explanation.replace('"', '').replace("'", "").strip()
                
                # Ensure it starts with lowercase
                if explanation and explanation[0].isupper() and not explanation.startswith(('AI', 'ML', 'GPU', 'CPU')):
                    explanation = explanation[0].lower() + explanation[1:]
                
                return explanation
            else:
                print(f"❌ Request failed for model {model}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error generating explanation with {model}: {e}")
            return None

    def run_model_comparison_experiment(self, test_cases: List[Dict] = None) -> Dict:
        """
        Run comprehensive model comparison using a subset of test cases.
        Tests each available model on the same test cases for fair comparison.
        """
        
        if test_cases is None:
            # Use a representative subset of test cases
            test_cases = TEST_CASES[:6]  # First 6 cases for comprehensive testing
            
        available_models = self.check_available_models()
        if not available_models:
            print("❌ No models available for testing")
            return {"error": "No models available"}
        
        print(f"🧪 Starting model comparison experiment with {len(available_models)} models and {len(test_cases)} test cases")
        
        # Use standard length target for fair comparison
        standard_length = next(lt for lt in LENGTH_TARGETS if lt["description"] == "standard")
        
        model_results = {}
        
        for model in available_models:
            print(f"\n🔬 Testing model: {model}")
            model_explanations = []
            
            for i, test_case in enumerate(test_cases):
                print(f"  Case {i+1}/{len(test_cases)}: {test_case['source_title']} → {test_case['target_title']}")
                
                explanation = self.generate_explanation(
                    model=model,
                    source_title=test_case["source_title"],
                    source_content=test_case["source_content"],
                    target_title=test_case["target_title"], 
                    target_content=test_case["target_content"],
                    connection_type=test_case["connection_type"],
                    length_target=standard_length
                )
                
                if explanation:
                    model_explanations.append({
                        "test_case_id": test_case["id"],
                        "source_title": test_case["source_title"],
                        "source_content": test_case["source_content"],
                        "target_title": test_case["target_title"],
                        "target_content": test_case["target_content"],
                        "connection_type": test_case["connection_type"],
                        "explanation": explanation,
                        "model": model,
                        "length_target": standard_length["description"],
                        "word_count": len(explanation.split())
                    })
                    
                time.sleep(1)  # Rate limiting
            
            # Evaluate all explanations for this model
            print(f"  📊 Evaluating {len(model_explanations)} explanations...")
            evaluated_explanations = self.judge.batch_evaluate(model_explanations)
            
            # Calculate model statistics
            if evaluated_explanations:
                scores = [result["evaluation"]["overall_score"] for result in evaluated_explanations]
                word_counts = [result["word_count"] for result in evaluated_explanations]
                
                model_stats = {
                    "model": model,
                    "total_explanations": len(evaluated_explanations),
                    "average_score": statistics.mean(scores),
                    "median_score": statistics.median(scores),
                    "score_std": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "average_word_count": statistics.mean(word_counts),
                    "explanations": evaluated_explanations
                }
                
                # Criteria breakdown
                criteria_scores = {}
                for criterion in ["relevance", "clarity", "conciseness", "usefulness", "accuracy", "uniqueness"]:
                    criterion_scores = [result["evaluation"][criterion] for result in evaluated_explanations if criterion in result["evaluation"]]
                    if criterion_scores:
                        criteria_scores[criterion] = statistics.mean(criterion_scores)
                
                model_stats["criteria_scores"] = criteria_scores
                model_results[model] = model_stats
                
                print(f"  ✅ {model}: Avg Score = {model_stats['average_score']:.2f}, Avg Length = {model_stats['average_word_count']:.1f} words")
            else:
                print(f"  ❌ {model}: No successful explanations generated")
                
        return {
            "experiment_type": "model_comparison",
            "timestamp": datetime.now().isoformat(),
            "models_tested": available_models,
            "test_cases_count": len(test_cases),
            "length_target": standard_length,
            "results": model_results
        }

    def run_length_optimization_experiment(self, best_model: str = None) -> Dict:
        """
        Run length optimization experiment using the best performing model
        or a specified model to find optimal explanation length.
        """
        
        if best_model is None:
            available_models = self.check_available_models()
            best_model = available_models[0] if available_models else "qwen2.5:7b"
            
        print(f"🧪 Starting length optimization experiment with model: {best_model}")
        
        # Use diverse test cases for length testing
        test_cases = [TEST_CASES[i] for i in [0, 1, 2, 4, 6]]  # Diverse subset
        
        length_results = {}
        
        for length_target in LENGTH_TARGETS:
            print(f"\n📏 Testing length target: {length_target['description']} ({length_target['min_words']}-{length_target['max_words']} words)")
            
            length_explanations = []
            
            for test_case in test_cases:
                explanation = self.generate_explanation(
                    model=best_model,
                    source_title=test_case["source_title"],
                    source_content=test_case["source_content"],
                    target_title=test_case["target_title"],
                    target_content=test_case["target_content"],
                    connection_type=test_case["connection_type"],
                    length_target=length_target
                )
                
                if explanation:
                    length_explanations.append({
                        "test_case_id": test_case["id"],
                        "source_title": test_case["source_title"],
                        "source_content": test_case["source_content"], 
                        "target_title": test_case["target_title"],
                        "target_content": test_case["target_content"],
                        "connection_type": test_case["connection_type"],
                        "explanation": explanation,
                        "model": best_model,
                        "length_target": length_target["description"],
                        "word_count": len(explanation.split()),
                        "target_range": f"{length_target['min_words']}-{length_target['max_words']}"
                    })
                    
                time.sleep(0.5)
            
            # Evaluate explanations for this length
            evaluated_explanations = self.judge.batch_evaluate(length_explanations)
            
            if evaluated_explanations:
                scores = [result["evaluation"]["overall_score"] for result in evaluated_explanations]
                word_counts = [result["word_count"] for result in evaluated_explanations]
                
                # Check length adherence
                in_range = sum(1 for wc in word_counts if length_target["min_words"] <= wc <= length_target["max_words"])
                adherence_rate = in_range / len(word_counts) if word_counts else 0
                
                length_stats = {
                    "length_target": length_target,
                    "average_score": statistics.mean(scores),
                    "average_word_count": statistics.mean(word_counts),
                    "length_adherence_rate": adherence_rate,
                    "explanations": evaluated_explanations
                }
                
                length_results[length_target["description"]] = length_stats
                print(f"  ✅ {length_target['description']}: Avg Score = {length_stats['average_score']:.2f}, "
                      f"Avg Length = {length_stats['average_word_count']:.1f}, "
                      f"Adherence = {adherence_rate:.1%}")
                      
        return {
            "experiment_type": "length_optimization", 
            "timestamp": datetime.now().isoformat(),
            "model_used": best_model,
            "test_cases_count": len(test_cases),
            "results": length_results
        }

    def save_results(self, results: Dict, filename: str = None):
        """Save experiment results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{results['experiment_type']}_{timestamp}.json"
            
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Results saved to: {filepath}")
        return filepath

    def generate_recommendations(self, model_results: Dict, length_results: Dict = None) -> Dict:
        """
        Generate data-driven recommendations based on experiment results.
        This is the key output that guides model selection.
        """
        
        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "recommendations": {}
        }
        
        if "results" in model_results:
            # Analyze model performance
            model_scores = []
            for model, stats in model_results["results"].items():
                model_scores.append((model, stats["average_score"], stats.get("criteria_scores", {})))
            
            # Sort by average score
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            best_model = model_scores[0]
            worst_model = model_scores[-1]
            
            recommendations["analysis"]["model_comparison"] = {
                "best_model": {
                    "name": best_model[0],
                    "average_score": best_model[1],
                    "criteria_breakdown": best_model[2]
                },
                "worst_model": {
                    "name": worst_model[0], 
                    "average_score": worst_model[1],
                    "criteria_breakdown": worst_model[2]
                },
                "performance_gap": best_model[1] - worst_model[1],
                "all_model_scores": [(model, score) for model, score, _ in model_scores]
            }
            
            # Generate model recommendation
            if recommendations["analysis"]["model_comparison"]["performance_gap"] > 1.0:
                recommendations["recommendations"]["model"] = {
                    "recommended": best_model[0],
                    "confidence": "high", 
                    "reasoning": f"{best_model[0]} significantly outperforms other models with {best_model[1]:.2f} average score vs {worst_model[1]:.2f} for the worst model."
                }
            elif recommendations["analysis"]["model_comparison"]["performance_gap"] > 0.5:
                recommendations["recommendations"]["model"] = {
                    "recommended": best_model[0],
                    "confidence": "medium",
                    "reasoning": f"{best_model[0]} shows moderate advantage with {best_model[1]:.2f} average score."
                }
            else:
                recommendations["recommendations"]["model"] = {
                    "recommended": best_model[0],
                    "confidence": "low", 
                    "reasoning": f"Models perform similarly. {best_model[0]} has slight edge but consider other factors like speed/cost."
                }
        
        if length_results and "results" in length_results:
            # Analyze length optimization
            length_scores = []
            for length_desc, stats in length_results["results"].items():
                length_scores.append((length_desc, stats["average_score"], stats["average_word_count"], stats["length_adherence_rate"]))
            
            length_scores.sort(key=lambda x: x[1], reverse=True)
            best_length = length_scores[0]
            
            recommendations["analysis"]["length_optimization"] = {
                "best_length": {
                    "description": best_length[0],
                    "average_score": best_length[1], 
                    "average_word_count": best_length[2],
                    "adherence_rate": best_length[3]
                },
                "all_length_scores": [(desc, score, wc) for desc, score, wc, _ in length_scores]
            }
            
            recommendations["recommendations"]["length"] = {
                "recommended": best_length[0],
                "reasoning": f"Length target '{best_length[0]}' achieved highest average score of {best_length[1]:.2f} with {best_length[2]:.1f} average words."
            }
        
        return recommendations

    def run_full_experiment_suite(self) -> Dict:
        """
        Run the complete experiment suite: model comparison + length optimization + recommendations.
        This is the main entry point for comprehensive testing.
        """
        
        print("🚀 Starting Full LLM Optimization Experiment Suite")
        print("=" * 60)
        
        # Step 1: Model Comparison
        print("\n📊 PHASE 1: Model Comparison")
        model_results = self.run_model_comparison_experiment()
        
        if "error" in model_results:
            return model_results
            
        model_filepath = self.save_results(model_results, "model_comparison_latest.json")
        
        # Step 2: Find best model for length optimization
        best_model = None
        if "results" in model_results:
            model_scores = [(model, stats["average_score"]) for model, stats in model_results["results"].items()]
            if model_scores:
                best_model = max(model_scores, key=lambda x: x[1])[0]
                print(f"🏆 Best performing model: {best_model}")
        
        # Step 3: Length Optimization  
        print(f"\n📏 PHASE 2: Length Optimization (using {best_model})")
        length_results = self.run_length_optimization_experiment(best_model)
        length_filepath = self.save_results(length_results, "length_optimization_latest.json")
        
        # Step 4: Generate Recommendations
        print("\n🎯 PHASE 3: Generating Recommendations")
        recommendations = self.generate_recommendations(model_results, length_results)
        rec_filepath = self.save_results(recommendations, "recommendations_latest.json")
        
        # Print summary
        self.print_experiment_summary(recommendations)
        
        return {
            "experiment_suite": "complete",
            "model_results": model_results,
            "length_results": length_results, 
            "recommendations": recommendations,
            "files": {
                "model_comparison": str(model_filepath),
                "length_optimization": str(length_filepath),
                "recommendations": str(rec_filepath)
            }
        }

    def print_experiment_summary(self, recommendations: Dict):
        """Print a clean summary of the experiment results and recommendations"""
        
        print("\n" + "="*60)
        print("🎉 EXPERIMENT SUITE COMPLETE - SUMMARY REPORT")
        print("="*60)
        
        if "model_comparison" in recommendations.get("analysis", {}):
            model_analysis = recommendations["analysis"]["model_comparison"]
            
            print(f"\n🏆 BEST MODEL: {model_analysis['best_model']['name']}")
            print(f"   Average Score: {model_analysis['best_model']['average_score']:.2f}/10")
            print(f"   Performance Gap: +{model_analysis['performance_gap']:.2f} vs worst model")
            
            print(f"\n📊 MODEL RANKINGS:")
            for i, (model, score) in enumerate(model_analysis['all_model_scores'][:5], 1):
                print(f"   {i}. {model}: {score:.2f}/10")
        
        if "length_optimization" in recommendations.get("analysis", {}):
            length_analysis = recommendations["analysis"]["length_optimization"]
            
            print(f"\n📏 OPTIMAL LENGTH: {length_analysis['best_length']['description']}")
            print(f"   Average Score: {length_analysis['best_length']['average_score']:.2f}/10")
            print(f"   Average Words: {length_analysis['best_length']['average_word_count']:.1f}")
            
        print(f"\n🎯 FINAL RECOMMENDATIONS:")
        if "model" in recommendations.get("recommendations", {}):
            model_rec = recommendations["recommendations"]["model"]
            print(f"   Model: {model_rec['recommended']} (confidence: {model_rec['confidence']})")
            print(f"   Reason: {model_rec['reasoning']}")
            
        if "length" in recommendations.get("recommendations", {}):
            length_rec = recommendations["recommendations"]["length"]
            print(f"   Length: {length_rec['recommended']}")
            print(f"   Reason: {length_rec['reasoning']}")
        
        print("\n💡 NEXT STEPS:")
        print("   1. Update cross_refs.py to use the recommended model")
        print("   2. Adjust prompt to target the optimal explanation length")
        print("   3. Test with a small batch of real cross-references")
        print("   4. Deploy to production if results look good")
        
        print("\n📁 Detailed results saved in scripts/llm_experiments/results/")
        print("="*60) 