#!/usr/bin/env python3
"""
Quick Model Comparison for Cross-Reference Explanations

Tests all 5 available Ollama models on a small set of examples
to identify the best performer before running larger experiments.
"""

import requests
import json
import time
from enhanced_design_space import AVAILABLE_MODELS

def test_model(model_name: str, test_prompt: str, max_retries: int = 3) -> dict:
    """Test a single model with a prompt and return results."""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 50
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                explanation = response.json().get("response", "").strip()
                
                # Basic cleanup
                explanation = explanation.replace('"', '').replace("'", "").strip()
                if explanation and explanation[0].isupper() and not explanation.startswith(('AI', 'ML', 'GPU', 'CPU')):
                    explanation = explanation[0].lower() + explanation[1:]
                
                word_count = len(explanation.split())
                
                return {
                    "success": True,
                    "explanation": explanation,
                    "word_count": word_count,
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                print(f"   ❌ HTTP {response.status_code} on attempt {attempt + 1}")
                
        except Exception as e:
            print(f"   ⚠️ Error on attempt {attempt + 1}: {str(e)[:50]}...")
            time.sleep(1)
    
    return {"success": False, "explanation": "", "word_count": 0, "response_time": 0}

def run_quick_model_comparison():
    """Run a quick comparison of all available models."""
    
    # Quick test cases 
    test_cases = [
        {
            "source": "Introduction to Neural Networks",
            "target": "Backpropagation Algorithm", 
            "prompt": """Write a natural 4-7 word explanation that completes: "See also: Backpropagation Algorithm - [your explanation]"

Source: Introduction to Neural Networks - covers basic concepts of artificial neurons and network architectures.
Target: Backpropagation Algorithm - explains how neural networks learn through gradient-based optimization.

Write ONLY a natural explanation phrase (no labels, no "Explanation:" prefix):"""
        },
        {
            "source": "Deep Learning Fundamentals", 
            "target": "GPU Acceleration",
            "prompt": """Write a natural 4-7 word explanation that completes: "See also: GPU Acceleration - [your explanation]"

Source: Deep Learning Fundamentals - introduces core concepts of deep neural networks and training.
Target: GPU Acceleration - explains how graphics processors speed up neural network computations.

Write ONLY a natural explanation phrase (no labels, no "Explanation:" prefix):"""
        }
    ]
    
    print("🚀 QUICK MODEL COMPARISON")
    print("=" * 50)
    print(f"Testing {len(AVAILABLE_MODELS)} models on {len(test_cases)} examples...")
    print()
    
    results = {}
    
    for model in AVAILABLE_MODELS:
        print(f"🔬 Testing {model}...")
        model_results = []
        total_time = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"  Example {i}/{len(test_cases)}: ", end="", flush=True)
            
            result = test_model(model, test_case["prompt"])
            
            if result["success"]:
                print(f"✅ \"{result['explanation']}\" ({result['word_count']} words)")
                model_results.append(result)
                total_time += result["response_time"]
            else:
                print("❌ Failed")
        
        # Calculate stats
        if model_results:
            avg_words = sum(r["word_count"] for r in model_results) / len(model_results)
            avg_time = total_time / len(model_results)
            success_rate = len(model_results) / len(test_cases)
            
            results[model] = {
                "success_rate": success_rate,
                "avg_words": avg_words,
                "avg_time": avg_time,
                "examples": [r["explanation"] for r in model_results]
            }
            
            print(f"  📊 Success: {success_rate:.0%}, Avg: {avg_words:.1f} words, Time: {avg_time:.1f}s")
        else:
            results[model] = {"success_rate": 0, "avg_words": 0, "avg_time": 0, "examples": []}
            print("  📊 All tests failed")
            
        print()
    
    # Summary
    print("🏆 SUMMARY RANKINGS")
    print("=" * 30)
    
    # Sort by success rate, then by word count closeness to target (5.5 words)
    target_words = 5.5
    sorted_models = sorted(
        results.items(), 
        key=lambda x: (x[1]["success_rate"], -abs(x[1]["avg_words"] - target_words)),
        reverse=True
    )
    
    for rank, (model, stats) in enumerate(sorted_models, 1):
        if stats["success_rate"] > 0:
            print(f"{rank}. {model}")
            print(f"   Success: {stats['success_rate']:.0%}")
            print(f"   Words: {stats['avg_words']:.1f} (target: 4-7)")
            print(f"   Speed: {stats['avg_time']:.1f}s")
            print(f"   Examples: {stats['examples']}")
        else:
            print(f"{rank}. {model} - ❌ Failed all tests")
        print()
    
    # Save results
    with open("results/quick_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("💾 Results saved to: results/quick_model_comparison.json")
    
    return sorted_models[0][0] if sorted_models[0][1]["success_rate"] > 0 else None

if __name__ == "__main__":
    best_model = run_quick_model_comparison()
    if best_model:
        print(f"🎯 RECOMMENDATION: Use {best_model} for cross-reference generation")
    else:
        print("⚠️ No models succeeded - check Ollama setup") 