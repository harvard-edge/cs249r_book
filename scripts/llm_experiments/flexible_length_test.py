#!/usr/bin/env python3
"""
Flexible Length Test - Does giving more room improve explanation quality?

Compares restrictive vs flexible length guidance to see impact on quality scores.
"""

import requests
import json
import time
from quality_model_comparison import RoleBasedJudge

def test_length_flexibility(model: str = "gemma2:9b"):
    """Test how length flexibility affects explanation quality."""
    
    # Test cases
    test_cases = [
        {
            "source": "Introduction to Neural Networks",
            "target": "Backpropagation Algorithm",
            "context": "Neural networks → Learning algorithm"
        },
        {
            "source": "Deep Learning Fundamentals", 
            "target": "GPU Acceleration",
            "context": "Training concepts → Hardware optimization"
        }
    ]
    
    # Length configurations to test
    length_configs = [
        {
            "name": "restrictive_current",
            "range": "4-7 words",
            "instruction": "Write a natural 4-7 word explanation"
        },
        {
            "name": "balanced_proposed", 
            "range": "6-12 words",
            "instruction": "Write a natural 6-12 word explanation"
        },
        {
            "name": "flexible_quality",
            "range": "8-15 words", 
            "instruction": "Write a natural 8-15 word explanation that provides helpful context"
        },
        {
            "name": "quality_first",
            "range": "flexible",
            "instruction": "Write a helpful, natural explanation (aim for clarity and usefulness over strict length)"
        }
    ]
    
    judge = RoleBasedJudge()
    results = {}
    
    print("🧪 FLEXIBLE LENGTH QUALITY TEST")
    print("=" * 50)
    print(f"Testing model: {model}")
    print()
    
    for config in length_configs:
        print(f"📏 Testing {config['name']} ({config['range']})...")
        config_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"  Example {i}: ", end="", flush=True)
            
            # Create prompt with current length instruction
            prompt = f"""{config['instruction']} that completes: "See also: {test_case['target']} - [your explanation]"

Source: {test_case['source']} - covers basic concepts and foundational knowledge.
Target: {test_case['target']} - explains advanced concepts and techniques.

Use varied, engaging language. Focus on WHY the connection matters and what value it provides to readers.

Write ONLY a natural explanation phrase (no labels, no "Explanation:" prefix):"""

            # Generate explanation
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "max_tokens": 80  # More tokens for longer explanations
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    explanation = response.json().get("response", "").strip()
                    explanation = explanation.replace('"', '').replace("'", "").strip()
                    
                    if explanation and explanation[0].isupper() and not explanation.startswith(('AI', 'ML', 'GPU', 'CPU')):
                        explanation = explanation[0].lower() + explanation[1:]
                    
                    word_count = len(explanation.split())
                    print(f'"{explanation}" ({word_count} words)')
                    
                    # Quick quality evaluation (just overall score for speed)
                    student_eval = judge.evaluate_as_student(explanation, test_case["source"], test_case["target"])
                    editor_eval = judge.evaluate_as_editor(explanation, test_case["source"], test_case["target"])  
                    educator_eval = judge.evaluate_as_educator(explanation, test_case["source"], test_case["target"])
                    
                    avg_quality = (student_eval['overall_score'] + editor_eval['overall_score'] + educator_eval['overall_score']) / 3
                    
                    config_results.append({
                        "explanation": explanation,
                        "word_count": word_count,
                        "quality_score": avg_quality,
                        "student_score": student_eval['overall_score'],
                        "editor_score": editor_eval['overall_score'],
                        "educator_score": educator_eval['overall_score']
                    })
                    
                    print(f"    Quality: {avg_quality:.1f}/10 (S:{student_eval['overall_score']:.1f} E:{editor_eval['overall_score']:.1f} Ed:{educator_eval['overall_score']:.1f})")
                else:
                    print("❌ Generation failed")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:30]}...")
        
        # Calculate averages for this configuration
        if config_results:
            avg_quality = sum(r["quality_score"] for r in config_results) / len(config_results)
            avg_words = sum(r["word_count"] for r in config_results) / len(config_results)
            avg_student = sum(r["student_score"] for r in config_results) / len(config_results)
            avg_editor = sum(r["editor_score"] for r in config_results) / len(config_results)
            avg_educator = sum(r["educator_score"] for r in config_results) / len(config_results)
            
            results[config["name"]] = {
                "config": config,
                "avg_quality": avg_quality,
                "avg_words": avg_words,
                "avg_student": avg_student,
                "avg_editor": avg_editor,
                "avg_educator": avg_educator,
                "examples": config_results
            }
            
            print(f"  📊 Average: {avg_quality:.1f}/10 quality, {avg_words:.1f} words")
        else:
            results[config["name"]] = {"config": config, "avg_quality": 0, "avg_words": 0}
            print("  📊 No successful results")
        
        print()
    
    # Compare results
    print("🏆 LENGTH FLEXIBILITY RESULTS")
    print("=" * 40)
    
    successful_configs = {k: v for k, v in results.items() if v["avg_quality"] > 0}
    if successful_configs:
        ranked = sorted(successful_configs.items(), key=lambda x: x[1]["avg_quality"], reverse=True)
        
        for rank, (config_name, stats) in enumerate(ranked, 1):
            quality_improvement = stats["avg_quality"] - ranked[-1][1]["avg_quality"] if len(ranked) > 1 else 0
            
            print(f"{rank}. {config_name.replace('_', ' ').title()}")
            print(f"   🎯 Quality: {stats['avg_quality']:.1f}/10 (+{quality_improvement:.1f} vs worst)")
            print(f"   📏 Length: {stats['avg_words']:.1f} words ({stats['config']['range']})")
            print(f"   🎓 Student: {stats['avg_student']:.1f}/10")
            print(f"   📝 Editor: {stats['avg_editor']:.1f}/10") 
            print(f"   👨‍🏫 Educator: {stats['avg_educator']:.1f}/10")
            
            if stats["examples"]:
                print(f"   💡 Example: \"{stats['examples'][0]['explanation']}\"")
            print()
        
        best_config = ranked[0]
        quality_gain = best_config[1]["avg_quality"] - ranked[-1][1]["avg_quality"] if len(ranked) > 1 else 0
        
        print(f"🎯 WINNER: {best_config[0].replace('_', ' ').title()}")
        print(f"   Quality improvement: +{quality_gain:.1f} points")
        print(f"   Recommended range: {best_config[1]['config']['range']}")
        
        # Save results
        with open("results/flexible_length_test.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("💾 Results saved to: results/flexible_length_test.json")
        
        return best_config[1]["config"]
    else:
        print("⚠️ No configurations succeeded")
        return None

if __name__ == "__main__":
    best_config = test_length_flexibility()
    if best_config:
        print(f"\n🚀 RECOMMENDATION: Use '{best_config['instruction']}' for better quality") 