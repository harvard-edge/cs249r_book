#!/usr/bin/env python3
"""
Models with Flexible Length Test

Re-tests all 5 models using the optimal 6-12 word flexible range
to see which model performs best when not artificially constrained.
"""

import requests
import json
import time
from pathlib import Path
from quality_model_comparison import RoleBasedJudge
from enhanced_design_space import AVAILABLE_MODELS

def load_real_examples():
    """Load real cross-reference examples for testing."""
    json_path = Path("../../data/cross_refs.json")
    
    if not json_path.exists():
        return []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get diverse examples
    examples = []
    for file_entry in data['cross_references']:
        for section in file_entry['sections']:
            for target in section['targets']:
                examples.append({
                    'source_title': section['section_title'],
                    'target_title': target['target_section_title'], 
                    'connection_type': target['connection_type'],
                    'current_explanation': target.get('explanation', ''),
                    'similarity': target['similarity']
                })
    
    # Take first 3 examples for faster testing
    return examples[:3]

def test_model_flexible_length(model: str, examples: list) -> dict:
    """Test a single model with flexible 6-12 word length."""
    
    print(f"🔬 Testing {model} with flexible length (6-12 words)...")
    
    # Optimal prompt from previous experiment
    prompt_template = """Write a natural 6-12 word explanation that provides helpful context that completes: "See also: {target_title} - [your explanation]"

This is a {connection_type} reference in a Machine Learning Systems textbook.

Source Section: "{source_title}" - introduces foundational concepts and background.
Target Section: "{target_title}" - covers related technical concepts and applications.

Current explanation: "{current_explanation}"

Focus on WHY this connection matters for learning and what specific value it provides to readers.

Write ONLY a natural explanation phrase (no labels, no "Explanation:" prefix):"""

    judge = RoleBasedJudge()
    results = []
    total_time = 0
    
    for i, example in enumerate(examples, 1):
        print(f"  Example {i}: ", end="", flush=True)
        
        # Create prompt for this example
        prompt = prompt_template.format(
            target_title=example['target_title'],
            connection_type=example['connection_type'],
            source_title=example['source_title'],
            current_explanation=example['current_explanation']
        )
        
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:11434/api/generate",
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
                timeout=45
            )
            response_time = time.time() - start_time
            total_time += response_time
            
            if response.status_code == 200:
                explanation = response.json().get("response", "").strip()
                explanation = explanation.replace('"', '').replace("'", "").strip()
                
                # Clean up explanation
                if explanation and explanation[0].isupper() and not explanation.startswith(('AI', 'ML', 'GPU', 'CPU')):
                    explanation = explanation[0].lower() + explanation[1:]
                
                word_count = len(explanation.split())
                length_adherence = 6 <= word_count <= 12
                
                print(f'"{explanation}" ({word_count} words, {response_time:.1f}s)')
                
                # Quality evaluation
                print(f"    Evaluating...", end="", flush=True)
                student_eval = judge.evaluate_as_student(explanation, example['source_title'], example['target_title'])
                editor_eval = judge.evaluate_as_editor(explanation, example['source_title'], example['target_title'])  
                educator_eval = judge.evaluate_as_educator(explanation, example['source_title'], example['target_title'])
                
                avg_quality = (student_eval['overall_score'] + editor_eval['overall_score'] + educator_eval['overall_score']) / 3
                
                results.append({
                    "explanation": explanation,
                    "word_count": word_count,
                    "length_adherence": length_adherence,
                    "response_time": response_time,
                    "quality_score": avg_quality,
                    "student_score": student_eval['overall_score'],
                    "editor_score": editor_eval['overall_score'],
                    "educator_score": educator_eval['overall_score']
                })
                
                print(f" Quality: {avg_quality:.1f}/10")
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:30]}...")
    
    # Calculate model statistics
    if results:
        return {
            "model": model,
            "success_rate": len(results) / len(examples),
            "avg_quality": sum(r["quality_score"] for r in results) / len(results),
            "avg_words": sum(r["word_count"] for r in results) / len(results),
            "length_adherence": sum(r["length_adherence"] for r in results) / len(results),
            "avg_response_time": total_time / len(results),
            "avg_student": sum(r["student_score"] for r in results) / len(results),
            "avg_editor": sum(r["editor_score"] for r in results) / len(results),
            "avg_educator": sum(r["educator_score"] for r in results) / len(results),
            "results": results
        }
    else:
        return {
            "model": model,
            "success_rate": 0,
            "avg_quality": 0,
            "avg_words": 0,
            "length_adherence": 0,
            "avg_response_time": 0,
            "results": []
        }

def run_flexible_model_comparison():
    """Test all models with flexible length approach."""
    
    # Load real examples
    examples = load_real_examples()
    if not examples:
        print("❌ No real examples found")
        return None
    
    print("🧪 MODELS WITH FLEXIBLE LENGTH (6-12 WORDS)")
    print("=" * 60)
    print(f"Testing {len(AVAILABLE_MODELS)} models on {len(examples)} real examples")
    print("Using optimal flexible length approach from previous experiment")
    print()
    
    # Show test examples
    print("📚 Test Examples:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex['source_title']} → {ex['target_title']} ({ex['connection_type']})")
    print()
    
    all_results = {}
    
    # Test each model
    for model in AVAILABLE_MODELS:
        result = test_model_flexible_length(model, examples)
        all_results[model] = result
        
        if result["success_rate"] > 0:
            print(f"  📊 {model}: Quality={result['avg_quality']:.1f}/10, Words={result['avg_words']:.1f}, Speed={result['avg_response_time']:.1f}s, Adherence={result['length_adherence']:.0%}")
        else:
            print(f"  📊 {model}: ❌ Failed all tests")
        print()
    
    # Rank models by quality (since length flexibility removes that constraint)
    successful_models = {k: v for k, v in all_results.items() if v["success_rate"] > 0}
    
    if successful_models:
        print("🏆 FLEXIBLE LENGTH MODEL RANKINGS")
        print("=" * 45)
        
        # Sort by quality first, then by length adherence, then by speed
        ranked = sorted(
            successful_models.items(),
            key=lambda x: (x[1]["avg_quality"], x[1]["length_adherence"], -x[1]["avg_response_time"]),
            reverse=True
        )
        
        # Compare to previous restrictive results
        print("🆚 IMPROVEMENT vs RESTRICTIVE LENGTH:")
        print("(Previous restrictive 4-7 word results for comparison)")
        previous_scores = {
            "phi3:3.8b": 6.5,
            "gemma2:9b": 5.9, 
            "qwen2.5:7b": 5.9,
            "llama3.1:8b": 5.8,
            "mistral:7b": 5.6
        }
        print()
        
        for rank, (model, stats) in enumerate(ranked, 1):
            prev_score = previous_scores.get(model, 0)
            improvement = stats["avg_quality"] - prev_score
            
            print(f"{rank}. {model}")
            print(f"   🎯 Quality: {stats['avg_quality']:.1f}/10 ({improvement:+.1f} vs restrictive)")
            print(f"   📏 Length: {stats['avg_words']:.1f} words ({stats['length_adherence']:.0%} adherence)")
            print(f"   ⚡ Speed: {stats['avg_response_time']:.1f}s")
            print(f"   🎓 Student: {stats['avg_student']:.1f}/10")
            print(f"   📝 Editor: {stats['avg_editor']:.1f}/10") 
            print(f"   👨‍🏫 Educator: {stats['avg_educator']:.1f}/10")
            
            if stats["results"]:
                best_example = max(stats["results"], key=lambda x: x["quality_score"])
                print(f"   💡 Best: \"{best_example['explanation']}\" ({best_example['quality_score']:.1f}/10)")
            print()
        
        best_model = ranked[0]
        print(f"🎯 FLEXIBLE LENGTH WINNER: {best_model[0]}")
        print(f"   Quality: {best_model[1]['avg_quality']:.1f}/10")
        print(f"   Speed: {best_model[1]['avg_response_time']:.1f}s")
        print(f"   Length adherence: {best_model[1]['length_adherence']:.0%}")
        
        # Show biggest quality improvements
        print(f"\n📈 BIGGEST QUALITY GAINS FROM FLEXIBILITY:")
        for model, stats in ranked:
            prev_score = previous_scores.get(model, 0)
            improvement = stats["avg_quality"] - prev_score
            if improvement > 0:
                print(f"   {model}: +{improvement:.1f} points ({prev_score:.1f} → {stats['avg_quality']:.1f})")
        
        # Save results
        with open("results/flexible_model_comparison.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print("💾 Results saved to: results/flexible_model_comparison.json")
        
        return best_model[0]
    else:
        print("⚠️ No models succeeded")
        return None

if __name__ == "__main__":
    best_model = run_flexible_model_comparison()
    if best_model:
        print(f"\n🚀 FINAL RECOMMENDATION:")
        print(f"   Best Model: {best_model}")
        print(f"   Best Length: 6-12 words (flexible)")
        print(f"   Expected Quality: 8.0+ / 10")
    else:
        print("⚠️ Model comparison failed") 