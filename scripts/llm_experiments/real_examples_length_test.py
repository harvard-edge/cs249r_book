#!/usr/bin/env python3
"""
Real Examples Length Test - Using actual book cross-references

Tests length flexibility on REAL cross-reference examples from the ML Systems book
to see if loosening word count constraints improves explanation quality.
"""

import requests
import json
import time
import sys
from pathlib import Path
from quality_model_comparison import RoleBasedJudge

def load_real_cross_references():
    """Load real cross-reference examples from the book data."""
    
    # Path to cross-references JSON (relative to project root)
    json_path = Path("../../data/cross_refs.json")
    
    if not json_path.exists():
        print(f"❌ Could not find cross-references file at {json_path}")
        return []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract examples with diverse characteristics
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
    
    # Select diverse examples for testing
    selected = []
    
    # Get examples with different connection types and lengths
    preview_examples = [ex for ex in examples if ex['connection_type'] == 'Preview']
    background_examples = [ex for ex in examples if ex['connection_type'] == 'Background'] 
    
    # Take 2 Preview + 2 Background examples
    selected.extend(preview_examples[:2])
    selected.extend(background_examples[:2])
    
    return selected

def test_real_length_flexibility(model: str = "gemma2:9b"):
    """Test length flexibility using real book examples."""
    
    # Load real examples
    real_examples = load_real_cross_references()
    
    if not real_examples:
        print("❌ No real examples found - check data/cross_refs.json path")
        return None
    
    print("🧪 REAL EXAMPLES LENGTH FLEXIBILITY TEST")
    print("=" * 60)
    print(f"Testing model: {model}")
    print(f"Using {len(real_examples)} real cross-reference examples from the book")
    print()
    
    # Show what we're testing with
    print("📚 Test Examples:")
    for i, ex in enumerate(real_examples, 1):
        current_words = len(ex['current_explanation'].split())
        print(f"  {i}. {ex['source_title']} → {ex['target_title']}")
        print(f"     {ex['connection_type']} | Current: \"{ex['current_explanation']}\" ({current_words} words)")
    print()
    
    # Length configurations to test
    length_configs = [
        {
            "name": "current_restrictive",
            "range": "4-7 words",
            "instruction": "Write a natural 4-7 word explanation",
            "description": "Our current restrictive target"
        },
        {
            "name": "balanced_flexible", 
            "range": "6-12 words",
            "instruction": "Write a natural 6-12 word explanation that provides helpful context",
            "description": "Proposed balanced approach"
        },
        {
            "name": "quality_focused",
            "range": "8-15 words", 
            "instruction": "Write a natural 8-15 word explanation that clearly explains why this connection is valuable",
            "description": "Quality-first approach"
        },
        {
            "name": "unrestricted",
            "range": "flexible",
            "instruction": "Write a clear, helpful explanation that tells readers exactly why they should follow this connection (prioritize usefulness over brevity)",
            "description": "No length restrictions"
        }
    ]
    
    judge = RoleBasedJudge()
    results = {}
    
    for config in length_configs:
        print(f"📏 Testing {config['name']} ({config['range']})...")
        print(f"    {config['description']}")
        config_results = []
        
        for i, example in enumerate(real_examples, 1):
            print(f"  Example {i}: {example['source_title'][:30]}... → {example['target_title'][:30]}...")
            print(f"    Current: \"{example['current_explanation']}\"")
            print(f"    New: ", end="", flush=True)
            
            # Create prompt for this real example
            prompt = f"""{config['instruction']} that completes: "See also: {example['target_title']} - [your explanation]"

This is a {example['connection_type']} reference in a Machine Learning Systems textbook.

Source Section: "{example['source_title']}" - introduces foundational concepts and background.
Target Section: "{example['target_title']}" - covers related technical concepts and applications.

Current explanation: "{example['current_explanation']}"

Focus on WHY this connection matters for learning and what specific value it provides to readers.

Write ONLY a natural explanation phrase (no labels, no "Explanation:" prefix):"""

            # Generate new explanation
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
                            "max_tokens": 100  # More tokens for flexible lengths
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    new_explanation = response.json().get("response", "").strip()
                    new_explanation = new_explanation.replace('"', '').replace("'", "").strip()
                    
                    # Clean up explanation
                    if new_explanation and new_explanation[0].isupper() and not new_explanation.startswith(('AI', 'ML', 'GPU', 'CPU')):
                        new_explanation = new_explanation[0].lower() + new_explanation[1:]
                    
                    word_count = len(new_explanation.split())
                    print(f'"{new_explanation}" ({word_count} words)')
                    
                    # Quality evaluation with all three roles
                    print(f"    Evaluating quality...", end="", flush=True)
                    student_eval = judge.evaluate_as_student(new_explanation, example['source_title'], example['target_title'])
                    editor_eval = judge.evaluate_as_editor(new_explanation, example['source_title'], example['target_title'])  
                    educator_eval = judge.evaluate_as_educator(new_explanation, example['source_title'], example['target_title'])
                    
                    avg_quality = (student_eval['overall_score'] + editor_eval['overall_score'] + educator_eval['overall_score']) / 3
                    
                    config_results.append({
                        "example": example,
                        "new_explanation": new_explanation,
                        "current_explanation": example['current_explanation'],
                        "word_count": word_count,
                        "current_word_count": len(example['current_explanation'].split()),
                        "quality_score": avg_quality,
                        "student_score": student_eval['overall_score'],
                        "editor_score": editor_eval['overall_score'],
                        "educator_score": educator_eval['overall_score'],
                        "improvement_over_current": True  # Will calculate later
                    })
                    
                    print(f" Quality: {avg_quality:.1f}/10")
                else:
                    print("❌ Generation failed")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:40]}...")
        
        # Calculate averages for this configuration
        if config_results:
            avg_quality = sum(r["quality_score"] for r in config_results) / len(config_results)
            avg_words = sum(r["word_count"] for r in config_results) / len(config_results)
            avg_current_words = sum(r["current_word_count"] for r in config_results) / len(config_results)
            avg_student = sum(r["student_score"] for r in config_results) / len(config_results)
            avg_editor = sum(r["editor_score"] for r in config_results) / len(config_results)
            avg_educator = sum(r["educator_score"] for r in config_results) / len(config_results)
            
            results[config["name"]] = {
                "config": config,
                "avg_quality": avg_quality,
                "avg_words": avg_words,
                "avg_current_words": avg_current_words,
                "avg_student": avg_student,
                "avg_editor": avg_editor,
                "avg_educator": avg_educator,
                "examples": config_results
            }
            
            print(f"  📊 Average: {avg_quality:.1f}/10 quality, {avg_words:.1f} words (vs current {avg_current_words:.1f})")
        else:
            results[config["name"]] = {"config": config, "avg_quality": 0, "avg_words": 0}
            print("  📊 No successful results")
        
        print()
    
    # Compare results and rank by quality
    print("🏆 REAL EXAMPLES - LENGTH FLEXIBILITY RESULTS")
    print("=" * 50)
    
    successful_configs = {k: v for k, v in results.items() if v["avg_quality"] > 0}
    if successful_configs:
        ranked = sorted(successful_configs.items(), key=lambda x: x[1]["avg_quality"], reverse=True)
        
        for rank, (config_name, stats) in enumerate(ranked, 1):
            quality_improvement = stats["avg_quality"] - ranked[-1][1]["avg_quality"] if len(ranked) > 1 else 0
            length_change = stats["avg_words"] - stats["avg_current_words"]
            
            print(f"{rank}. {config_name.replace('_', ' ').title()}")
            print(f"   🎯 Quality: {stats['avg_quality']:.1f}/10 (+{quality_improvement:.1f} vs worst)")
            print(f"   📏 Length: {stats['avg_words']:.1f} words ({length_change:+.1f} vs current {stats['avg_current_words']:.1f})")
            print(f"   🎓 Student: {stats['avg_student']:.1f}/10")
            print(f"   📝 Editor: {stats['avg_editor']:.1f}/10") 
            print(f"   👨‍🏫 Educator: {stats['avg_educator']:.1f}/10")
            print(f"   📄 Range: {stats['config']['range']}")
            
            if stats["examples"]:
                best_example = max(stats["examples"], key=lambda x: x["quality_score"])
                print(f"   💡 Best: \"{best_example['new_explanation']}\" ({best_example['quality_score']:.1f}/10)")
            print()
        
        best_config = ranked[0]
        quality_gain = best_config[1]["avg_quality"] - ranked[-1][1]["avg_quality"] if len(ranked) > 1 else 0
        
        print(f"🎯 WINNER: {best_config[0].replace('_', ' ').title()}")
        print(f"   Quality improvement: +{quality_gain:.1f} points over worst approach")
        print(f"   Recommended instruction: \"{best_config[1]['config']['instruction']}\"")
        
        # Save detailed results
        with open("results/real_examples_length_test.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("💾 Results saved to: results/real_examples_length_test.json")
        
        return best_config[1]["config"]
    else:
        print("⚠️ No configurations succeeded")
        return None

if __name__ == "__main__":
    best_config = test_real_length_flexibility()
    if best_config:
        print(f"\n🚀 PRODUCTION RECOMMENDATION:")
        print(f"   Use: '{best_config['instruction']}'")
        print(f"   Range: {best_config['range']}")
    else:
        print("⚠️ Experiment failed - check Ollama setup") 