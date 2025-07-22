#!/usr/bin/env python3
"""
Quality-Focused Model Comparison for Cross-Reference Explanations

Tests all 5 available Ollama models using LLM-as-judge evaluation
with different professional perspectives to assess explanation quality.
"""

import requests
import json
import time
from enhanced_design_space import AVAILABLE_MODELS

class RoleBasedJudge:
    """LLM judge that evaluates explanations from different professional perspectives."""
    
    def __init__(self, judge_model: str = "qwen2.5:7b", ollama_url: str = "http://localhost:11434"):
        self.judge_model = judge_model
        self.ollama_url = ollama_url
    
    def _make_request(self, prompt: str, max_retries: int = 3) -> str:
        """Make request to Ollama with retries."""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.judge_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temp for consistent evaluation
                            "top_p": 0.9,
                            "max_tokens": 300
                        }
                    },
                    timeout=45
                )
                
                if response.status_code == 200:
                    return response.json().get("response", "").strip()
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
        return ""
    
    def evaluate_as_student(self, explanation: str, source_title: str, target_title: str) -> dict:
        """Evaluate from a student's perspective - would this help me learn?"""
        
        prompt = f"""You are a COMPUTER SCIENCE STUDENT learning about Machine Learning Systems. You're reading "{source_title}" and see a cross-reference to "{target_title}" with this explanation:

EXPLANATION: "{explanation}"

As a student, evaluate this explanation on a 1-10 scale:

1. HELPFULNESS: Would this explanation help me decide if I should read the target section now?
2. CLARITY: Do I understand what this explanation means without confusion?
3. MOTIVATION: Does this make me actually want to click and read the target section?
4. LEARNING VALUE: Does this explanation teach me something useful about the connection?

Please respond in this exact JSON format:
{{
    "helpfulness": X,
    "clarity": X, 
    "motivation": X,
    "learning_value": X,
    "overall_score": X.X,
    "student_feedback": "Brief comment on what works/doesn't work for a learner"
}}"""

        response = self._make_request(prompt)
        return self._parse_evaluation(response, default_role="student")
    
    def evaluate_as_editor(self, explanation: str, source_title: str, target_title: str) -> dict:
        """Evaluate from a textbook editor's perspective - is this good writing?"""
        
        prompt = f"""You are a PROFESSIONAL TEXTBOOK EDITOR with 15 years of experience editing technical books. You're reviewing a cross-reference explanation:

SOURCE: "{source_title}"  
TARGET: "{target_title}"
EXPLANATION: "{explanation}"

As an editor, evaluate this explanation on a 1-10 scale:

1. WRITING QUALITY: Is this well-written, concise, and professional?
2. CONSISTENCY: Does this match the style of other educational cross-references?
3. PRECISION: Is the language precise and technically accurate?
4. ENGAGEMENT: Is this engaging without being too casual or too dry?

Please respond in this exact JSON format:
{{
    "writing_quality": X,
    "consistency": X,
    "precision": X, 
    "engagement": X,
    "overall_score": X.X,
    "editor_feedback": "Brief editorial comment on the writing quality"
}}"""

        response = self._make_request(prompt)
        return self._parse_evaluation(response, default_role="editor")
    
    def evaluate_as_educator(self, explanation: str, source_title: str, target_title: str) -> dict:
        """Evaluate from an ML educator's perspective - does this support learning?"""
        
        prompt = f"""You are a MACHINE LEARNING PROFESSOR who has taught ML systems for 10 years. You're reviewing a cross-reference in your course materials:

SOURCE: "{source_title}"
TARGET: "{target_title}"  
EXPLANATION: "{explanation}"

As an educator, evaluate this explanation on a 1-10 scale:

1. PEDAGOGICAL VALUE: Does this support the learning progression?
2. CONCEPTUAL ACCURACY: Is the technical relationship described correctly?
3. SCAFFOLDING: Does this help students build knowledge step-by-step?
4. TIMING: Is this the right level of detail for a cross-reference?

Please respond in this exact JSON format:
{{
    "pedagogical_value": X,
    "conceptual_accuracy": X,
    "scaffolding": X,
    "timing": X, 
    "overall_score": X.X,
    "educator_feedback": "Brief comment on the educational effectiveness"
}}"""

        response = self._make_request(prompt)
        return self._parse_evaluation(response, default_role="educator")
    
    def _parse_evaluation(self, response: str, default_role: str) -> dict:
        """Parse JSON evaluation response with fallback."""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                result['evaluation_success'] = True
                return result
        except:
            pass
        
        # Fallback scores
        return {
            'evaluation_success': False,
            'overall_score': 5.0,
            f'{default_role}_feedback': 'Evaluation failed - using default scores'
        }

def test_model_with_quality(model_name: str) -> dict:
    """Test a model and evaluate the quality of its responses."""
    
    # Test cases for model evaluation
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
    
    judge = RoleBasedJudge()
    model_results = []
    
    print(f"🔬 Testing {model_name}...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"  📝 Example {i}: ", end="", flush=True)
        
        # Generate explanation
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": test_case["prompt"],
                    "stream": False,
                    "options": {"temperature": 0.3, "top_p": 0.9, "max_tokens": 50}
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print("❌ Generation failed")
                continue
                
            explanation = response.json().get("response", "").strip()
            explanation = explanation.replace('"', '').replace("'", "").strip()
            
            if explanation and explanation[0].isupper() and not explanation.startswith(('AI', 'ML', 'GPU', 'CPU')):
                explanation = explanation[0].lower() + explanation[1:]
            
            word_count = len(explanation.split())
            print(f'"{explanation}" ({word_count} words)')
            
            # Evaluate with all three roles
            print(f"    🎓 Student eval: ", end="", flush=True)
            student_eval = judge.evaluate_as_student(explanation, test_case["source"], test_case["target"])
            print(f"{student_eval['overall_score']:.1f}/10")
            
            print(f"    📝 Editor eval: ", end="", flush=True)
            editor_eval = judge.evaluate_as_editor(explanation, test_case["source"], test_case["target"])
            print(f"{editor_eval['overall_score']:.1f}/10")
            
            print(f"    👨‍🏫 Educator eval: ", end="", flush=True)
            educator_eval = judge.evaluate_as_educator(explanation, test_case["source"], test_case["target"])
            print(f"{educator_eval['overall_score']:.1f}/10")
            
            result = {
                "explanation": explanation,
                "word_count": word_count,
                "student_evaluation": student_eval,
                "editor_evaluation": editor_eval,
                "educator_evaluation": educator_eval,
                "average_quality": (student_eval['overall_score'] + editor_eval['overall_score'] + educator_eval['overall_score']) / 3
            }
            
            model_results.append(result)
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}...")
    
    # Calculate model statistics
    if model_results:
        avg_quality = sum(r["average_quality"] for r in model_results) / len(model_results)
        avg_words = sum(r["word_count"] for r in model_results) / len(model_results)
        length_adherence = sum(1 for r in model_results if 4 <= r["word_count"] <= 7) / len(model_results)
        
        return {
            "model": model_name,
            "success_rate": len(model_results) / len(test_cases),
            "average_quality": avg_quality,
            "average_words": avg_words,
            "length_adherence": length_adherence,
            "results": model_results
        }
    else:
        return {"model": model_name, "success_rate": 0, "average_quality": 0, "average_words": 0, "length_adherence": 0, "results": []}

def run_quality_model_comparison():
    """Run comprehensive model comparison with quality evaluation."""
    
    print("🎯 QUALITY-FOCUSED MODEL COMPARISON")
    print("=" * 60)
    print("Testing models with Student 🎓 + Editor 📝 + Educator 👨‍🏫 evaluation")
    print()
    
    all_results = {}
    
    for model in AVAILABLE_MODELS:
        result = test_model_with_quality(model)
        all_results[model] = result
        
        if result["success_rate"] > 0:
            print(f"  📊 {model}: Quality={result['average_quality']:.1f}/10, Words={result['average_words']:.1f}, Adherence={result['length_adherence']:.0%}")
        else:
            print(f"  📊 {model}: ❌ Failed all tests")
        print()
    
    # Rank by combined score (quality + length adherence)
    successful_models = {k: v for k, v in all_results.items() if v["success_rate"] > 0}
    
    if successful_models:
        print("🏆 QUALITY RANKINGS")
        print("=" * 40)
        
        # Sort by quality first, then length adherence
        ranked = sorted(
            successful_models.items(),
            key=lambda x: (x[1]["average_quality"], x[1]["length_adherence"]),
            reverse=True
        )
        
        for rank, (model, stats) in enumerate(ranked, 1):
            print(f"{rank}. {model}")
            print(f"   🎯 Quality Score: {stats['average_quality']:.1f}/10")
            print(f"   📏 Length: {stats['average_words']:.1f} words ({stats['length_adherence']:.0%} adherence)")
            print(f"   ✅ Success Rate: {stats['success_rate']:.0%}")
            
            # Show role-specific feedback
            if stats["results"]:
                example = stats["results"][0]
                print(f"   🎓 Student: {example['student_evaluation'].get('student_feedback', 'N/A')}")
                print(f"   📝 Editor: {example['editor_evaluation'].get('editor_feedback', 'N/A')}")
                print(f"   👨‍🏫 Educator: {example['educator_evaluation'].get('educator_feedback', 'N/A')}")
            print()
        
        best_model = ranked[0][0]
        print(f"🎯 QUALITY WINNER: {best_model}")
        print(f"   Overall Quality: {ranked[0][1]['average_quality']:.1f}/10")
        
        # Save detailed results
        with open("results/quality_model_comparison.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print("💾 Detailed results saved to: results/quality_model_comparison.json")
        return best_model
    else:
        print("⚠️ No models succeeded - check Ollama setup")
        return None

if __name__ == "__main__":
    best_model = run_quality_model_comparison() 