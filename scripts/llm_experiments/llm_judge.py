# LLM-as-Judge Evaluation System for Cross-Reference Explanations
#
# This module implements a sophisticated evaluation system using a powerful LLM
# to judge the quality of cross-reference explanations across multiple criteria.

import json
import requests
import time
from typing import Dict, List, Tuple, Optional
from test_cases import EVALUATION_CRITERIA, TEST_CASES

class LLMJudge:
    """
    LLM-based evaluation system for cross-reference explanations.
    Uses a powerful model (GPT-4 via Ollama or similar) to score explanations.
    """
    
    def __init__(self, judge_model: str = "qwen2.5:32b", ollama_url: str = "http://localhost:11434"):
        self.judge_model = judge_model
        self.ollama_url = ollama_url
        self.evaluation_history = []
        
    def _make_ollama_request(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Make a request to Ollama with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.judge_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for consistent evaluation
                            "top_p": 0.9,
                            "max_tokens": 1000
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    return response.json().get("response", "").strip()
                else:
                    print(f"Request failed with status {response.status_code}")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return None

    def evaluate_explanation(self, 
                           source_title: str,
                           source_content: str, 
                           target_title: str,
                           target_content: str,
                           explanation: str,
                           connection_type: str = "Preview") -> Dict:
        """
        Evaluate a single explanation across all criteria.
        Returns scores (1-10) for each evaluation criterion.
        """
        
        evaluation_prompt = f"""You are an expert evaluator of educational cross-references in a Machine Learning Systems textbook. 

TASK: Evaluate the quality of a cross-reference explanation across multiple criteria.

SOURCE SECTION: "{source_title}"
Content: {source_content[:500]}...

TARGET SECTION: "{target_title}" 
Content: {target_content[:500]}...

CONNECTION TYPE: {connection_type} ({"Preview = forward reference to later material" if connection_type == "Preview" else "Background = reference to earlier foundational material"})

EXPLANATION TO EVALUATE: "{explanation}"

EVALUATION CRITERIA:
1. RELEVANCE (1-10): How well does the explanation capture the actual relationship between the sections?
2. CLARITY (1-10): Is the explanation clear and easy to understand for students?
3. CONCISENESS (1-10): Is the explanation appropriately concise without being too brief or verbose?
4. USEFULNESS (1-10): Would this explanation actually help a student decide to follow the cross-reference?
5. ACCURACY (1-10): Is the explanation factually correct about the content domains?
6. UNIQUENESS (1-10): Does the explanation add value beyond just restating the section titles?

SCORING GUIDELINES:
- 1-3: Poor quality, significant issues
- 4-6: Average quality, some issues  
- 7-8: Good quality, minor issues
- 9-10: Excellent quality, exemplary

Please provide your evaluation in this EXACT JSON format:
{{
    "relevance": X,
    "clarity": X, 
    "conciseness": X,
    "usefulness": X,
    "accuracy": X,
    "uniqueness": X,
    "overall_score": X.X,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"], 
    "reasoning": "Brief explanation of your scoring decisions"
}}

EVALUATION:"""

        response = self._make_ollama_request(evaluation_prompt)
        
        if not response:
            print(f"Failed to get evaluation for explanation: {explanation}")
            return self._get_default_scores()
            
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                print(f"No JSON found in response: {response[:200]}...")
                return self._get_default_scores()
                
            json_str = response[json_start:json_end]
            evaluation = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["relevance", "clarity", "conciseness", "usefulness", "accuracy", "uniqueness"]
            for field in required_fields:
                if field not in evaluation:
                    evaluation[field] = 5  # Default score
                    
            # Calculate overall score if missing
            if "overall_score" not in evaluation:
                scores = [evaluation[field] for field in required_fields]
                evaluation["overall_score"] = sum(scores) / len(scores)
                
            # Store evaluation history
            self.evaluation_history.append({
                "explanation": explanation,
                "evaluation": evaluation,
                "timestamp": time.time()
            })
            
            return evaluation
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON evaluation: {e}")
            print(f"Response was: {response}")
            return self._get_default_scores()
    
    def _get_default_scores(self) -> Dict:
        """Return default scores when evaluation fails"""
        return {
            "relevance": 5,
            "clarity": 5,
            "conciseness": 5, 
            "usefulness": 5,
            "accuracy": 5,
            "uniqueness": 5,
            "overall_score": 5.0,
            "strengths": ["evaluation_failed"],
            "weaknesses": ["evaluation_failed"],
            "reasoning": "Evaluation failed - using default scores"
        }

    def batch_evaluate(self, explanations: List[Dict]) -> List[Dict]:
        """
        Evaluate multiple explanations and return comprehensive results.
        
        Args:
            explanations: List of dicts with keys: source_title, source_content, 
                         target_title, target_content, explanation, connection_type
        """
        results = []
        
        for i, exp_data in enumerate(explanations):
            print(f"Evaluating explanation {i+1}/{len(explanations)}: {exp_data['explanation'][:50]}...")
            
            evaluation = self.evaluate_explanation(
                exp_data["source_title"],
                exp_data["source_content"], 
                exp_data["target_title"],
                exp_data["target_content"],
                exp_data["explanation"],
                exp_data.get("connection_type", "Preview")
            )
            
            result = {
                **exp_data,
                "evaluation": evaluation,
                "word_count": len(exp_data["explanation"].split())
            }
            
            results.append(result)
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
            
        return results

    def compare_explanations(self, explanations: List[str], context: Dict) -> Dict:
        """
        Direct comparison of multiple explanations for the same cross-reference.
        Returns ranking and relative scores.
        """
        
        comparison_prompt = f"""You are evaluating multiple cross-reference explanations for the same connection in a Machine Learning Systems textbook.

SOURCE: "{context['source_title']}" 
TARGET: "{context['target_title']}"
CONNECTION: {context['connection_type']}

EXPLANATIONS TO COMPARE:
"""
        
        for i, explanation in enumerate(explanations, 1):
            comparison_prompt += f"{i}. \"{explanation}\"\n"
            
        comparison_prompt += f"""
TASK: Rank these explanations from BEST to WORST and provide scores.

Consider:
- Which explanation most accurately captures the relationship?
- Which would be most helpful to a student?
- Which is the right length and clarity level?
- Which adds the most value beyond the section titles?

Provide your analysis in this EXACT JSON format:
{{
    "ranking": [1, 3, 2],
    "scores": [8.5, 6.2, 7.1],
    "best_explanation": "explanation text",
    "worst_explanation": "explanation text", 
    "reasoning": "Why you ranked them this way"
}}

COMPARISON:"""

        response = self._make_ollama_request(comparison_prompt)
        
        if not response:
            return {"error": "Failed to get comparison"}
            
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except:
            return {"error": "Failed to parse comparison"}

    def get_evaluation_summary(self) -> Dict:
        """Get summary statistics of all evaluations performed"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
            
        all_scores = []
        criteria_scores = {criterion: [] for criterion in EVALUATION_CRITERIA.keys()}
        
        for eval_record in self.evaluation_history:
            evaluation = eval_record["evaluation"]
            all_scores.append(evaluation["overall_score"])
            
            for criterion in criteria_scores.keys():
                if criterion in evaluation:
                    criteria_scores[criterion].append(evaluation[criterion])
        
        summary = {
            "total_evaluations": len(self.evaluation_history),
            "average_score": sum(all_scores) / len(all_scores),
            "score_range": {"min": min(all_scores), "max": max(all_scores)},
            "criteria_averages": {}
        }
        
        for criterion, scores in criteria_scores.items():
            if scores:
                summary["criteria_averages"][criterion] = sum(scores) / len(scores)
                
        return summary 