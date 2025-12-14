#!/usr/bin/env python3
"""
LLM-Enhanced Cross-Reference Generator
======================================

Enhances the concept-driven cross-references with LLM analysis for better explanations
and connection discovery. Uses local Ollama for privacy and no API costs.
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
import sys

class LLMEnhancer:
    def __init__(self, model: str = "gemma2:27b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.available = self.check_availability()

    def check_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                return self.model in models
        except:
            pass
        return False

    def enhance_explanation(self, source_chapter: str, target_chapter: str,
                           connection_type: str, concepts: List[str]) -> str:
        """Use LLM to generate better explanations for connections."""
        if not self.available:
            return f"Related through {connection_type}: {', '.join(concepts[:3])}"

        prompt = f"""You are an expert in machine learning systems education.

Given a connection between two textbook chapters:
- Source chapter: {source_chapter}
- Target chapter: {target_chapter}
- Connection type: {connection_type}
- Shared concepts: {', '.join(concepts)}

Write a concise (1-2 sentences) explanation of why a student reading the source chapter would benefit from the cross-reference to the target chapter. Focus on the educational value and learning progression.

Explanation:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 100
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                # Clean up the response
                result = result.replace('\n', ' ').replace('  ', ' ')
                if result and len(result) > 20:
                    return result

        except Exception as e:
            print(f"LLM error for {source_chapter} -> {target_chapter}: {e}")

        # Fallback
        return f"Related through {connection_type}: {', '.join(concepts[:3])}"

    def discover_additional_connections(self, source_concepts: Dict, target_concepts: Dict,
                                      source_chapter: str, target_chapter: str) -> Optional[Dict]:
        """Use LLM to discover additional conceptual connections."""
        if not self.available:
            return None

        source_text = f"Primary: {', '.join(source_concepts.get('primary_concepts', []))}"
        target_text = f"Primary: {', '.join(target_concepts.get('primary_concepts', []))}"

        prompt = f"""You are analyzing connections between ML textbook chapters.

Chapter A ({source_chapter}): {source_text}
Chapter B ({target_chapter}): {target_text}

Are there important educational connections between these chapters beyond obvious keyword matches? Consider:
- Learning prerequisites (A needed before B)
- Implementation relationships (A theory, B practice)
- Complementary perspectives on same problems

Respond with JSON only:
{{"has_connection": true/false, "connection_type": "prerequisite/implementation/complementary/none", "strength": 0.0-1.0, "concepts": ["concept1", "concept2"], "explanation": "brief explanation"}}

If no strong educational connection exists, respond: {{"has_connection": false}}

JSON:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.8,
                        "num_predict": 150
                    }
                },
                timeout=45
            )

            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                try:
                    # Extract JSON from response
                    start = result.find('{')
                    end = result.rfind('}') + 1
                    if start != -1 and end != -1:
                        json_str = result[start:end]
                        data = json.loads(json_str)
                        if data.get('has_connection', False) and data.get('strength', 0) > 0.3:
                            return data
                except:
                    pass

        except Exception as e:
            print(f"LLM discovery error for {source_chapter} -> {target_chapter}: {e}")

        return None

def enhance_xrefs_with_llm(chapters_dir: Path, model: str = "llama3.1:8b") -> None:
    """Enhance existing cross-references with LLM analysis."""
    enhancer = LLMEnhancer(model)

    if not enhancer.available:
        print(f"❌ LLM model {model} not available. Please run: ollama run {model}")
        return

    print(f"🤖 Enhancing cross-references with {model}...")

    enhanced_count = 0
    new_connections = 0

    for chapter_dir in sorted(chapters_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue

        chapter = chapter_dir.name
        xref_file = chapter_dir / f"{chapter}_xrefs.json"

        if not xref_file.exists():
            continue

        print(f"  📝 Enhancing {chapter}...")

        with open(xref_file) as f:
            xrefs = json.load(f)

        # Enhance existing connections
        for section, connections in xrefs.get('cross_references', {}).items():
            for conn in connections:
                # Enhance explanation
                new_explanation = enhancer.enhance_explanation(
                    chapter,
                    conn['target_chapter'],
                    conn['connection_type'],
                    conn['concepts']
                )
                if new_explanation != conn['explanation']:
                    conn['explanation'] = new_explanation
                    conn['llm_enhanced'] = True
                    enhanced_count += 1

        # Save enhanced version
        xrefs['llm_enhanced'] = True
        xrefs['llm_model'] = model

        with open(xref_file, 'w', encoding='utf-8') as f:
            json.dump(xrefs, f, indent=2, ensure_ascii=False)

    print(f"✅ Enhanced {enhanced_count} explanations using {model}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhance cross-references with LLM")
    parser.add_argument('-d', '--chapters-dir', default='/Users/VJ/GitHub/MLSysBook/quarto/contents/core')
    parser.add_argument('-m', '--model', default='llama3.1:8b')
    parser.add_argument('--test', action='store_true', help='Test LLM connection only')

    args = parser.parse_args()

    if args.test:
        enhancer = LLMEnhancer(args.model)
        if enhancer.available:
            print(f"✅ {args.model} is available")
            # Test explanation enhancement
            test_explanation = enhancer.enhance_explanation(
                "training", "hw_acceleration", "implementation",
                ["distributed training", "GPU computing", "parallel processing"]
            )
            print(f"📝 Test explanation: {test_explanation}")
        else:
            print(f"❌ {args.model} is not available")
        return

    enhance_xrefs_with_llm(Path(args.chapters_dir), args.model)

if __name__ == "__main__":
    main()
