#!/usr/bin/env python3
"""
Regenerate all cross-references for the MLSysBook
Uses Gemma 2 27B model with Ollama for enhanced explanations
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import requests
import time

def check_ollama():
    """Check if Ollama is running and has gemma2:27b model"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            if 'gemma2:27b' in models:
                print("✅ Ollama is running with gemma2:27b model")
                return True
            else:
                print("⚠️  gemma2:27b model not found in Ollama")
                print("   Please run: ollama run gemma2:27b")
                return False
    except:
        print("❌ Ollama is not running")
        print("   Please start Ollama and run: ollama run gemma2:27b")
        return False

def clean_existing_xrefs():
    """Remove all existing _xrefs.json files"""
    base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")
    count = 0
    for xref_file in base_dir.glob("**/*_xrefs.json"):
        xref_file.unlink()
        count += 1
    print(f"🧹 Cleaned up {count} existing _xrefs.json files")

def generate_xrefs_with_production_script():
    """Use the production script to generate cross-references"""
    script_path = Path(__file__).parent / "production_xref_generator.py"

    print("\n📊 Generating cross-references with production script...")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=script_path.parent
        )

        if result.returncode == 0:
            print("✅ Cross-references generated successfully")
            return True
        else:
            print(f"❌ Error generating cross-references: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Failed to run production script: {e}")
        return False

def enhance_with_llm():
    """Enhance cross-references with LLM explanations using Gemma 2"""
    base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")

    print("\n🤖 Enhancing cross-references with Gemma 2 27B...")

    chapters = [
        'introduction', 'ml_systems', 'dl_primer', 'workflow', 'data_engineering',
        'frameworks', 'training', 'efficient_ai', 'optimizations', 'hw_acceleration',
        'benchmarking', 'ondevice_learning', 'ops', 'privacy_security', 'responsible_ai',
        'sustainable_ai', 'ai_for_good', 'robust_ai', 'generative_ai', 'frontiers',
        'emerging_topics', 'conclusion'
    ]

    for chapter in chapters:
        xref_file = base_dir / chapter / f"{chapter}_xrefs.json"
        if not xref_file.exists():
            continue

        print(f"  Enhancing {chapter}...")

        with open(xref_file, 'r') as f:
            data = json.load(f)

        # Enhance each cross-reference with better explanations
        for section_id, refs in data.get('cross_references', {}).items():
            for ref in refs:
                # Generate enhanced explanation using Gemma 2
                enhanced_explanation = generate_llm_explanation(
                    chapter,
                    ref.get('target_chapter'),
                    ref.get('connection_type'),
                    ref.get('concepts', [])
                )
                if enhanced_explanation:
                    ref['explanation'] = enhanced_explanation

        # Save enhanced version
        with open(xref_file, 'w') as f:
            json.dump(data, f, indent=2)

    print("✅ Enhancement complete")

def generate_llm_explanation(source_chapter: str, target_chapter: str,
                            connection_type: str, concepts: list) -> str:
    """Generate explanation using Gemma 2 27B"""

    if not target_chapter or not concepts:
        return ""

    # Format concepts for display
    concept_str = ", ".join(concepts[:3]) if len(concepts) > 3 else ", ".join(concepts)

    prompt = f"""You are an expert in machine learning systems education.

Given a connection between textbook chapters:
- Source: {source_chapter}
- Target: {target_chapter}
- Type: {connection_type}
- Key concepts: {concept_str}

Write a brief (max 15 words) explanation of why this connection is valuable for students.
Focus on learning value, not just listing concepts.

Explanation:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma2:27b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 50
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            # Clean up the response
            result = result.replace('\n', ' ').replace('  ', ' ')

            # Limit length
            if len(result) > 80:
                result = result[:77] + "..."

            return result

    except Exception as e:
        # Fallback to simple explanation
        pass

    # Default fallback
    if connection_type == 'prerequisite':
        return f"Essential foundation in {concept_str}"
    elif connection_type == 'extends':
        return f"Advances concepts to {concept_str}"
    elif connection_type == 'foundation':
        return f"Builds on {concept_str}"
    else:
        return f"Related through {concept_str}"

def main():
    print("🚀 Cross-Reference Regeneration Tool")
    print("=" * 50)

    # Step 1: Check Ollama
    if not check_ollama():
        print("\n⚠️  Continuing without LLM enhancement...")
        use_llm = False
    else:
        use_llm = True

    # Step 2: Clean existing files
    clean_existing_xrefs()

    # Step 3: Generate base cross-references
    if not generate_xrefs_with_production_script():
        print("❌ Failed to generate cross-references")
        return 1

    # Step 4: Enhance with LLM if available
    if use_llm:
        enhance_with_llm()

    print("\n✅ Cross-reference regeneration complete!")
    print("\n📝 To use these cross-references:")
    print("   1. The files are already in the correct locations")
    print("   2. Build the PDF: ./binder pdf intro")
    print("   3. The inject-xrefs.lua filter will automatically use them")

    return 0

if __name__ == "__main__":
    sys.exit(main())
