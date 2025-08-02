#!/usr/bin/env python3
"""
AI-powered release notes generator for MLSysBook

This script generates intelligent release notes using Ollama AI models.
It analyzes recent changes and creates comprehensive release documentation.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_ollama_command(prompt, model="gemma2:9b", timeout=300):
    """Run Ollama command and return the response."""
    try:
        # Construct the Ollama command
        cmd = ["ollama", "run", model, prompt]
        
        # Run the command with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"❌ Ollama command failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ Ollama command timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"❌ Error running Ollama: {e}")
        return None


def generate_release_notes(version, previous_version, description, model="gemma2:9b", verbose=False):
    """Generate AI-powered release notes."""
    
    if verbose:
        print(f"🤖 Generating release notes with {model}...")
        print(f"📋 Version: {version}")
        print(f"📋 Previous: {previous_version}")
        print(f"📋 Description: {description}")
    
    # Create the prompt for AI
    prompt = f"""
You are an expert technical writer for a Machine Learning Systems textbook. 
Generate comprehensive, professional release notes for version {version} of the MLSysBook.

Previous version: {previous_version}
Release description: {description}

Please create release notes that include:

1. A clear summary of what's new in this version
2. Key improvements and changes
3. Technical details about the build (Linux, HTML + PDF)
4. Links to access the textbook
5. Professional academic tone

Format the response as clean Markdown without any AI disclaimers or meta-commentary.
Focus on being helpful and informative for students and educators.

Generate the complete release notes now:
"""
    
    if verbose:
        print("📝 Sending prompt to AI...")
    
    # Get AI response
    ai_response = run_ollama_command(prompt, model, timeout=300)
    
    if not ai_response:
        print("❌ Failed to get AI response")
        return None
    
    if verbose:
        print("✅ Received AI response")
        print("📄 Generated content:")
        print("-" * 50)
        print(ai_response)
        print("-" * 50)
    
    # Create the final release notes
    release_notes = f"""## 📚 Release {version}

{ai_response}

### 📋 Release Information
- **Type**: Release
- **Previous Version**: {previous_version}
- **Published at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Build Platform**: Linux (HTML + PDF)

### 🔗 Quick Links
- 🌐 [Read Online](https://harvard-edge.github.io/cs249r_book)
- 📄 [Download PDF](https://github.com/harvard-edge/cs249r_book/releases/download/{version}/Machine-Learning-Systems.pdf)
- 📄 [Direct PDF Access](https://harvard-edge.github.io/cs249r_book/assets/Machine-Learning-Systems.pdf)

### 🏗️ Build Information
- **Platform**: Linux
- **Outputs**: HTML + PDF
- **Deployment**: GitHub Pages
- **PDF Generation**: Quarto with LaTeX

---
*AI-generated release notes using {model}*
"""
    
    return release_notes


def main():
    parser = argparse.ArgumentParser(description="Generate AI-powered release notes")
    parser.add_argument("--release-notes", action="store_true", help="Generate release notes")
    parser.add_argument("--version", required=True, help="New version number")
    parser.add_argument("--previous-version", required=True, help="Previous version number")
    parser.add_argument("--description", required=True, help="Release description")
    parser.add_argument("--model", default="gemma2:9b", help="Ollama model to use")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.release_notes:
        print("📝 Generating release notes...")
        
        # Generate the release notes
        release_notes = generate_release_notes(
            version=args.version,
            previous_version=args.previous_version,
            description=args.description,
            model=args.model,
            verbose=args.verbose
        )
        
        if release_notes:
            # Write to file
            filename = f"release_notes_{args.version}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(release_notes)
            
            print(f"✅ Release notes written to: {filename}")
            print(f"📊 File size: {len(release_notes)} characters")
        else:
            print("❌ Failed to generate release notes")
            sys.exit(1)
    else:
        print("❌ No action specified. Use --release-notes to generate release notes.")
        sys.exit(1)


if __name__ == "__main__":
    main() 