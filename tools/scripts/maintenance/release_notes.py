#!/usr/bin/env python3
"""
Generate release notes with optional changelog integration.

This script generates release notes for GitHub releases. It can either:
1. Use the latest changelog entry to generate AI-powered summaries
2. Generate a simple template-based release notes
3. Use AI to summarize the changelog content
"""

import argparse
import os
import sys
import re
from datetime import datetime

def extract_latest_changelog_section(changelog_file="CHANGELOG.md"):
    """Extract the latest changelog section for use in release notes."""
    if not os.path.exists(changelog_file):
        print(f"⚠️ Changelog file not found: {changelog_file}")
        return None
    
    try:
        with open(changelog_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the latest section (first section after the header)
        sections = content.split('## ')
        if len(sections) < 2:
            print("⚠️ No changelog sections found")
            return None
        
        # Get the first section (latest)
        latest_section = sections[1]
        
        # Extract the year and content
        lines = latest_section.split('\n')
        year = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()
        
        return {
            'year': year,
            'content': content
        }
        
    except Exception as e:
        print(f"❌ Error reading changelog: {e}")
        return None

def call_ollama(prompt, model="gemma2:9b", url="http://localhost:11434"):
    """Call Ollama API to generate AI summaries."""
    try:
        import requests
        import json
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(f"{url}/api/generate", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            print(f"⚠️ Ollama API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Error calling Ollama: {e}")
        return None

def generate_ai_release_summary(changelog_content, version, description, model="gemma2:9b", url="http://localhost:11434"):
    """Generate AI summary of changelog content for release notes."""
    prompt = f"""Based on this changelog content for version {version}, generate a concise and engaging release summary.

Version: {version}
Description: {description}

Changelog Content:
{changelog_content}

Generate a 2-3 sentence summary that highlights the most important changes and improvements in this release:"""
    
    ai_summary = call_ollama(prompt, model, url)
    
    if ai_summary:
        return ai_summary
    else:
        # Fallback to simple summary
        return f"This release includes various improvements and updates to the Machine Learning Systems textbook."

def generate_release_notes_from_changelog(version, previous_version, description, changelog_entry, verbose=False, ai_mode=False, ollama_model="gemma2:9b", ollama_url="http://localhost:11434"):
    """Generate release notes using changelog content."""
    if verbose:
        print(f"📝 Generating release notes from changelog for version {version}")
    
    # Extract changelog content
    changelog_content = changelog_entry.get('content', '') if changelog_entry else ""
    
    # Generate summary based on AI mode
    if ai_mode and changelog_content:
        summary = generate_ai_release_summary(changelog_content, version, description, ollama_model, ollama_url)
    else:
        summary = f"This release includes various improvements and updates to the Machine Learning Systems textbook."
    
    # Build release notes template
    release_notes = f"""## 📚 Release {version}

**{description}**

### 📋 Release Information
- **Type**: Release
- **Previous Version**: {previous_version}
- **Published by**: GitHub Actions
- **Published at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Build Platform**: Linux (HTML + PDF)

### 📖 What's New
{summary}

### 🔗 Quick Links
- 🌐 [Web](https://mlsysbook.ai)
- 📄 [PDF](https://mlsysbook.ai/pdf)

### 📖 Detailed Changes
For a complete list of all changes, improvements, and updates, see the [detailed changelog](https://www.mlsysbook.ai/contents/frontmatter/changelog/changelog).

### 🏗️ Build Information
- **Platform**: Linux
- **Outputs**: HTML + PDF
- **Deployment**: GitHub Pages
- **PDF Generation**: Quarto with LaTeX

---
*Generated automatically by GitHub Actions*"""
    
    return release_notes

def generate_release_notes(version, previous_version, description, verbose=False, use_changelog=True, ai_mode=False, ollama_model="gemma2:9b", ollama_url="http://localhost:11434"):
    """Generate release notes with optional changelog integration."""
    if verbose:
        print(f"📝 Generating release notes for version {version}")
    
    # Get changelog content if requested
    changelog_entry = None
    if use_changelog:
        changelog_entry = extract_latest_changelog_section()
        if changelog_entry and verbose:
            print(f"📄 Found changelog entry for {changelog_entry['year']}")
    
    # Generate release notes
    release_notes = generate_release_notes_from_changelog(
        version=version,
        previous_version=previous_version,
        description=description,
        changelog_entry=changelog_entry,
        verbose=verbose,
        ai_mode=ai_mode,
        ollama_model=ollama_model,
        ollama_url=ollama_url
    )
    
    # Write to temporary file
    filename = f"release_notes_{version}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(release_notes)
    
    if verbose:
        print(f"✅ Release notes written to: {filename}")
    
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate release notes with optional changelog integration.")
    
    # Required arguments
    parser.add_argument("--version", type=str, required=True, help="Version number for release notes.")
    parser.add_argument("--previous-version", type=str, required=True, help="Previous version number for release notes.")
    parser.add_argument("--description", type=str, required=True, help="Release description for release notes.")
    
    # Options
    parser.add_argument("-t", "--test", action="store_true", help="Run without writing to file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--no-changelog", action="store_true", help="Don't use changelog content (generate template only).")
    
    # AI options
    parser.add_argument("--ai-mode", action="store_true", help="Enable AI-generated summaries from changelog.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL for AI summaries.")
    parser.add_argument("--ollama-model", default="gemma2:9b", help="Ollama model to use for AI summaries.")

    args = parser.parse_args()
    
    try:
        # Print configuration header
        print("=" * 60)
        print("📝 RELEASE NOTES GENERATION CONFIG")
        print("=" * 60)
        print(f"🎯 Version: {args.version}")
        print(f"📋 Previous: {args.previous_version}")
        print(f"📢 Description: {args.description}")
        print(f"🔧 Test Mode: {'ON' if args.test else 'OFF'}")
        print(f"📢 Verbose: {'ON' if args.verbose else 'OFF'}")
        print(f"📄 Use Changelog: {'OFF' if args.no_changelog else 'ON'}")
        print(f"🤖 AI Mode: {'ON' if args.ai_mode else 'OFF'}")
        if args.ai_mode:
            print(f"🤖 AI Model: {args.ollama_model}")
            print(f"🤖 AI URL: {args.ollama_url}")
        print("=" * 60)
        print()
        
        print("🚀 Starting release notes generation...")
        
        # Generate release notes
        filename = generate_release_notes(
            version=args.version,
            previous_version=args.previous_version,
            description=args.description,
            verbose=args.verbose,
            use_changelog=not args.no_changelog,
            ai_mode=args.ai_mode,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url
        )
        
        if args.test:
            # Read and display the content for test mode
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            print("🧪 TEST MODE - Release notes content:")
            print("=" * 60)
            print(content)
            print("=" * 60)
            print(f"📊 File size: {len(content)} characters")
            # Clean up test file
            os.remove(filename)
            print("🧹 Test file cleaned up")
        else:
            print(f"✅ Release notes saved to: {filename}")
            print(f"📊 File size: {os.path.getsize(filename)} bytes")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 