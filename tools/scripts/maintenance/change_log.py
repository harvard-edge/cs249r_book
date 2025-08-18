#!/usr/bin/env python3
"""
Generate changelog entries for the ML Systems textbook.

This script analyzes Git changes since the last publication and generates organized
changelog entries with AI-powered summaries, impact visualization, and structured
organization by content type (frontmatter, chapters, labs, appendix).

Key Features:
- AI-generated summaries using Ollama (default: gemma2:9b)
- Dynamic impact bars based on change volume
- Chapter ordering preserves textbook pedagogical sequence
- Lab organization follows hardware platform structure
- Period-specific impact thresholds for meaningful visualization
"""

import argparse
import os
import sys
import re
from datetime import datetime, timedelta
from collections import defaultdict
import yaml

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
CHANGELOG_FILE = "CHANGELOG.md"

# Configuration file paths
DEFAULT_QUARTO_PDF_CONFIG = "quarto/config/_quarto-pdf.yml"
DEFAULT_QUARTO_HTML_CONFIG = "quarto/config/_quarto-html.yml"

# Lab structure from quarto config
LAB_STRUCTURE = None

# Updated to match your actual file structure
chapter_lookup = [
    # MAIN chapters
    ("contents/core/introduction/introduction.qmd", "Introduction", 1),
    ("contents/core/ml_systems/ml_systems.qmd", "ML Systems", 2),
    ("contents/core/dl_primer/dl_primer.qmd", "DL Primer", 3),
    ("contents/core/dnn_architectures/dnn_architectures.qmd", "DNN Architectures", 4),
    ("contents/core/workflow/workflow.qmd", "AI Workflow", 5),
    ("contents/core/data_engineering/data_engineering.qmd", "Data Engineering", 6),
    ("contents/core/frameworks/frameworks.qmd", "AI Frameworks", 7),
    ("contents/core/training/training.qmd", "AI Training", 8),
    ("contents/core/efficient_ai/efficient_ai.qmd", "Efficient AI", 9),
    ("contents/core/optimizations/optimizations.qmd", "Model Optimizations", 10),
    ("contents/core/hw_acceleration/hw_acceleration.qmd", "AI Acceleration", 11),
    ("contents/core/benchmarking/benchmarking.qmd", "Benchmarking AI", 12),
    ("contents/core/ops/ops.qmd", "ML Operations", 13),
    ("contents/core/ondevice_learning/ondevice_learning.qmd", "On-Device Learning", 14),
    ("contents/core/privacy_security/privacy_security.qmd", "Security & Privacy", 15),
    ("contents/core/responsible_ai/responsible_ai.qmd", "Responsible AI", 16),
    ("contents/core/sustainable_ai/sustainable_ai.qmd", "Sustainable AI", 17),
    ("contents/core/robust_ai/robust_ai.qmd", "Robust AI", 18),
    ("contents/core/ai_for_good/ai_for_good.qmd", "AI for Good", 19),
    ("contents/core/conclusion/conclusion.qmd", "Conclusion", 20),
    
    # LAB sections
    ("contents/labs/overview.qmd", "Labs Overview", 100),
    ("contents/labs/getting_started.qmd", "Lab Setup", 101),
    
    # Arduino Nicla Vision Labs
    ("contents/labs/arduino/nicla_vision/setup/setup.qmd", "Arduino Setup", 102),
    ("contents/labs/arduino/nicla_vision/image_classification/image_classification.qmd", "Arduino Image Classification", 103),
    ("contents/labs/arduino/nicla_vision/object_detection/object_detection.qmd", "Arduino Object Detection", 104),
    ("contents/labs/arduino/nicla_vision/kws/kws.qmd", "Arduino Keyword Spotting", 105),
    ("contents/labs/arduino/nicla_vision/motion_classification/motion_classification.qmd", "Arduino Motion Classification", 106),
    
    # Seeed XIAO ESP32S3 Labs
    ("contents/labs/seeed/xiao_esp32s3/setup/setup.qmd", "XIAO Setup", 107),
    ("contents/labs/seeed/xiao_esp32s3/image_classification/image_classification.qmd", "XIAO Image Classification", 108),
    ("contents/labs/seeed/xiao_esp32s3/object_detection/object_detection.qmd", "XIAO Object Detection", 109),
    ("contents/labs/seeed/xiao_esp32s3/kws/kws.qmd", "XIAO Keyword Spotting", 110),
    ("contents/labs/seeed/xiao_esp32s3/motion_classification/motion_classification.qmd", "XIAO Motion Classification", 111),
    
    # Raspberry Pi Labs
    ("contents/labs/raspi/setup/setup.qmd", "Raspberry Pi Setup", 112),
    ("contents/labs/raspi/image_classification/image_classification.qmd", "Pi Image Classification", 113),
    ("contents/labs/raspi/object_detection/object_detection.qmd", "Pi Object Detection", 114),
    ("contents/labs/raspi/llm/llm.qmd", "Pi Large Language Models", 115),
    ("contents/labs/raspi/vlm/vlm.qmd", "Pi Vision Language Models", 116),
    
    # Frontmatter
    ("contents/frontmatter/foreword.qmd", "Foreword", 200),
    ("contents/frontmatter/about/about.qmd", "About", 201),
    ("contents/frontmatter/changelog/changelog.qmd", "Changelog", 202),
    ("contents/frontmatter/acknowledgements/acknowledgements.qmd", "Acknowledgements", 203),
    ("contents/frontmatter/socratiq/socratiq.qmd", "SocratiQ", 204),
    
    # Appendix
    ("contents/appendix/phd_survival_guide.qmd", "PhD Survival Guide", 300),
]

chapter_order = []

def load_lab_structure(quarto_file=DEFAULT_QUARTO_HTML_CONFIG):
    """Load lab structure from quarto HTML config file."""
    global LAB_STRUCTURE
    
    if not os.path.exists(quarto_file):
        print(f"⚠️ Quarto config file not found: {quarto_file}")
        return None
    
    try:
        with open(quarto_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Extract lab sections from the sidebar structure
        lab_sections = {}
        
        if 'website' in config and 'sidebar' in config['website']:
            sidebar = config['website']['sidebar']
            
            # The sidebar is a list, and we need to find the main content section
            for sidebar_item in sidebar:
                if isinstance(sidebar_item, dict) and 'contents' in sidebar_item:
                    # This contains the main sections
                    for section in sidebar_item['contents']:
                        if isinstance(section, dict) and 'section' in section:
                            section_title = section['section']
                            section_id = section.get('id', '')
                            
                            # Check if this is a lab section based on known lab section names
                            lab_keywords = ['arduino', 'seeed', 'grove', 'raspberry', 'shared', 'hands-on']
                            if any(keyword in section_title.lower() or keyword in section_id.lower() for keyword in lab_keywords):
                                lab_sections[section_title] = []
                                
                                # Extract file paths from contents
                                if 'contents' in section:
                                    for item in section['contents']:
                                        if isinstance(item, dict) and 'href' in item:
                                            file_path = item['href']
                                            lab_sections[section_title].append(file_path)
                                        elif isinstance(item, str):
                                            lab_sections[section_title].append(item)
        
        LAB_STRUCTURE = lab_sections
        if lab_sections:
            print(f"✅ Loaded lab structure with {len(lab_sections)} groups:")
            for group_name, files in lab_sections.items():
                print(f"  📁 {group_name}: {len(files)} files")
        else:
            print("⚠️ No lab structure found in config")
        return lab_sections
        
    except Exception as e:
        print(f"❌ Error loading lab structure: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_lab_group_for_file(file_path):
    """Determine which lab group a file belongs to based on the structure."""
    if not LAB_STRUCTURE:
        return None
    
    # Normalize the file path for comparison
    normalized_path = file_path.replace('book/', '')
    
    for group_name, files in LAB_STRUCTURE.items():
        for group_file in files:
            # Convert group file path to normalized format
            group_file_normalized = group_file.replace('book/', '')
            
            # Check if the file matches this group
            if normalized_path == group_file_normalized:
                return group_name
    
    return None

def debug_lab_changes():
    """Debug mode to analyze lab file changes and help troubleshoot issues."""
    print("🐛 DEBUG MODE: Analyzing lab file changes")
    print("=" * 60)
    
    # Get the latest publication date
    print("🔄 Starting Git data fetch...")
    run_git_command(["git", "fetch", "origin", "gh-pages:refs/remotes/origin/gh-pages"], verbose=True)
    run_git_command(["git", "fetch", "origin", "dev:refs/remotes/origin/dev"], verbose=True)
    
    # Get latest gh-pages commit
    def get_latest_gh_pages_commit():
        output = run_git_command(["git", "log", "--pretty=format:%H %aI", "--grep=Built site for gh-pages", "origin/gh-pages"], verbose=True)
        if output.strip():
            first_line = output.split('\n')[0]
            parts = first_line.split(" ", 1)
            return (parts[0], parts[1]) if len(parts) == 2 else (None, None)
        return (None, None)
    
    commit_hash, latest_date = get_latest_gh_pages_commit()
    if latest_date:
        print(f"📅 Latest publication: {format_friendly_date(latest_date)}")
        print(f"📅 Analyzing changes from: {latest_date} to current dev branch")
    else:
        print("📅 No previous publication found, analyzing all changes")
    
    print("\n🔍 Lab structure loaded:")
    if LAB_STRUCTURE:
        for group_name, files in LAB_STRUCTURE.items():
            print(f"  📁 {group_name}:")
            for file_path in files:
                print(f"    📄 {file_path}")
    else:
        print("  ⚠️ No lab structure loaded!")
        return
    
    print("\n🔍 Testing Git commands for lab files:")
    
    # Test each lab file individually
    for group_name, files in LAB_STRUCTURE.items():
        print(f"\n📁 Testing {group_name}:")
        for file_path in files:
            print(f"  📄 Testing: {file_path}")
            
            # Test different path variations
            test_paths = [
                file_path,
                file_path.replace('contents/', ''),
                f"quarto/{file_path}",
                f"book/{file_path}"
            ]
            
            for test_path in test_paths:
                # Build git command
                git_cmd = f"git log --numstat --since {latest_date} origin/dev -- {test_path}"
                if not latest_date:
                    git_cmd = f"git log --numstat origin/dev -- {test_path}"
                
                print(f"    🔧 Testing path: {test_path}")
                print(f"    📦 Command: {git_cmd}")
                
                try:
                    import subprocess
                    result = subprocess.run(git_cmd.split(), capture_output=True, text=True, cwd=".")
                    if result.returncode == 0:
                        output_lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
                        if output_lines:
                            print(f"    ✅ Found {len(output_lines)} lines of output")
                            # Show first few lines
                            for i, line in enumerate(output_lines[:3]):
                                print(f"      {i+1}: {line}")
                            if len(output_lines) > 3:
                                print(f"      ... and {len(output_lines) - 3} more lines")
                        else:
                            print(f"    ⚠️ No output (file may not exist or have changes)")
                    else:
                        print(f"    ❌ Command failed: {result.stderr.strip()}")
                except Exception as e:
                    print(f"    ❌ Error running command: {e}")
                
                print()
    
    print("🐛 Debug analysis complete!")

def organize_labs_by_structure(lab_entries, thresholds=None):
    """Organize lab entries according to the structure from quarto config."""
    if not LAB_STRUCTURE:
        # Fallback to flat list if no structure loaded
        return sort_by_impact_level(lab_entries, thresholds)
    
    # Group lab entries by their hardware platform using the actual LAB_STRUCTURE
    lab_groups = defaultdict(list)
    
    for entry in lab_entries:
        # Try to match entry to lab groups based on file paths or content
        matched = False
        
        # First try to match based on the actual LAB_STRUCTURE
        for group_name, file_paths in LAB_STRUCTURE.items():
            # Check if any keywords from this group appear in the entry
            group_keywords = []
            if "Arduino" in group_name:
                group_keywords = ["Arduino", "nicla"]
            elif "Seeed" in group_name:
                group_keywords = ["Seeed", "xiao", "esp32s3"]
            elif "Grove" in group_name:
                group_keywords = ["Grove", "grove"]
            elif "Raspberry" in group_name:
                group_keywords = ["Raspberry", "raspi", "pi "]
            elif "Shared" in group_name:
                group_keywords = ["Shared", "shared", "kws_feature", "dsp_spectral"]
            elif "Hands-on" in group_name:
                group_keywords = ["Hands-on", "labs"]
            
            # Check if any keywords match
            if any(keyword.lower() in entry.lower() for keyword in group_keywords):
                lab_groups[group_name].append(entry)
                matched = True
                break
        
        # If no match found, put in Other Labs
        if not matched:
            lab_groups["Other Labs"].append(entry)
    
    # Sort each group by impact level and build the organized output
    organized_labs = []
    
    # Use the order from the quarto config
    for group_name in LAB_STRUCTURE.keys():
        if group_name in lab_groups:
            sorted_entries = sort_by_impact_level(lab_groups[group_name], thresholds)
            if sorted_entries:
                # Add group header with collapsible details
                organized_labs.append(f"**{group_name}**")
                organized_labs.append("")
                
                # Add each entry in the group
                for entry in sorted_entries:
                    organized_labs.append(entry)
                
                organized_labs.append("")  # Add spacing between groups
    
    # Handle any unmatched entries
    if "Other Labs" in lab_groups:
        sorted_entries = sort_by_impact_level(lab_groups["Other Labs"], thresholds)
        if sorted_entries:
            organized_labs.append("**Other Labs**")
            organized_labs.append("")
            for entry in sorted_entries:
                organized_labs.append(entry)
    
    return organized_labs

def load_chapter_order(quarto_file=None):
    """Load chapter order from quarto config file."""
    global chapter_order
    
    if not quarto_file:
        quarto_file = DEFAULT_QUARTO_PDF_CONFIG
    
    if not os.path.exists(quarto_file):
        print(f"⚠️ Quarto config file not found: {quarto_file}")
        return
    
    try:
        with open(quarto_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        def find_chapters(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "chapters":
                        return value
                    result = find_chapters(value)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = find_chapters(item)
                    if result:
                        return result
            return None

        def extract_qmd_paths(items):
            paths = []
            for item in items:
                if isinstance(item, str) and item.endswith(".qmd"):
                    paths.append(item)
                elif isinstance(item, dict):
                    if "chapters" in item:
                        paths.extend(extract_qmd_paths(item["chapters"]))
                    elif "part" in item and isinstance(item["part"], str):
                        if item["part"].endswith(".qmd"):
                            paths.append(item["part"])
                        if "chapters" in item:
                            paths.extend(extract_qmd_paths(item["chapters"]))
            return paths

        chapters_section = find_chapters(config)
        chapter_order = extract_qmd_paths(chapters_section) if chapters_section else []
        
        print(f"📚 Loaded {len(chapter_order)} chapters from {quarto_file}")
        
    except Exception as e:
        print(f"❌ Error loading chapter order: {e}")
        chapter_order = []

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

def generate_ai_summary(chapter_title, commit_messages, file_path, verbose=False):
    """Generate AI summary for a file based on commit messages."""
    if not commit_messages.strip():
        return f"Updated content with minor changes"
    
    # Create a prompt for AI summary with JSON output
    prompt = f"""You are a technical writer creating changelog entries. Analyze these Git commit messages for {chapter_title} ({file_path}) and return ONLY a valid JSON response.

        Commit messages:
        {commit_messages}

        Return ONLY valid JSON in this exact format:
        {{
          "summary": "Brief 1-2 sentence summary of what changed"
        }}

        Requirements for the summary:
        - INSTRUCTOR/READER PERSPECTIVE: What would an instructor or student need to know about educational content changes?
        - FOCUS ONLY ON LEARNING CONTENT: new concepts, explanations, examples, exercises, fixes to educational material, improved clarity
        - COMPLETELY IGNORE: file renames, directory moves, path changes, case sensitivity fixes, build systems, CI/CD, infrastructure
        - COMPLETELY IGNORE: "converted filenames to lowercase", "updated references", "renamed directory", "standardized naming"
        - Be specific about what educational value was added or improved
        - Use factual language focused on learning outcomes
        - No conversational text, questions, or offers to help
        - DO NOT repeat the chapter name/number in your summary since it's already shown in the title
        - Start directly with what educational content changed (e.g., "Added new section..." not "Chapter X was updated with...")
        - If commits only contain technical/structural changes with no educational content updates, return: {{"summary": "No educational content changes"}}

        Example valid responses for EDUCATIONAL CONTENT changes:
        {{"summary": "Added transformer architecture section with attention mechanism diagrams and corrected backpropagation equations"}}
        {{"summary": "Fixed mathematical notation errors and improved code examples for GPU optimization"}}
        {{"summary": "Enhanced privacy-preserving techniques section with new federated learning examples"}}
        {{"summary": "Updated lab exercises with new hardware setup instructions and troubleshooting guide"}}
        {{"summary": "Added TikZ figure illustrating neural network architecture and improved explanation clarity"}}
        {{"summary": "Clarified dropout's role in uncertainty estimation and elaborated on adversarial example detection"}}

        Example for NON-EDUCATIONAL changes:
        {{"summary": "No educational content changes"}}

        Return only the JSON, nothing else:"""
    
    if verbose:
        print(f"🤖 Generating AI summary for {chapter_title}...")
    
    ai_summary = call_ollama(prompt)
    
    if ai_summary:
        try:
            # Parse JSON response
            import json
            
            # Clean up the response - remove any non-JSON text
            cleaned_response = ai_summary.strip()
            
            # Find JSON content between curly braces
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = cleaned_response[start_idx:end_idx + 1]
                parsed_response = json.loads(json_str)
                
                if 'summary' in parsed_response and parsed_response['summary'].strip():
                    summary = parsed_response['summary'].strip()
                    
                    # Additional validation - reject conversational responses
                    conversational_phrases = [
                        "let me know", "if you need", "please provide", 
                        "once you give", "i need", "feel free", "hope this helps"
                    ]
                    
                    if any(phrase in summary.lower() for phrase in conversational_phrases):
                        raise ValueError("Conversational response detected")
                    
                    # Ensure minimum quality
                    if len(summary) < 10:
                        raise ValueError("Summary too short")
                    
                    # Skip entries that are only organizational changes or have no educational content
                    skip_phrases = [
                        "internal repository reorganization",
                        "no content changes", 
                        "no educational content changes",
                        "converted filenames to lowercase",
                        "updated references",
                        "renamed directory",
                        "standardized naming"
                    ]
                    if any(phrase in summary.lower() for phrase in skip_phrases):
                        return None
                        
                    return summary
                else:
                    raise ValueError("No valid summary in JSON response")
                    
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            if verbose:
                print(f"      ⚠️ Failed to parse AI response as JSON: {e}")
                print(f"      📝 Raw response: {ai_summary[:100]}...")
            
            # Fall through to fallback
    
    # Fallback to simple summary
    commit_count = len([msg for msg in commit_messages.split('\n') if msg.strip()])
    return f"Updated content with {commit_count} changes"

def run_git_command(cmd, verbose=False, retries=3):
    """Run a git command and return the output."""
    import subprocess
    
    for attempt in range(retries):
        if verbose:
            print(f"📦 Running: {' '.join(cmd)} (attempt {attempt + 1})")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        
        if attempt < retries - 1:
            print(f"⚠️ Git command failed, retrying in 2s: {result.stderr}")
            import time
            time.sleep(2)
        else:
            raise RuntimeError(f"Git command failed after {retries} attempts: {' '.join(cmd)}\n{result.stderr}")

def extract_chapter_title(file_path):
    """Extract chapter title from file path using lookup table."""
    # Normalize path by removing quarto/ prefix for comparison
    normalized_path = file_path.replace('quarto/', '')
    
    # Try exact path match first
    for fname, title, number in chapter_lookup:
        if fname == normalized_path:
            if number <= 20:
                return f"Chapter {number}: {title}"
            elif number <= 199:
                return f"Lab: {title}"
            elif number <= 299:
                return title  # Frontmatter - just use title
            else:
                return title  # Appendix - just use title
    
    # Fallback: try basename matching for backwards compatibility
    base = os.path.basename(file_path)
    for fname, title, number in chapter_lookup:
        if os.path.basename(fname) == base:
            if number <= 20:
                return f"Chapter {number}: {title}"
            elif number <= 199:
                return f"Lab: {title}"
            elif number <= 299:
                return title
            else:
                return title
    
    # Final fallback: generate from path
    if "contents/core/" in file_path:
        return f"Chapter: {base.replace('_', ' ').replace('.qmd', '').title()}"
    elif "contents/labs/" in file_path:
        return f"Lab: {base.replace('_', ' ').replace('.qmd', '').title()}"
    elif "contents/frontmatter/" in file_path:
        return base.replace('_', ' ').replace('.qmd', '').title()
    elif "contents/appendix/" in file_path:
        return base.replace('_', ' ').replace('.qmd', '').title()
    else:
        return base.replace('_', ' ').replace('.qmd', '').title()

def analyze_changes_for_period(changes_by_file):
    """Analyze changes in current period to determine appropriate impact bar thresholds."""
    if not changes_by_file:
        return None
    
    # Get all change counts for this period
    change_counts = []
    for file_path, (added, removed) in changes_by_file.items():
        total_changes = added + removed
        if total_changes > 0:  # Only include files with actual changes
            change_counts.append(total_changes)
    
    if not change_counts:
        return None
    
    # Sort to calculate percentiles
    change_counts.sort()
    n = len(change_counts)
    
    if n == 1:
        # Only one file changed
        return {
            'major': change_counts[0],
            'large': change_counts[0],
            'medium': change_counts[0],
            'small': change_counts[0],
            'tiny': 1
        }
    
    # Calculate percentiles for this specific period
    percentiles = {
        'p95': change_counts[int(0.95 * n)] if n > 1 else change_counts[-1],
        'p75': change_counts[int(0.75 * n)] if n > 1 else change_counts[-1],
        'p50': change_counts[int(0.50 * n)] if n > 1 else change_counts[-1],
        'p25': change_counts[int(0.25 * n)] if n > 1 else change_counts[0],
        'max': change_counts[-1],
        'min': change_counts[0]
    }
    
    # Set thresholds based on this period's distribution
    thresholds = {
        'major': max(percentiles['p95'], 50),  # Top 5% or at least 50 lines
        'large': max(percentiles['p75'], 25),  # Top 25% or at least 25 lines
        'medium': max(percentiles['p50'], 10), # Median or at least 10 lines
        'small': max(percentiles['p25'], 3),   # Bottom 75% or at least 3 lines
        'tiny': 1
    }
    
    return thresholds

def generate_impact_bar(change_count, thresholds=None):
    """Generate impact bar based on number of line changes (added + removed).
    
    If thresholds are provided, use period-specific thresholds.
    Otherwise, fall back to global repository thresholds.
    """
    if thresholds:
        # Use period-specific thresholds
        if change_count >= thresholds['major']:
            return "█████"  # Major: Top 5% of changes in this period
        elif change_count >= thresholds['large']:
            return "████░"  # Large: Top 25% of changes in this period
        elif change_count >= thresholds['medium']:
            return "███░░"  # Medium: Above median for this period
        elif change_count >= thresholds['small']:
            return "██░░░"  # Small: Above bottom 25% for this period
        else:
            return "█░░░░"  # Tiny: Bottom 25% of changes
    else:
        # Fall back to global repository thresholds
        if change_count >= 225:
            return "█████"  # Major: 225+ lines (global top 10%)
        elif change_count >= 72:
            return "████░"  # Large: 72-224 lines (global 75th-90th percentile)  
        elif change_count >= 15:
            return "███░░"  # Medium: 15-71 lines (global 50th-75th percentile)
        elif change_count >= 5:
            return "██░░░"  # Small: 5-14 lines (global common minor updates)
        else:
            return "█░░░░"  # Tiny: 1-4 lines (global small fixes)

def sort_by_impact_level(updates, thresholds=None):
    """Sort updates by impact level (number of changes) and add impact bars."""
    def extract_impact_level(update):
        # Extract number of changes from the update
        # Look for patterns like "Updated content with 25 changes" or "description (150 changes)"
        match = re.search(r'\((\d+) changes\)|(\d+) changes', update)
        if match:
            # Return the first non-None group
            return int(match.group(1) or match.group(2))
        return 0
    
    # Sort by impact level (highest first)
    sorted_updates = sorted(updates, key=extract_impact_level, reverse=True)
    
    # Add impact bars to each update
    enhanced_updates = []
    for update in sorted_updates:
        change_count = extract_impact_level(update)
        impact_bar = generate_impact_bar(change_count, thresholds)
        
        # Clean up the display by removing the change count from AI summaries
        # but keep it for non-AI summaries
        if " changes)" in update:
            # AI mode: remove the change count from display but use it for sorting
            cleaned_update = re.sub(r' \(\d+ changes\)', '', update)
            enhanced_update = cleaned_update.replace("- **", f"- `{impact_bar}` **")
        else:
            # Non-AI mode: keep the change count in display
            enhanced_update = update.replace("- **", f"- `{impact_bar}` **")
        
        enhanced_updates.append(enhanced_update)
    
    return enhanced_updates

def get_changes_in_dev_since(date_start, date_end=None, verbose=False):
    """Get all changes in dev branch since a specific date."""
    cmd = ["git", "log", "--numstat", "--since", date_start]
    if date_end:
        cmd += ["--until", date_end]
    cmd += ["origin/dev", "--", "quarto/contents/**/*.qmd"]
    return run_git_command(cmd, verbose=verbose)

def get_commit_messages_for_file(file_path, since, until=None, verbose=False):
    """Get commit messages for a specific file since a date."""
    cmd = ["git", "log", "--pretty=format:%s", "--since", since]
    if until:
        cmd += ["--until", until]
    cmd += ["origin/dev", "--", file_path]
    messages = run_git_command(cmd, verbose=verbose)
    
    # Return all commit messages - let AI determine importance
    meaningful_messages = []
    for message in messages.splitlines():
        if message.strip():
            meaningful_messages.append(message.strip())
    
    return "\n".join(meaningful_messages)

def format_friendly_date(date_str):
    """Format date string to friendly format."""
    try:
        # Try ISO format first (with T separator)
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str)
        else:
            # Fallback to space-separated format
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")
        # Format as "January 28 at 02:36 PM" (full month name)
        return dt.strftime("%B %d at %I:%M %p")
    except:
        return date_str

def normalized_path(path):
    """Normalize path for comparison."""
    return os.path.normpath(path).lower()

def generate_entry(start_date, end_date=None, verbose=False, is_latest=False, ai_mode=False, ollama_url="http://localhost:11434", ollama_model="gemma2:9b", sort_processing="chapter"):
    """Generate a changelog entry for the specified time period."""
    if verbose:
        print(f"📁 Processing changes from {start_date} to {end_date or 'now'}")
    print(f"🔍 Analyzing Git changes...")
    changes = get_changes_in_dev_since(start_date, end_date, verbose=verbose)
    if not changes.strip():
        print("  ⚠️ No changes found in specified period")
        return None

    print("📊 Categorizing changes by file...")
    changes_by_file = defaultdict(lambda: [0, 0])
    for line in changes.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, removed, file_path = parts
        added = int(added) if added.isdigit() else 0
        removed = int(removed) if removed.isdigit() else 0
        changes_by_file[file_path][0] += added
        changes_by_file[file_path][1] += removed

    current_date = datetime.now().strftime('%B %d at %I:%M %p') if not end_date else format_friendly_date(end_date)
    entry = f"### 📅 {current_date}\n\n"

    frontmatter, chapters, labs, appendix = [], [], [], []

    # Sort files based on processing preference
    # Default: Chapter order preserves textbook pedagogical sequence
    if sort_processing == "impact":
        # Sort by impact (total changes) - highest first
        ordered_files = sorted(
            changes_by_file,
            key=lambda f: changes_by_file[f][0] + changes_by_file[f][1],  # added + removed
            reverse=True
        )
        print(f"📊 Processing files by impact level (highest changes first)...")
    else:
        # Chapter ordering maintains intended learning progression
        # This preserves the textbook's pedagogical flow for better readability
        ordered_files = sorted(
            changes_by_file,
            key=lambda f: next(
                (i for i, ch in enumerate(chapter_order) if normalized_path(f).endswith(normalized_path(ch))),
                float('inf')
            )
        )
        print(f"📚 Processing files by chapter order (preserving textbook sequence)...")

    total_files = len(ordered_files)
    print(f"📝 Processing {total_files} changed files...")
    print()
    
    for idx, file_path in enumerate(ordered_files, 1):
        added, removed = changes_by_file[file_path]
        total = added + removed
        # Skip references
        if "references.qmd" in file_path:
            continue
            
        commit_msgs = get_commit_messages_for_file(file_path, start_date, end_date, verbose=verbose)
        
        # Skip if no meaningful commits
        if not commit_msgs.strip():
            if verbose:
                print(f"⏭️ Skipping {file_path} - no meaningful changes")
            continue
        
        # Get actual change data for impact calculation
        added, removed = changes_by_file[file_path]
        total_changes = added + removed
        
        # Skip entries with zero changes
        if total_changes == 0:
            if verbose:
                print(f"⏭️ Skipping {file_path} - no line changes detected")
            continue
        
        # Generate summary based on AI mode
        chapter_title = extract_chapter_title(file_path)
        if ai_mode:
            summary_text = generate_ai_summary(chapter_title, commit_msgs, file_path, verbose=verbose)
            # Skip if this is only organizational changes or no educational content
            # But be less strict for historical periods (only apply strict filtering to recent changes)  
            if summary_text is None:
                if is_latest:
                    # For recent changes, skip organizational changes
                    if verbose:
                        print(f"⏭️ Skipping {file_path} - no educational content changes")
                    continue
                else:
                    # For historical periods, create a basic summary to avoid empty periods
                    commit_count = len([msg for msg in commit_msgs.split('\n') if msg.strip()])
                    summary_text = f"Updated content ({commit_count} commits)"
            # Include change data for impact bar calculation
            summary = f"- **{chapter_title}**: {summary_text} ({total_changes} changes)"
        else:
            # Create simple summary based on file path and commit count
            commit_count = len([msg for msg in commit_msgs.split('\n') if msg.strip()])
            summary_text = f"Updated content with {commit_count} changes"
            summary = f"- **{chapter_title}**: {summary_text}"
        
        # Show clean output: file name and summary
        impact_bar = generate_impact_bar(total_changes)
        print(f"📄 {os.path.basename(file_path)} ({added}+ {removed}-)")
        print(f"   `{impact_bar}` {summary_text}")
        print()
        
        # Categorize by content type (handle both quarto/contents/ and contents/ paths)
        if "contents/frontmatter/" in file_path or "quarto/contents/frontmatter/" in file_path:
            frontmatter.append(summary)
        elif "contents/labs/" in file_path or "quarto/contents/labs/" in file_path:
            labs.append(summary)
        elif "contents/appendix/" in file_path or "quarto/contents/appendix/" in file_path:
            appendix.append(summary)
        else:
            chapters.append(summary)

    print(f"📋 Organizing into sections...")
    print(f"  📄 Frontmatter: {len(frontmatter)} entries")
    print(f"  📖 Chapters: {len(chapters)} entries")
    print(f"  🧑‍💻 Labs: {len(labs)} entries")
    print(f"  📚 Appendix: {len(appendix)} entries")

    # Analyze changes for this period to set appropriate impact bar thresholds
    print(f"📊 Analyzing impact distribution for this period...")
    thresholds = analyze_changes_for_period(changes_by_file)
    if thresholds and verbose:
        print(f"  🎯 Period-specific thresholds:")
        print(f"    █████ Major: {thresholds['major']}+ lines")
        print(f"    ████░ Large: {thresholds['large']}-{thresholds['major']-1} lines")
        print(f"    ███░░ Medium: {thresholds['medium']}-{thresholds['large']-1} lines")
        print(f"    ██░░░ Small: {thresholds['small']}-{thresholds['medium']-1} lines")
        print(f"    █░░░░ Tiny: 1-{thresholds['small']-1} lines")

    # Determine if sections should be open or closed
    # All entries should be closed by default - let users choose what to explore
    details_state = ""  # Always closed for better UX

    # Add sections in order: Frontmatter, Chapters, Labs, Appendix
    if frontmatter:
        entry += f"<details {details_state}>\n<summary>**📄 Frontmatter**</summary>\n\n" + "\n".join(sort_by_impact_level(frontmatter, thresholds)) + "\n\n</details>\n\n"
    if chapters:
        entry += f"<details {details_state}>\n<summary>**📖 Chapters**</summary>\n\n" + "\n".join(sort_by_impact_level(chapters, thresholds)) + "\n\n</details>\n\n"
    if labs:
        # Organize labs according to the structure from quarto config
        organized_labs = organize_labs_by_structure(labs, thresholds)
        entry += f"<details {details_state}>\n<summary>**🧑‍💻 Labs**</summary>\n\n" + "\n".join(organized_labs) + "\n\n</details>\n\n"
    if appendix:
        entry += f"<details {details_state}>\n<summary>**📚 Appendix**</summary>\n\n" + "\n".join(sort_by_impact_level(appendix, thresholds)) + "\n\n</details>\n"

    # If no content sections were added, return None (empty entry)
    if not frontmatter and not chapters and not labs and not appendix:
        print("  ⚠️ No meaningful content changes found - skipping entry")
        return None

    print("✅ Entry generation complete")
    return entry

def generate_changelog(mode="incremental", verbose=False, ai_mode=False, ollama_url="http://localhost:11434", ollama_model="gemma2:9b", include_unpublished=False, sort_processing="chapter"):
    """Generate changelog entries."""
    if verbose:
        print("🔄 Starting Git data fetch...")
        print("  📦 Fetching gh-pages branch...")
    run_git_command(["git", "fetch", "origin", "gh-pages:refs/remotes/origin/gh-pages"], verbose=verbose)
    if verbose:
        print("  📦 Fetching dev branch...")
    run_git_command(["git", "fetch", "origin", "dev:refs/remotes/origin/dev"], verbose=verbose)
    if verbose:
        print("✅ Git data fetch complete")
    else:
        print("🔄 Fetching latest Git data...")

    def get_latest_gh_pages_commit():
        if verbose:
            print("🔍 Looking for latest publication commit...")
        # Look for actual publication commits, not administrative ones
        output = run_git_command(["git", "log", "--pretty=format:%H %aI", "--grep=Built site for gh-pages", "origin/gh-pages"], verbose=verbose)
        if output.strip():
            first_line = output.split('\n')[0]
            parts = first_line.split(" ", 1)
            result = (parts[0], parts[1]) if len(parts) == 2 else (None, None)
            if result[0] and verbose:
                print(f"  📅 Found latest commit: {result[0][:8]} from {result[1]}")
            return result
        print("  ⚠️ No publication commits found")
        return (None, None)

    def get_all_gh_pages_commits():
        if verbose:
            print("🔍 Scanning all publication commits...")
        # Look for actual publication commits, not administrative ones
        output = run_git_command(["git", "log", "--pretty=format:%H %aI", "--grep=Built site for gh-pages", "origin/gh-pages"], verbose=verbose)
        commits = []
        for line in output.splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2:
                commits.append((parts[0], parts[1]))
        print(f"  📊 Found {len(commits)} publication commits")
        return commits

    def extract_year_from_date(date_str):
        try:
            # Try ISO format first (2023-09-16T22:16:31-04:00)
            return datetime.fromisoformat(date_str.replace('Z', '+00:00')).year
        except:
            try:
                # Try the old format as fallback
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z").year
            except:
                return datetime.now().year

    latest_commit, latest_date = get_latest_gh_pages_commit()

    if mode == "full":
        if verbose:
            print("🔁 Running full regeneration...")
        commits = get_all_gh_pages_commits()
        
        # Group commits by date (YYYY-MM-DD) to merge same-day publishes
        def extract_date_only(date_str):
            try:
                # Try ISO format first (2023-09-16T22:16:31-04:00)
                return datetime.fromisoformat(date_str.replace('Z', '+00:00')).strftime("%Y-%m-%d")
            except:
                try:
                    # Try the old format as fallback
                    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z").strftime("%Y-%m-%d")
                except:
                    return date_str.split()[0]  # fallback to first part
        
        # Group commits by publication date
        commits_by_date = defaultdict(list)
        for commit, date in commits:
            date_key = extract_date_only(date)
            commits_by_date[date_key].append((commit, date))
        
        # Sort dates and get unique publication periods
        unique_dates = sorted(commits_by_date.keys(), reverse=True)  # newest first
        
        print(f"📊 Found {len(unique_dates)} unique publication dates...")
        
        # Show what will be processed before starting
        print("\n📋 PROCESSING PLAN:")
        print("=" * 50)
        for i in range(len(unique_dates) - 1):
            current_date_key = unique_dates[i]
            previous_date_key = unique_dates[i + 1]
            
            current_commits = commits_by_date[current_date_key]
            latest_current = max(current_commits, key=lambda x: x[1])
            
            previous_commits = commits_by_date[previous_date_key]
            earliest_previous = min(previous_commits, key=lambda x: x[1])
            
            current_date = latest_current[1]
            previous_date = earliest_previous[1]
            pub_year = extract_year_from_date(current_date)
            
            print(f"📅 Period {i+1}/{len(unique_dates)-1}: {format_friendly_date(previous_date)} → {format_friendly_date(current_date)} [{pub_year}]")
        
        if include_unpublished and unique_dates:
            latest_date_key = unique_dates[0]
            latest_commits = commits_by_date[latest_date_key]
            latest_publication = max(latest_commits, key=lambda x: x[1])
            print(f"📅 Period {len(unique_dates)}/{len(unique_dates)}: {format_friendly_date(latest_publication[1])} → TODAY [Current Dev Changes]")
        
        print("=" * 50)
        print(f"🎯 Total periods to process: {len(unique_dates) + (1 if include_unpublished else 0)}")
        print(f"🤖 AI summaries: {'Current dev changes only' if not ai_mode else 'Disabled for historical periods, enabled for current dev'}")
        print("=" * 50)
        print()
        
        # Group entries by year
        entries_by_year = defaultdict(list)
        
        # Process periods between publication dates
        for i in range(len(unique_dates) - 1):
            current_date_key = unique_dates[i]
            previous_date_key = unique_dates[i + 1]
            
            # Get the latest commit from current date for the "published on" date
            current_commits = commits_by_date[current_date_key]
            latest_current = max(current_commits, key=lambda x: x[1])  # latest timestamp
            
            # Get the earliest commit from previous date as the "since" date
            previous_commits = commits_by_date[previous_date_key]
            earliest_previous = min(previous_commits, key=lambda x: x[1])  # earliest timestamp
            
            current_date = latest_current[1]
            previous_date = earliest_previous[1]
            
            # Extract year from current_date (the publication date)
            pub_year = extract_year_from_date(current_date)
            
            if verbose:
                print(f"📅 Processing period {i+1}/{len(unique_dates)}: {format_friendly_date(previous_date)} → {format_friendly_date(current_date)} [{pub_year}]")
            else:
                print(f"📅 Processing {pub_year} period {i+1}/{len(unique_dates)}...")
            # For historical periods, use simple summaries to avoid slow AI processing
            # Only use AI for the most recent period (current dev changes)
            historical_ai_mode = False  # Disable AI for historical periods
            entry = generate_entry(previous_date, current_date, verbose=verbose, is_latest=False, ai_mode=historical_ai_mode, ollama_url=ollama_url, ollama_model=ollama_model, sort_processing=sort_processing)
            if entry:
                entries_by_year[pub_year].append(entry)
        
        # CRITICAL: Process all unpublished dev changes as one release preparation entry
        if include_unpublished and unique_dates:
            latest_date_key = unique_dates[0]  # Most recent publication date
            latest_commits = commits_by_date[latest_date_key]
            latest_publication = max(latest_commits, key=lambda x: x[1])
            
            if verbose:
                print(f"📅 Processing unpublished dev changes: {format_friendly_date(latest_publication[1])} → TODAY")
            else:
                print(f"📅 Processing current dev changes...")
            
            # Generate entry from last publication to today (no end_date = up to current dev)
            # This represents all changes being prepared for the next gh-pages release
            # Use full AI mode for current dev changes
            entry = generate_entry(latest_publication[1], end_date=None, verbose=verbose, is_latest=True, ai_mode=ai_mode, ollama_url=ollama_url, ollama_model=ollama_model, sort_processing=sort_processing)
            if entry:
                current_year = datetime.now().year
                entries_by_year[current_year].append(entry)
        
        if not entries_by_year:
            return "_No updates found._"
        
        # Build output with year headers, newest years first
        output_sections = []
        for year in sorted(entries_by_year.keys(), reverse=True):
            year_header = f"## {year} Updates"
            year_entries = "\n\n".join(entries_by_year[year])
            output_sections.append(f"{year_header}\n\n{year_entries}")
        
        return "\n\n---\n\n".join(output_sections) + "\n"
        
    else:
        if verbose:
            print("⚡ Running update mode...")
        
        # Determine the end date based on include_unpublished flag
        end_date = None if include_unpublished else latest_date
        
        if include_unpublished:
            print(f"📅 Processing changes: {format_friendly_date(latest_date) if latest_date else 'beginning'} → {format_friendly_date(datetime.now().isoformat())}")
        else:
            print(f"📅 Processing changes since: {format_friendly_date(latest_date) if latest_date else 'beginning'}")
            
        entry = generate_entry(latest_date, end_date=end_date, verbose=verbose, is_latest=True, ai_mode=ai_mode, ollama_url=ollama_url, ollama_model=ollama_model, sort_processing=sort_processing)
        if not entry:
            return "_No updates found._"
        
        # Use current year for both published and unpublished changes
        current_year = datetime.now().year
        year_header = f"## {current_year} Updates"
        return f"{year_header}\n\n{entry}"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate changelog entries for the ML Systems textbook with AI summaries and structured organization.")
    
    # Main mode arguments
    parser.add_argument("--full", action="store_true", help="Generate complete changelog from all publications to current dev branch (everything up to today).")
    parser.add_argument("--update", action="store_true", help="Add latest changes since last publication to the top of existing changelog.")
    
    # Options
    parser.add_argument("--dry-run", action="store_true", help="Show output without writing to file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("-o", "--output", type=str, default=CHANGELOG_FILE, help=f"Output file path for changelog (default: {CHANGELOG_FILE})")
    
    # AI options
    parser.add_argument("--ai", type=lambda x: x.lower() == 'true', default=True, help="Enable AI-generated summaries (default: true). Use --ai=false to disable.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL for AI summaries.")
    parser.add_argument("--ollama-model", default="gemma2:9b", help="Ollama model to use for AI summaries.")
    
    # Sorting options
    parser.add_argument("--sort", choices=["chapter", "impact"], default="impact", help="Sort order: 'impact' sorts by change volume (default), 'chapter' maintains textbook sequence.")

    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    

    
    # Determine mode
    if args.full and args.update:
        print("❌ Error: Cannot specify both --full and --update modes")
        exit(1)
    elif args.full:
        mode = "full"
    elif args.update:
        mode = "update"
    else:
        print("❌ Error: Must specify either --full or --update")
        print("💡 Use --help for usage information")
        exit(1)

    try:
        load_chapter_order(DEFAULT_QUARTO_PDF_CONFIG)
        # Load lab structure from HTML config (not PDF config)
        load_lab_structure(DEFAULT_QUARTO_HTML_CONFIG)
        
        # Print configuration header
        print("=" * 60)
        print("📝 CHANGELOG GENERATION CONFIG")
        print("=" * 60)
        print(f"🎯 Mode: {mode.upper()}")
        print(f"🔧 Dry Run Mode: {'ON' if args.dry_run else 'OFF'}")
        print(f"📢 Verbose: {'ON' if args.verbose else 'OFF'}")
        print(f"📄 Output File: {args.output}")
        print(f"🤖 AI Mode: {'ON' if args.ai else 'OFF'}")
        if args.ai:
            print(f"🤖 AI Model: {args.ollama_model}")
            print(f"🤖 AI URL: {args.ollama_url}")
        print(f"📋 Features: Impact bars, chapter ordering (default), AI summaries")
        print(f"📚 Processing Order: {'Impact level (highest changes first)' if args.sort == 'impact' else 'Chapter sequence'}")
        print("=" * 60)
        print()
        
        print(f"🚀 Starting changelog generation in {mode} mode...")
        if mode == "full":
            print("📋 Generating complete changelog from all publications + current dev changes")
        else:
            print("📋 Adding latest changes since last publication")

        # Both modes now include unpublished changes (everything up to today)
        # --full: Complete changelog from all publications to current dev branch
        # --update: Add latest changes since last publication to the top
        include_unpublished = True
        
        new_entry = generate_changelog(mode=mode, verbose=args.verbose, ai_mode=args.ai, ollama_url=args.ollama_url, ollama_model=args.ollama_model, include_unpublished=include_unpublished, sort_processing=args.sort)

        if args.dry_run:
            print("🧪 DRY RUN OUTPUT ONLY:\n")
            print(new_entry)
        else:
            existing = ""
            if os.path.exists(args.output):
                with open(args.output, "r", encoding="utf-8") as f:
                    existing = f.read()

            current_year = datetime.now().year
            year_header = f"## {current_year} Updates"

            # For update mode, insert new entry after the year header
            if mode == "full":
                # For full mode, replace entire content (already includes year headers)
                updated_content = new_entry.strip()
            else:
                # For incremental, insert new entry after year header
                existing_lines = existing.splitlines()
                new_lines = []
                inserted = False
                
                for line in existing_lines:
                    new_lines.append(line)
                    # Insert new entry right after the year header
                    if not inserted and line.strip() == year_header:
                        # Add the new entry (without year header since it's already in the file)
                        new_entry_lines = new_entry.strip().splitlines()
                        # Skip the first line (year header) since we're inserting after existing year header
                        if new_entry_lines and new_entry_lines[0].strip() == year_header:
                            new_entry_lines = new_entry_lines[1:]
                        new_lines.extend(new_entry_lines)
                        new_lines.append("")  # Add blank line
                        inserted = True
                
                if not inserted:
                    # If no year header found, prepend to beginning
                    new_lines = new_entry.strip().splitlines() + [""] + existing_lines
                
                updated_content = "\n".join(new_lines)

            with open(args.output, "w", encoding="utf-8") as f:
                f.write(updated_content.strip() + "\n")

            print(f"\n✅ Changelog written to {args.output}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 