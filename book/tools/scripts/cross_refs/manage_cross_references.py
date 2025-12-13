#!/usr/bin/env python3
"""
ML Systems Cross-Reference Generator
====================================

Complete toolkit for domain adaptation and cross-reference generation using pypandoc for clean content extraction.

MODES:
    Training Mode:
        python3 cross_referencing.py --train -d ../../contents/core/ -o ./my_model
        python3 cross_referencing.py --train --dirs ../../contents/core/ --output ./my_model --base-model sentence-t5-base --epochs 5

    Generation Mode (Domain-adapted):
        python3 cross_referencing.py -g -m ./t5-mlsys-domain-adapted -o cross_refs.json
        python3 cross_referencing.py --generate --model ./t5-mlsys-domain-adapted --output cross_refs.json

    Generation Mode (Base model):
        python3 cross_referencing.py -g -m sentence-t5-base -o cross_refs.json
        python3 cross_referencing.py --generate --model all-MiniLM-L6-v2 --output cross_refs.json

CONTENT EXTRACTION:
    • Uses pypandoc to intelligently clean markdown files
    • Removes all Quarto divs (:::), callouts, TikZ code, citations, footnotes
    • Extracts only ## level sections for focused embeddings
    • Applies quality filtering via filters.yml to exclude meta-content
    • Produces clean semantic content ideal for embeddings

CONFIGURATION:
    • Section filtering rules defined in filters.yml
    • Excludes meta-sections like "Purpose", "Overview", "Learning Objectives"
    • Configurable content filters for length and quality
    • Pattern matching for quiz/exercise sections

TRAINING:
    • Extracts content from specified directories
    • Excludes introduction/conclusion chapters (champion approach)
    • Creates sophisticated training examples with nuanced similarity labels
    • Domain-adapts base model using contrastive learning
    • Saves trained model for later use

GENERATION:
    • Works with domain-adapted models OR base sentence-transformer models
    • Generates embeddings from cleaned section content
    • Finds cross-references with 65%+ similarity threshold
    • Outputs Lua-compatible JSON for inject_xrefs.lua

REQUIREMENTS:
    pip install sentence-transformers scikit-learn numpy torch pyyaml pypandoc requests

For AI explanations (--explain flag):
    brew install ollama  # macOS
    # or: curl -fsSL https://ollama.ai/install.sh | sh  # Linux
    ollama run gemma2:27b  # Use larger Gemma 2 model for better quality
    # Use --ollama-model to specify a different model (e.g., --ollama-model llama3.1:8b)

    Core libraries:
    • sentence-transformers: Embedding generation and model handling
    • scikit-learn: Nearest neighbors for similarity search
    • numpy: Numerical operations
    • torch: PyTorch backend for transformers
    • pyyaml: YAML configuration file processing
    • pypandoc: Markdown cleaning and conversion
"""

import json
import numpy as np
import argparse
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, cast, Tuple

try:
    import yaml
except ImportError:
    print("❌ Error: PyYAML not installed")
    print("   Install with: pip install pyyaml")
    sys.exit(1)

# Add for explanation generation
try:
    import requests
    import json as json_lib
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def load_section_filters() -> Dict:
    """Load section filtering configuration from filters.yml."""
    filter_file = Path(__file__).parent / "filters.yml"

    if not filter_file.exists():
        print(f"📄 No filters.yml found at {filter_file}, using default filters")
        # Default config if file doesn't exist
        return {
            'exclude_files': {
                'patterns': ['.*introduction.*', '.*quiz.*', '.*exercise.*']
            },
            'exclude_sections': {
                'patterns': ['^purpose$', '^overview$', '^learning objectives$', '^prerequisites$', '.*quiz.*', '.*exercise.*']
            },
            'content_filters': {
                'min_length': 200,
                'max_length': 15000,
                'exclude_if_contains': ['learning outcome', 'this chapter will']
            },
            'quality_filters': {
                'max_list_ratio': 0.7,
                'max_code_ratio': 0.8,
                'max_citation_ratio': 0.3
            }
        }

    try:
        with open(filter_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("filters.yml must contain a valid YAML dictionary")

        # Validate required sections
        required_sections = ['exclude_files', 'exclude_sections', 'content_filters', 'quality_filters']
        for section in required_sections:
            if section not in config:
                print(f"⚠️  Warning: Missing '{section}' in filters.yml, using defaults")

        return config

    except yaml.YAMLError as e:
        print(f"❌ Error parsing filters.yml: {e}")
        print(f"   Please check YAML syntax in {filter_file}")
        return {}
    except FileNotFoundError:
        print(f"❌ Could not find filters.yml at {filter_file}")
        return {}
    except Exception as e:
        print(f"❌ Unexpected error loading filters.yml: {e}")
        return {}


def should_exclude_file(file_path: str, config: Dict) -> tuple[bool, str]:
    """
    Determine if an entire file should be excluded from processing.

    Returns:
        (should_exclude: bool, reason: str)
    """
    if not config:
        return False, ""

    # Extract filename and relative path for pattern matching
    filename = Path(file_path).name.lower()
    try:
        relative_path = str(Path(file_path).relative_to(Path.cwd())).lower()
    except ValueError:
        relative_path = str(Path(file_path)).lower()

    # Check file exclusion patterns
    file_patterns = config.get('exclude_files', {}).get('patterns', [])
    for pattern in file_patterns:
        try:
            if re.search(pattern.lower(), filename) or re.search(pattern.lower(), relative_path):
                return True, f"file pattern match: '{pattern}'"
        except re.error:
            continue

    return False, ""


def should_exclude_section(title: str, content: str, config: Dict) -> tuple[bool, str]:
    """
    Determine if section should be excluded from cross-referencing.

    Returns:
        (should_exclude: bool, reason: str)
    """
    if not config:
        return False, ""

    title_lower = title.lower().strip()
    content_lower = content.lower()

    # Pattern matching (simplified - no more exact vs patterns)
    patterns = config.get('exclude_sections', {}).get('patterns', [])
    for pattern in patterns:
        try:
            if re.search(pattern.lower(), title_lower):
                return True, f"section pattern match: '{pattern}'"
        except re.error:
            continue

    # Content length filters
    content_filters = config.get('content_filters', {})
    min_length = content_filters.get('min_length', 0)
    max_length = content_filters.get('max_length', float('inf'))

    if len(content) < min_length:
        return True, f"too short ({len(content)} < {min_length} chars)"

    if len(content) > max_length:
        return True, f"too long ({len(content)} > {max_length} chars)"

    # Meta-content detection
    exclude_keywords = content_filters.get('exclude_if_contains', [])
    for keyword in exclude_keywords:
        if keyword.lower() in content_lower:
            return True, f"contains meta-content: '{keyword}'"

    # Quality filters
    quality_filters = config.get('quality_filters', {})

    # Check list ratio
    lines = content.split('\n')
    list_lines = sum(1 for line in lines if line.strip().startswith(('-', '*', '•')))
    list_ratio = list_lines / len(lines) if lines else 0
    max_list_ratio = quality_filters.get('max_list_ratio', 1.0)

    if list_ratio > max_list_ratio:
        return True, f"too much list content ({list_ratio:.1%} > {max_list_ratio:.1%})"

    # Check code ratio (lines starting with spaces/tabs or in code blocks)
    code_lines = sum(1 for line in lines if line.startswith(('    ', '\t', '```')))
    code_ratio = code_lines / len(lines) if lines else 0
    max_code_ratio = quality_filters.get('max_code_ratio', 1.0)

    if code_ratio > max_code_ratio:
        return True, f"too much code ({code_ratio:.1%} > {max_code_ratio:.1%})"

    return False, ""


def get_quarto_file_order(quiet: bool = False) -> List[str]:
    """
    Parse _quarto.yml using proper YAML processing to get the intended chapter order.
    Returns list of file paths in the order they appear in the chapters section.
    Processes ALL chapters (both commented and uncommented) to get complete ordering.
    """
    quarto_yml_path = Path.cwd() / "_quarto.yml"

    if not quarto_yml_path.exists():
        print(f"❌ Error: _quarto.yml not found at {quarto_yml_path}")
        print("   Please run this script from the project root directory where _quarto.yml is located.")
        sys.exit(1)

    try:
        # Read the raw YAML content to preserve comments
        with open(quarto_yml_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse with PyYAML to get the main structure
        try:
            yaml_data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            if not quiet:
                print(f"⚠️  YAML parsing error: {e}")
            return []

        ordered_files = []

        # Get chapters list from parsed YAML (uncommented entries)
        book_data = yaml_data.get('book', {})
        chapters = book_data.get('chapters', [])

        for chapter in chapters:
            if isinstance(chapter, str):
                if chapter.endswith('.qmd') and 'contents/core/' in chapter:
                    ordered_files.append(chapter)
            elif isinstance(chapter, dict):
                # Handle 'part:' entries
                if 'part' in chapter:
                    part_file = chapter['part']
                    if part_file.endswith('.qmd') and 'contents/core/' in part_file:
                        ordered_files.append(part_file)

        # Extract commented chapters from raw content (YAML ignores these)
        lines = content.split('\n')
        in_chapters_section = False

        for line in lines:
            stripped = line.strip()

            # Detect chapters section
            if stripped.startswith('chapters:'):
                in_chapters_section = True
                continue

            # End of chapters section (next major section at root level)
            if in_chapters_section and stripped and ':' in stripped and not stripped.startswith((' ', '\t', '-', '#')):
                break

            if in_chapters_section:
                # Extract commented chapter entries
                if line.strip().startswith('# - '):
                    # Remove comment and list markers
                    file_part = line.strip()[4:].strip()  # Remove '# - '

                    # Handle 'part:' entries
                    if file_part.startswith('part: '):
                        file_part = file_part[6:].strip()  # Remove 'part: '

                    # Add if it's a core chapter file and not already in list
                    if (file_part.endswith('.qmd') and
                        'contents/core/' in file_part and
                        file_part not in ordered_files):
                        ordered_files.append(file_part)

        if not quiet:
            print(f"📋 Found {len(ordered_files)} ordered files in _quarto.yml (ALL chapters processed)")
            if ordered_files:
                print("🔍 _quarto.yml file order preview:")
                for i, file_path in enumerate(ordered_files[:5], 1):
                    print(f"    {i}. {file_path}")
                if len(ordered_files) > 5:
                    print(f"    ... and {len(ordered_files) - 5} more")

        return ordered_files

    except Exception as e:
        if not quiet:
            print(f"⚠️  Warning: Could not parse _quarto.yml: {e}")
        return []

def find_qmd_files(directories: List[str], quiet: bool = False) -> List[str]:
    """Find all .qmd files in directories, ordered by _quarto.yml if available."""
    # Get all qmd files
    all_qmd_files = []
    for directory in directories:
        for path in Path(directory).rglob("*.qmd"):
            all_qmd_files.append(str(path))

    # Get order from _quarto.yml
    quarto_order = get_quarto_file_order(quiet=quiet)

    if not quarto_order:
        # Fallback to alphabetical sorting
        if not quiet:
            print("📂 Using alphabetical file ordering (no _quarto.yml order found)")
        return sorted(list(set(all_qmd_files)))

    # Create mapping for efficient lookup
    all_files_set = set(all_qmd_files)
    ordered_files = []

    # First, add files in _quarto.yml order
    for ordered_file in quarto_order:
        # Try different path combinations to match
        possible_paths = [
            ordered_file,
            str(Path.cwd() / "../../" / ordered_file),
            str(Path(ordered_file).resolve()) if Path(ordered_file).exists() else None
        ]

        # Also try matching by filename pattern
        filename = Path(ordered_file).name
        for discovered_file in all_files_set.copy():
            if Path(discovered_file).name == filename:
                # Extra check: ensure it's the same chapter directory
                ordered_chapter = Path(ordered_file).parent.name
                discovered_chapter = Path(discovered_file).parent.name
                if ordered_chapter == discovered_chapter:
                    ordered_files.append(discovered_file)
                    all_files_set.remove(discovered_file)
                    break
        else:
            # If no filename match, try the path-based matching
            for possible_path in possible_paths:
                if possible_path and possible_path in all_files_set:
                    ordered_files.append(possible_path)
                    all_files_set.remove(possible_path)
                    break

    # Add any remaining files alphabetically
    remaining_files = sorted(list(all_files_set))
    ordered_files.extend(remaining_files)

    if not quiet:
        print(f"📊 File ordering: {len(ordered_files)} total ({len(ordered_files) - len(remaining_files)} from _quarto.yml, {len(remaining_files)} alphabetical)")

    return ordered_files

def extract_sections(file_path: str, verbose: bool = False, quiet: bool = False) -> List[Dict]:
    """
    Extract ## level sections from Quarto markdown files using pypandoc for clean content.

    1. First extracts original section IDs and titles from raw markdown
    2. Uses pypandoc to convert markdown to clean text (removes all Quarto markup)
    3. Matches cleaned content with original section IDs
    4. Applies section-level filtering to remove meta-content
    5. Creates embeddings from this cleaned content
    """
    sections = []

    # Load filtering configuration
    filter_config = load_section_filters()
    excluded_sections = []

    try:
        # Step 1: Extract original section headers with IDs from raw markdown
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        original_sections = _extract_original_headers(original_content)

        # Step 2: Use pypandoc to clean the entire file
        cleaned_markdown = _process_markdown_with_pypandoc(file_path)
        if not cleaned_markdown:
            print(f"⚠️  Failed to process {file_path} with pypandoc")
            return []

        # Step 3: Extract content from cleaned markdown and match with original headers
        cleaned_sections = _extract_h2_sections_from_text(cleaned_markdown)

        # Step 4: Match cleaned content with original headers
        for original_title, original_id in original_sections:
            # Find matching cleaned content by title similarity
            cleaned_content = None
            for cleaned_title, content in cleaned_sections:
                # Match by normalized title (remove extra spaces, case-insensitive)
                if _normalize_title(original_title) == _normalize_title(cleaned_title):
                    cleaned_content = content
                    break

            if not cleaned_content:
                continue

            # Apply section filtering
            should_exclude, exclude_reason = should_exclude_section(
                original_title, cleaned_content, filter_config
            )

            if should_exclude:
                excluded_sections.append((original_title, exclude_reason))
                if verbose and not quiet:
                    print(f"      🚫 Excluded '{original_title}': {exclude_reason}")
                continue

            # Only include sections with valid IDs and content
            if original_id and cleaned_content.strip():
                sections.append({
                    'file_path': file_path,
                    'title': original_title,  # Use exact original title
                    'section_id': original_id,  # Use exact original ID
                    'content': cleaned_content.strip()[:2000]  # Limit for embeddings
                })

        # Show filtering summary
        if excluded_sections and verbose and not quiet:
            chapter_name = Path(file_path).parent.name
            print(f"      📋 [{chapter_name}] Excluded {len(excluded_sections)} sections:")
            for title, reason in excluded_sections[:3]:  # Show first 3
                print(f"         • '{title}' ({reason})")
            if len(excluded_sections) > 3:
                print(f"         • ... and {len(excluded_sections) - 3} more")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return sections


def _extract_original_headers(content: str) -> List[tuple[str, str]]:
    """
    Extract original section headers with their exact IDs from raw markdown.

    Returns list of (title, section_id) tuples.
    """
    headers = []
    lines = content.split('\n')

    for line in lines:
        if line.startswith('## ') and not line.startswith('### '):
            # Extract title and ID from header line
            header_match = re.match(r'^##\s+(.+?)(?:\s*\{([^}]+)\})?\s*$', line)
            if header_match:
                title = header_match.group(1).strip()
                id_part = header_match.group(2)

                # Extract section ID from {#sec-something} or skip if no ID
                section_id = None
                if id_part and id_part.startswith('#'):
                    section_id = id_part[1:]  # Remove the # prefix

                # Only include sections with valid IDs
                if section_id and section_id.startswith('sec-'):
                    headers.append((title, section_id))

    return headers


def _normalize_title(title: str) -> str:
    """Normalize title for matching (remove extra spaces, lowercase)."""
    return re.sub(r'\s+', ' ', title.strip().lower())


def _process_markdown_with_pypandoc(file_path: str) -> str:
    """
    Process markdown file using pypandoc for robust content extraction and cleaning.

    Converts markdown to markdown format while:
    - Preserving headers with --markdown-headings=atx
    - Removing Quarto-specific divs and constructs
    - Cleaning LaTeX, citations, and other markup
    - Preprocessing problematic table formats that can break pypandoc

    Args:
        file_path: Path to the markdown file

    Returns:
        Cleaned markdown content as string
    """
    try:
        import pypandoc
        import tempfile
        import os

        # Read and preprocess content to handle pypandoc issues
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Preprocess problematic ASCII-style tables that break pypandoc
        content = _preprocess_tables(content)

        # Write preprocessed content to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Convert using pypandoc with markdown output format
            cleaned_content = pypandoc.convert_file(
                temp_path,
                'markdown',
                format='markdown',
                extra_args=['--markdown-headings=atx']
            )

            # Apply additional cleanup
            cleaned_content = _additional_cleanup(cleaned_content)

            return cleaned_content

        finally:
            # Clean up temp file
            os.unlink(temp_path)

    except Exception as e:
        print(f"❌ Error processing {file_path} with pypandoc: {e}")
        # Fallback to basic reading without pypandoc
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return _additional_cleanup(content)


def _preprocess_tables(content: str) -> str:
    """
    Preprocess content to handle ASCII-style tables that can break pypandoc.

    Converts ASCII-style tables (+----+----+) to simple markdown tables.
    """
    import re

    lines = content.split('\n')
    processed_lines = []
    in_ascii_table = False
    table_buffer = []

    for line in lines:
        # Detect ASCII-style table borders
        if re.match(r'^\s*\+[-=]+\+', line):
            if not in_ascii_table:
                in_ascii_table = True
                table_buffer = []
                # Add placeholder for table
                processed_lines.append("| Table content removed for processing |")
                processed_lines.append("|-----|")
            table_buffer.append(line)
        elif in_ascii_table:
            table_buffer.append(line)
            # Check if table has ended (empty line or new section)
            if line.strip() == '' or line.strip().startswith('#'):
                in_ascii_table = False
                # Don't add the table buffer, just continue with the line
                if line.strip() != '':
                    processed_lines.append(line)
            # Continue collecting table lines
        else:
            processed_lines.append(line)

    return '\n'.join(processed_lines)


def check_ollama_setup() -> Tuple[bool, List[str]]:
    """
    Check if Ollama is running and return available models.

    Returns:
        (is_running, available_models)
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return True, models
        return False, []
    except:
        return False, []


def setup_ollama_interactive() -> str:
    """
    Interactive setup for Ollama with model selection.

    Returns:
        Selected model name or empty string if setup failed
    """
    print("\n🤖 AI EXPLANATION SETUP")
    print("=" * 40)

    # Check if Ollama is running
    is_running, models = check_ollama_setup()

    if not is_running:
        print("❌ Ollama is not running or not installed.")
        print("\n📥 INSTALLATION INSTRUCTIONS:")
        print("   macOS:  brew install ollama")
        print("   Linux:  curl -fsSL https://ollama.ai/install.sh | sh")
        print("   Windows: Download from https://ollama.ai")
        print("\n🚀 AFTER INSTALLATION:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Download a model: ollama run qwen2.5:7b")
        print("   3. Re-run this script with --explain")
        return ""

    print("✅ Ollama is running!")

    if not models:
        print("\n📥 NO MODELS FOUND. Recommended models for explanations:")
        print("   1. ollama run gemma2:27b     (Recommended - high quality Gemma 2 model)")
        print("   2. ollama run llama3.1:8b    (Alternative - 8.0/10 quality from systematic experiments)")
        print("   3. ollama run qwen2.5:7b     (Fast alternative - 7.8/10 quality)")
        print("   3. ollama run phi3:3.8b      (High quality but verbose)")
        print("\nRun one of the above commands and then re-run this script.")
        return ""

    print(f"\n📚 Found {len(models)} model(s):")

    # Recommend best model for explanations (based on systematic experiments)
    recommended_models = [
        "gemma2:27b",   # Default: High quality Gemma 2 model
        "llama3.1:8b",  # Alternative: Best quality (8.0/10) from experiments
        "qwen2.5:7b",   # Fast alternative with good quality (7.8/10)
        "gemma2:9b",    # Good balance of quality and length adherence
        "phi3:3.8b",    # High quality but verbose
        "qwen2.5:14b", "qwen2.5:3b",
        "llama3.1:7b",
        "phi3:medium", "phi3:mini"
    ]

    best_model = None
    for rec in recommended_models:
        for model in models:
            if rec in model:
                best_model = model
                break
        if best_model:
            break

    for i, model in enumerate(models, 1):
        marker = " ⭐ (Recommended)" if model == best_model else ""
        print(f"   {i}. {model}{marker}")

    if best_model:
        print(f"\n✅ Using recommended model: {best_model}")
        return best_model
    else:
        print(f"\n🔄 Using first available: {models[0]}")
        return models[0]


def generate_explanation(source_content: str, target_content: str, source_title: str, target_title: str,
                        model: str = "gemma2:27b", max_retries: int = 3) -> str:
    """
    Generate a concise explanation of why two sections are connected using local LLM.

    Args:
        source_content: Content of the source section
        target_content: Content of the target section
        source_title: Title of the source section
        target_title: Title of the target section
        model: Ollama model to use
        max_retries: Maximum number of retry attempts

    Returns:
        Short explanation string or empty string if generation fails
    """
    if not REQUESTS_AVAILABLE:
        return ""

    # Truncate content to manageable size for LLM
    source_snippet = source_content[:800] + "..." if len(source_content) > 800 else source_content
    target_snippet = target_content[:800] + "..." if len(target_content) > 800 else target_content



    prompt = f"""You are writing cross-reference explanations for a Machine Learning Systems textbook. Create natural, flowing explanations that tell students WHY they should follow the connection.

Source Section: "{source_title}"
{source_snippet}

Target Section: "{target_title}"
{target_snippet}

Write ONLY the explanation text - a 6-12 word phrase starting with an action verb.

DO NOT include arrows, section titles, or references - just the explanation.

Your text will complete this: "→ Section Title (ref) [YOUR TEXT HERE]"

JUST WRITE THE EXPLANATION PART. Examples of what YOU should generate:
✓ "provides essential mathematical foundations for neural networks"
✓ "explains practical deployment strategies for edge devices"
✓ "demonstrates optimization techniques for mobile inference"
✓ "explores advanced architectural design patterns"

DO NOT WRITE:
✗ "→ provides essential foundations"
✗ "Section explains advanced topics"
✗ "← explores implementation details"

GOOD examples - EXACTLY what you should write:
- "provides essential mathematical foundations for neural networks"
- "demonstrates practical optimization techniques for deployment"
- "explores advanced GPU acceleration strategies"
- "explains fundamental concepts driving ML system design"
- "reveals critical performance bottlenecks in inference"
- "contrasts different approaches to model compression"
- "introduces key principles for scalable architectures"

BAD examples - DO NOT write these:
- "→ provides essential foundations" (has arrow - forbidden!)
- "Section explains advanced topics" (includes section reference)
- "← explores implementation details" (has arrow - forbidden!)
- "understands how this works" (section doesn't understand)
- "essential background" (missing action verb)
- "helps readers understand" (section doesn't help)

CRITICAL: Write ONLY the explanation text. No arrows (→←), no section names, no references.

Start with verbs like: provides, shows, explains, demonstrates, covers, explores, reveals, contrasts, introduces, details, established, introduced, covered, defined, built, etc.

Generate ONE explanation phrase only:"""

    for attempt in range(max_retries):
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
                        "max_tokens": 80
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                explanation = result.get("response", "").strip()

                # Sophisticated cleanup for verbose model responses
                explanation = _extract_core_explanation(explanation)

                # If we got a valid explanation, return it
                if explanation and len(explanation.strip()) > 3:
                    return explanation
                else:
                    # Empty or too short explanation, try again
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return ""  # Give up after max retries
            else:
                if attempt < max_retries - 1:
                    continue  # Try again
                else:
                    print(f"⚠️  Ollama API error after {max_retries} attempts: {response.status_code}")
                    return ""

        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                continue  # Try again
            else:
                return ""  # Give up - connection issues
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                continue  # Try again
            else:
                return ""  # Give up - timeout issues
        except Exception:
            if attempt < max_retries - 1:
                continue  # Try again
            else:
                return ""  # Give up - other errors

    return ""  # Fallback


def _extract_core_explanation(raw_response: str) -> str:
    """
    Extract the core explanation from verbose LLM responses.
    Handles cases where models generate too much text or include unwanted prefixes.
    """
    if not raw_response or not raw_response.strip():
        return ""

    explanation = raw_response.strip()

    # Step 1: Handle multi-line responses - extract relevant lines
    lines = explanation.split('\n')
    useful_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip instructional text
        if any(skip in line.lower() for skip in [
            'here are', 'explanations:', 'cross-reference', 'source section',
            'target section', 'write only', 'instruction', 'follow'
        ]):
            continue

        # Look for lines that contain actual explanations
        if any(indicator in line.lower() for indicator in [
            'see also:', 'reveals', 'explains', 'shows', 'demonstrates',
            'provides', 'covers', 'explores', 'highlights', 'helps'
        ]):
            useful_lines.append(line)

    # If we found useful lines, use the best one
    if useful_lines:
        explanation = useful_lines[0]  # Take the first useful line

    # Step 2: Extract just the explanation part after "see also:" or similar
    patterns_to_extract = [
        r'see also:.*?-\s*(.+?)(?:\.|$)',  # "See also: Title - explanation"
        r'see also:\s*(.+?)(?:\.|$)',      # "See also: explanation"
        r'-\s*(.+?)(?:\.|$)',              # "Title - explanation"
        r':\s*(.+?)(?:\.|$)',              # "Label: explanation"
    ]

    for pattern in patterns_to_extract:
        import re
        match = re.search(pattern, explanation, re.IGNORECASE | re.DOTALL)
        if match:
            explanation = match.group(1).strip()
            break

    # Step 3: Remove unwanted prefixes/suffixes
    prefixes_to_remove = [
        "explanation:", "- ", "• ", "*", '"', "'", "`",
        "contextual:", "foundational:", "practical:", "detailed:", "comparative:",
        "reveals", "explains", "shows", "demonstrates", "provides", "covers",
        "here", "this", "see also", "target", "source"
    ]

    for prefix in prefixes_to_remove:
        if explanation.lower().startswith(prefix.lower()):
            explanation = explanation[len(prefix):].strip()

    # Step 4: Clean up punctuation and formatting
    explanation = explanation.replace('"', '').replace("'", '').replace('*', '')
    explanation = explanation.replace('\n', ' ').replace('\t', ' ')
    explanation = ' '.join(explanation.split())  # Normalize whitespace

    # Step 5: Ensure proper capitalization
    if explanation and explanation[0].isupper() and not explanation.startswith(('AI', 'ML', 'GPU', 'CPU', 'API')):
        explanation = explanation[0].lower() + explanation[1:]

    # Step 6: Ensure reasonable length (6-12 words is optimal from experiments)
    words = explanation.split()
    if len(words) > 15:  # If too long, take first meaningful part
        explanation = ' '.join(words[:12])
    elif len(words) < 3:  # If too short, it's probably incomplete
        return ""

    # Step 7: Remove trailing punctuation except periods
    explanation = explanation.rstrip('.,!?;:')

    return explanation


def _additional_cleanup(content: str) -> str:
    """
    Additional cleanup to remove any remaining Quarto artifacts and LaTeX/TikZ code.

    Specifically targets ::: divs, TikZ blocks, and other markup that shouldn't be in embeddings.
    """
    import re

    lines = content.split('\n')
    cleaned_lines = []
    in_div_block = False
    in_tikz_block = False
    in_latex_block = False
    div_level = 0

    for line in lines:
        stripped_line = line.strip()

        # Handle TikZ blocks - remove them completely
        if '\\begin{tikzpicture}' in stripped_line:
            in_tikz_block = True
            continue
        if '\\end{tikzpicture}' in stripped_line:
            in_tikz_block = False
            continue
        if in_tikz_block:
            continue

        # Handle other LaTeX environments
        if re.match(r'\\begin\{(axis|figure|table|equation|align|gather|multline)\}', stripped_line):
            in_latex_block = True
            continue
        if re.match(r'\\end\{(axis|figure|table|equation|align|gather|multline)\}', stripped_line):
            in_latex_block = False
            continue
        if in_latex_block:
            continue

        # Handle ::: div blocks - remove them completely
        if stripped_line.startswith(':::'):
            if stripped_line == ':::':
                # Closing div
                if in_div_block:
                    div_level -= 1
                    if div_level == 0:
                        in_div_block = False
                continue
            else:
                # Opening div like ::: {.callout-note}
                in_div_block = True
                div_level += 1
                continue

        # Skip content inside div blocks
        if in_div_block:
            # Check for nested divs
            if stripped_line.startswith(':::') and not stripped_line == ':::':
                div_level += 1
            elif stripped_line == ':::':
                div_level -= 1
                if div_level == 0:
                    in_div_block = False
            continue

        # Remove LaTeX commands and environments
        line = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*', '', line)  # LaTeX commands
        line = re.sub(r'\$[^$]*\$', '', line)  # Inline math
        line = re.sub(r'\$\$[^$]*\$\$', '', line)  # Display math

        # Remove any remaining Quarto-specific patterns
        line = re.sub(r'\{\{<.*?>\}\}', '', line)  # Quarto shortcodes
        line = re.sub(r'@fig-[a-zA-Z0-9-_]+', '', line)  # Figure references
        line = re.sub(r'@lst-[a-zA-Z0-9-_]+', '', line)  # List references
        line = re.sub(r'@vid-[a-zA-Z0-9-_]+', '', line)  # Video references
        line = re.sub(r'@tbl-[a-zA-Z0-9-_]+', '', line)  # Table references
        line = re.sub(r'\{#[^}]+\}', '', line)  # Section IDs
        line = re.sub(r'\{\.[\w-]+\}', '', line)  # CSS classes
        line = re.sub(r'!\[.*?\]\([^)]+\)', '', line)  # Images
        line = re.sub(r'<[^>]+>', '', line)  # HTML tags

        # Remove academic citations and footnotes
        line = re.sub(r'\[@[^\]]+\]', '', line)  # Academic citations [@author2024]
        line = re.sub(r'\[[0-9]+\]', '', line)  # Footnote numbers [1], [2], etc.

        # Remove image captions (DALL·E prompts and similar)
        line = re.sub(r'\[DALL·E[^\]]*\]', '', line)  # DALL·E prompts
        line = re.sub(r'\[.*?Prompt:.*?\]', '', line)  # Other prompts
        line = re.sub(r'\[Source:.*?\]', '', line)  # Source attributions

        # Fix broken references to removed figures
        line = re.sub(r'(\w+,?\s+)?(shown|depicted|illustrated|presented)\s+in\s*,?\s*', r'\1', line)
        line = re.sub(r',\s*as\s+(shown|depicted|illustrated)\s+in\s*,?\s*', ', ', line)
        line = re.sub(r'the timeline shows', 'the timeline showed', line)
        line = re.sub(r'(\w+)\s+shows?\s+', r'\1 demonstrates ', line)

        # Remove code blocks (lines that look like pure code)
        if (stripped_line.startswith('\\') and
            any(cmd in stripped_line for cmd in ['draw', 'node', 'fill', 'coordinate', 'clip'])):
            continue

        # Skip lines that are mostly LaTeX syntax
        if re.match(r'^[\\{}()[\],;\s%\-\d\.]*$', stripped_line) and len(stripped_line) > 10:
            continue

        # Skip footnote definitions (lines starting with [number] Definition:)
        if re.match(r'^\[\d+\]\s+[A-Z][\w\s]+:', stripped_line):
            continue

        # Clean up the line and add if not empty and meaningful
        cleaned_line = line.strip()
        if (cleaned_line and
            len(cleaned_line) > 3 and  # Minimum meaningful length
            not re.match(r'^[\\{}()[\],;\s%\-\d\.]*$', cleaned_line)):  # Not just syntax
            cleaned_lines.append(cleaned_line)

    # Join and clean up multiple spaces/newlines
    cleaned_content = '\n'.join(cleaned_lines)
    cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)  # Multiple newlines to double
    cleaned_content = re.sub(r'[ \t]+', ' ', cleaned_content)  # Multiple spaces to single

    # Final cleanup pass for any remaining artifacts
    cleaned_content = re.sub(r'\s*,\s*,\s*', ', ', cleaned_content)  # Multiple commas
    cleaned_content = re.sub(r'\s*\.\s*\.\s*', '. ', cleaned_content)  # Multiple periods
    cleaned_content = re.sub(r'\s+([,.;:!?])', r'\1', cleaned_content)  # Space before punctuation
    cleaned_content = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', cleaned_content)  # Proper sentence spacing

    return cleaned_content.strip()


def _extract_h2_sections_from_text(text: str) -> List[tuple]:
    """
    Extract ## level headers and their content from cleaned text.

    Only extracts content between ## headers, stopping at the next ##.
    """
    lines = text.split('\n')
    sections = []
    current_section_title = None
    current_section_lines = []

    for line in lines:
        stripped_line = line.strip()

        # Check for ## header (exactly 2 #, not more)
        if stripped_line.startswith('## ') and not stripped_line.startswith('### '):
            # Save previous section if it exists
            if current_section_title and current_section_lines:
                section_content = '\n'.join(current_section_lines).strip()
                if section_content:  # Only add non-empty sections
                    sections.append((current_section_title, section_content))

            # Start new section
            current_section_title = stripped_line.replace('##', '').strip()
            # No need to clean title - we preserve original titles exactly
            current_section_lines = []

        elif current_section_title:
            # Add content to current section
            current_section_lines.append(line)

    # Don't forget the last section
    if current_section_title and current_section_lines:
        section_content = '\n'.join(current_section_lines).strip()
        if section_content:
            sections.append((current_section_title, section_content))

    return sections


def load_content(directories: List[str], exclude_chapters: Optional[List[str]] = None, verbose: bool = False, quiet: bool = False) -> List[Dict]:
    """Load and filter content from directories using pypandoc for clean extraction."""
    if exclude_chapters is None:
        exclude_chapters = []  # Default to including all chapters

    if not quiet:
        print(f"📚 Loading content from: {', '.join(directories)}")
        print(f"🔧 Using pypandoc-based content extraction with section filtering")
    qmd_files = find_qmd_files(directories, quiet=quiet)

    if not quiet:
        print(f"📋 Found {len(qmd_files)} .qmd files:")
        for i, file_path in enumerate(qmd_files, 1):
            try:
                relative_path = str(Path(file_path).relative_to(Path.cwd()))
            except ValueError:
                relative_path = str(Path(file_path))
            print(f"    {i:2d}. {relative_path}")

    all_sections = []
    processed_files = []
    excluded_files = []
    total_excluded_sections = 0

    # Load filtering configuration once
    filter_config = load_section_filters()

    for file_path in qmd_files:
        # Apply file-level filtering
        should_exclude_file_result = should_exclude_file(file_path, filter_config)
        if should_exclude_file_result[0]:
            excluded_files.append(file_path)
            if verbose and not quiet:
                print(f"      🚫 Excluding entire file '{file_path}': {should_exclude_file_result[1]}")
            continue

        sections = extract_sections(file_path, verbose=verbose and not quiet, quiet=quiet)
        if sections:
            # Apply filtering
            filtered_sections = []
            chapter_name = Path(file_path).parent.name
            is_excluded = any(excluded in chapter_name.lower() for excluded in exclude_chapters)

            if is_excluded:
                excluded_files.append(file_path)
            else:
                filtered_sections = sections
                processed_files.append(file_path)
                all_sections.extend(filtered_sections)

    if not quiet:
        print(f"\n✅ PROCESSING SUMMARY:")
        print(f"📖 Processing {len(processed_files)} files:")
        for i, file_path in enumerate(processed_files, 1):
            try:
                relative_path = str(Path(file_path).relative_to(Path.cwd()))
            except ValueError:
                relative_path = str(Path(file_path))
            chapter_name = Path(file_path).parent.name
            sections_count = len([s for s in all_sections if s['file_path'] == file_path])
            print(f"    {i:2d}. {relative_path} [{chapter_name}] ({sections_count} sections)")

    if excluded_files:
        print(f"\n🚫 Excluded {len(excluded_files)} files ({', '.join(exclude_chapters)} chapters):")
        for i, file_path in enumerate(excluded_files, 1):
            try:
                relative_path = str(Path(file_path).relative_to(Path.cwd()))
            except ValueError:
                relative_path = str(Path(file_path))
            chapter_name = Path(file_path).parent.name
            print(f"    {i:2d}. {relative_path} [{chapter_name}]")

    if not quiet:
        print(f"\n📊 FINAL COUNTS:")
        print(f"    • Total files found: {len(qmd_files)}")
        print(f"    • Files processed: {len(processed_files)}")
        print(f"    • Files excluded: {len(excluded_files)}")
        print(f"    • Sections extracted: {len(all_sections)}")
        print(f"    • Quality filtering: Sections excluded for being meta-content, too short, or low-quality")
    else:
        print(f"📊 Processed {len(processed_files)} files, extracted {len(all_sections)} sections")

    return all_sections

def create_training_examples(sections: List[Dict]) -> List:
    """Create sophisticated training examples for domain adaptation."""
    try:
        from sentence_transformers import InputExample
    except ImportError:
        print("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
        return []

    train_examples = []

    # Group by chapter
    chapters = {}
    for section in sections:
        chapter = Path(section['file_path']).parent.name
        if chapter not in chapters:
            chapters[chapter] = []
        chapters[chapter].append(section)

    chapter_list = list(chapters.items())

    print("🎯 Creating sophisticated training examples...")

    # Same chapter = high similarity (75-85%)
    for chapter_sections in chapters.values():
        if len(chapter_sections) > 1:
            for i, s1 in enumerate(chapter_sections):
                for s2 in chapter_sections[i+1:]:
                    similarity = np.random.uniform(0.75, 0.85)
                    train_examples.append(InputExample(
                        texts=[s1['content'], s2['content']],
                        label=similarity
                    ))

    # Adjacent chapters = medium similarity (40-60%)
    for i, (_, sections1) in enumerate(chapter_list[:-1]):
        _, sections2 = chapter_list[i+1]
        for s1 in sections1[:2]:
            for s2 in sections2[:2]:
                similarity = np.random.uniform(0.4, 0.6)
                train_examples.append(InputExample(
                    texts=[s1['content'], s2['content']],
                    label=similarity
                ))

    # Distant chapters = low similarity (10-30%)
    for i, (_, sections1) in enumerate(chapter_list):
        for j, (_, sections2) in enumerate(chapter_list):
            if abs(i - j) >= 3:
                for s1 in sections1[:1]:
                    for s2 in sections2[:1]:
                        similarity = np.random.uniform(0.1, 0.3)
                        train_examples.append(InputExample(
                            texts=[s1['content'], s2['content']],
                            label=similarity
                        ))

    # Random negative examples
    import random
    random.seed(42)
    for _ in range(50):
        s1, s2 = random.sample(sections, 2)
        similarity = np.random.uniform(0.05, 0.25)
        train_examples.append(InputExample(
            texts=[s1['content'], s2['content']],
            label=similarity
        ))

    print(f"✅ Created {len(train_examples)} training examples")
    return train_examples

def train_model(directories: List[str], output_path: str, base_model: str = "sentence-t5-base",
                epochs: int = 5, exclude_chapters: Optional[List[str]] = None, verbose: bool = False, quiet: bool = False) -> bool:
    """Train a domain-adapted model."""
    print("🔥 TRAINING MODE: Domain Adaptation")
    print("=" * 50)

    try:
        from sentence_transformers import SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("❌ Required packages not installed. Run:")
        print("   pip install sentence-transformers torch")
        return False

    # Load content
    all_sections = load_content(directories, exclude_chapters, verbose=verbose, quiet=quiet)
    if len(all_sections) < 50:
        print(f"❌ Need at least 50 sections for training, got {len(all_sections)}")
        return False

    # Create training examples
    train_examples = create_training_examples(all_sections)
    if not train_examples:
        return False

    # Load base model
    print(f"🧠 Loading base model: {base_model}")
    try:
        model = SentenceTransformer(base_model)
        print(f"✅ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return False

    # Setup training
    train_loss = losses.MultipleNegativesRankingLoss(model)

    def collate_fn(batch):
        return batch

    train_dataloader = DataLoader(
        cast(Any, train_examples),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    warmup_steps = int(len(train_examples) // 8 * 0.2)

    # Train model
    print(f"🚀 Training for {epochs} epochs...")
    print(f"📊 Training examples: {len(train_examples)}")
    print(f"📊 Batch size: 8")
    print(f"📊 Warmup steps: {warmup_steps}")

    import time
    start_time = time.time()

    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
            output_path=output_path
        )

        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"💾 Model saved to: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def generate_cross_references(model_path: str, directories: List[str], output_file: str,
                            exclude_chapters: Optional[List[str]] = None,
                            max_suggestions: int = 5,
                            similarity_threshold: float = 0.65,
                            verbose: bool = False, quiet: bool = False,
                            explain: bool = False, ollama_model: Optional[str] = None) -> Dict:
    """Generate cross-references using any sentence-transformer model.

    Args:
        explain: If True, generate AI explanations for each cross-reference using local LLM (requires Ollama)
        ollama_model: Specific ollama model to use for explanations (e.g. 'llama3.1:8b', 'qwen2.5:7b')
    """

    if not quiet:
        print("🚀 GENERATION MODE: Cross-Reference Generation")
        print("=" * 50)

    # Handle AI explanation setup
    selected_model = "gemma2:27b"  # Default - high quality Gemma 2 model
    if explain:
        if not REQUESTS_AVAILABLE:
            print("❌ requests library not available. Install with: pip install requests")
            return {}

        # Use user-specified model if provided
        if ollama_model:
            if not quiet:
                print(f"🤖 Using specified ollama model: {ollama_model}")
            selected_model = ollama_model

            # Verify the model is available
            is_running, available_models = check_ollama_setup()
            if not is_running:
                print("❌ Ollama is not running. Please start it with: ollama serve")
                return {}

            if available_models and ollama_model not in available_models:
                print(f"⚠️  Warning: Model '{ollama_model}' not found in available models.")
                print(f"📚 Available models: {', '.join(available_models)}")
                print(f"💡 Download with: ollama run {ollama_model}")
                explain = False
            else:
                if not quiet:
                    print(f"✅ Model '{ollama_model}' is available and will be used for explanations.")
        else:
            # Use interactive setup if no model specified
            selected_model = setup_ollama_interactive()
            if not selected_model:
                print("\n⚠️  Skipping explanation generation - setup incomplete.")
                print("   Cross-references will be generated without explanations.")
                explain = False

    # Determine if model_path is a local path or HuggingFace model name
    is_local_model = Path(model_path).exists()
    model_type = "Domain-adapted" if is_local_model else "Base HuggingFace"

    if not quiet:
        print(f"📂 Model: {model_path} ({model_type})")

    # Load model
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path)
        if not quiet:
            print(f"✅ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return {}

    # Suppress pypandoc warnings in quiet mode
    if quiet:
        import os
        os.environ['PYPANDOC_PANDOC'] = os.environ.get('PYPANDOC_PANDOC', 'pandoc')
        # Redirect stderr to suppress warnings
        import subprocess
        devnull = open(os.devnull, 'w')
        old_stderr = sys.stderr
        sys.stderr = devnull

    try:
        # Load content
        all_sections = load_content(directories, exclude_chapters, verbose=verbose, quiet=quiet)
        if not all_sections:
            if not quiet:
                print("❌ No content loaded")
            return {}

        # Generate embeddings
        if not quiet:
            print("🧮 Generating embeddings...")
        embeddings = model.encode([section['content'] for section in all_sections],
                                show_progress_bar=not quiet)

        # Find similar sections
        if not quiet:
            print("🔍 Finding cross-references...")

    finally:
        # Restore stderr if we suppressed it
        if quiet:
            sys.stderr = old_stderr
            devnull.close()

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("❌ scikit-learn not installed. Run: pip install scikit-learn")
        return {}

    # Get file order for determining connection type
    file_order = find_qmd_files(directories, quiet=quiet)
    file_index_map = {file_path: i for i, file_path in enumerate(file_order)}

    nn_model = NearestNeighbors(n_neighbors=min(10, len(all_sections)), metric='cosine')
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors(embeddings)

    # Use a dictionary to collect cross-references by file and section
    file_section_refs = {}



    for i, section in enumerate(all_sections):
        source_file = section['file_path']
        source_filename = Path(source_file).name
        source_chapter = Path(section['file_path']).parent.name

        # Use the exact section ID from extraction
        source_section_id = section.get('section_id')
        if not source_section_id:
            continue  # Skip sections without valid IDs

        if source_filename not in file_section_refs:
            file_section_refs[source_filename] = {}

        if source_section_id not in file_section_refs[source_filename]:
            file_section_refs[source_filename][source_section_id] = {
                'section_title': section['title'],  # Use exact original title
                'targets': []
            }

        suggestions_count = 0

        for j in range(1, min(10, len(indices[i]))):
            if suggestions_count >= max_suggestions:
                break

            target_idx = indices[i][j]
            target_section = all_sections[target_idx]
            target_chapter = Path(target_section['file_path']).parent.name

            similarity = 1 - distances[i][j]

            # Filter: different chapter, good similarity
            if (similarity > similarity_threshold and
                source_chapter != target_chapter):

                # Use the exact section ID from extraction
                target_id = target_section.get('section_id')
                if not target_id:
                    continue  # Skip targets without valid IDs

                # Determine connection type
                source_idx = file_index_map.get(source_file, -1)
                target_idx_map = file_index_map.get(target_section['file_path'], -1)
                connection_type = "related" # Default
                if source_idx != -1 and target_idx_map != -1:
                    if target_idx_map > source_idx:
                        connection_type = "Preview"
                    else:
                        connection_type = "Background"

                # Generate explanation if requested
                explanation = ""
                if explain:
                    if not quiet:
                        print(f"   🤖 Generating explanation for: {section['title']} → {target_section['title']}")
                    explanation = generate_explanation(
                        section['content'],
                        target_section['content'],
                        section['title'],
                        target_section['title'],
                        model=selected_model
                    )

                    # Show the generated explanation nicely formatted
                    if not quiet and explanation:
                        direction_symbol = "→" if connection_type == "Preview" else "←" if connection_type == "Background" else "•"
                        full_ref = f"      {direction_symbol} {target_section['title']} (§\\ref{{{target_id}}}) {explanation}"
                        print(f"      ✨ {full_ref}")
                    elif not quiet:
                        print(f"      ❌ No explanation generated")

                target_data = {
                        'target_section_id': target_id,
                    'target_section_title': target_section['title'],  # Use exact original title
                        'connection_type': connection_type,
                        'similarity': float(similarity)
                }

                # Add explanation if generated
                if explanation:
                    target_data['explanation'] = explanation

                file_section_refs[source_filename][source_section_id]['targets'].append(target_data)
                suggestions_count += 1

    # Convert to final array structure
    cross_references = []
    for filename, sections in file_section_refs.items():
        if sections:  # Only include files that have sections with targets
            file_entry = {
                'file': filename,
                'sections': []
            }

            for section_id, section_data in sections.items():
                if section_data['targets']:  # Only include sections that have targets
                    file_entry['sections'].append({
                        'section_id': section_id,
                        'section_title': section_data['section_title'],
                        'targets': section_data['targets']
                    })

            if file_entry['sections']:  # Only add file if it has sections with targets
                cross_references.append(file_entry)

    total_refs = sum(len(section['targets']) for file_entry in cross_references for section in file_entry['sections'])

    # Save results
    result = {
        'metadata': {
            'generated_at': str(np.datetime64('now')),
            'model_used': model_path,
            'model_type': model_type,
            'total_sections': len(all_sections),
            'total_cross_references': total_refs,
            'approach': 'domain_adapted_t5' if is_local_model else 'base_model'
        },
        'cross_references': cross_references
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    if not quiet:
        print(f"\n✅ Generated {total_refs} cross-references across {len(cross_references)} files.")
        all_sims = [target['similarity'] for file_entry in cross_references for section in file_entry['sections'] for target in section['targets']]
        print(f"📊 Average similarity: {np.mean(all_sims):.3f}" if all_sims else "📊 No valid cross-references")
        print(f"📄 Results saved to: {output_file}")
        print(f"🎯 Model type: {model_type}")
    else:
        print(f"✅ Generated {total_refs} cross-references → {output_file}")

    return result

def main():
    """Main function with full CLI support."""
    parser = argparse.ArgumentParser(
        description="ML Systems Cross-Reference Generator with Domain Adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train a domain-adapted model with intelligent extraction
    python3 cross_refs.py --train -d ../../contents/core/ -o ./t5-mlsys-domain-adapted

    # Generate with domain-adapted model (uses pypandoc by default)
    python3 cross_refs.py -g -m ./t5-mlsys-domain-adapted -o cross_refs.json -d ../../contents/core/

    # Generate with base model (no training needed)
    python3 cross_refs.py -g -m sentence-t5-base -o cross_refs.json -d ../../contents/core/

    # Generate with AI explanations (requires Ollama + any model)
    python3 cross_refs.py -g -m sentence-t5-base -o cross_refs.json -d ../../contents/core/ --explain

    # Generate with specific ollama model for explanations
    python3 cross_refs.py -g -m sentence-t5-base -o cross_refs.json -d ../../contents/core/ --explain --ollama-model gemma2:27b

    # Test extraction methods
    python3 test_intelligent_extraction.py
        """)

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Training mode: Domain-adapt a base model')
    mode_group.add_argument('-g', '--generate', action='store_true', help='Generation mode: Generate cross-references')

    # Common arguments
    parser.add_argument('-d', '--dirs', nargs='+', required=True,
                       help='Directories containing .qmd files')
    parser.add_argument('-o', '--output', required=True,
                       help='Output path (model directory for training, JSON file for generation)')

    # Training-specific arguments
    parser.add_argument('--base-model', default='sentence-t5-base',
                       help='Base model for training (default: sentence-t5-base)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')

    # Generation-specific arguments
    parser.add_argument('-m', '--model',
                       help='Model path (for generation): local path or HuggingFace name')

    # Optional arguments
    parser.add_argument('--exclude-chapters', nargs='*', default=[],
                        help='Space-separated list of chapter folder names to exclude (e.g., introduction conclusion)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--max-suggestions', type=int, default=5,
                        help='Max cross-references per section (default: 5)')
    parser.add_argument('-t', '--threshold', type=float, default=0.65,
                        help='Minimum similarity for cross-references (default: 0.65)')
    parser.add_argument('--explain', action='store_true',
                        help='Generate AI explanations for cross-references using local LLM (requires Ollama)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity (only show essential information)')
    parser.add_argument('--ollama-model',
                        help='Specific Ollama model to use for explanations (e.g., llama3.1:8b, qwen2.5:7b)')

    args = parser.parse_args()

    # Validate arguments
    if args.generate and not args.model:
        print("❌ Generation mode requires --model argument")
        return 1

    # Validate directories
    for directory in args.dirs:
        if not Path(directory).exists():
            print(f"❌ Directory not found: {directory}")
            return 1

    try:
        if args.train:
            # Training mode
            success = train_model(
                directories=args.dirs,
                output_path=args.output,
                base_model=args.base_model,
                epochs=args.epochs,
                exclude_chapters=args.exclude_chapters,
                verbose=args.verbose,
                quiet=args.quiet
            )
            return 0 if success else 1

        elif args.generate:
            # Generation mode
            result = generate_cross_references(
                model_path=args.model,
                directories=args.dirs,
                output_file=args.output,
                exclude_chapters=args.exclude_chapters,
                max_suggestions=args.max_suggestions,
                similarity_threshold=args.threshold,
                verbose=args.verbose,
                quiet=args.quiet,
                explain=args.explain,
                ollama_model=args.ollama_model
            )

            if result and 'cross_references' in result and result['cross_references']:
                if not args.quiet:
                    print(f"🎉 Success! Ready for Quarto injection. NOTE: lua/inject_xrefs.lua may need updates for the new JSON structure.")
                return 0
            else:
                if not args.quiet:
                    print("⚠️  No cross-references generated")
                return 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
