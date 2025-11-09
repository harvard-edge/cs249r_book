#!/usr/bin/env python3
"""
Generate Alt Text for Images

This script generates accessibility alt text for images in the Machine Learning Systems book.
It works by:
1. Building the book locally (using binder)
2. Parsing the HTML to find figures with their IDs
3. Sending images to a vision model (OpenAI or Ollama)
4. Matching figure IDs back to source .qmd files
5. Adding fig-alt attributes to the source

Usage:
    # Using OpenAI (requires OPENAI_API_KEY)
    python generate_alt_text.py --chapter intro --provider openai
    
    # Using Ollama (local, requires llava model)
    python generate_alt_text.py --chapter intro --provider ollama
    
    # Dry run (no changes to source files)
    python generate_alt_text.py --chapter intro --provider ollama --dry-run
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("generate_alt_text.log")
    ]
)

# Constants
REPO_ROOT = Path(__file__).resolve().parents[3]
QUARTO_DIR = REPO_ROOT / "quarto"
BUILD_DIR = QUARTO_DIR / "_build" / "html"

# Prompt template for alt text generation
ALT_TEXT_PROMPT = """You are an expert at creating accessible alt text for images in technical textbooks.

Your task is to write concise, informative alt text for this image from a machine learning systems textbook.

Guidelines:
- Be concise but descriptive (aim for 1-2 sentences, max 125 characters if possible)
- Describe what's visually important, not what's obvious from the caption
- Focus on the key information the image conveys
- For diagrams: describe the structure, flow, and relationships
- For graphs: describe the trend, comparison, or key insight
- For screenshots: describe the UI element and its purpose
- Don't start with "Image of" or "Figure showing"
- Don't repeat information from the caption

Context:
Caption: {caption}
Section: {section}

Provide only the alt text, nothing else."""


class OllamaClient:
    """Simple wrapper for Ollama API to match OpenAI client interface."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llava"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        
    class ChatCompletions:
        def __init__(self, parent):
            self.parent = parent
            
        def create(self, model: str, messages: List[Dict], **kwargs):
            """Create a chat completion using Ollama API with vision support."""
            # Convert OpenAI format to Ollama format
            ollama_messages = []
            
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    # Handle vision messages with image content
                    text_parts = []
                    images = []
                    
                    for item in msg["content"]:
                        if item["type"] == "text":
                            text_parts.append(item["text"])
                        elif item["type"] == "image_url":
                            # Extract base64 image data
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                # Extract base64 part
                                base64_data = image_url.split(",", 1)[1]
                                images.append(base64_data)
                    
                    ollama_messages.append({
                        "role": msg["role"],
                        "content": " ".join(text_parts),
                        "images": images
                    })
                else:
                    ollama_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Call Ollama API
            url = f"{self.parent.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": ollama_messages,
                "stream": False
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Convert to OpenAI-like response format
                class Choice:
                    def __init__(self, content):
                        self.message = type('obj', (object,), {'content': content})()
                
                class Response:
                    def __init__(self, content):
                        self.choices = [Choice(content)]
                
                return Response(result["message"]["content"])
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    
    @property
    def chat(self):
        if not hasattr(self, '_chat'):
            self._chat = type('obj', (object,), {'completions': self.ChatCompletions(self)})()
        return self._chat


class FigureInfo:
    """Information about a figure extracted from HTML."""
    
    def __init__(self, figure_id: str, image_path: str, caption: str, section: str = ""):
        self.figure_id = figure_id
        self.image_path = image_path
        self.caption = caption
        self.section = section
        self.alt_text = None
        
    def __repr__(self):
        return f"FigureInfo(id={self.figure_id}, path={self.image_path})"


def build_chapter(chapter: str) -> bool:
    """
    Build a specific chapter using binder.
    
    Args:
        chapter: Chapter name (e.g., 'intro', 'introduction')
    
    Returns:
        True if build succeeded, False otherwise
    """
    logging.info(f"Building chapter: {chapter}")
    
    # Check if binder exists
    binder_path = REPO_ROOT / "binder"
    if not binder_path.exists():
        logging.error("binder script not found")
        return False
    
    try:
        # Run binder html <chapter>
        result = subprocess.run(
            ["./binder", "html", chapter],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logging.info(f"Successfully built chapter: {chapter}")
            return True
        else:
            logging.error(f"Build failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("Build timed out after 5 minutes")
        return False
    except Exception as e:
        logging.error(f"Build error: {e}")
        return False


def find_html_file(chapter: str) -> Optional[Path]:
    """
    Find the HTML file for a chapter in the build directory.
    
    Args:
        chapter: Chapter name
    
    Returns:
        Path to HTML file or None if not found
    """
    # Common chapter name mappings
    chapter_files = {
        "intro": "introduction.html",
        "introduction": "introduction.html",
        "ml_systems": "ml_systems.html",
        "dl_primer": "dl_primer.html",
        # Add more mappings as needed
    }
    
    # Try direct mapping
    if chapter in chapter_files:
        html_file = BUILD_DIR / "contents" / "core" / chapter_files[chapter]
        if html_file.exists():
            return html_file
    
    # Try searching for files
    patterns = [
        BUILD_DIR / "contents" / "core" / f"{chapter}.html",
        BUILD_DIR / "contents" / "core" / f"*{chapter}*.html",
    ]
    
    for pattern in patterns:
        matches = list(BUILD_DIR.glob(str(pattern.relative_to(BUILD_DIR))))
        if matches:
            return matches[0]
    
    logging.error(f"Could not find HTML file for chapter: {chapter}")
    return None


def extract_figures_from_html(html_path: Path) -> List[FigureInfo]:
    """
    Extract figure information from HTML file.
    
    Args:
        html_path: Path to HTML file
    
    Returns:
        List of FigureInfo objects
    """
    logging.info(f"Extracting figures from: {html_path}")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    figures = []
    
    # Find all figure elements
    for figure in soup.find_all('figure', class_='quarto-float-fig'):
        # Extract figure ID from aria-describedby or figcaption id
        figure_id = None
        aria_desc = figure.find('div', attrs={'aria-describedby': True})
        if aria_desc:
            caption_id = aria_desc['aria-describedby']
            # Extract fig-xxx from fig-xxx-caption-...
            match = re.match(r'(fig-[^-]+(?:-[^-]+)*)-caption', caption_id)
            if match:
                figure_id = match.group(1)
        
        if not figure_id:
            logging.warning("Found figure without ID, skipping")
            continue
        
        # Extract image path
        img = figure.find('img')
        if not img or 'src' not in img.attrs:
            logging.warning(f"Figure {figure_id} has no image, skipping")
            continue
        
        image_path = img['src']
        
        # Extract caption
        figcaption = figure.find('figcaption')
        caption = figcaption.get_text(strip=True) if figcaption else ""
        
        # Try to find section heading
        section = ""
        # Look for nearest preceding heading
        prev_heading = figure.find_previous(['h1', 'h2', 'h3'])
        if prev_heading:
            section = prev_heading.get_text(strip=True)
        
        fig_info = FigureInfo(figure_id, image_path, caption, section)
        figures.append(fig_info)
        logging.info(f"Found figure: {figure_id}")
    
    logging.info(f"Extracted {len(figures)} figures")
    return figures


def encode_image(image_path: Path) -> str:
    """
    Encode image to base64 string.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_alt_text_openai(client: OpenAI, figure: FigureInfo, image_path: Path) -> str:
    """
    Generate alt text using OpenAI's vision model.
    
    Args:
        client: OpenAI client
        figure: Figure information
        image_path: Path to image file
    
    Returns:
        Generated alt text
    """
    logging.info(f"Generating alt text for {figure.figure_id} using OpenAI")
    
    # Encode image
    base64_image = encode_image(image_path)
    
    # Prepare prompt
    prompt = ALT_TEXT_PROMPT.format(
        caption=figure.caption,
        section=figure.section
    )
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",  # Updated model that supports vision
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    alt_text = response.choices[0].message.content.strip()
    logging.info(f"Generated alt text: {alt_text}")
    return alt_text


def generate_alt_text_ollama(client: OllamaClient, figure: FigureInfo, image_path: Path) -> str:
    """
    Generate alt text using Ollama's vision model.
    
    Args:
        client: Ollama client
        figure: Figure information
        image_path: Path to image file
    
    Returns:
        Generated alt text
    """
    logging.info(f"Generating alt text for {figure.figure_id} using Ollama")
    
    # Encode image
    base64_image = encode_image(image_path)
    
    # Prepare prompt
    prompt = ALT_TEXT_PROMPT.format(
        caption=figure.caption,
        section=figure.section
    )
    
    # Call Ollama API
    response = client.chat.completions.create(
        model=client.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )
    
    alt_text = response.choices[0].message.content.strip()
    logging.info(f"Generated alt text: {alt_text}")
    return alt_text


def find_qmd_files(chapter: str) -> List[Path]:
    """
    Find .qmd files for a chapter.
    
    Args:
        chapter: Chapter name
    
    Returns:
        List of .qmd file paths
    """
    # Search in contents/core
    core_dir = QUARTO_DIR / "contents" / "core"
    
    # Try to find directory matching chapter name
    chapter_patterns = [
        chapter,
        f"*{chapter}*",
    ]
    
    qmd_files = []
    for pattern in chapter_patterns:
        matches = list(core_dir.glob(f"{pattern}/**/*.qmd"))
        qmd_files.extend(matches)
        
        # Also try direct files
        direct_matches = list(core_dir.glob(f"{pattern}.qmd"))
        qmd_files.extend(direct_matches)
    
    # Remove duplicates
    qmd_files = list(set(qmd_files))
    
    logging.info(f"Found {len(qmd_files)} .qmd files for chapter {chapter}")
    return qmd_files


def find_figure_in_qmd(figure_id: str, qmd_path: Path) -> Optional[Tuple[int, str]]:
    """
    Find a figure reference in a .qmd file.
    
    Args:
        figure_id: Figure ID to find (e.g., 'fig-ai-timeline')
        qmd_path: Path to .qmd file
    
    Returns:
        Tuple of (line_number, line_content) if found, None otherwise
    """
    with open(qmd_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Look for figure ID in various forms
    patterns = [
        rf'\{{#{figure_id}\}}',  # {#fig-xxx}
        rf'#\| label: {figure_id}',  # #| label: fig-xxx
        rf'id="{figure_id}"',  # id="fig-xxx"
    ]
    
    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.search(pattern, line):
                return (i, line)
    
    return None


def add_alt_text_to_qmd(figure_id: str, alt_text: str, qmd_path: Path, dry_run: bool = False) -> bool:
    """
    Add or update fig-alt attribute in a .qmd file.
    
    Args:
        figure_id: Figure ID
        alt_text: Alt text to add
        qmd_path: Path to .qmd file
        dry_run: If True, don't actually modify the file
    
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Adding alt text to {qmd_path} for {figure_id}")
    
    with open(qmd_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the figure
    result = find_figure_in_qmd(figure_id, qmd_path)
    if not result:
        logging.warning(f"Could not find {figure_id} in {qmd_path}")
        return False
    
    line_num, line_content = result
    
    # Check if fig-alt already exists
    if 'fig-alt' in line_content:
        logging.info(f"Figure {figure_id} already has fig-alt, updating")
        # Update existing fig-alt
        updated_line = re.sub(
            r'fig-alt="[^"]*"',
            f'fig-alt="{alt_text}"',
            line_content
        )
    else:
        # Add fig-alt attribute
        # Try to add it to the same line if it's an inline figure
        if '{#' in line_content:
            # Inline figure: ![caption](path){#fig-xxx}
            updated_line = line_content.replace(
                f'{{#{figure_id}}}',
                f'{{#{figure_id} fig-alt="{alt_text}"}}'
            )
        else:
            # Block figure with #| label:
            # Add fig-alt on the next line
            lines.insert(line_num + 1, f'#| fig-alt: "{alt_text}"\n')
            updated_line = None
    
    if updated_line:
        lines[line_num] = updated_line
    
    if dry_run:
        logging.info(f"[DRY RUN] Would update {qmd_path}")
        logging.info(f"[DRY RUN] Line {line_num}: {line_content.strip()}")
        if updated_line:
            logging.info(f"[DRY RUN] New line: {updated_line.strip()}")
        return True
    
    # Write back to file
    with open(qmd_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    logging.info(f"Successfully updated {qmd_path}")
    return True


def process_chapter(chapter: str, provider: str, model: str, dry_run: bool = False) -> Dict:
    """
    Process a chapter to generate and add alt text.
    
    Args:
        chapter: Chapter name
        provider: 'openai' or 'ollama'
        model: Model name
        dry_run: If True, don't modify files
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "total_figures": 0,
        "alt_text_generated": 0,
        "files_updated": 0,
        "errors": 0
    }
    
    # Step 1: Build the chapter
    logging.info(f"=== Processing chapter: {chapter} ===")
    if not build_chapter(chapter):
        logging.error("Build failed, aborting")
        return stats
    
    # Step 2: Find the HTML file
    html_path = find_html_file(chapter)
    if not html_path:
        logging.error("Could not find HTML file, aborting")
        return stats
    
    # Step 3: Extract figures
    figures = extract_figures_from_html(html_path)
    stats["total_figures"] = len(figures)
    
    if not figures:
        logging.info("No figures found in this chapter")
        return stats
    
    # Step 4: Initialize client
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.error("OPENAI_API_KEY not set")
            return stats
        client = OpenAI(api_key=api_key)
    else:  # ollama
        client = OllamaClient(model=model)
    
    # Step 5: Generate alt text for each figure
    for figure in figures:
        try:
            # Find the actual image file
            image_rel_path = figure.image_path
            # Remove leading / if present
            if image_rel_path.startswith('/'):
                image_rel_path = image_rel_path[1:]
            
            # Try different possible locations
            possible_paths = [
                html_path.parent / image_rel_path,
                BUILD_DIR / image_rel_path,
                html_path.parent / Path(image_rel_path).name,
            ]
            
            image_path = None
            for path in possible_paths:
                if path.exists():
                    image_path = path
                    break
            
            if not image_path:
                logging.warning(f"Could not find image file: {figure.image_path}")
                stats["errors"] += 1
                continue
            
            # Generate alt text
            if provider == "openai":
                alt_text = generate_alt_text_openai(client, figure, image_path)
            else:
                alt_text = generate_alt_text_ollama(client, figure, image_path)
            
            figure.alt_text = alt_text
            stats["alt_text_generated"] += 1
            
        except Exception as e:
            logging.error(f"Error generating alt text for {figure.figure_id}: {e}")
            stats["errors"] += 1
            continue
    
    # Step 6: Find .qmd files
    qmd_files = find_qmd_files(chapter)
    
    # Step 7: Update .qmd files
    for figure in figures:
        if not figure.alt_text:
            continue
        
        # Try to find which .qmd file contains this figure
        found = False
        for qmd_path in qmd_files:
            if find_figure_in_qmd(figure.figure_id, qmd_path):
                if add_alt_text_to_qmd(figure.figure_id, figure.alt_text, qmd_path, dry_run):
                    stats["files_updated"] += 1
                    found = True
                    break
        
        if not found:
            logging.warning(f"Could not find {figure.figure_id} in any .qmd file")
            stats["errors"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate alt text for images in ML Systems book",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--chapter", "-c",
        required=True,
        help="Chapter to process (e.g., 'intro', 'ml_systems')"
    )
    
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["openai", "ollama"],
        help="AI provider to use (default: ollama)"
    )
    
    parser.add_argument(
        "--model",
        help="Model to use (default: gpt-4o for OpenAI, llava for Ollama)"
    )
    
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't modify files, just show what would be done"
    )
    
    args = parser.parse_args()
    
    # Set default model
    if not args.model:
        if args.provider == "openai":
            args.model = "gpt-4o"
        else:
            args.model = "llava"
    
    logging.info(f"Starting alt text generation for chapter: {args.chapter}")
    logging.info(f"Provider: {args.provider}, Model: {args.model}")
    if args.dry_run:
        logging.info("DRY RUN MODE - No files will be modified")
    
    # Process the chapter
    stats = process_chapter(args.chapter, args.provider, args.model, args.dry_run)
    
    # Print summary
    print("\n" + "="*60)
    print("ALT TEXT GENERATION SUMMARY")
    print("="*60)
    print(f"Total figures found: {stats['total_figures']}")
    print(f"Alt text generated: {stats['alt_text_generated']}")
    print(f"Files updated: {stats['files_updated']}")
    print(f"Errors: {stats['errors']}")
    print("="*60)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files were actually modified")
    else:
        print(f"\n✅ Successfully processed chapter: {args.chapter}")
        print(f"📝 Check the log file for details: generate_alt_text.log")


if __name__ == "__main__":
    main()


