#!/usr/bin/env python3
"""
Content Caption Improvement Script (Unified Version)

A comprehensive tool for analyzing, validating, and improving figure and table captions
in a Quarto-based textbook using local Ollama models.

Workflow:
1. Build content map from QMD files (--build-qmd-map)
2. Validate QMD mapping (--validate) 
3. Check caption quality (--check/-c)
4. Repair only broken captions (--repair/-r)
5. Update all captions (--update)
"""

import argparse
import base64
import json
import os
import re
import requests
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from titlecase import titlecase

class CaptionQualityChecker:
    """Analyzes caption quality and identifies issues."""
    
    def __init__(self):
        self.quality_rules = {
            'missing_punctuation': self._check_punctuation,
            'poor_capitalization': self._check_capitalization, 
            'too_generic': self._check_generic,
            'missing_bold_pattern': self._check_bold_pattern,
            'broken_formatting': self._check_formatting
        }
    
    def _check_punctuation(self, caption: str) -> Tuple[bool, str]:
        """Check if caption ends with proper punctuation."""
        if not caption or caption.strip().endswith(('.', '!', '?')):
            return True, ""
        return False, "Missing period"
    
    def _check_capitalization(self, caption: str) -> Tuple[bool, str]:
        """Check if capitalization follows style guide."""
        if not caption:
            return True, ""
        
        # Check for obvious issues like all caps or all lowercase
        if caption.isupper():
            return False, "All caps"
        if caption.islower() and not caption.startswith('**'):
            return False, "All lowercase"
        
        # More sophisticated checks could be added here
        return True, ""
    
    def _check_generic(self, caption: str) -> Tuple[bool, str]:
        """Check for overly generic captions."""
        if not caption:
            return True, ""
        
        generic_patterns = [
            r'^Figure shows',
            r'^Figure \d+',
            r'^Table shows',
            r'^Table \d+',
            r'^Diagram of',
            r'^Image of',
            r'^Screenshot',
            r'^Example$',
            r'^Overview$',
            r'^Comparison$',
        ]
        
        for pattern in generic_patterns:
            if re.search(pattern, caption, re.IGNORECASE):
                return False, "Too generic"
        
        return True, ""
    
    def _check_bold_pattern(self, caption: str) -> Tuple[bool, str]:
        """Check if caption follows **Concept**: explanation pattern."""
        if not caption:
            return True, ""
        
        # This is optional - some captions might not need bold pattern
        # For now, we'll be lenient and not require it
        return True, ""
    
    def _check_formatting(self, caption: str) -> Tuple[bool, str]:
        """Check for broken markdown formatting."""
        if not caption:
            return True, ""
        
        # Check for unmatched bold markers
        bold_count = caption.count('**')
        if bold_count % 2 != 0:
            return False, "Unmatched ** markers"
        
        # Check for other formatting issues
        if '{{' in caption or '}}' in caption:
            return False, "LaTeX artifacts"
        
        return True, ""
    
    def analyze_caption(self, caption: str) -> Dict[str, any]:
        """Analyze a single caption and return quality report."""
        issues = []
        suggestions = []
        
        for rule_name, rule_func in self.quality_rules.items():
            is_good, issue_desc = rule_func(caption)
            if not is_good:
                issues.append({
                    'type': rule_name,
                    'description': issue_desc
                })
        
        return {
            'caption': caption,
            'issues': issues,
            'needs_repair': len(issues) > 0,
            'suggestions': suggestions
        }

class FigureCaptionImprover:
    def __init__(self, model_name="llava:7b"):
        self.model_name = model_name
        self.figure_pattern = re.compile(r'@fig-([a-zA-Z0-9_-]+)')
        self.stats = {
            'files_processed': 0,
            'figures_found': 0,
            'figures_improved': 0,
            'tables_found': 0,
            'tables_improved': 0,
            'images_found': 0,
            'images_missing': 0,
            'json_success': 0,
            'json_failed': 0,
            'errors': []
        }
        self.content_map_file = "content_map.json"
        self.quality_checker = CaptionQualityChecker()
        self.quarto_config_file = "_quarto.yml"
        
    def find_qmd_files(self, directory: str) -> List[Path]:
        """Find all .qmd files in a directory recursively."""
        directory_path = Path(directory)
        return list(directory_path.rglob("*.qmd"))
    
    def get_book_chapters_from_quarto(self) -> Dict[str, List[str]]:
        """Parse _quarto.yml and return active and commented chapter files."""
        if not os.path.exists(self.quarto_config_file):
            print(f"❌ Quarto config not found: {self.quarto_config_file}")
            return {'active': [], 'commented': []}
        
        try:
            with open(self.quarto_config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            chapters = config.get('book', {}).get('chapters', [])
            active_chapters = []
            
            for chapter in chapters:
                # Handle different chapter formats
                if isinstance(chapter, str):
                    # Simple string chapter (e.g., "index.qmd")
                    if chapter.endswith('.qmd'):
                        active_chapters.append(chapter)
                elif isinstance(chapter, dict):
                    # Part or complex chapter structure
                    if 'part' in chapter and chapter['part'].endswith('.qmd'):
                        active_chapters.append(chapter['part'])
                    # Could add more complex handling here if needed
            
            # Also read the raw file to find commented chapters
            with open(self.quarto_config_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Find commented chapter lines
            commented_chapters = []
            for line in raw_content.split('\n'):
                line = line.strip()
                if line.startswith('# - ') and line.endswith('.qmd'):
                    # Remove '# - ' prefix and clean up
                    commented_chapter = line[4:].strip()
                    commented_chapters.append(commented_chapter)
            
            print(f"📚 Found {len(active_chapters)} active chapters, {len(commented_chapters)} commented chapters")
            return {
                'active': active_chapters,
                'commented': commented_chapters
            }
            
        except Exception as e:
            print(f"❌ Error parsing {self.quarto_config_file}: {e}")
            return {'active': [], 'commented': []}
    
    def find_qmd_files_in_order(self, directories: List[str]) -> List[Path]:
        """Find QMD files following the book's chapter order from _quarto.yml."""
        book_structure = self.get_book_chapters_from_quarto()
        active_chapters = book_structure.get('active', [])
        
        if not active_chapters:
            print("⚠️  No book structure found, falling back to directory scan")
            # Fallback to original method
            all_files = []
            for directory in directories:
                all_files.extend(self.find_qmd_files(directory))
            return all_files
        
        # Filter book chapters to only those in specified directories
        filtered_chapters = []
        directory_set = {os.path.normpath(d) for d in directories}
        
        for chapter_path in active_chapters:
            chapter_full_path = Path(chapter_path)
            
            # Check if this chapter is within any of the specified directories
            for directory in directory_set:
                try:
                    # Try to see if chapter is within directory
                    chapter_full_path.relative_to(directory)
                    if chapter_full_path.exists():
                        filtered_chapters.append(chapter_full_path)
                    break
                except ValueError:
                    # Not within this directory, continue
                    continue
        
        print(f"📖 Processing {len(filtered_chapters)} chapters in book order")
        return filtered_chapters
    
    def check_commented_chapters_in_directories(self, directories: List[str]) -> Dict:
        """Check if any chapters are commented out within the target directories."""
        book_structure = self.get_book_chapters_from_quarto()
        commented_chapters = book_structure.get('commented', [])
        
        # Normalize directory paths
        directory_set = {os.path.normpath(d) for d in directories}
        
        issues = {
            'commented_in_target_dirs': [],
            'total_issues': 0,
            'should_halt': False
        }
        
        # Check if any commented chapters are within our target directories
        for commented_chapter in commented_chapters:
            chapter_path = Path(commented_chapter)
            
            # Check if this commented chapter is within any target directory
            for directory in directory_set:
                try:
                    chapter_path.relative_to(directory)
                    # If we get here, the commented chapter is within this directory
                    issues['commented_in_target_dirs'].append({
                        'chapter': commented_chapter,
                        'directory': directory
                    })
                    break
                except ValueError:
                    # Not within this directory, continue
                    continue
        
        issues['total_issues'] = len(issues['commented_in_target_dirs'])
        issues['should_halt'] = issues['total_issues'] > 0
        
        return issues
    
    def print_commented_chapter_issues(self, issues: Dict):
        """Print issues about commented chapters and halt if necessary."""
        if issues['total_issues'] == 0:
            return False
            
        print(f"\n🚨 CRITICAL ISSUE:")
        print(f"Found {issues['total_issues']} commented chapters in target directories")
        print(f"Processing cannot continue as QMD files will be inconsistent.")
        
        print(f"\n📁 Commented chapters in target directories:")
        for item in issues['commented_in_target_dirs']:
            print(f"  • {item['chapter']} (in directory: {item['directory']})")
        
        print(f"\n💡 To fix:")
        print(f"   1. Uncomment these chapters in _quarto.yml, OR")
        print(f"   2. Exclude these directories from processing, OR") 
        print(f"   3. Run 'quarto render --to titlepage-pdf' after uncommenting")
        print(f"\n❌ HALTING EXECUTION - Please resolve these issues first.")
        
        return True  # Should halt
    
    def normalize_caption_punctuation(self, caption: str) -> str:
        """Ensure caption ends with a period for academic formatting."""
        if not caption:
            return caption
            
        caption = caption.strip()
        
        # Don't add period if already ends with punctuation
        if caption and not caption.endswith(('.', '!', '?')):
            caption += '.'
            
        return caption

    def normalize_caption_case(self, caption: str) -> str:
        """Normalize caption case using proper English title case."""
        if not caption:
            return caption
        
        # Use titlecase library for proper English title case
        return titlecase(caption)
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode an image to base64 for multimodal models."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding image {image_path}: {e}")
            return None
    
    def extract_section_context(self, content: str, figure_or_table_id: str) -> Dict[str, str]:
        """Extract the section context around a figure or table reference."""
        lines = content.split('\n')
        context_lines = []
        section_title = "Unknown Section"
        
        # Find the line containing the figure/table reference
        ref_line_idx = None
        for i, line in enumerate(lines):
            if figure_or_table_id in line and (line.strip().startswith('![') or line.strip().startswith(':') or ':::' in line):
                ref_line_idx = i
                break
        
        if ref_line_idx is None:
            # Fallback: search for @fig- or @tbl- references
            for i, line in enumerate(lines):
                if f"@{figure_or_table_id}" in line:
                    ref_line_idx = i
                    break
        
        if ref_line_idx is not None:
            # Extract context around the reference (±20 lines)
            start_idx = max(0, ref_line_idx - 20)
            end_idx = min(len(lines), ref_line_idx + 20)
            
            # Find section heading above the reference
            for i in range(ref_line_idx, -1, -1):
                line = lines[i].strip()
                if line.startswith('#') and not line.startswith('###'):
                    # Extract section title (remove # and {#...} tags)
                    section_title = re.sub(r'^#+\s*', '', line)
                    section_title = re.sub(r'\s*\{#[^}]+\}.*$', '', section_title)
                    break
            
            context_lines = lines[start_idx:end_idx]
        else:
            # No specific reference found, use first few paragraphs
            context_lines = lines[:50]
        
        context_text = '\n'.join(context_lines).strip()
        
        return {
            'title': section_title,
            'content': context_text
        }
    
    def generate_caption_with_ollama(self, section_title: str, section_text: str, 
                                   figure_id: str, current_caption: str, 
                                   image_path: Optional[str] = None) -> Optional[str]:
        """Generate improved caption using Ollama multimodal model."""
        
        # Construct the prompt requesting **bold**: explanation format
        prompt = f"""You are a textbook editor expert improving figure and table captions for educational materials.

Context:
- Section: {section_title}
- Figure/Table ID: {figure_id}
- Current caption: {current_caption}

Section content:
{section_text[:2000]}  # Limit context to avoid token limits

Task: Create an improved caption that:
1. Uses EXACTLY this format: **Bold Key Concept**: Clear educational explanation
2. Is more informative and educational than the current caption
3. Helps students understand the figure/table's educational purpose
4. Uses proper academic language
5. Is concise but comprehensive

CRITICAL: Your response must be ONLY the improved caption in this exact format:
**Bold Key Concept**: Educational explanation that helps students learn

Example good format:
**Machine Learning Pipeline**: Comprehensive workflow showing data preprocessing, model training, and evaluation stages for developing robust AI systems.

Respond with ONLY the improved caption, nothing else:"""

        try:
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent formatting
                    "num_predict": 150   # Limit response length
                }
            }
            
            # Add image if provided (for multimodal models)
            if image_path and os.path.exists(image_path):
                encoded_image = self.encode_image(image_path)
                if encoded_image:
                    payload["images"] = [encoded_image]
            
            # Make request to Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                new_caption = result.get('response', '').strip()
                
                # Clean up any markdown code blocks
                if new_caption.startswith('```') and new_caption.endswith('```'):
                    new_caption = new_caption.strip('`').strip()
                if new_caption.startswith('json\n'):
                    new_caption = new_caption[5:].strip()
                
                # Validate the format contains **bold**: 
                if '**' in new_caption and ':' in new_caption:
                    return new_caption
                else:
                    print(f"      ⚠️  Generated caption doesn't follow **bold**: format: {new_caption[:100]}")
                    return None
            else:
                print(f"      ❌ Ollama API error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"      ❌ Request error: {e}")
            return None
        except Exception as e:
            print(f"      ❌ Unexpected error: {e}")
            return None
    
    def compile_tikz_to_image(self, tikz_code: str, figure_id: str) -> Optional[str]:
        """Compile TikZ code to a PNG image for multimodal processing."""
        temp_dir = Path("temp_tikz")
        temp_dir.mkdir(exist_ok=True)
        
        tex_file = temp_dir / f"{figure_id}.tex"
        pdf_file = temp_dir / f"{figure_id}.pdf"
        png_file = temp_dir / f"{figure_id}.png"
        
        # Create minimal LaTeX document with TikZ
        latex_content = f"""\\documentclass{{standalone}}
\\usepackage{{tikz}}
\\usepackage{{pgfplots}}
\\usetikzlibrary{{positioning,arrows,shapes,calc}}
\\begin{{document}}
{tikz_code}
\\end{{document}}"""
        
        try:
            # Write LaTeX file
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            # Compile to PDF
            result = subprocess.run(
                ["pdflatex", "-output-directory", str(temp_dir), str(tex_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0 or not pdf_file.exists():
                print(f"      ❌ LaTeX compilation failed for {figure_id}")
                return None
            
            # Convert PDF to PNG using ImageMagick
            result = subprocess.run(
                ["magick", "convert", "-density", "150", str(pdf_file), str(png_file)],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode != 0 or not png_file.exists():
                print(f"      ❌ PDF to PNG conversion failed for {figure_id}")
                return None
            
            return str(png_file)
            
        except subprocess.TimeoutExpired:
            print(f"      ❌ Compilation timeout for {figure_id}")
            return None
        except Exception as e:
            print(f"      ❌ Compilation error for {figure_id}: {e}")
            return None
        finally:
            # Clean up intermediate files
            for temp_file in [tex_file, pdf_file, tex_file.with_suffix('.aux'), 
                            tex_file.with_suffix('.log')]:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
    
    def parse_sections(self, content: str) -> List[Dict[str, any]]:
        """Parse QMD content to extract sections with their positions."""
        lines = content.split('\n')
        sections = []
        current_section = None
        
        for i, line in enumerate(lines):
            # Detect section headers (## or ###)
            if re.match(r'^##\s+', line) and not line.startswith('###'):
                # Save previous section
                if current_section:
                    current_section['end_line'] = i - 1
                    current_section['content'] = '\n'.join(lines[current_section['start_line']:i])
                    sections.append(current_section)
                
                # Start new section
                title = re.sub(r'^##\s*', '', line)
                title = re.sub(r'\s*\{#[^}]+\}.*$', '', title)  # Remove {#id} tags
                
                current_section = {
                    'title': title.strip(),
                    'start_line': i,
                    'end_line': len(lines) - 1,  # Will be updated when next section found
                    'content': ''
                }
        
        # Don't forget the last section
        if current_section:
            current_section['end_line'] = len(lines) - 1
            current_section['content'] = '\n'.join(lines[current_section['start_line']:])
            sections.append(current_section)
        
        return sections
    
    def load_content_map(self) -> Dict:
        """Load existing content map from JSON file."""
        if not os.path.exists(self.content_map_file):
            print(f"❌ Content map not found: {self.content_map_file}")
            print(f"💡 Run: python {os.path.basename(__file__)} --build-map")
            return {}
        
        try:
            with open(self.content_map_file, 'r', encoding='utf-8') as f:
                content_map = json.load(f)
            print(f"📋 Loaded content map: {len(content_map.get('figures', {}))} figures, {len(content_map.get('tables', {}))} tables")
            return content_map
        except Exception as e:
            error_msg = f"Error loading content map: {e}"
            print(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return {}
    
    def save_content_map(self, content_map: Dict):
        """Save content map to JSON file with proper serialization."""
        def convert_paths_to_strings(obj):
            """Recursively convert Path objects to strings for JSON serialization."""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: convert_paths_to_strings(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths_to_strings(item) for item in obj]
            else:
                return obj
        
        try:
            # Convert any Path objects to strings
            serializable_map = convert_paths_to_strings(content_map)
            
            with open(self.content_map_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_map, f, indent=2, ensure_ascii=False)
            print(f"💾 Content map saved to: {self.content_map_file}")
        except Exception as e:
            print(f"❌ Error saving content map: {e}")
            raise

    def find_qmd_files(self, directories: List[str]) -> List[str]:
        """Find all .qmd files in specified directories."""
        qmd_files = []
        for directory in directories:
            if os.path.isfile(directory) and directory.endswith('.qmd'):
                qmd_files.append(directory)
            elif os.path.isdir(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.qmd'):
                            qmd_files.append(os.path.join(root, file))
        return qmd_files

    def find_figure_definition_in_qmd(self, content: str, fig_id: str) -> Optional[Dict[str, str]]:
        """
        Unified figure detection across all supported formats.
        
        Tries each detection method in order:
        1. Code-generated figures (most specific)
        2. TikZ/Div figures 
        3. Standard markdown figures
        
        Args:
            content: QMD file content
            fig_id: Full figure ID (e.g., "fig-example")
            
        Returns:
            Dict with type-specific information or None if not found
        """
        # Try each detection method in order of specificity
        for detector in [self.detect_code_figure, self.detect_tikz_figure, self.detect_markdown_figure]:
            result = detector(content, fig_id)
            if result:
                return result
        return None
    
    def detect_markdown_figure(self, content: str, fig_id: str) -> Optional[Dict[str, str]]:
        """
        Detect standard markdown figures.
        
        Format: ![caption](path){#fig-id}
        
        Examples:
            ![AI timeline](images/ai-timeline.png){#fig-ai-timeline}
            ![Complex caption with [citations]](path.png){width=80% #fig-id}
            ![Caption](path.png){#fig-id width=80% height=60%}
        
        Args:
            content: QMD file content
            fig_id: Full figure ID (e.g., "fig-ai-timeline")
            
        Returns:
            Dict with 'caption', 'path', 'full_match' or None if not found
        """
        # Fixed pattern: Use .*? to handle nested brackets in captions (like citations)
        # Pattern: ![caption](path){...#fig-id...}
        pattern = rf'!\[(.*?)\]\(([^)]+(?:\\.[^)]*)*)\)\s*\{{[^}}]*#{re.escape(fig_id)}(?:\s|[^}}])*\}}'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return {
                'type': 'markdown',
                'caption': match.group(1).strip(),
                'path': match.group(2).strip(),
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            }
        return None
    
    def detect_tikz_figure(self, content: str, fig_id: str) -> Optional[Dict[str, str]]:
        """
        Detect TikZ/Div block figures.
        
        Format: 
            ::: {#fig-id}
            ```{.tikz}
            % TikZ code here
            ```
            Caption text here
            :::
        
        Examples:
            ::: {#fig-neural-network}
            ::: {width=80% #fig-neural-network}
            ::: {#fig-neural-network .column-margin}
        
        Args:
            content: QMD file content
            fig_id: Full figure ID (e.g., "fig-neural-network")
            
        Returns:
            Dict with 'caption', 'tikz_code', 'full_match' or None if not found
        """
        # More flexible pattern: fig-id can be anywhere in the attributes
        pattern = rf':::\s*\{{[^}}]*#{re.escape(fig_id)}(?:\s|[^}}])*\}}(.*?):::'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            div_content = match.group(1)
            
            # Extract TikZ code block
            tikz_match = re.search(r'```\{\.tikz\}(.*?)```', div_content, re.DOTALL)
            tikz_code = tikz_match.group(1).strip() if tikz_match else ""
            
            # Extract caption (usually the last text line before :::)
            lines = [line.strip() for line in div_content.split('\n') if line.strip()]
            caption = ""
            for line in reversed(lines):
                if line and not line.startswith('\\') and not line.startswith('```'):
                    caption = line
                    break
            
            return {
                'type': 'tikz',
                'caption': caption,
                'tikz_code': tikz_code,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            }
        return None
    
    def detect_code_figure(self, content: str, fig_id: str) -> Optional[Dict[str, str]]:
        """
        Detect code-generated figures (R/Python blocks).
        
        Format:
            ```{r}
            #| label: fig-id
            #| fig-cap: "Caption text"
            #| other-options: values
            
            # R or Python code here
            ggplot(data) + ...
            ```
        
        Examples:
            ```{r}
            #| label: fig-datacenter-energy
            #| fig-cap: "Energy usage over time"
            #| echo: false
            
            library(ggplot2)
            ggplot(data, aes(x=year, y=usage)) + geom_line()
            ```
        
        Args:
            content: QMD file content
            fig_id: Full figure ID (e.g., "fig-datacenter-energy")
            
        Returns:
            Dict with 'caption', 'code', 'language', 'full_match' or None if not found
        """
        # Pattern: ```{r|python} ... #| label: fig-id ... #| fig-cap: "caption" ... ```
        pattern = rf'```\{{(r|python)[^}}]*\}}([^`]*?#\|\s*label:\s*{re.escape(fig_id)}[^`]*?)```'
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        
        if match:
            language = match.group(1)
            code_block = match.group(2)
            
            # Extract fig-cap from the code block
            cap_pattern = r'#\|\s*fig-cap:\s*["\']?([^"\'\n]+)["\']?'
            cap_match = re.search(cap_pattern, code_block)
            caption = cap_match.group(1).strip() if cap_match else ""
            
            return {
                'type': 'code',
                'caption': caption,
                'language': language,
                'code': code_block.strip(),
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            }
        return None
    
    def detect_table(self, content: str, tbl_id: str) -> Optional[Dict[str, str]]:
        """
        Detect table captions.
        
        Supports both formats:
        - Old: : Caption text {#tbl-id}
        - New: Caption text {#tbl-id}
        
        Examples:
            : AI model comparison {#tbl-models}  (old format)
            AI model comparison {#tbl-models}   (new format)
            Performance metrics {width=80% #tbl-performance}
        
        Args:
            content: QMD file content
            tbl_id: Full table ID (e.g., "tbl-models")
            
        Returns:
            Dict with 'caption', 'full_match' or None if not found
        """
        # Try new format first (without colon)
        pattern_new = rf'^([^{{\n:]+?)\s*\{{[^}}]*#{re.escape(tbl_id)}(?:\s|[^}}])*\}}\s*$'
        match = re.search(pattern_new, content, re.MULTILINE)
        
        if not match:
            # Fall back to old format (with colon)
            pattern_old = rf'^:\s*([^{{\n]+?)\s*\{{[^}}]*#{re.escape(tbl_id)}(?:\s|[^}}])*\}}\s*$'
            match = re.search(pattern_old, content, re.MULTILINE)
        
        if match:
            return {
                'type': 'table',
                'caption': match.group(1).strip(),
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            }
        return None
    
    def find_table_definition_in_qmd(self, content: str, tbl_id: str) -> Optional[Dict[str, str]]:
        """
        Unified table detection using the specialized detect_table function.
        
        Args:
            content: QMD file content
            tbl_id: Full table ID (e.g., "tbl-models")
            
        Returns:
            Dict with table information or None if not found
        """
        return self.detect_table(content, tbl_id)

    # ================================================================
    # SPECIALIZED UPDATE FUNCTIONS FOR DIFFERENT FIGURE/TABLE TYPES  
    # ================================================================
    
    def update_markdown_figure(self, content: str, fig_id: str, new_caption: str) -> str:
        """
        Update caption in standard markdown figures.
        
        Updates: ![old_caption](path){#fig-id} → ![new_caption](path){#fig-id}
        
        Args:
            content: QMD file content
            fig_id: Full figure ID (e.g., "fig-ai-timeline")
            new_caption: New caption text
            
        Returns:
            Updated content
        """
        # Fixed pattern: Use .*? to handle nested brackets in captions (like citations)
        pattern = rf'(!\[).*?(\]\([^)]+\)\s*\{{[^}}]*#{re.escape(fig_id)}(?:\s|[^}}])*\}})'
        replacement = rf'\g<1>{new_caption}\g<2>'
        return re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    def update_tikz_figure(self, content: str, fig_id: str, new_caption: str) -> str:
        """
        Update caption in TikZ/Div block figures.
        
        Updates the caption text between the closing ``` and :::
        
        Args:
            content: QMD file content  
            fig_id: Full figure ID (e.g., "fig-neural-network")
            new_caption: New caption text
            
        Returns:
            Updated content
        """
        # Method 1: Replace caption after tikz block  
        # More flexible pattern: fig-id can be anywhere in the attributes
        pattern = rf'(:::\s*\{{[^}}]*#{re.escape(fig_id)}(?:\s|[^}}])*\}}.*?```\s*\n\s*)([^:]+?)((?:\s*:::))'
        
        def replace_caption(match):
            before = match.group(1)
            after = match.group(3)
            return f"{before}{new_caption}{after}"
        
        updated_content = re.sub(pattern, replace_caption, content, flags=re.MULTILINE | re.DOTALL)
        
        # Method 2: If that didn't work, try simpler approach
        if updated_content == content:
            div_pattern = rf'(:::\s*\{{[^}}]*#{re.escape(fig_id)}(?:\s|[^}}])*\}}.*?```\s*\n)([^:]*?)(:::'
            
            def replace_div_caption(match):
                before = match.group(1)
                after = match.group(3)
                return f"{before}{new_caption}\n{after}"
            
            updated_content = re.sub(div_pattern, replace_div_caption, content, flags=re.MULTILINE | re.DOTALL)
        
        return updated_content
    
    def update_code_figure(self, content: str, fig_id: str, new_caption: str) -> str:
        """
        Update caption in code-generated figures (R/Python blocks).
        
        Updates: #| fig-cap: "old caption" → #| fig-cap: "new caption"
        
        Args:
            content: QMD file content
            fig_id: Full figure ID (e.g., "fig-datacenter-energy")  
            new_caption: New caption text
            
        Returns:
            Updated content
        """
        # Find the fig-cap line specifically for this figure's code block
        pattern = rf'(```\{{(r|python)[^}}]*\}}[^`]*?#\|\s*label:\s*{re.escape(fig_id)}[^`]*?#\|\s*fig-cap:\s*)([^\n]+)'
        
        def replace_fig_cap(match):
            before = match.group(1)
            return f'{before}"{new_caption}"'
        
        return re.sub(pattern, replace_fig_cap, content, flags=re.MULTILINE | re.DOTALL)
    
    def update_table_caption(self, content: str, tbl_id: str, new_caption: str) -> str:
        """
        Update table captions.
        
        Converts old format to new format:
        - Old: : old caption {#tbl-id} → new caption {#tbl-id}
        - New: old caption {#tbl-id} → new caption {#tbl-id}
        
        Args:
            content: QMD file content
            tbl_id: Full table ID (e.g., "tbl-models")
            new_caption: New caption text
            
        Returns:
            Updated content
        """
        # Try old format first (with colon) - convert to new format
        pattern_old = rf'^:\s*([^{{\n]+?)(\s*\{{[^}}]*#{re.escape(tbl_id)}(?:\s|[^}}])*\}})\s*$'
        match_old = re.search(pattern_old, content, re.MULTILINE)
        
        if match_old:
            # Convert from old format to new format (remove colon)
            replacement = rf'{new_caption}\g<2>'
            return re.sub(pattern_old, replacement, content, flags=re.MULTILINE)
        else:
            # Handle new format (no colon)
            pattern_new = rf'^([^{{\n:]+?)(\s*\{{[^}}]*#{re.escape(tbl_id)}(?:\s|[^}}])*\}})\s*$'
            replacement = rf'{new_caption}\g<2>'
            return re.sub(pattern_new, replacement, content, flags=re.MULTILINE)
        
        return content
    
    def update_table_caption_in_qmd(self, content: str, tbl_id: str, new_caption: str) -> str:
        """
        Unified table caption update using the specialized update_table_caption function.
        
        Args:
            content: QMD file content
            tbl_id: Full table ID (e.g., "tbl-models")
            new_caption: New caption text
            
        Returns:
            Updated content
        """
        return self.update_table_caption(content, tbl_id, new_caption)
    
    # ================================================================
    # UNIFIED UPDATE FUNCTION
    # ================================================================
    
    def update_figure_caption_in_qmd(self, content: str, fig_id: str, new_caption: str) -> str:
        """
        Unified figure caption update across all supported formats.
        
        Detects the figure type and calls the appropriate update function.
        
        Args:
            content: QMD file content
            fig_id: Full figure ID (e.g., "fig-example")
            new_caption: New caption text
            
        Returns:
            Updated content
        """
        # First, determine what type of figure this is
        fig_def = self.find_figure_definition_in_qmd(content, fig_id)
        if not fig_def:
            return content
        
        # Route to appropriate update function based on type
        if fig_def['type'] == 'markdown':
            return self.update_markdown_figure(content, fig_id, new_caption)
        elif fig_def['type'] == 'tikz':
            return self.update_tikz_figure(content, fig_id, new_caption)
        elif fig_def['type'] == 'code':
            return self.update_code_figure(content, fig_id, new_caption)
        else:
            # Fallback to markdown method
            return self.update_markdown_figure(content, fig_id, new_caption)

    def update_caption_in_content(self, content: str, figure_def: Dict, new_caption: str) -> str:
        """Update the caption in the file content."""
        lines = content.split('\n')
        line_num = figure_def['line_number']
        old_line = lines[line_num]
        
        # Replace the caption in the figure definition
        # Handle ![old_caption](image){#fig-id} format
        caption_pattern = r'!\[([^\]]*)\](\([^)]+\)\s*\{[^}]+\})'
        new_line = re.sub(caption_pattern, f'![{new_caption}]\\2', old_line)
        
        lines[line_num] = new_line
        return '\n'.join(lines)
    
    def process_file(self, file_path: Path) -> None:
        """Process a single .qmd file to improve figure captions."""
        print(f"\n📄 Processing: {file_path.name}")
        print(f"   Path: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse sections
            sections = self.parse_sections(content)
            
            # Find figure references
            figure_refs = self.find_figure_references(content)
            
            # Find TikZ figures
            tikz_figures = self.find_tikz_figures_with_pypandoc(content)
            
            total_figures = len(figure_refs) + len(tikz_figures)
            if total_figures == 0:
                print(f"   ℹ️  No figure references or TikZ figures found")
                return
            
            print(f"   📊 Found {len(figure_refs)} regular figure(s) and {len(tikz_figures)} TikZ figure(s)")
            self.stats['figures_found'] += total_figures
            
            modified = False
            current_content = content
            
            for fig_ref in figure_refs:
                figure_id = fig_ref['figure_id']
                ref_line = fig_ref['line_number']
                
                # Find the section containing this reference
                section = self.get_section_for_line(sections, ref_line)
                if not section:
                    error_msg = f"Could not find section for figure {figure_id}"
                    print(f"      ❌ {error_msg}")
                    self.stats['errors'].append(error_msg)
                    continue
                
                # Find the figure definition
                fig_def = self.find_figure_definition(current_content, figure_id)
                if not fig_def:
                    error_msg = f"Could not find definition for figure {figure_id}"
                    print(f"      ❌ {error_msg}")
                    self.stats['errors'].append(error_msg)
                    continue
                
                print(f"\n   🎯 Processing: {figure_id}")
                print(f"      📝 Current: '{fig_def['current_caption'][:60]}{'...' if len(fig_def['current_caption']) > 60 else ''}'")
                print(f"      📑 Section: '{section['title'][:50]}{'...' if len(section['title']) > 50 else ''}'")
                
                # Find the actual image file
                image_file = None
                if fig_def['image_path']:
                    image_file = self.find_image_file(file_path, fig_def['image_path'])
                
                # Extract section content for context
                section_text = self.extract_section_text(section)
                if len(section_text.strip()) < 50:  # Skip very short sections
                    error_msg = f"Section too short for {figure_id}, skipping"
                    print(f"      ⚠️  {error_msg}")
                    self.stats['errors'].append(error_msg)
                    continue
                
                # Generate improved caption using multimodal model
                print(f"      🤖 Generating improved caption...")
                new_caption = self.generate_caption_with_ollama(
                    section['title'], section_text, figure_id, 
                    fig_def['current_caption'], image_file
                )
                
                if new_caption and new_caption != fig_def['current_caption']:
                    print(f"      ✅ New: '{new_caption[:80]}{'...' if len(new_caption) > 80 else ''}'")
                    
                    # Update the content
                    current_content = self.update_caption_in_content(
                        current_content, fig_def, new_caption
                    )
                    modified = True
                    self.stats['figures_improved'] += 1
                    
                    # Update sections for next iteration (line numbers may have changed)
                    sections = self.parse_sections(current_content)
                else:
                    print(f"      ⚠️  No improvement generated or same as original")
            
            # Process TikZ figures
            for tikz_fig in tikz_figures:
                figure_id = tikz_fig['figure_id']
                
                # Find the section containing this TikZ figure
                section = self.get_section_for_line(sections, tikz_fig['line_number'])
                if not section:
                    error_msg = f"Could not find section for TikZ figure {figure_id}"
                    print(f"      ❌ {error_msg}")
                    self.stats['errors'].append(error_msg)
                    continue
                
                print(f"\n   🎯 Processing TikZ: {figure_id}")
                print(f"      📝 Current: '{tikz_fig['raw_caption'][:60]}{'...' if len(tikz_fig['raw_caption']) > 60 else ''}'")
                print(f"      📑 Section: '{section['title'][:50]}{'...' if len(section['title']) > 50 else ''}'")
                
                # Extract section content for context
                section_text = self.extract_section_text(section)
                if len(section_text.strip()) < 50:  # Skip very short sections
                    error_msg = f"Section too short for TikZ {figure_id}, skipping"
                    print(f"      ⚠️  {error_msg}")
                    self.stats['errors'].append(error_msg)
                    continue
                
                # Compile TikZ to image
                print(f"      🔨 Compiling TikZ to image...")
                compiled_image = self.compile_tikz_to_image(tikz_fig['tikz_code'], figure_id)
                
                # Generate improved caption using multimodal model
                print(f"      🤖 Generating improved caption...")
                new_caption = self.generate_caption_with_ollama(
                    section['title'], section_text, figure_id, 
                    tikz_fig['raw_caption'], compiled_image
                )
                
                if new_caption and new_caption != tikz_fig['raw_caption']:
                    print(f"      ✅ New: '{new_caption[:80]}{'...' if len(new_caption) > 80 else ''}'")
                    
                    # Update the content
                    current_content = self.update_tikz_caption_in_content(
                        current_content, tikz_fig, new_caption
                    )
                    modified = True
                    self.stats['figures_improved'] += 1
                    
                    # Update sections for next iteration (line numbers may have changed)
                    sections = self.parse_sections(current_content)
                else:
                    print(f"      ⚠️  No improvement generated or same as original")
                
                # Clean up compiled image
                if compiled_image and os.path.exists(compiled_image):
                    try:
                        os.unlink(compiled_image)
                    except:
                        pass  # Ignore cleanup errors
            
            # Save the file if modified
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(current_content)
                print(f"   💾 File updated successfully!")
            else:
                print(f"   ℹ️  No changes made")
            
            self.stats['files_processed'] += 1
                
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            print(f"   ❌ {error_msg}")
            self.stats['errors'].append(error_msg)
    
    def print_summary(self) -> None:
        """Print a summary of the processing results."""
        print(f"\n{'='*60}")
        print(f"📊 CAPTION IMPROVEMENT SUMMARY")
        print(f"{'='*60}")
        
        print(f"Files processed:     {self.stats['files_processed']}")
        print(f"Figures found:       {self.stats['figures_found']}")
        print(f"Figures improved:    {self.stats['figures_improved']} ✅")
        print(f"Tables found:        {self.stats['tables_found']}")
        print(f"Tables improved:     {self.stats['tables_improved']} ✅")
        print(f"Images found:        {self.stats['images_found']} 🖼️")
        print(f"Images missing:      {self.stats['images_missing']} ⚠️")
        print(f"JSON success:        {self.stats['json_success']} 📋")
        print(f"JSON failed:         {self.stats['json_failed']} 🚫")
        
        if self.stats['figures_improved'] > 0:
            improvement_rate = (self.stats['figures_improved'] / self.stats['figures_found']) * 100
            print(f"Improvement rate:    {improvement_rate:.1f}%")
        
        if self.stats['errors']:
            print(f"\n⚠️  Issues encountered ({len(self.stats['errors'])}):")
            for i, error in enumerate(self.stats['errors'], 1):
                print(f"   {i}. {error}")
        
        if self.stats['images_missing'] > 0:
            print(f"\n💡 Tip: {self.stats['images_missing']} images were not found.")
            print(f"   Consider checking image paths or using text-only processing.")
        
        if self.stats['tikz_found'] > 0:
            print(f"\n🔧 TikZ Processing: Found {self.stats['tikz_found']} TikZ figures.")
            if self.stats['tikz_failed'] > 0:
                print(f"   ⚠️  {self.stats['tikz_failed']} TikZ compilations failed.")
                print(f"   Ensure you have LaTeX (pdflatex) and ImageMagick (magick) installed.")
            
        print(f"{'='*60}")

    def run(self, files: List[str] = None, directories: List[str] = None) -> None:
        """Run the caption improvement process."""
        if files:
            file_paths = [Path(f) for f in files]
        elif directories:
            file_paths = []
            for directory in directories:
                dir_files = self.find_qmd_files(directory)
                file_paths.extend(dir_files)
                print(f"📂 Found {len(dir_files)} .qmd files in {directory}")
        else:
            raise ValueError("Must specify either files or directories")
        
        print(f"🚀 Starting caption improvement with model: {self.model_name}")
        print(f"📁 Found {len(file_paths)} .qmd files to process")
        
        for file_path in file_paths:
            self.process_file(file_path)
        
        self.print_summary()

    def validate_qmd_mapping(self, directories: List[str], content_map: Dict) -> Dict:
        """Scan QMD files and validate mapping for all figures/tables."""
        print(f"🔍 Validating QMD mapping...")
        
        # Check for commented chapters in target directories first
        commented_issues = self.check_commented_chapters_in_directories(directories)
        should_halt = self.print_commented_chapter_issues(commented_issues)
        if should_halt:
            return content_map
        
        qmd_files = self.find_qmd_files_in_order(directories)
        print(f"📁 Scanning {len(qmd_files)} QMD files")
        
        # Track what we find in QMD files
        found_figures = {}
        found_tables = {}
        missing_figures = []
        missing_tables = []
        
        # Scan all QMD files for figure/table references
        for qmd_file in qmd_files:
            try:
                with open(qmd_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"  📄 Scanning: {qmd_file}")
                
                # Check each figure from content map
                for fig_id in content_map.get('figures', {}):
                    if fig_id not in found_figures:
                        fig_def = self.find_figure_definition_in_qmd(content, fig_id)
                        if fig_def:
                            found_figures[fig_id] = {
                                'qmd_file': qmd_file,
                                'current_caption': fig_def['caption'],
                                'definition': fig_def
                            }
                            print(f"    ✅ Found figure: {fig_id}")
                
                # Check each table from content map  
                for tbl_id in content_map.get('tables', {}):
                    if tbl_id not in found_tables:
                        tbl_def = self.find_table_definition_in_qmd(content, tbl_id)
                        if tbl_def:
                            found_tables[tbl_id] = {
                                'qmd_file': qmd_file,
                                'current_caption': tbl_def['caption'],
                                'definition': tbl_def
                            }
                            print(f"    ✅ Found table: {tbl_id}")
                            
            except Exception as e:
                print(f"    ❌ Error scanning {qmd_file}: {e}")
        
        # Identify missing items
        for fig_id in content_map.get('figures', {}):
            if fig_id not in found_figures:
                missing_figures.append(fig_id)
        
        for tbl_id in content_map.get('tables', {}):
            if tbl_id not in found_tables:
                missing_tables.append(tbl_id)
        
        # Update content map with QMD locations
        for fig_id, fig_info in found_figures.items():
            content_map['figures'][fig_id]['source_file'] = fig_info['qmd_file']
            content_map['figures'][fig_id]['qmd_caption'] = fig_info['current_caption']
        
        for tbl_id, tbl_info in found_tables.items():
            content_map['tables'][tbl_id]['source_file'] = tbl_info['qmd_file']
            content_map['tables'][tbl_id]['qmd_caption'] = tbl_info['current_caption']
        
        # Report validation results
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Figures: {len(found_figures)}/{len(content_map.get('figures', {}))} found in QMD")
        print(f"   Tables:  {len(found_tables)}/{len(content_map.get('tables', {}))} found in QMD")
        
        if missing_figures:
            print(f"\n❌ Missing figures ({len(missing_figures)}):")
            for fig_id in missing_figures:
                print(f"   - {fig_id}")
        
        if missing_tables:
            print(f"\n❌ Missing tables ({len(missing_tables)}):")
            for tbl_id in missing_tables:
                print(f"   - {tbl_id}")
        
        if not missing_figures and not missing_tables:
            print(f"\n✅ Perfect mapping! All items found in QMD files.")
        
        # Check already performed at start of validation
        
        return content_map

    def check_caption_quality(self, directories: List[str]) -> Dict[str, any]:
        """Analyze all captions and generate quality report."""
        print("🔍 Analyzing caption quality...")
        
        # Check for commented chapters in target directories first
        commented_issues = self.check_commented_chapters_in_directories(directories)
        should_halt = self.print_commented_chapter_issues(commented_issues)
        if should_halt:
            return {}
        
        # Load content map
        content_map = self.load_content_map()
        if not content_map:
            print("❌ No content map found. Run --build-map first.")
            return {}
        
        # Find all QMD files
        qmd_files = self.find_qmd_files_in_order(directories)
        
        report = {
            'total_captions': 0,
            'captions_needing_repair': 0,
            'issues_by_type': {},
            'detailed_issues': []
        }
        
        # Analyze figures
        for fig_id, fig_data in content_map.get('figures', {}).items():
            current_caption = fig_data.get('current_caption', '')
            analysis = self.quality_checker.analyze_caption(current_caption)
            
            report['total_captions'] += 1
            if analysis['needs_repair']:
                report['captions_needing_repair'] += 1
                report['detailed_issues'].append({
                    'id': fig_id,
                    'type': 'figure',
                    'current_caption': current_caption,
                    'issues': analysis['issues']
                })
                
                # Count issues by type
                for issue in analysis['issues']:
                    issue_type = issue['type']
                    report['issues_by_type'][issue_type] = report['issues_by_type'].get(issue_type, 0) + 1
        
        # Analyze tables  
        for tbl_id, tbl_data in content_map.get('tables', {}).items():
            current_caption = tbl_data.get('current_caption', '')
            analysis = self.quality_checker.analyze_caption(current_caption)
            
            report['total_captions'] += 1
            if analysis['needs_repair']:
                report['captions_needing_repair'] += 1
                report['detailed_issues'].append({
                    'id': tbl_id,
                    'type': 'table', 
                    'current_caption': current_caption,
                    'issues': analysis['issues']
                })
                
                # Count issues by type
                for issue in analysis['issues']:
                    issue_type = issue['type']
                    report['issues_by_type'][issue_type] = report['issues_by_type'].get(issue_type, 0) + 1
        
        return report
    
    def print_quality_report(self, report: Dict[str, any]):
        """Print a nicely formatted quality report."""
        if not report:
            return
            
        total = report['total_captions']
        needing_repair = report['captions_needing_repair']
        percentage = (needing_repair / total * 100) if total > 0 else 0
        
        print(f"\n📊 Caption Quality Report:")
        print(f"={'=' * 60}")
        print(f"Total captions: {total}")
        print(f"Need repair: {needing_repair} ({percentage:.1f}%)")
        
        if report['issues_by_type']:
            print(f"\n🔍 Issues by type:")
            for issue_type, count in report['issues_by_type'].items():
                print(f"  • {issue_type.replace('_', ' ').title()}: {count}")
        
        if report['detailed_issues']:
            print(f"\n📝 Detailed Issues:")
            print(f"┌─{'─' * 18}─┬─{'─' * 12}─┬─{'─' * 35}─┐")
            print(f"│ {'ID':<18} │ {'Issue':<12} │ {'Current Caption':<35} │")
            print(f"├─{'─' * 18}─┼─{'─' * 12}─┼─{'─' * 35}─┤")
            
            for issue_item in report['detailed_issues'][:20]:  # Limit to first 20
                item_id = issue_item['id'][:18]
                caption = issue_item['current_caption'][:35]
                issues_desc = ', '.join([issue['description'] for issue in issue_item['issues']])[:12]
                
                print(f"│ {item_id:<18} │ {issues_desc:<12} │ {caption:<35} │")
            
            print(f"└─{'─' * 18}─┴─{'─' * 12}─┴─{'─' * 35}─┘")
            
            if len(report['detailed_issues']) > 20:
                print(f"... and {len(report['detailed_issues']) - 20} more issues")
        
        if needing_repair > 0:
            print(f"\n💡 To fix these issues, run:")
            print(f"   python {__file__} --repair -d {' -d '.join(['contents/core/'])}")
        else:
            print(f"\n✅ All captions look good!")
    
    def repair_captions(self, directories: List[str]):
        """Repair only captions that need fixing."""
        print("🔧 Repairing captions that need fixing...")
        
        # Check for commented chapters in target directories first
        commented_issues = self.check_commented_chapters_in_directories(directories)
        should_halt = self.print_commented_chapter_issues(commented_issues)
        if should_halt:
            return
        
        # First check what needs repair
        report = self.check_caption_quality(directories)
        if not report or report['captions_needing_repair'] == 0:
            print("✅ No captions need repair!")
            return
        
        print(f"🎯 Found {report['captions_needing_repair']} captions needing repair")
        
        # Load content map
        content_map = self.load_content_map()
        if not content_map:
            print("❌ No content map found. Run --build-map first.")
            return
        
        # Create repair list - only items that need fixing
        repair_items = []
        for issue_item in report['detailed_issues']:
            item_id = issue_item['id']
            item_type = issue_item['type']
            
            if item_type == 'figure' and item_id in content_map.get('figures', {}):
                repair_items.append(('figure', item_id, content_map['figures'][item_id]))
            elif item_type == 'table' and item_id in content_map.get('tables', {}):
                repair_items.append(('table', item_id, content_map['tables'][item_id]))
        
        # Apply basic fixes (punctuation, capitalization)
        fixed_count = 0
        for item_type, item_id, item_data in repair_items:
            current_caption = item_data.get('current_caption', '')
            
            # Apply normalization fixes
            fixed_caption = self.normalize_caption_punctuation(current_caption)
            fixed_caption = self.normalize_caption_case(fixed_caption)
            
            # Only update if it actually changed
            if fixed_caption != current_caption:
                item_data['new_caption'] = fixed_caption
                item_data['current_caption'] = fixed_caption  # Update current_caption too
                print(f"🔧 {item_id}: '{current_caption}' → '{fixed_caption}'")
                fixed_count += 1
        
        if fixed_count > 0:
            # Save updated content map
            self.save_content_map(content_map)
            
            # Update QMD files
            self.process_qmd_files(directories, content_map)
            print(f"✅ Repaired {fixed_count} captions")
        else:
            print("ℹ️  No automatic repairs possible. Manual review may be needed.")

    # ================================================================
    # QMD-FOCUSED CONTENT MAP BUILDING
    # ================================================================
    
    def build_content_map_from_qmd(self, directories: List[str]) -> Dict:
        """
        Build comprehensive content map by scanning QMD files directly.
        
        This QMD-focused approach:
        1. Scans all .qmd files in specified directories
        2. Uses specialized detection functions for each format type
        3. Extracts current captions and metadata
        4. Stores everything in a clean JSON structure
        5. Independent of .tex builds or rendering
        
        Args:
            directories: List of directories to scan for .qmd files
            
        Returns:
            Dict with figures, tables, metadata, and extraction stats
        """
        print(f"📄 Building content map from QMD files...")
        
        # Check for commented chapters first
        commented_issues = self.check_commented_chapters_in_directories(directories)
        should_halt = self.print_commented_chapter_issues(commented_issues)
        if should_halt:
            return {}
        
        # Get ordered QMD files
        qmd_files = self.find_qmd_files_in_order(directories)
        print(f"📖 Scanning {len(qmd_files)} QMD files in book order")
        
        content_map = {
            'figures': {},
            'tables': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'source': 'qmd_direct_scan',
                'directories': directories,
                'qmd_files_scanned': len(qmd_files),
                'extraction_stats': {
                    'figures_found': 0,
                    'tables_found': 0,
                    'markdown_figures': 0,
                    'tikz_figures': 0, 
                    'code_figures': 0,
                    'extraction_failures': 0,
                    'files_with_issues': []
                }
            }
        }
        
        stats = content_map['metadata']['extraction_stats']
        
        for qmd_file in qmd_files:
            try:
                print(f"  📄 Scanning: {qmd_file}")
                
                with open(qmd_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_figures = 0
                file_tables = 0
                
                # Find all potential figure IDs in the content using regex
                fig_id_pattern = r'#(fig-[a-zA-Z0-9_-]+)'
                potential_fig_ids = set(re.findall(fig_id_pattern, content))
                
                # Find all potential table IDs
                tbl_id_pattern = r'#(tbl-[a-zA-Z0-9_-]+)'
                potential_tbl_ids = set(re.findall(tbl_id_pattern, content))
                
                # Process each potential figure ID
                for fig_id in potential_fig_ids:
                    try:
                        fig_def = self.find_figure_definition_in_qmd(content, fig_id)
                        if fig_def:
                            # Normalize caption
                            current_caption = fig_def['caption']
                            normalized_caption = self.normalize_caption_punctuation(current_caption)
                            normalized_caption = self.normalize_caption_case(normalized_caption)
                            
                            content_map['figures'][fig_id] = {
                                'current_caption': normalized_caption,
                                'original_caption': current_caption,
                                'new_caption': '',
                                'type': fig_def['type'],
                                'source_file': qmd_file
                            }
                            
                            print(f"    ✅ Found figure: {fig_id} ({fig_def['type']})")
                            file_figures += 1
                            stats['figures_found'] += 1
                            
                            # Count by type
                            if fig_def['type'] == 'markdown':
                                stats['markdown_figures'] += 1
                            elif fig_def['type'] == 'tikz':
                                stats['tikz_figures'] += 1
                            elif fig_def['type'] == 'code':
                                stats['code_figures'] += 1
                                
                        else:
                            print(f"    ⚠️  Failed to extract: {fig_id}")
                            stats['extraction_failures'] += 1
                            if qmd_file not in stats['files_with_issues']:
                                stats['files_with_issues'].append(qmd_file)
                                
                    except Exception as e:
                        print(f"    ❌ Error processing {fig_id}: {e}")
                        stats['extraction_failures'] += 1
                        if qmd_file not in stats['files_with_issues']:
                            stats['files_with_issues'].append(qmd_file)
                
                # Process each potential table ID
                for tbl_id in potential_tbl_ids:
                    try:
                        tbl_def = self.detect_table(content, tbl_id)
                        if tbl_def:
                            # Normalize caption
                            current_caption = tbl_def['caption']
                            normalized_caption = self.normalize_caption_punctuation(current_caption)
                            normalized_caption = self.normalize_caption_case(normalized_caption)
                            
                            content_map['tables'][tbl_id] = {
                                'current_caption': normalized_caption,
                                'original_caption': current_caption,
                                'new_caption': '',
                                'type': 'table',
                                'source_file': qmd_file
                            }
                            
                            print(f"    ✅ Found table: {tbl_id}")
                            file_tables += 1
                            stats['tables_found'] += 1
                            
                        else:
                            print(f"    ⚠️  Failed to extract: {tbl_id}")
                            stats['extraction_failures'] += 1
                            if qmd_file not in stats['files_with_issues']:
                                stats['files_with_issues'].append(qmd_file)
                                
                    except Exception as e:
                        print(f"    ❌ Error processing {tbl_id}: {e}")
                        stats['extraction_failures'] += 1
                        if qmd_file not in stats['files_with_issues']:
                            stats['files_with_issues'].append(qmd_file)
                
                # Summary for this file
                if file_figures > 0 or file_tables > 0:
                    print(f"    📊 File summary: {file_figures} figures, {file_tables} tables")
                    
            except Exception as e:
                print(f"  ❌ Error reading {qmd_file}: {e}")
                stats['extraction_failures'] += 1
                if qmd_file not in stats['files_with_issues']:
                    stats['files_with_issues'].append(qmd_file)
        
        # Final summary
        print(f"\n📊 QMD EXTRACTION SUMMARY:")
        print(f"   📊 Figures: {stats['figures_found']} found")
        print(f"      • Markdown: {stats['markdown_figures']}")
        print(f"      • TikZ: {stats['tikz_figures']}") 
        print(f"      • Code: {stats['code_figures']}")
        print(f"   📋 Tables: {stats['tables_found']} found")
        print(f"   ⚠️  Extraction failures: {stats['extraction_failures']}")
        
        if stats['files_with_issues']:
            print(f"   📁 Files with issues: {len(stats['files_with_issues'])}")
            for file in stats['files_with_issues'][:5]:  # Show first 5
                print(f"      • {file}")
            if len(stats['files_with_issues']) > 5:
                print(f"      • ... and {len(stats['files_with_issues']) - 5} more")
        
        # Calculate success rate
        total_ids = stats['figures_found'] + stats['tables_found'] + stats['extraction_failures']
        success_rate = (stats['figures_found'] + stats['tables_found']) / total_ids * 100 if total_ids > 0 else 0
        print(f"   ✅ Success rate: {success_rate:.1f}%")
        
        return content_map

    def process_qmd_files(self, directories: List[str], content_map: Dict):
        """
        Process QMD files to update captions based on content map.
        
        Groups figures/tables by source file and updates each file once
        with all caption changes to minimize file I/O operations.
        
        Args:
            directories: List of directories to process
            content_map: Content map with figures and tables data
        """
        print("📝 Processing QMD files for caption updates...")
        
        # Group all items by source file
        files_to_update = {}
        
        # Collect figures that need updates
        for fig_id, fig_data in content_map.get('figures', {}).items():
            if 'new_caption' in fig_data and fig_data.get('new_caption'):
                source_file = fig_data.get('source_file')
                if source_file:
                    if source_file not in files_to_update:
                        files_to_update[source_file] = {'figures': [], 'tables': []}
                    files_to_update[source_file]['figures'].append((fig_id, fig_data['new_caption']))
        
        # Collect tables that need updates  
        for tbl_id, tbl_data in content_map.get('tables', {}).items():
            if 'new_caption' in tbl_data and tbl_data.get('new_caption'):
                source_file = tbl_data.get('source_file')
                if source_file:
                    if source_file not in files_to_update:
                        files_to_update[source_file] = {'figures': [], 'tables': []}
                    files_to_update[source_file]['tables'].append((tbl_id, tbl_data['new_caption']))
        
        if not files_to_update:
            print("ℹ️  No caption updates needed (no new_caption entries found)")
            return
        
        # Process each file once with all its updates
        total_figures_updated = 0
        total_tables_updated = 0
        
        for file_path, updates in files_to_update.items():
            try:
                print(f"📄 Updating {file_path}...")
                print(f"   📊 {len(updates['figures'])} figures, {len(updates['tables'])} tables")
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply all figure updates
                for fig_id, new_caption in updates['figures']:
                    content = self.update_figure_caption_in_qmd(content, fig_id, new_caption)
                    total_figures_updated += 1
                
                # Apply all table updates  
                for tbl_id, new_caption in updates['tables']:
                    content = self.update_table_caption_in_qmd(content, tbl_id, new_caption)
                    total_tables_updated += 1
                
                # Write back only if content changed
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"   ✅ Updated successfully")
                else:
                    print(f"   ⚠️  No changes applied (patterns may not have matched)")
                    
            except Exception as e:
                print(f"   ❌ Error updating {file_path}: {e}")
        
        print(f"📊 Summary: Updated {total_figures_updated} figures and {total_tables_updated} tables across {len(files_to_update)} files")

    def improve_captions_with_llm(self, directories: List[str], content_map: Dict):
        """Improve captions using LLM based on extracted content map and context."""
        print("🤖 Improving captions with LLM...")
        
        total_figures = len(content_map.get('figures', {}))
        total_tables = len(content_map.get('tables', {}))
        
        if total_figures == 0 and total_tables == 0:
            print("❌ No figures or tables found in content map")
            return content_map
        
        print(f"📊 Processing: {total_figures} figures, {total_tables} tables")
        
        improved_count = 0
        
        # Process figures
        for fig_id, fig_data in content_map.get('figures', {}).items():
            source_file = fig_data.get('source_file')
            if not source_file:
                continue
                
            print(f"  📊 Processing figure: {fig_id}")
            
            try:
                # Read the source file for context extraction
                with open(source_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Extract context around this figure
                context = self.extract_section_context(file_content, fig_id)
                
                # Find image path if it's a markdown figure
                image_path = None
                if fig_data.get('type') == 'markdown':
                    # Try to extract image path from the figure definition
                    image_pattern = rf'!\[[^\]]*\]\(([^)]+)\)[^{{]*{{[^}}]*#{re.escape(fig_id)}'
                    match = re.search(image_pattern, file_content)
                    if match:
                        relative_path = match.group(1)
                        # Resolve relative to the source file directory
                        source_dir = Path(source_file).parent
                        image_path = str(source_dir / relative_path)
                        if not os.path.exists(image_path):
                            image_path = None
                
                # Generate improved caption
                current_caption = fig_data.get('current_caption', '')
                new_caption = self.generate_caption_with_ollama(
                    context['title'], 
                    context['content'], 
                    fig_id, 
                    current_caption, 
                    image_path
                )
                
                if new_caption and new_caption != current_caption:
                    fig_data['new_caption'] = new_caption
                    improved_count += 1
                    print(f"    ✅ Improved: {new_caption[:80]}{'...' if len(new_caption) > 80 else ''}")
                else:
                    print(f"    ⚠️  No improvement generated")
                    
            except Exception as e:
                print(f"    ❌ Error processing {fig_id}: {e}")
        
        # Process tables
        for tbl_id, tbl_data in content_map.get('tables', {}).items():
            source_file = tbl_data.get('source_file')
            if not source_file:
                continue
                
            print(f"  📋 Processing table: {tbl_id}")
            
            try:
                # Read the source file for context extraction
                with open(source_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Extract context around this table
                context = self.extract_section_context(file_content, tbl_id)
                
                # Generate improved caption (no image for tables)
                current_caption = tbl_data.get('current_caption', '')
                new_caption = self.generate_caption_with_ollama(
                    context['title'], 
                    context['content'], 
                    tbl_id, 
                    current_caption, 
                    None  # No image for tables
                )
                
                if new_caption and new_caption != current_caption:
                    tbl_data['new_caption'] = new_caption
                    improved_count += 1
                    print(f"    ✅ Improved: {new_caption[:80]}{'...' if len(new_caption) > 80 else ''}")
                else:
                    print(f"    ⚠️  No improvement generated")
                    
            except Exception as e:
                print(f"    ❌ Error processing {tbl_id}: {e}")
        
        print(f"🎉 LLM improvement complete: {improved_count} captions improved")
        return content_map


def main():
    parser = argparse.ArgumentParser(
        description="Improve figure and table captions using local Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build content map from .qmd files directly (QMD-focused approach)
  python improve_figure_captions.py --build-qmd-map -d contents/core/
  python improve_figure_captions.py --build-qmd-map -d contents/core/ -d contents/frontmatter/
  
  # Build content map and save to JSON file for review
  python improve_figure_captions.py --build-qmd-map --save-json -d contents/core/
  
  # Improve captions using LLM (requires existing content map)
  python improve_figure_captions.py --improve -d contents/core/
  
  # Check caption quality
  python improve_figure_captions.py --check -d contents/core/
  
  # Repair caption issues automatically  
  python improve_figure_captions.py --repair -d contents/core/
  
  # Validate QMD mapping (requires existing content_map.json)
  python improve_figure_captions.py --validate -d contents/core/
  
  # Update QMD files with improved captions (requires content_map.json with new_caption fields)
  python improve_figure_captions.py --update -d contents/core/
  
  # Complete workflow: Build map → Improve → Update
  python improve_figure_captions.py --build-qmd-map -d contents/core/
  python improve_figure_captions.py --improve -d contents/core/
  python improve_figure_captions.py --update -d contents/core/
"""
    )
    
    # Multiple file/directory input
    parser.add_argument('-f', '--files', action='append',
                       help='QMD files to process (can be used multiple times)')
    parser.add_argument('-d', '--directories', action='append',
                       help='Directories to scan for QMD files (can be used multiple times)')

    # Mode selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--build-qmd-map', action='store_true',
                      help='Build content map from QMD files directly')
    group.add_argument('--improve', action='store_true',
                      help='Improve captions using LLM (requires existing content map)')
    group.add_argument('--validate', action='store_true',
                      help='Validate QMD files against content map')
    group.add_argument('--update', action='store_true',
                      help='Update QMD files with improved captions')
    group.add_argument('--check', '-c', action='store_true',
                      help='Check caption quality without making changes')
    group.add_argument('--repair', '-r', action='store_true',
                      help='Repair caption formatting issues')

    # Output options
    parser.add_argument('--save-json', action='store_true',
                       help='Save content map to JSON file (for --build-qmd-map)')

    args = parser.parse_args()
    
    # Validate that we have input files/directories for certain operations
    if args.build_qmd_map or args.improve or args.validate or args.update or args.check or args.repair:
        if not args.files and not args.directories:
            print("❌ Error: --files or --directories required for this operation")
            return 1
    
    # Determine which files/directories to process
    directories = []
    if args.directories:
        directories.extend(args.directories)
    if args.files:
        # For individual files, add their parent directories
        for file in args.files:
            parent_dir = str(Path(file).parent)
            if parent_dir not in directories:
                directories.append(parent_dir)
    
    improver = FigureCaptionImprover()
    
    try:
        if args.build_qmd_map:
            # QMD-focused approach: Build content map from .qmd files directly
            print("🔍 QMD-Focused: Building content map from .qmd files...")
            content_map = improver.build_content_map_from_qmd(directories)
            if content_map:
                print("✅ QMD content map building completed!")
                
                # Save JSON if requested
                if args.save_json:
                    improver.save_content_map(content_map)
                
                # Show extraction report
                stats = content_map['metadata']['extraction_stats']
                if stats['extraction_failures'] == 0:
                    print("🎉 Perfect extraction! All figures and tables successfully processed.")
                else:
                    print(f"⚠️  {stats['extraction_failures']} extraction failures detected.")
                    print("💡 Consider reviewing the files with issues for manual fixes.")
                
                # Show brief summary of what was found
                print(f"\n📋 CONTENT SUMMARY:")
                print(f"   📊 Figures: {stats['figures_found']} total")
                print(f"      • Markdown: {stats['markdown_figures']}")
                print(f"      • TikZ: {stats['tikz_figures']}")
                print(f"      • Code: {stats['code_figures']}")
                print(f"   📋 Tables: {stats['tables_found']} total")
                print(f"   📁 Files processed: {content_map['metadata']['qmd_files_scanned']}")
                
                if args.save_json:
                    print(f"\n💾 Content map saved to: content_map.json")
                    print(f"📄 You can now review the complete JSON structure!")
                
                if stats['extraction_failures'] > 0:
                    print(f"\n💡 Next steps:")
                    print(f"   1. Review extraction failures to improve detection patterns")
                    print(f"   2. Use this content map for caption improvements")
                    print(f"   3. Update QMD files directly with improved captions")
                
            else:
                print("❌ QMD content map building failed!")
                return 1
                
        elif args.improve:
            # Improve captions using LLM
            print("🤖 Improving captions using LLM...")
            content_map = improver.load_content_map()
            if not content_map:
                print("❌ No content map found. Run --build-qmd-map first.")
                return 1
            
            # Run LLM improvement
            improved_content_map = improver.improve_captions_with_llm(directories, content_map)
            
            # Save the updated content map with new captions
            improver.save_content_map(improved_content_map)
            print("✅ Caption improvement completed! Use --update to apply changes to QMD files.")
                
        elif args.check:
            # Check caption quality
            print("🔍 Checking caption quality...")
            improver.check_caption_quality(directories)
            
        elif args.repair:
            # Repair captions automatically  
            print("🔧 Repairing captions...")
            content_map = improver.repair_captions(directories)
            if content_map:
                improver.save_content_map(content_map)
                print("✅ Caption repair completed!")
            
        elif args.validate:
            # Validate QMD mapping
            print("🔍 Validating QMD mapping...")
            improver.validate_qmd_mapping(directories)
            
        elif args.update:
            # Update QMD files
            print("✏️ Updating QMD files...")
            content_map = improver.load_content_map()
            if content_map:
                improver.process_qmd_files(directories, content_map)
                print("✅ QMD file updates completed!")
            else:
                print("❌ No content map found. Run --build-map or --build-qmd-map first.")
                return 1
        else:
            # Default: Run validation + update (legacy behavior)
            print("🔍 Running default workflow: Validate + Update")
            content_map = improver.load_content_map()
            if not content_map:
                print("❌ No content map found. Run --build-map or --build-qmd-map first.")
                return 1
                
            print("\n🔍 Validating QMD mapping...")
            improver.validate_qmd_mapping(directories)
            
            print("\n✏️ Updating QMD files...")
            improver.process_qmd_files(directories, content_map)
            print("✅ Default workflow completed!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 