#!/usr/bin/env python3
"""
Figure Caption Improvement Script

A streamlined tool for improving figure and table captions in Quarto-based textbooks 
using local Ollama LLM models with strong, educational language.

Main Modes:
1. --improve/-i: LLM caption improvement and file updates (default)
2. --build-map/-b: Build content map from QMD files and save to JSON
3. --analyze/-a: Quality analysis and file structure validation
4. --repair/-r: Fix formatting issues only (no LLM)

Features:
- Follows _quarto.yml chapter ordering
- 100% extraction success (270 figures, 92 tables)
- Strong language improvements (removes weak starters)
- Proper formatting (spacing, capitalization, table prefixes)
- Context-aware processing with paragraph-level analysis
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
import pypandoc

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
    """
    Main class for improving figure and table captions using local Ollama LLM models.
    
    Provides streamlined modes:
    - build_map: Extract content structure and save to JSON
    - analyze: Quality analysis and file validation  
    - repair: Fix formatting issues only
    - improve: Complete LLM caption improvement (default)
    
    Features:
    - Follows _quarto.yml chapter ordering
    - 100% extraction success rate (270 figures, 92 tables)
    - Strong language improvements with educational focus
    - Proper formatting and spacing normalization
    - Context-aware processing with retry logic
    """
    
    def __init__(self, model_name="qwen2.5:7b"):
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
    
    def apply_sentence_case(self, text: str) -> str:
        """
        Apply proper sentence case using comprehensive rules for technical content.
        
        Rules:
        - Capitalize first word
        - Preserve proper nouns, acronyms, and technical terms  
        - Handle contractions and possessives correctly
        - Don't lowercase words that should stay capitalized
        """
        if not text:
            return text
        
        # Comprehensive list of terms to preserve (case-sensitive)
        preserve_exact = {
            # Technical acronyms
            'AI', 'ML', 'IoT', 'GPU', 'CPU', 'API', 'UI', 'UX', 'PDF', 'HTML', 'JSON', 'XML',
            'HTTP', 'HTTPS', 'SQL', 'NoSQL', 'REST', 'SOAP', 'TCP', 'UDP', 'IP', 'DNS',
            'TinyML', 'MLOps', 'DevOps', 'CI/CD', 'SDK', 'IDE', 'CLI', 'GUI',
            
            # Companies and products
            'AlexNet', 'FarmBeats', 'TikZ', 'LaTeX', 'GitHub', 'YouTube', 'Microsoft', 
            'Google', 'Amazon', 'Facebook', 'Netflix', 'Tesla', 'OpenAI', 'NVIDIA',
            'PyTorch', 'TensorFlow', 'Keras', 'Scikit-learn',
            
            # Research terms
            'CNN', 'RNN', 'LSTM', 'GRU', 'BERT', 'GPT', 'ResNet', 'VGG', 'YOLO',
            'IoU', 'mAP', 'BLEU', 'ROUGE', 'F1', 'ROC', 'AUC', 'MSE', 'RMSE',
            'SGD', 'Adam', 'AdaGrad', 'RMSprop',
            
            # File formats and standards
            'PNG', 'JPEG', 'SVG', 'CSV', 'YAML', 'TOML', 'HDF5', 'ONNX',
            'Docker', 'Kubernetes', 'AWS', 'GCP', 'Azure', 'S3',
            
            # Programming languages and tools
            'Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'C#', 'Go', 'Rust',
            'React', 'Vue', 'Angular', 'Node.js', 'MongoDB', 'PostgreSQL', 'Redis'
        }
        
        # Split into words while preserving spaces and punctuation
        tokens = re.findall(r'\b\w+(?:\'\w+)?\b|\s+|[^\w\s]', text)
        result_tokens = []
        word_index = 0  # Track actual words (not spaces/punctuation)
        
        for token in tokens:
            if re.match(r'\s+', token):  # Preserve whitespace
                result_tokens.append(token)
            elif not re.match(r'\w', token):  # Preserve punctuation
                result_tokens.append(token)
            else:  # Process words
                # First word is always capitalized
                if word_index == 0:
                    # Check if it's a preserved term first
                    if token.upper() in [p.upper() for p in preserve_exact]:
                        # Find the exact preserved form
                        preserved_form = next(p for p in preserve_exact if p.upper() == token.upper())
                        result_tokens.append(preserved_form)
                    else:
                        result_tokens.append(token.capitalize())
                else:
                    # Check if word should be preserved as-is
                    if token.upper() in [p.upper() for p in preserve_exact]:
                        # Find the exact preserved form
                        preserved_form = next(p for p in preserve_exact if p.upper() == token.upper())
                        result_tokens.append(preserved_form)
                    elif token.isupper() and len(token) > 1:
                        # Preserve all-caps words (likely acronyms)
                        result_tokens.append(token)
                    else:
                        # Apply lowercase for regular words
                        result_tokens.append(token.lower())
                
                word_index += 1
        
        return ''.join(result_tokens)

    def format_bold_explanation_caption(self, caption: str) -> str:
        """
        Format caption to ensure proper **bold**: explanation capitalization.
        Bold part: Title Case, Explanation part: Proper sentence case
        """
        if not caption or '**' not in caption or ':' not in caption:
            return caption
        
        # Parse **bold**: explanation format
        match = re.match(r'^\*\*([^*]+)\*\*:\s*(.+)$', caption.strip())
        if not match:
            return caption
        
        bold_part = match.group(1).strip()
        explanation_part = match.group(2).strip()
        
        # Apply title case to bold part
        bold_part = titlecase(bold_part)
        
        # Apply proper sentence case to explanation part
        explanation_part = self.apply_sentence_case(explanation_part)
        
        return f"**{bold_part}**: {explanation_part}"
    
    def fix_capitalization_after_periods(self, text: str) -> str:
        """
        Fix capitalization after periods, handling edge cases properly.
        """
        if not text:
            return text
        
        # Common abbreviations that shouldn't trigger capitalization of next word
        abbreviations = {
            'dr.', 'prof.', 'mr.', 'mrs.', 'ms.', 'vs.', 'etc.', 'i.e.', 'e.g.',
            'fig.', 'tbl.', 'eq.', 'sec.', 'ch.', 'vol.', 'no.', 'p.', 'pp.',
            'ml.', 'ai.', 'gpu.', 'cpu.', 'api.', 'url.', 'http.', 'https.'
        }
        
        # Split into sentences while preserving the structure
        sentences = re.split(r'(\. )', text)
        result_parts = []
        
        for i, part in enumerate(sentences):
            if i == 0:
                # First part - capitalize first letter
                if part and part[0].islower():
                    part = part[0].upper() + part[1:]
            elif part == '. ' and i + 1 < len(sentences):
                # Period followed by space - check next part
                next_part = sentences[i + 1] if i + 1 < len(sentences) else ""
                if next_part:
                    # Check if previous word was an abbreviation
                    prev_parts = ''.join(sentences[:i])
                    words = prev_parts.split()
                    last_word = words[-1].lower() if words else ""
                    
                    # If not an abbreviation, capitalize next sentence
                    if last_word not in abbreviations:
                        if next_part and next_part[0].islower():
                            sentences[i + 1] = next_part[0].upper() + next_part[1:]
            
            result_parts.append(part)
        
        return ''.join(result_parts)
    
    def improve_sentence_starters(self, text: str) -> str:
        """
        Replace weak sentence starters and mid-sentence weak patterns with stronger, more direct language.
        """
        if not text:
            return text
        
        # Split into sentences and improve each one
        sentences = re.split(r'(\. )', text)
        improved_sentences = []
        
        for sentence in sentences:
            if sentence == '. ':
                improved_sentences.append(sentence)
                continue
            
            original = sentence.strip()
            if not original:
                improved_sentences.append(sentence)
                continue
            
            # Apply improvements to this sentence
            improved = original
            
            # Patterns for beginning-of-sentence weak starters
            beginning_patterns = [
                # "Illustrates how X" -> "X" (direct approach)
                (r'^illustrates how (.+)$', r'\1'),
                (r'^shows how (.+)$', r'\1'),
                (r'^demonstrates how (.+)$', r'\1'),
                (r'^depicts how (.+)$', r'\1'),
                (r'^reveals how (.+)$', r'\1'),
                (r'^highlights how (.+)$', r'\1'),
                
                # "Illustrates the X" -> "The X" (remove weak verb)
                (r'^illustrates the (.+)$', r'The \1'),
                (r'^shows the (.+)$', r'The \1'),
                (r'^demonstrates the (.+)$', r'The \1'),
                (r'^depicts the (.+)$', r'The \1'),
                (r'^reveals the (.+)$', r'The \1'),
                (r'^highlights the (.+)$', r'The \1'),
                
                # Generic weak starters at beginning - remove entirely
                (r'^illustrates (.+)$', r'\1'),
                (r'^shows (.+)$', r'\1'),
                (r'^demonstrates (.+)$', r'\1'),
                (r'^depicts (.+)$', r'\1'),
                (r'^reveals (.+)$', r'\1'),
                (r'^highlights (.+)$', r'\1'),
            ]
            
            # Apply beginning-of-sentence patterns first
            for pattern, replacement in beginning_patterns:
                if re.search(pattern, improved, re.IGNORECASE):
                    improved = re.sub(pattern, replacement, improved, flags=re.IGNORECASE)
                    break
            
            # Patterns for mid-sentence weak constructions
            mid_sentence_patterns = [
                # "X illustrates how Y" -> stronger constructions
                (r'(.+?)\s+illustrates how (.+)', r'\2 through \1'),
                (r'(.+?)\s+demonstrates how (.+)', r'\2 via \1'),
                (r'(.+?)\s+depicts how (.+)', r'\2 using \1'),
                (r'(.+?)\s+reveals how (.+)', r'\2 through \1'),
                (r'(.+?)\s+highlights how (.+)', r'\2 via \1'),
                
                # "X illustrates that Y" -> "X confirms that Y" / "X establishes that Y"
                (r'(.+?)\s+illustrates that (.+)', r'\1 confirms that \2'),
                (r'(.+?)\s+demonstrates that (.+)', r'\1 establishes that \2'),
                (r'(.+?)\s+depicts that (.+)', r'\1 confirms that \2'),
                (r'(.+?)\s+reveals that (.+)', r'\1 establishes that \2'),
                (r'(.+?)\s+highlights that (.+)', r'\1 emphasizes that \2'),
                
                # "X illustrates Y" -> "X enables Y" / "X provides Y"
                (r'(.+?)\s+illustrates (.+)', r'\1 enables \2'),
                (r'(.+?)\s+demonstrates (.+)', r'\1 provides \2'),
                (r'(.+?)\s+depicts (.+)', r'\1 presents \2'),
                (r'(.+?)\s+reveals (.+)', r'\1 exposes \2'),
                (r'(.+?)\s+highlights (.+)', r'\1 emphasizes \2'),
            ]
            
            # Apply mid-sentence patterns
            for pattern, replacement in mid_sentence_patterns:
                if re.search(pattern, improved, re.IGNORECASE):
                    improved = re.sub(pattern, replacement, improved, flags=re.IGNORECASE)
                    break
            
            # Ensure first letter is capitalized
            if improved and improved[0].islower():
                improved = improved[0].upper() + improved[1:]
            
            # Add the improved sentence (spacing will be normalized later)
            improved_sentences.append(improved)
        
        return ''.join(improved_sentences)
    
    def normalize_spacing(self, text: str) -> str:
        """
        Normalize spacing in text - remove multiple spaces, leading/trailing spaces.
        """
        if not text:
            return text
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text
    
    def validate_and_improve_caption(self, caption: str, is_table: bool = False) -> str:
        """
        Apply all quality improvements to a caption.
        
        Args:
            caption: The caption to improve
            is_table: Whether this is a table caption (affects format requirements)
            
        Returns:
            Improved caption with proper capitalization, strong language, and format
        """
        if not caption:
            return caption
        
        # Normalize input spacing first
        caption = self.normalize_spacing(caption)
        
        # Check if caption already has table prefix and remove it for processing
        has_table_prefix = caption.startswith(': ')
        if has_table_prefix:
            caption = caption[2:]  # Remove ': ' prefix
        
        # Clean up any extra leading colons and spaces (handles edge cases like ::, :  :, etc.)
        caption = re.sub(r'^:+\s*', '', caption)
        
        # Parse **bold**: explanation format (handle spaces around colon)
        match = re.match(r'^(\*\*[^*]+\*\*)\s*:\s*(.+)$', caption)
        if not match:
            return caption
        
        bold_part = match.group(1)
        explanation = match.group(2)
        
        # Improve sentence starters in explanation
        explanation = self.improve_sentence_starters(explanation)
        
        # Fix capitalization after periods
        explanation = self.fix_capitalization_after_periods(explanation)
        
        # Normalize spacing in explanation
        explanation = self.normalize_spacing(explanation)
        
        # Combine with proper single space after colon
        improved = f"{bold_part}: {explanation}"
        
        # For tables, ensure proper : prefix format (don't double up)
        if is_table:
            improved = f": {improved}"
        
        return improved
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode an image to base64 for multimodal models."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding image {image_path}: {e}")
            return None
    
    def extract_paragraph_context(self, content: str, figure_or_table_id: str) -> Dict[str, str]:
        """Extract focused context: paragraph with figure + adjacent paragraphs."""
        
        # Split content into paragraphs (double newlines or section breaks)
        # Keep section headers with their following content
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Find which paragraph contains our figure/table
        target_paragraph_idx = None
        section_title = "Unknown Section"
        
        for i, paragraph in enumerate(paragraphs):
            # Check if this paragraph contains the figure definition or reference
            if (figure_or_table_id in paragraph or 
                f"@{figure_or_table_id}" in paragraph):
                target_paragraph_idx = i
                break
        
        if target_paragraph_idx is None:
            # Fallback: search in full content and extract around location
            return self._extract_fallback_context(content, figure_or_table_id)
        
        # Extract section title from content before the target paragraph
        full_content_before = '\n\n'.join(paragraphs[:target_paragraph_idx + 1])
        section_headers = re.findall(r'^##\s+([^#\n]+?)(?:\s*\{#[^}]+\}.*)?$', 
                                   full_content_before, re.MULTILINE)
        if section_headers:
            section_title = section_headers[-1].strip()  # Get the most recent header
        
        # Collect context paragraphs: [previous, current, next]
        context_paragraphs = []
        
        # Previous paragraph (if exists)
        if target_paragraph_idx > 0:
            prev_para = paragraphs[target_paragraph_idx - 1]
            # Skip if it's just a section header
            if not re.match(r'^##\s+', prev_para):
                context_paragraphs.append(prev_para)
        
        # Current paragraph (with the figure)
        context_paragraphs.append(paragraphs[target_paragraph_idx])
        
        # Next paragraph (if exists)
        if target_paragraph_idx + 1 < len(paragraphs):
            next_para = paragraphs[target_paragraph_idx + 1]
            # Skip if it's a section header
            if not re.match(r'^##\s+', next_para):
                context_paragraphs.append(next_para)
        
        context_content = '\n\n'.join(context_paragraphs)
        
        return {
            'title': section_title,
            'content': context_content
        }
    
    def _extract_fallback_context(self, content: str, figure_or_table_id: str) -> Dict[str, str]:
        """Fallback: extract ±300 words around figure reference."""
        lines = content.split('\n')
        section_title = "Unknown Section"
        
        for i, line in enumerate(lines):
            if figure_or_table_id in line or f"@{figure_or_table_id}" in line:
                # Find section heading
                for j in range(i, -1, -1):
                    check_line = lines[j].strip()
                    if check_line.startswith('##') and not check_line.startswith('###'):
                        section_title = re.sub(r'^#+\s*', '', check_line)
                        section_title = re.sub(r'\s*\{#[^}]+\}.*$', '', section_title)
                        break
                
                # Extract context around reference (±10 lines, then expand to word boundaries)
                start_idx = max(0, i - 10)
                end_idx = min(len(lines), i + 10)
                context_text = '\n'.join(lines[start_idx:end_idx]).strip()
                
                # Limit to ~300 words around the figure
                words = context_text.split()
                if len(words) > 300:
                    # Find the figure reference position in words
                    fig_word_pos = None
                    for word_idx, word in enumerate(words):
                        if figure_or_table_id in word:
                            fig_word_pos = word_idx
                            break
                    
                    if fig_word_pos:
                        start_word = max(0, fig_word_pos - 150)
                        end_word = min(len(words), fig_word_pos + 150)
                        context_text = ' '.join(words[start_word:end_word])
                
                return {
                    'title': section_title,
                    'content': context_text
                }
        
        # Ultimate fallback
        return {
            'title': "Unknown Section", 
            'content': content[:1000]
        }

    def extract_section_context(self, content: str, figure_or_table_id: str) -> Dict[str, str]:
        """Extract context around a figure - now uses paragraph-based approach."""
        return self.extract_paragraph_context(content, figure_or_table_id)
    
    def generate_caption_with_ollama(self, section_title: str, section_text: str, 
                                   figure_id: str, current_caption: str, 
                                   image_path: Optional[str] = None, is_table: bool = False) -> Optional[str]:
        """Generate improved caption using Ollama multimodal model with retry logic."""
        import time
        
        # Construct a focused, context-aware prompt
        prompt = f"""You are an expert at editing a caption for a visual (figure or table) in a technical AI/ML systems textbook.

Your task is to improve the caption so that it *teaches*. The goal is to help students understand what the visual conveys in the context of machine learning systems.

SECTION: {section_title}  
ORIGINAL CAPTION: {current_caption}  

TEXTBOOK CONTEXT (for reference):
{section_text[:1500]}

🧠 TASK: Rewrite the caption to make it educational, precise, and aligned with the visual’s teaching purpose.

✍️ FORMAT:
**<Key Phrase>**: Explanation sentence(s)

✅ REQUIREMENTS:

1. **Key Phrase**: A single bolded noun phrase (1–5 words) that captures the main idea. Avoid full sentences or multiple bolded phrases. If similar figures exist in this section, choose a unique but relevant phrase.

2. **Explanation**: 1–2 concise, natural sentences that express what the student *learns* from the figure or table. Use active voice. Avoid simply describing what the figure “shows”—explain what *insight* it provides or how it advances understanding.

3. **Terminology**: Use domain-specific language from the original caption if helpful, but rephrase it to clarify meaning for students.

4. **No Weak Openers**: Do not begin with “This figure...”, “This table...”, or “This diagram...”. Begin with the concept or the takeaway.

5. **Clarity & Precision**: Be specific, pedagogical, and concrete. Emphasize learning outcomes over general description.

6. **Tone**: Use a textbook tone. Use technical but student-friendly language appropriate for upper-level undergraduates or early graduate students. Avoid jargon unless it is defined or central to the concept. 

7. **Sources**: If the original caption includes a source (e.g., “Source: IEEE Spectrum”), retain it at the end of the caption in italics. Append it after a period.

📌 STRONG EXAMPLES:
**Attention Weights**: Shows how transformer models compute attention using query, key, and value interactions, enabling dynamic focus across input sequences.  
**Edge Deployment**: Demonstrates how AI and IoT technologies are integrated at the farm edge to optimize agricultural practices through real-world ML applications.  
**Dataset Splits**: Explains how ML workflows partition data into training, validation, and testing sets to support model development and evaluation.

🚫 AVOID:
- Starting with “This figure shows…” or “This table illustrates…”
- Using a full sentence or list as the bold key phrase
- Repeating the section title or being too vague (e.g., **AI System**)

🖊️ OUTPUT: Write only the improved caption below:

🚫 AVOID WEAK SENTENCE STARTERS:
- Do NOT use: "Illustrates", "Shows", "Demonstrates", "Depicts", "Reveals", "Highlights" 
- Instead use direct language: "The system processes...", "Machine learning models...", "This approach enables..."
- Be direct and specific about what the student learns

💡 EXAMPLES OF STRONG vs WEAK:
❌ Weak: "Illustrates how neural networks process data"  
✅ Strong: "Neural networks process data through layered transformations"
❌ Weak: "Shows the relationship between accuracy and efficiency"
✅ Strong: "Higher accuracy typically requires more computational resources"
"""
        
        # Retry logic: up to 3 attempts with exponential backoff
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Prepare the request payload
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,  # Higher temperature for more diverse, creative captions
                        "num_predict": 120,  # Slightly shorter for focused responses
                        "top_p": 0.9        # Add nucleus sampling for better variety
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
                    
                    # Sanity check: Reject overly long captions (likely hallucination)
                    word_count = len(new_caption.split())
                    if word_count > 100:
                        print(f"      ⚠️  Generated caption too long ({word_count} words, max 100): {new_caption[:100]}...")
                        # Don't retry for long captions - this is a formatting issue, not API error
                        return None
                    
                    # Validate the format contains **bold**: 
                    if '**' in new_caption and ':' in new_caption:
                        # Apply comprehensive quality improvements
                        formatted_caption = self.format_bold_explanation_caption(new_caption)
                        improved_caption = self.validate_and_improve_caption(formatted_caption, is_table)
                        
                        # Double-check word count after improvements
                        final_word_count = len(improved_caption.split())
                        if final_word_count > 100:
                            print(f"      ⚠️  Improved caption too long ({final_word_count} words): {improved_caption[:100]}...")
                            return None
                        
                        return improved_caption
                    else:
                        print(f"      ⚠️  Generated caption doesn't follow **bold**: format: {new_caption[:100]}")
                        # Don't retry for format issues - this is a generation problem, not API error
                        return None
                else:
                    # API error - this is worth retrying
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"      ⚠️  Ollama API error {response.status_code}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"      ❌ Ollama API error: {response.status_code} (all {max_retries} attempts failed)")
                        return None
                        
            except requests.exceptions.RequestException as e:
                # Network/connection error - worth retrying
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"      ⚠️  Request error: {e}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"      ❌ Request error: {e} (all {max_retries} attempts failed)")
                    return None
            except Exception as e:
                # Unexpected error - worth retrying once but likely a code issue
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"      ⚠️  Unexpected error: {e}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"      ❌ Unexpected error: {e} (all {max_retries} attempts failed)")
                    return None
        
        # Should never reach here due to the loop structure, but just in case
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
        """Parse QMD content to extract sections (content-based, no line numbers)."""
        # Find all section headers using regex
        section_pattern = r'^##\s+([^#\n]+?)(?:\s*\{#[^}]+\}.*)?$'
        sections = []
        
        # Split content by section headers
        parts = re.split(section_pattern, content, flags=re.MULTILINE)
        
        if len(parts) > 1:
            # First part is content before any section (if any)
            if parts[0].strip():
                sections.append({
                    'title': 'Introduction',
                    'content': parts[0].strip()
                })
            
            # Process section pairs (title, content)
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    title = parts[i].strip()
                    content_part = parts[i + 1].strip()
                    sections.append({
                        'title': title,
                        'content': content_part
                    })
        
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
        # Enhanced pattern: Handle nested brackets in captions like [Source](url)
        # We need to balance brackets properly for captions containing citations
        
        # First, find the start of our figure pattern
        start_pattern = rf'!\['
        id_pattern = rf'\s*\{{[^}}]*#{re.escape(fig_id)}(?:\s|[^}}])*\}}'
        
        # Use a more sophisticated approach to extract the caption
        # Look for the pattern and then parse it properly
        simple_pattern = rf'!\[.*?\]\([^)]+\)\s*\{{[^}}]*#{re.escape(fig_id)}(?:\s|[^}}])*\}}'
        rough_match = re.search(simple_pattern, content, re.MULTILINE)
        
        if rough_match:
            full_text = rough_match.group(0)
            # Now carefully extract the caption and path
            # Find the balanced caption between ![...] 
            caption = self._extract_balanced_caption(full_text)
            if caption is not None:
                # Extract the path - handle escaped characters properly
                path = self._extract_balanced_path(full_text)
                if path is not None:
                    return {
                        'type': 'markdown',
                        'caption': caption.strip(),
                        'path': path.strip(),
                        'full_match': full_text,
                        'start': rough_match.start(),
                        'end': rough_match.end()
                    }
        
        return None
        
    def _extract_balanced_caption(self, figure_text: str) -> Optional[str]:
        """
        Extract caption from ![caption] handling nested [brackets](links).
        
        Args:
            figure_text: The full figure markdown text starting with ![
            
        Returns:
            The caption text or None if parsing fails
        """
        if not figure_text.startswith('!['):
            return None
            
        # Start after ![
        pos = 2
        bracket_count = 0
        caption_end = None
        
        # Walk through the text, counting brackets
        for i in range(pos, len(figure_text)):
            char = figure_text[i]
            if char == '[':
                bracket_count += 1
            elif char == ']':
                if bracket_count == 0:
                    # This is the closing ] for the main caption
                    caption_end = i
                    break
                else:
                    bracket_count -= 1
        
        if caption_end is not None:
            return figure_text[2:caption_end]  # Skip ![ and get text before ]
        
        return None
    
    def _extract_balanced_path(self, figure_text: str) -> Optional[str]:
        """
        Extract path from ](path) handling escaped characters like \\(partially\\).
        
        Args:
            figure_text: The full figure markdown text
            
        Returns:
            The path text or None if parsing fails
        """
        # Find the ]( that starts the path
        path_start_pattern = r'\]\('
        path_start_match = re.search(path_start_pattern, figure_text)
        if not path_start_match:
            return None
        
        # Start after ](
        start_pos = path_start_match.end()
        pos = start_pos
        path_end = None
        
        # Walk through characters, handling escapes
        while pos < len(figure_text):
            char = figure_text[pos]
            
            if char == '\\':
                # Skip the escaped character
                pos += 2  # Skip both \ and the next character
                continue
            elif char == ')':
                # Found the closing )
                path_end = pos
                break
            else:
                pos += 1
        
        if path_end is not None:
            return figure_text[start_pos:path_end]
        
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
            **Special Function Units**: Details... {#tbl-sfu}
        
        Args:
            content: QMD file content
            tbl_id: Full table ID (e.g., "tbl-models")
            
        Returns:
            Dict with 'caption', 'full_match' or None if not found
        """
        # Try new format first (without leading colon) - allow colons in caption text
        pattern_new = rf'^([^{{\n]+?)\s*\{{[^}}]*#{re.escape(tbl_id)}(?:\s|[^}}])*\}}\s*$'
        match = re.search(pattern_new, content, re.MULTILINE)
        
        if not match:
            # Fall back to old format (with leading colon)
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
                    'failed_extractions': [],
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
                            # Store original caption as-is from the file
                            original_caption = fig_def['caption']
                            
                            content_map['figures'][fig_id] = {
                                'original_caption': original_caption,
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
                            stats['failed_extractions'].append(fig_id)
                            if qmd_file not in stats['files_with_issues']:
                                stats['files_with_issues'].append(qmd_file)
                                
                    except Exception as e:
                        print(f"    ❌ Error processing {fig_id}: {e}")
                        stats['extraction_failures'] += 1
                        stats['failed_extractions'].append(fig_id)
                        if qmd_file not in stats['files_with_issues']:
                            stats['files_with_issues'].append(qmd_file)
                
                # Process each potential table ID
                for tbl_id in potential_tbl_ids:
                    try:
                        tbl_def = self.detect_table(content, tbl_id)
                        if tbl_def:
                            # Store original caption as-is from the file
                            original_caption = tbl_def['caption']
                            
                            content_map['tables'][tbl_id] = {
                                'original_caption': original_caption,
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
                            stats['failed_extractions'].append(tbl_id)
                            if qmd_file not in stats['files_with_issues']:
                                stats['files_with_issues'].append(qmd_file)
                                
                    except Exception as e:
                        print(f"    ❌ Error processing {tbl_id}: {e}")
                        stats['extraction_failures'] += 1
                        stats['failed_extractions'].append(tbl_id)
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
        
        # Show specific failed extractions
        if stats['extraction_failures'] > 0 and 'failed_extractions' in stats:
            print(f"   📋 Failed extractions:")
            for failed_id in stats['failed_extractions']:
                print(f"      • {failed_id}")
        
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

    def process_qmd_files(self, directories: List[str], content_map: Optional[Dict] = None):
        """
        Process QMD files to update captions using targeted search-and-replace.
        
        Uses individual search-and-replace operations for each caption change
        to preserve file integrity, encoding, and formatting.
        
        Args:
            directories: List of directories to process
            content_map: Content map with figures and tables data (optional, will build if None)
        """
        print("📝 Processing QMD files for caption updates...")
        
        # Build content map if not provided
        if content_map is None:
            print("📄 Building content map from QMD files...")
            content_map = self.build_content_map_from_qmd(directories)
            if not content_map:
                print("❌ Failed to build content map")
                return
        
        # Collect all items that need updates
        updates_to_apply = []
        
        # Collect figures that need updates
        for fig_id, fig_data in content_map.get('figures', {}).items():
            if 'new_caption' in fig_data and fig_data.get('new_caption'):
                source_file = fig_data.get('source_file')
                original_caption = fig_data.get('original_caption', '')
                new_caption = fig_data.get('new_caption', '')
                
                if source_file and original_caption and new_caption:
                    updates_to_apply.append({
                        'file': source_file,
                        'id': fig_id,
                        'type': 'figure',
                        'original_caption': original_caption,
                        'new_caption': new_caption
                    })
        
        # Collect tables that need updates  
        for tbl_id, tbl_data in content_map.get('tables', {}).items():
            if 'new_caption' in tbl_data and tbl_data.get('new_caption'):
                source_file = tbl_data.get('source_file')
                original_caption = tbl_data.get('original_caption', '')
                new_caption = tbl_data.get('new_caption', '')
                
                if source_file and original_caption and new_caption:
                    updates_to_apply.append({
                        'file': source_file,
                        'id': tbl_id,
                        'type': 'table',
                        'original_caption': original_caption,
                        'new_caption': new_caption
                    })
        
        if not updates_to_apply:
            print("ℹ️  No caption updates needed (no new_caption entries found)")
            return
        
        # Apply targeted search-and-replace for each update
        total_successful = 0
        total_failed = 0
        
        for update in updates_to_apply:
            try:
                file_path = update['file']
                item_id = update['id']
                item_type = update['type']
                original_caption = update['original_caption']
                new_caption = update['new_caption']
                
                print(f"📄 Updating {item_id} in {file_path}")
                
                # Create targeted search pattern for this specific caption
                success = self.apply_targeted_caption_update(
                    file_path, item_id, item_type, original_caption, new_caption
                )
                
                if success:
                    total_successful += 1
                    print(f"   ✅ Updated: {item_id}")
                else:
                    total_failed += 1
                    print(f"   ❌ Failed: {item_id} (pattern not found)")
                    
            except Exception as e:
                total_failed += 1
                print(f"   ❌ Error updating {update['id']}: {e}")
        
        print(f"📊 Summary: {total_successful} successful updates, {total_failed} failed")
        
    def apply_targeted_caption_update(self, file_path: str, item_id: str, item_type: str, 
                                    original_caption: str, new_caption: str) -> bool:
        """
        Apply a single targeted caption update using precise search-and-replace.
        
        Args:
            file_path: Path to the QMD file
            item_id: Figure or table ID (e.g., "fig-example", "tbl-data")
            item_type: "figure" or "table"
            original_caption: Current caption text to find
            new_caption: New caption text to replace with
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Read current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Build targeted search pattern based on type
            if item_type == 'figure':
                old_pattern, new_pattern = self.build_figure_search_patterns(
                    item_id, original_caption, new_caption, content
                )
            else:  # table
                old_pattern, new_pattern = self.build_table_search_patterns(
                    item_id, original_caption, new_caption, content
                )
            
            if not old_pattern:
                return False
            
            # Verify the pattern exists exactly once
            if content.count(old_pattern) != 1:
                print(f"      ⚠️  Pattern occurs {content.count(old_pattern)} times (expected 1)")
                return False
            
            # Apply the replacement
            new_content = content.replace(old_pattern, new_pattern)
            
            # Write back the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True
            
        except Exception as e:
            print(f"      ❌ Error in targeted update: {e}")
            return False
    
    def build_figure_search_patterns(self, fig_id: str, original_caption: str, 
                                   new_caption: str, content: str) -> Tuple[str, str]:
        """
        Build precise search patterns for figure caption replacement.
        
        Returns:
            Tuple of (old_pattern, new_pattern) or (None, None) if not found
        """
        # Try different figure formats in order of specificity
        
        # 1. Markdown figure: ![caption](path){#fig-id}
        markdown_pattern = rf'!\[{re.escape(original_caption)}\](\([^)]+\)\s*\{{[^}}]*#{re.escape(fig_id)}[^}}]*\}})'
        if re.search(markdown_pattern, content):
            old_pattern = re.search(markdown_pattern, content).group(0)
            new_pattern = f'![{new_caption}]' + re.search(markdown_pattern, content).group(1)
            return old_pattern, new_pattern
        
        # 2. TikZ figure: look for caption line in div block
        tikz_div_pattern = rf'(:::\s*\{{[^}}]*#{re.escape(fig_id)}[^}}]*\}}.*?```\s*\n\s*){re.escape(original_caption)}(\s*)(:::)'
        match = re.search(tikz_div_pattern, content, re.DOTALL)
        if match:
            old_pattern = match.group(0)
            # Ensure there's a line break before the closing :::
            line_break = match.group(2) if match.group(2) else '\n'
            new_pattern = match.group(1) + new_caption + line_break + match.group(3)
            return old_pattern, new_pattern
        
        # 3. Code figure: #| fig-cap: "caption"
        code_pattern = rf'(#\|\s*fig-cap:\s*["\']?){re.escape(original_caption)}(["\']?)'
        if re.search(code_pattern, content):
            old_pattern = re.search(code_pattern, content).group(0)
            new_pattern = re.search(code_pattern, content).group(1) + new_caption + re.search(code_pattern, content).group(2)
            return old_pattern, new_pattern
        
        return None, None
    
    def build_table_search_patterns(self, tbl_id: str, original_caption: str, 
                                  new_caption: str, content: str) -> Tuple[str, str]:
        """
        Build precise search patterns for table caption replacement.
        
        Returns:
            Tuple of (old_pattern, new_pattern) or (None, None) if not found
        """
        # Try both old and new table formats, and handle cases with/without line breaks
        
        # 1. Old format: : Caption {#tbl-id} (with proper line break)
        old_format_pattern = rf'^:\s*{re.escape(original_caption)}(\s*\{{[^}}]*#{re.escape(tbl_id)}[^}}]*\}})(\s*)$'
        match = re.search(old_format_pattern, content, re.MULTILINE)
        if match:
            old_pattern = match.group(0)
            # Ensure there's a line break after the caption AND preserve the : prefix
            line_break = match.group(2) if match.group(2) else '\n'
            new_pattern = ': ' + new_caption + match.group(1) + line_break
            return old_pattern, new_pattern
        
        # 2. Old format: : Caption {#tbl-id} (content stuck to same line - problematic case)
        old_format_stuck_pattern = rf'^:\s*{re.escape(original_caption)}(\s*\{{[^}}]*#{re.escape(tbl_id)}[^}}]*\}})([^\n]*)'
        match = re.search(old_format_stuck_pattern, content, re.MULTILINE)
        if match:
            old_pattern = match.group(0)
            # Force a line break before the following content AND preserve the : prefix
            following_content = match.group(2)
            new_pattern = ': ' + new_caption + match.group(1) + '\n' + following_content
            return old_pattern, new_pattern
        
        # 3. New format: Caption {#tbl-id} (with proper line break) - convert to old format with :
        new_format_pattern = rf'^{re.escape(original_caption)}(\s*\{{[^}}]*#{re.escape(tbl_id)}[^}}]*\}})(\s*)$'
        match = re.search(new_format_pattern, content, re.MULTILINE)
        if match:
            old_pattern = match.group(0)
            # Ensure there's a line break after the caption AND add : prefix for consistency
            line_break = match.group(2) if match.group(2) else '\n'
            new_pattern = ': ' + new_caption + match.group(1) + line_break
            return old_pattern, new_pattern
        
        # 4. New format: Caption {#tbl-id} (content stuck to same line) - convert to old format with :
        new_format_stuck_pattern = rf'^{re.escape(original_caption)}(\s*\{{[^}}]*#{re.escape(tbl_id)}[^}}]*\}})([^\n]*)'
        match = re.search(new_format_stuck_pattern, content, re.MULTILINE)
        if match:
            old_pattern = match.group(0)
            # Force a line break before the following content AND add : prefix for consistency
            following_content = match.group(2)
            new_pattern = ': ' + new_caption + match.group(1) + '\n' + following_content
            return old_pattern, new_pattern
        
        return None, None

    def improve_captions_with_llm(self, directories: List[str], content_map: Optional[Dict] = None):
        """Improve captions using LLM and immediately update each file after processing."""
        print("🤖 Improving captions with LLM...")
        
        # Build content map if not provided
        if content_map is None:
            print("📄 Building content map from QMD files...")
            content_map = self.build_content_map_from_qmd(directories)
            if not content_map:
                print("❌ Failed to build content map")
                return {}
        
        total_figures = len(content_map.get('figures', {}))
        total_tables = len(content_map.get('tables', {}))
        
        if total_figures == 0 and total_tables == 0:
            print("❌ No figures or tables found in content map")
            return content_map
        
        print(f"📊 Processing: {total_figures} figures, {total_tables} tables")
        
        # Group items by source file for efficient processing
        files_to_process = {}
        
        # Group figures by file
        for fig_id, fig_data in content_map.get('figures', {}).items():
            source_file = fig_data.get('source_file')
            if source_file:
                if source_file not in files_to_process:
                    files_to_process[source_file] = {'figures': [], 'tables': []}
                files_to_process[source_file]['figures'].append((fig_id, fig_data))
        
        # Group tables by file
        for tbl_id, tbl_data in content_map.get('tables', {}).items():
            source_file = tbl_data.get('source_file')
            if source_file:
                if source_file not in files_to_process:
                    files_to_process[source_file] = {'figures': [], 'tables': []}
                files_to_process[source_file]['tables'].append((tbl_id, tbl_data))
        
        total_improved = 0
        files_updated = 0
        
        # Process each file independently
        for source_file, items in files_to_process.items():
            print(f"\n📄 Processing file: {source_file}")
            
            try:
                # Read file content once for this file
                with open(source_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                file_improvements = []
                file_improved_count = 0
                
                # Process all figures in this file
                for fig_id, fig_data in items['figures']:
                    print(f"  📊 Processing figure: {fig_id}")
                    
                    try:
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
                        current_caption = fig_data.get('original_caption', '')
                        new_caption = self.generate_caption_with_ollama(
                            context['title'], 
                            context['content'], 
                            fig_id, 
                            current_caption, 
                            image_path,
                            is_table=False
                        )
                        
                        if new_caption and new_caption != current_caption:
                            fig_data['new_caption'] = new_caption
                            file_improvements.append({
                                'id': fig_id,
                                'type': 'figure',
                                'original': current_caption,
                                'new': new_caption
                            })
                            file_improved_count += 1
                            word_count = len(new_caption.split())
                            print(f"    ✅ Improved ({word_count} words): {new_caption[:80]}{'...' if len(new_caption) > 80 else ''}")
                        else:
                            print(f"    ⚠️  No improvement generated")
                            
                    except Exception as e:
                        print(f"    ❌ Error processing {fig_id}: {e}")
                
                # Process all tables in this file
                for tbl_id, tbl_data in items['tables']:
                    print(f"  📋 Processing table: {tbl_id}")
                    
                    try:
                        # Extract context around this table
                        context = self.extract_section_context(file_content, tbl_id)
                        
                        # Generate improved caption (no image for tables)
                        current_caption = tbl_data.get('original_caption', '')
                        new_caption = self.generate_caption_with_ollama(
                            context['title'], 
                            context['content'], 
                            tbl_id, 
                            current_caption, 
                            None,  # No image for tables
                            is_table=True
                        )
                        
                        if new_caption and new_caption != current_caption:
                            tbl_data['new_caption'] = new_caption
                            file_improvements.append({
                                'id': tbl_id,
                                'type': 'table',
                                'original': current_caption,
                                'new': new_caption
                            })
                            file_improved_count += 1
                            word_count = len(new_caption.split())
                            print(f"    ✅ Improved ({word_count} words): {new_caption[:80]}{'...' if len(new_caption) > 80 else ''}")
                        else:
                            print(f"    ⚠️  No improvement generated")
                            
                    except Exception as e:
                        print(f"    ❌ Error processing {tbl_id}: {e}")
                
                # Immediately update this file if we have improvements
                if file_improvements:
                    print(f"  ✏️  Updating file with {file_improved_count} improvements...")
                    self.update_single_file_captions(source_file, file_improvements)
                    files_updated += 1
                    print(f"  ✅ File updated successfully!")
                else:
                    print(f"  ℹ️  No improvements for this file - skipping update")
                
                total_improved += file_improved_count
                
            except Exception as e:
                print(f"  ❌ Error processing file {source_file}: {e}")
        
        print(f"\n🎉 LLM improvement complete!")
        print(f"   📊 Total captions improved: {total_improved}")
        print(f"   📁 Files updated: {files_updated}")
        return content_map
    
    def update_single_file_captions(self, file_path: str, improvements: List[Dict]):
        """
        Update a single QMD file with improved captions using targeted search-and-replace.
        
        Args:
            file_path: Path to the QMD file to update
            improvements: List of dicts with 'id', 'type', 'original', 'new' keys
        """
        if not improvements:
            return
        
        success_count = 0
        
        for improvement in improvements:
            item_id = improvement['id']
            item_type = improvement['type']
            original_caption = improvement['original']
            new_caption = improvement['new']
            
            try:
                success = self.apply_targeted_caption_update(
                    file_path, item_id, item_type, original_caption, new_caption
                )
                
                if success:
                    success_count += 1
                    print(f"    ✅ Updated {item_id}")
                else:
                    print(f"    ⚠️  Failed to update {item_id}")
                    
            except Exception as e:
                print(f"    ❌ Error updating {item_id}: {e}")
        
        print(f"    📊 Successfully updated {success_count}/{len(improvements)} items in {file_path}")
    
    def complete_caption_improvement_workflow(self, directories: List[str], save_json: bool = False):
        """
        Complete LLM caption improvement process (used by --improve and default mode).
        
        Process: Extract → Analyze → Improve with LLM → Update files → Validate
        
        Args:
            directories: List of directories to process (follows _quarto.yml order)
            save_json: Whether to save detailed content map to JSON file
            
        Returns:
            Final content map with all improvements applied
        """
        print("🚀 Starting LLM caption improvement process...")
        
        # Step 1: Extract content map from QMD files
        print("\n📄 Step 1: Extracting figures and tables from QMD files...")
        content_map = self.build_content_map_from_qmd(directories)
        if not content_map:
            print("❌ Failed to build content map")
            return {}
        
        total_items = len(content_map.get('figures', {})) + len(content_map.get('tables', {}))
        print(f"✅ Found {total_items} items to process")
        
        # Optional: Save JSON for inspection
        if save_json:
            self.save_content_map(content_map)
            print("💾 Content map saved to content_map.json for inspection")
        
        # Step 2: Improve captions using LLM (files updated immediately during processing)
        print("\n🤖 Step 2: Improving captions with LLM...")
        improved_content_map = self.improve_captions_with_llm(directories, content_map)
        
        # Count improvements
        improved_count = 0
        for fig_data in improved_content_map.get('figures', {}).values():
            if fig_data.get('new_caption'):
                improved_count += 1
        for tbl_data in improved_content_map.get('tables', {}).values():
            if tbl_data.get('new_caption'):
                improved_count += 1
        
        if improved_count == 0:
            print("⚠️  No captions were improved. Workflow complete.")
            return improved_content_map
        
        print(f"✅ {improved_count} captions improved and files updated")
        
        # Step 3: Save improvements summary to JSON file  
        print("\n💾 Step 3: Saving improvements summary...")
        improvements_file = self.save_improvements_summary(improved_content_map, directories, improved_count, total_items)
        
        print("\n🎉 LLM caption improvement completed successfully!")
        print(f"📊 Total items processed: {total_items}")
        print(f"📝 Items improved: {improved_count}")
        print(f"📁 Directories: {', '.join(directories)}")
        print(f"📄 Improvements saved to: {improvements_file}")
        
        return improved_content_map
    
    def save_improvements_summary(self, content_map: Dict, directories: List[str], improved_count: int, total_items: int) -> str:
        """
        Save a comprehensive summary of caption improvements to a JSON file.
        
        Args:
            content_map: Content map with original and improved captions
            directories: Directories processed
            improved_count: Number of items improved
            total_items: Total items processed
            
        Returns:
            Path to the saved improvements file
        """
        from datetime import datetime
        
        # Create improvements summary
        improvements = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'directories_processed': directories,
                'total_items': total_items,
                'items_improved': improved_count,
                'success_rate': f"{(improved_count/total_items*100):.1f}%" if total_items > 0 else "0%",
                'workflow': 'complete_caption_improvement_workflow',
                'method': 'pypandoc_ast_context_extraction'
            },
            'improvements': {
                'figures': {},
                'tables': {}
            },
            'summary': {
                'figures_improved': 0,
                'tables_improved': 0,
                'no_change': 0
            }
        }
        
        # Process figures
        for fig_id, fig_data in content_map.get('figures', {}).items():
            original = fig_data.get('original_caption', '')
            improved = fig_data.get('new_caption', '')
            
            improvement_entry = {
                'id': fig_id,
                'type': fig_data.get('type', 'unknown'),
                'source_file': fig_data.get('source_file', ''),
                'original_caption': original,
                'improved_caption': improved,
                'status': 'improved' if improved and improved != original else 'no_change'
            }
            
            improvements['improvements']['figures'][fig_id] = improvement_entry
            
            if improved and improved != original:
                improvements['summary']['figures_improved'] += 1
            else:
                improvements['summary']['no_change'] += 1
        
        # Process tables
        for tbl_id, tbl_data in content_map.get('tables', {}).items():
            original = tbl_data.get('original_caption', '')
            improved = tbl_data.get('new_caption', '')
            
            improvement_entry = {
                'id': tbl_id,
                'type': 'table',
                'source_file': tbl_data.get('source_file', ''),
                'original_caption': original,
                'improved_caption': improved,
                'status': 'improved' if improved and improved != original else 'no_change'
            }
            
            improvements['improvements']['tables'][tbl_id] = improvement_entry
            
            if improved and improved != original:
                improvements['summary']['tables_improved'] += 1
            else:
                improvements['summary']['no_change'] += 1
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"caption_improvements_{timestamp}.json"
        
        # Convert any Path objects to strings for JSON serialization
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
        
        # Save to JSON file
        try:
            serializable_improvements = convert_paths_to_strings(improvements)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_improvements, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Improvements summary saved to: {filename}")
            print(f"📊 Summary: {improvements['summary']['figures_improved']} figures + {improvements['summary']['tables_improved']} tables improved")
            
            return filename
            
        except Exception as e:
            print(f"❌ Error saving improvements summary: {e}")
            return ""

    def extract_references_with_pypandoc(self, content: str, qmd_file_path: str) -> Dict[str, Dict]:
        """
        Use pypandoc to systematically extract figure/table references and their context.
        
        This approach:
        1. Parses QMD to structured AST using pypandoc
        2. Finds all @fig-id and @tbl-id references in paragraphs
        3. Extracts surrounding paragraph context systematically
        4. Maps references to their #fig-id definitions
        
        Returns:
            Dict mapping figure_id -> {
                'reference_context': paragraph context around @fig-id,
                'definition_info': info about #fig-id definition,
                'section_title': section containing the reference
            }
        """
        try:
            # Convert QMD to JSON AST using pypandoc
            ast_json = pypandoc.convert_text(
                content, 
                'json', 
                format='markdown+smart',
                extra_args=['--preserve-tabs']
            )
            ast = json.loads(ast_json)
            
            references_map = {}
            
            # Walk the AST to find references and context
            def walk_ast(element, section_title="Unknown Section", paragraph_context=None):
                if isinstance(element, dict):
                    element_type = element.get('t', '')
                    
                    # Track section headers
                    if element_type == 'Header':
                        level = element.get('c', [None, None, []])[0]
                        if level == 2:  # ## headers
                            inlines = element.get('c', [None, None, []])[2]
                            section_title = self._extract_text_from_inlines(inlines)
                    
                    # Process paragraphs to find cross-references
                    elif element_type == 'Para':
                        para_content = element.get('c', [])
                        para_text = self._extract_text_from_inlines(para_content)
                        
                        # Find figure/table references in this paragraph
                        fig_refs = re.findall(r'@(fig-[a-zA-Z0-9_-]+)', para_text)
                        tbl_refs = re.findall(r'@(tbl-[a-zA-Z0-9_-]+)', para_text)
                        
                        for ref_id in fig_refs + tbl_refs:
                            if ref_id not in references_map:
                                # Get definition info
                                def_info = self.find_figure_definition_in_qmd(content, ref_id) or \
                                          self.find_table_definition_in_qmd(content, ref_id)
                                
                                references_map[ref_id] = {
                                    'reference_paragraph': para_text,
                                    'section_title': section_title,
                                    'definition_info': def_info,
                                    'file_path': qmd_file_path
                                }
                    
                    # Recursively process content
                    if 'c' in element:
                        if isinstance(element['c'], list):
                            for item in element['c']:
                                walk_ast(item, section_title, paragraph_context)
                        else:
                            walk_ast(element['c'], section_title, paragraph_context)
                
                elif isinstance(element, list):
                    for item in element:
                        walk_ast(item, section_title, paragraph_context)
            
            # Start walking from the document blocks
            blocks = ast.get('blocks', [])
            walk_ast(blocks)
            
            # Now get adjacent paragraph context for each reference
            for ref_id, ref_data in references_map.items():
                context = self._get_adjacent_paragraphs_from_ast(ast, ref_data['reference_paragraph'])
                ref_data['context_paragraphs'] = context
            
            return references_map
            
        except Exception as e:
            print(f"⚠️  pypandoc parsing failed: {e}")
            print(f"   Falling back to regex-based approach")
            return {}
    
    def _extract_text_from_inlines(self, inlines: List) -> str:
        """Extract plain text from pypandoc inline elements."""
        text_parts = []
        
        def extract_from_element(element):
            if isinstance(element, dict):
                element_type = element.get('t', '')
                if element_type == 'Str':
                    return element.get('c', '')
                elif element_type == 'Space':
                    return ' '
                elif element_type in ['Emph', 'Strong', 'Code']:
                    # Extract text from emphasized/strong/code content
                    content = element.get('c', [])
                    if isinstance(content, list):
                        return ''.join(extract_from_element(item) for item in content)
                    return str(content)
                elif element_type == 'Link':
                    # Extract text from link content (first element of c)
                    link_content = element.get('c', [[], '', []])[0]
                    return ''.join(extract_from_element(item) for item in link_content)
                # Handle other inline types as needed
                elif 'c' in element:
                    content = element['c']
                    if isinstance(content, list):
                        return ''.join(extract_from_element(item) for item in content)
                    return str(content)
            elif isinstance(element, str):
                return element
            return ''
        
        for inline in inlines:
            text_parts.append(extract_from_element(inline))
        
        return ''.join(text_parts)
    
    def _get_adjacent_paragraphs_from_ast(self, ast: Dict, target_paragraph: str) -> List[str]:
        """
        Find the target paragraph in AST and return [previous, current, next] paragraphs.
        """
        blocks = ast.get('blocks', [])
        paragraphs = []
        
        # Extract all paragraph texts
        def collect_paragraphs(elements):
            for element in elements:
                if isinstance(element, dict) and element.get('t') == 'Para':
                    para_text = self._extract_text_from_inlines(element.get('c', []))
                    paragraphs.append(para_text)
                elif isinstance(element, dict) and 'c' in element:
                    if isinstance(element['c'], list):
                        collect_paragraphs(element['c'])
        
        collect_paragraphs(blocks)
        
        # Find target paragraph and get context
        target_idx = None
        for i, para in enumerate(paragraphs):
            if target_paragraph in para or para in target_paragraph:
                target_idx = i
                break
        
        if target_idx is None:
            return [target_paragraph]  # Fallback
        
        context_paragraphs = []
        
        # Previous paragraph
        if target_idx > 0:
            context_paragraphs.append(paragraphs[target_idx - 1])
        
        # Current paragraph
        context_paragraphs.append(paragraphs[target_idx])
        
        # Next paragraph
        if target_idx + 1 < len(paragraphs):
            context_paragraphs.append(paragraphs[target_idx + 1])
        
        return context_paragraphs

    def check_ollama_and_model(self, model_name: str) -> bool:
        """
        Check if Ollama is running and if the specified model is available.
        If model doesn't exist, automatically pull it.
        
        Returns:
            True if model is ready to use, False if there are issues
        """
        print(f"🔍 Checking Ollama and model: {model_name}")
        
        try:
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                print("❌ Ollama server not responding. Please start Ollama:")
                print("   brew services start ollama")
                print("   # or")  
                print("   ollama serve")
                return False
            
            # Get list of available models
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            print(f"📦 Available models: {len(available_models)} found")
            
            # Check if our model is available
            if model_name in available_models:
                print(f"✅ Model {model_name} is ready!")
                return True
            
            # Model not found - try to pull it
            print(f"📥 Model {model_name} not found. Pulling from Ollama registry...")
            print(f"⏳ This may take several minutes for large models...")
            
            # Pull the model
            pull_response = requests.post(
                "http://localhost:11434/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=1800  # 30 minutes timeout for large models
            )
            
            if pull_response.status_code == 200:
                # Stream the pull progress
                for line in pull_response.iter_lines():
                    if line:
                        try:
                            progress_data = json.loads(line)
                            status = progress_data.get('status', '')
                            if 'pulling' in status.lower():
                                if 'completed' in progress_data:
                                    completed = progress_data['completed']
                                    total = progress_data.get('total', completed)
                                    if total > 0:
                                        percent = (completed / total) * 100
                                        print(f"\r📥 Pulling {model_name}: {percent:.1f}%", end='', flush=True)
                            elif 'success' in status.lower():
                                print(f"\n✅ Successfully pulled {model_name}")
                                return True
                        except json.JSONDecodeError:
                            continue
                
                print(f"\n✅ Model {model_name} pull completed!")
                return True
            else:
                print(f"\n❌ Failed to pull model {model_name}: {pull_response.status_code}")
                print(f"💡 Try manually: ollama pull {model_name}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama. Please ensure Ollama is installed and running:")
            print("   1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
            print("   2. Start: ollama serve")
            print(f"   3. Pull model: ollama pull {model_name}")
            return False
        except requests.exceptions.Timeout:
            print("⏰ Timeout while pulling model. Large models can take 10+ minutes.")
            print("💡 Try running manually in another terminal:")
            print(f"   ollama pull {model_name}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error checking Ollama: {e}")
            return False

    def list_available_models(self) -> bool:
        """
        List all available Ollama models.
        
        Returns:
            True if successful, False if there are issues
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                print("❌ Ollama server not responding. Please start Ollama.")
                return False
            
            models_data = response.json()
            models = models_data.get('models', [])
            
            if not models:
                print("📦 No models found. Popular models to try:")
                print("   ollama pull qwen2.5:7b      # Fast, good quality")
                print("   ollama pull llama3.2:3b     # Smaller, faster")
                print("   ollama pull qwen2.5:14b     # Larger, better quality")
                print("   ollama pull mistral:7b      # Alternative option")
                return True
            
            print(f"📦 Available Ollama Models ({len(models)} found):")
            print("=" * 60)
            
            # Sort models by size for better display
            sorted_models = sorted(models, key=lambda x: x.get('size', 0))
            
            for model in sorted_models:
                name = model.get('name', 'Unknown')
                size = model.get('size', 0)
                size_gb = size / (1024**3) if size > 0 else 0
                modified = model.get('modified_at', '')
                
                # Format size nicely
                if size_gb >= 1:
                    size_str = f"{size_gb:.1f}GB"
                else:
                    size_mb = size / (1024**2)
                    size_str = f"{size_mb:.0f}MB"
                
                # Format date
                if modified:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(modified.replace('Z', '+00:00'))
                        date_str = dt.strftime('%Y-%m-%d')
                    except:
                        date_str = modified[:10]
                else:
                    date_str = "Unknown"
                
                print(f"  📊 {name:<25} │ {size_str:>8} │ {date_str}")
            
            print("=" * 60)
            print("💡 Usage: python improve_figure_captions.py -d contents/core/ --model MODEL_NAME")
            return True
            
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama. Please ensure Ollama is running.")
            return False
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return False

    def extract_section_context(self, content: str, figure_or_table_id: str) -> Dict[str, str]:
        """Extract context around a figure - now uses pypandoc AST when possible."""
        # Try pypandoc approach first
        try:
            references = self.extract_references_with_pypandoc(content, "temp_file")
            if figure_or_table_id in references:
                ref_data = references[figure_or_table_id]
                context_paragraphs = ref_data.get('context_paragraphs', [])
                
                return {
                    'title': ref_data.get('section_title', 'Unknown Section'),
                    'content': '\n\n'.join(context_paragraphs)
                }
        except Exception as e:
            print(f"   ⚠️  pypandoc approach failed: {e}")
        
        # Fallback to original paragraph-based approach
        return self.extract_paragraph_context(content, figure_or_table_id)


def main():
    parser = argparse.ArgumentParser(
        description="Improve figure and table captions using local Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Improve captions with LLM (recommended default):
  python improve_figure_captions.py -d contents/core/
  python improve_figure_captions.py --improve -d contents/core/
  
  # Using different models:
  python improve_figure_captions.py -d contents/core/ --model llama3.2:3b
  python improve_figure_captions.py -i -d contents/core/ -m qwen2.5:14b
  python improve_figure_captions.py -d contents/core/ --model mistral:7b
  
  # Analysis and utilities:
  python improve_figure_captions.py --build-map -d contents/core/
  python improve_figure_captions.py -b -d contents/core/
  python improve_figure_captions.py --analyze -d contents/core/
  python improve_figure_captions.py --repair -d contents/core/
  
  # Multiple directories:
  python improve_figure_captions.py -d contents/core/ -d contents/frontmatter/ -m llama3.2:3b
  
  # Save detailed JSON output:
  python improve_figure_captions.py -d contents/core/ --save-json
"""
    )
    
    # Multiple file/directory input
    parser.add_argument('-f', '--files', action='append',
                       help='QMD files to process (can be used multiple times)')
    parser.add_argument('-d', '--directories', action='append',
                       help='Directories to scan for QMD files (can be used multiple times)')

    # Mode selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--build-map', '-b', action='store_true',
                      help='Build content map from QMD files and save to JSON')
    group.add_argument('--analyze', '-a', action='store_true',
                      help='Analyze caption quality and validate file structure')
    group.add_argument('--repair', '-r', action='store_true',
                      help='Repair caption formatting issues only')
    group.add_argument('--improve', '-i', action='store_true',
                      help='Improve captions using LLM and update files (default mode)')

    # Model options
    parser.add_argument('--model', '-m', default='qwen2.5:7b',
                       help='Ollama model to use (default: qwen2.5:7b)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available Ollama models and exit')

    # Output options
    parser.add_argument('--save-json', action='store_true',
                       help='Save detailed content map to JSON file')

    args = parser.parse_args()
    
    # Handle --list-models flag
    if args.list_models:
        improver = FigureCaptionImprover()
        success = improver.list_available_models()
        return 0 if success else 1
    
    # Validate that we have input files/directories for other operations
    if not args.files and not args.directories:
        print("❌ Error: --files or --directories required")
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
    
    # Initialize improver with specified model
    improver = FigureCaptionImprover(model_name=args.model)
    
    # Check Ollama and model availability before proceeding
    if not improver.check_ollama_and_model(args.model):
        print(f"❌ Cannot proceed without properly configured Ollama and model {args.model}")
        return 1
    
    try:
        if args.build_map:
            # Build content map and save to JSON
            print("🔍 Building content map from QMD files...")
            content_map = improver.build_content_map_from_qmd(directories)
            if content_map:
                print("✅ Content map building completed!")
                
                # Always save JSON for --build-map
                improver.save_content_map(content_map)
                
                # Show extraction report
                stats = content_map['metadata']['extraction_stats']
                if stats['extraction_failures'] == 0:
                    print("🎉 Perfect extraction! All figures and tables successfully processed.")
                else:
                    print(f"⚠️  {stats['extraction_failures']} extraction failures detected.")
                    print("💡 Consider reviewing the files with issues for manual fixes.")
                
                # Show brief summary
                print(f"\n📋 CONTENT SUMMARY:")
                print(f"   📊 Figures: {stats['figures_found']} total")
                print(f"      • Markdown: {stats['markdown_figures']}")
                print(f"      • TikZ: {stats['tikz_figures']}")
                print(f"      • Code: {stats['code_figures']}")
                print(f"   📋 Tables: {stats['tables_found']} total")
                print(f"   📁 Files processed: {content_map['metadata']['qmd_files_scanned']}")
                
                print(f"\n💾 Content map saved to: content_map.json")
                print(f"📄 You can now review the complete JSON structure!")
                
            else:
                print("❌ Content map building failed!")
                return 1
                
        elif args.analyze:
            # Analyze caption quality and validate file structure
            print("🔍 Analyzing caption quality and file structure...")
            
            # Build content map for validation
            content_map = improver.build_content_map_from_qmd(directories)
            if not content_map:
                print("❌ Failed to build content map for analysis")
                return 1
            
            # Check caption quality
            improver.check_caption_quality(directories)
            
            # Validate QMD mapping
            improver.validate_qmd_mapping(directories, content_map)
            
            print("✅ Analysis completed!")
            
        elif args.repair:
            # Repair caption formatting issues only
            print("🔧 Repairing caption formatting issues...")
            content_map = improver.repair_captions(directories)
            if content_map and args.save_json:
                improver.save_content_map(content_map)
                print("💾 Repaired content map saved to content_map.json")
            print("✅ Caption repair completed!")
            
        elif args.improve:
            # LLM caption improvement mode (explicit)
            print("🚀 Improving captions with LLM...")
            improved_content_map = improver.complete_caption_improvement_workflow(directories, args.save_json)
            if not improved_content_map:
                return 1
                
        else:
            # Default: Same as --improve (LLM improvement)
            print("🚀 Improving captions with LLM (default mode)...")
            improved_content_map = improver.complete_caption_improvement_workflow(directories, args.save_json)
            if not improved_content_map:
                return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 