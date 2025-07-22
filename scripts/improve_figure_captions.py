#!/usr/bin/env python3
"""
Content Caption Improvement Script (Unified Version)

A comprehensive tool for analyzing, validating, and improving figure and table captions
in a Quarto-based textbook using local Ollama models.

Workflow:
1. Build content map from .tex file (--build-map/-b)
2. Validate QMD mapping (--validate/-v) 
3. Check caption quality (--check/-c)
4. Repair only broken captions (--repair/-r)
5. Update all captions (--update/-u)
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
    def __init__(self, model_name="llama3.2:3b"):
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
    
    def get_book_chapters_from_quarto(self) -> List[str]:
        """Parse _quarto.yml and return list of active chapter files in order."""
        if not os.path.exists(self.quarto_config_file):
            print(f"❌ Quarto config not found: {self.quarto_config_file}")
            return []
        
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
                
            print(f"📚 Found {len(active_chapters)} active chapters in book structure")
            return active_chapters
            
        except Exception as e:
            print(f"❌ Error parsing {self.quarto_config_file}: {e}")
            return []
    
    def find_qmd_files_in_order(self, directories: List[str]) -> List[Path]:
        """Find QMD files following the book's chapter order from _quarto.yml."""
        book_chapters = self.get_book_chapters_from_quarto()
        
        if not book_chapters:
            print("⚠️  No book structure found, falling back to directory scan")
            # Fallback to original method
            all_files = []
            for directory in directories:
                all_files.extend(self.find_qmd_files(directory))
            return all_files
        
        # Filter book chapters to only those in specified directories
        filtered_chapters = []
        directory_set = {os.path.normpath(d) for d in directories}
        
        for chapter_path in book_chapters:
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
    
    def build_content_map_from_tex(self, tex_file: str = "Machine-Learning-Systems.tex") -> Dict:
        """Build comprehensive content map by parsing generated .tex file."""
        if not os.path.exists(tex_file):
            print(f"❌ .tex file not found: {tex_file}")
            print(f"💡 Run: quarto render --to titlepage-pdf")
            return {}
        
        print(f"📄 Parsing .tex file: {tex_file}")
        
        content_map = {
            "figures": {},
            "tables": {},
            "metadata": {
                "generated_at": subprocess.run(['date', '-Iseconds'], capture_output=True, text=True).stdout.strip(),
                "tex_source": tex_file,
                "total_figures": 0,
                "total_tables": 0
            }
        }
        
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                tex_content = f.read()
            
            # Parse figures - handle both orders: includegraphics before or after caption
            figure_pattern = r'\\begin\{figure\}(.*?)\\caption\{\\label\{(fig-[^}]+)\}([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}(.*?)\\end\{figure\}'
            
            def extract_caption_text(raw_caption):
                """Extract clean caption text from LaTeX with nested braces."""
                # Handle nested braces by counting brace levels
                brace_count = 0
                caption_chars = []
                
                for char in raw_caption:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count < 0:  # End of caption
                            break
                    
                    if brace_count >= 0:
                        caption_chars.append(char)
                
                caption = ''.join(caption_chars).strip()
                
                # Clean up LaTeX formatting
                caption = re.sub(r'\\textbf\{([^}]*)\}', r'\1', caption)  # Remove \textbf{}
                caption = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', caption)  # Remove other LaTeX commands
                caption = re.sub(r'\s+', ' ', caption)  # Normalize whitespace
                caption = caption.strip()
                
                return caption

            for match in re.finditer(figure_pattern, tex_content, re.DOTALL):
                before_caption = match.group(1)
                fig_id = match.group(2)
                raw_caption = match.group(3)
                after_caption = match.group(4)
                
                caption = extract_caption_text(raw_caption)
                caption = self.normalize_caption_case(caption)
                caption = self.normalize_caption_punctuation(caption)
                
                # Find includegraphics path in either before or after caption
                pdf_path = None
                includegraphics_pattern = r'(?:\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}|\\pandocbounded\{\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\})'
                
                # Check before caption first
                img_match = re.search(includegraphics_pattern, before_caption)
                if not img_match:
                    # Check after caption
                    img_match = re.search(includegraphics_pattern, after_caption)
                
                if img_match:
                    pdf_path = img_match.group(1) or img_match.group(2)
                
                # Determine figure type
                fig_type = "tikz" if "figure-pdf" in (pdf_path or "") else "regular"
                
                content_map["figures"][fig_id] = {
                    "current_caption": caption,
                    "new_caption": "",
                    "pdf_path": pdf_path,
                    "source_file": "",  # Will be determined during processing
                    "tex_line": tex_content[:match.start()].count('\n') + 1
                }
            
            # Parse subfigures with \subcaption pattern
            subfigure_pattern = r'\\subcaption\{\\label\{(fig-[^}]+)\}([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            
            for match in re.finditer(subfigure_pattern, tex_content, re.DOTALL):
                fig_id = match.group(1)
                raw_caption = match.group(2)
                caption = extract_caption_text(raw_caption)
                caption = self.normalize_caption_case(caption)
                caption = self.normalize_caption_punctuation(caption)
                
                # For subfigures, find the nearest includegraphics in the same minipage
                # Look backwards from subcaption to find includegraphics in same minipage
                match_start = match.start()
                
                # Look backwards up to 300 chars to find the includegraphics
                context_start = max(0, match_start - 300)
                before_context = tex_content[context_start:match_start]
                
                pdf_path = None
                includegraphics_pattern = r'(?:\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}|\\pandocbounded\{\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\})'
                
                # Find all images in the before context and take the closest one (last one found)
                img_matches = list(re.finditer(includegraphics_pattern, before_context))
                if img_matches:
                    img_match = img_matches[-1]  # Take the last (closest) match
                
                if img_match:
                    pdf_path = img_match.group(1) or img_match.group(2)
                
                # Determine figure type
                fig_type = "tikz" if pdf_path and "figure-pdf" in pdf_path else "regular"
                
                content_map["figures"][fig_id] = {
                    "current_caption": caption,
                    "new_caption": "",
                    "pdf_path": pdf_path,
                    "source_file": "",  # Will be determined during processing
                    "tex_line": tex_content[:match.start()].count('\n') + 1
                }
                
            # Parse tables - format is \caption{Caption text}\label{tbl-id}\tabularnewline  
            # Look for caption followed immediately by table label
            table_pattern = r'\\caption\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\\label\{(tbl-[^}]+)\}'
            
            for match in re.finditer(table_pattern, tex_content, re.DOTALL):
                raw_caption = match.group(1)
                tbl_id = match.group(2)
                caption = extract_caption_text(raw_caption)
                caption = self.normalize_caption_case(caption)
                caption = self.normalize_caption_punctuation(caption)
                
                content_map["tables"][tbl_id] = {
                    "current_caption": caption,
                    "new_caption": "",
                    "source_file": "",  # Will be determined during processing
                    "tex_line": tex_content[:match.start()].count('\n') + 1
                }
            
            content_map["metadata"]["total_figures"] = len(content_map["figures"])
            content_map["metadata"]["total_tables"] = len(content_map["tables"])
            
            print(f"✅ Found {len(content_map['figures'])} figures and {len(content_map['tables'])} tables")
            
            # Save content map
            with open(self.content_map_file, 'w', encoding='utf-8') as f:
                json.dump(content_map, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Content map saved to: {self.content_map_file}")
            
        except Exception as e:
            error_msg = f"Error parsing .tex file: {e}"
            print(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
        
        return content_map

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
        """Normalize caption case: title case for bold parts, sentence case for explanations."""
        if not caption:
            return caption
            
        # Check if caption has the pattern **Bold Part**: explanation
        bold_pattern = re.match(r'^(\*\*[^*]+\*\*):?\s*(.*)$', caption)
        
        if bold_pattern:
            bold_part = bold_pattern.group(1)
            explanation_part = bold_pattern.group(2)
            
            # Title case the bold part (but preserve the ** markers)
            bold_text = bold_part.strip('*')
            bold_text = self.apply_title_case(bold_text)
            
            # Sentence case the explanation part
            if explanation_part:
                explanation_part = explanation_part.strip()
                if explanation_part:
                    # Capitalize first letter, keep rest as sentence case
                    explanation_part = explanation_part[0].upper() + explanation_part[1:]
            
            # Reconstruct with proper formatting
            if explanation_part:
                return f"**{bold_text}**: {explanation_part}"
            else:
                return f"**{bold_text}**"
        else:
            # No bold pattern, just apply title case to the whole thing
            return self.apply_title_case(caption)
    
    def apply_title_case(self, text: str) -> str:
        """Apply proper title case to text."""
        if not text:
            return text
            
        # Articles, prepositions, and conjunctions to keep lowercase (unless at start)
        small_words = {
            'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'if', 'in', 
            'nor', 'of', 'on', 'or', 'so', 'the', 'to', 'up', 'yet', 'vs'
        }
        
        # Preserve specific terms that should keep their original case
        preserve_case = {
            'AI', 'ML', 'IoT', 'GPU', 'CPU', 'API', 'UI', 'UX', 'PDF', 'HTML', 'JSON',
            'SocratiQ', 'TinyML', 'AlexNet', 'FarmBeats', 'TikZ', 'LaTeX', 'GitHub'
        }
        
        words = text.split()
        if not words:
            return text
            
        result = []
        
        for i, word in enumerate(words):
            # Remove punctuation for comparison but keep it for result
            clean_word = word.strip('.,!?;:()[]{}')
            punct = word[len(clean_word):] if len(word) > len(clean_word) else ''
            
            # Check if this exact word should be preserved
            if clean_word in preserve_case:
                result.append(clean_word + punct)
            # First word is always capitalized (unless it's a preserved term)
            elif i == 0:
                if clean_word.lower() in [p.lower() for p in preserve_case]:
                    preserved = next((p for p in preserve_case if p.lower() == clean_word.lower()), clean_word)
                    result.append(preserved + punct)
                else:
                    result.append(clean_word.capitalize() + punct)
            # Keep small words lowercase unless they're at the start
            elif clean_word.lower() in small_words:
                result.append(clean_word.lower() + punct)
            # Keep words that are already all caps (likely acronyms)
            elif clean_word.isupper() and len(clean_word) > 1:
                result.append(clean_word + punct)
            # Check for preserved case versions
            elif clean_word.lower() in [p.lower() for p in preserve_case]:
                preserved = next((p for p in preserve_case if p.lower() == clean_word.lower()), clean_word)
                result.append(preserved + punct)
            else:
                result.append(clean_word.capitalize() + punct)
        
        return ' '.join(result)

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
        """Save updated content map to JSON file."""
        try:
            with open(self.content_map_file, 'w', encoding='utf-8') as f:
                json.dump(content_map, f, indent=2, ensure_ascii=False)
        except Exception as e:
            error_msg = f"Error saving content map: {e}"
            print(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)

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

    def find_figure_definition_in_qmd(self, content: str, fig_id: str) -> Dict:
        """Find figure definition in QMD content."""
        # Extract just the ID part after 'fig-'
        id_part = fig_id.replace('fig-', '')
        
        # Pattern 1: Standard markdown image: ![caption](path){#fig-id ...}
        # Use a more robust pattern that handles nested brackets correctly
        pattern1 = rf'!\[(.*?)\]\([^)]+\)\s*\{{[^}}]*#fig-{re.escape(id_part)}[^}}]*\}}'
        
        match = re.search(pattern1, content, re.MULTILINE | re.DOTALL)
        if match:
            return {
                'found': True,
                'current_caption': match.group(1),
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'markdown'
            }
            
        # Pattern 2: TikZ/Div figures: ::: {#fig-id ...} ... caption ... :::
        pattern2 = rf':::\s*\{{[^}}]*#fig-{re.escape(id_part)}[^}}]*\}}(.*?):::'
        
        match = re.search(pattern2, content, re.MULTILINE | re.DOTALL)
        if match:
            div_content = match.group(1)
            
            # Extract caption - it's typically after the closing ``` and before :::
            # Look for text after a closing ``` block
            caption_pattern = r'```\s*\n\s*(.+?)(?=\s*$)'
            caption_match = re.search(caption_pattern, div_content, re.MULTILINE | re.DOTALL)
            
            if caption_match:
                caption = caption_match.group(1).strip()
            else:
                # Fallback: look for any text content after tikz block
                # Remove tikz code block and get remaining text
                cleaned_content = re.sub(r'```\{\.tikz\}.*?```', '', div_content, flags=re.DOTALL)
                caption = cleaned_content.strip()
                if not caption:
                    caption = f"TikZ figure {fig_id}"
            
            return {
                'found': True,
                'current_caption': caption,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'tikz'
            }
            
        return {'found': False}

    def find_table_definition_in_qmd(self, content: str, tbl_id: str) -> Dict:
        """Find table definition in QMD content."""
        # Extract just the ID part after 'tbl-'
        id_part = tbl_id.replace('tbl-', '')
        
        # Pattern: any line that starts with : and has the table ID in braces
        # `: anything {#tbl-id ...}`
        pattern = rf'^:\s*(.+?)\s*\{{[^}}]*#tbl-{re.escape(id_part)}[^}}]*\}}'
        
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return {
                'found': True,
                'current_caption': match.group(1).strip(),
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            }
        return {'found': False}

    def update_figure_caption_in_qmd(self, content: str, fig_id: str, new_caption: str) -> str:
        """Update figure caption in QMD content."""
        # Extract just the ID part after 'fig-'
        id_part = fig_id.replace('fig-', '')
        
        # First, determine what type of figure this is
        fig_def = self.find_figure_definition_in_qmd(content, fig_id)
        if not fig_def['found']:
            return content
            
        if fig_def.get('type') == 'markdown':
            # Pattern: ![old_caption](path){#fig-id ...}
            pattern = rf'(!\[)[^\]]*(\]\([^)]+\)\s*\{{[^}}]*#fig-{re.escape(id_part)}[^}}]*\}})'
            replacement = rf'\g<1>{new_caption}\g<2>'
            updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
            
        elif fig_def.get('type') == 'tikz':
            # For TikZ figures, replace the caption text after the tikz block but before :::
            # Pattern: find the div block and replace the caption part
            pattern = rf'(:::\s*\{{[^}}]*#fig-{re.escape(id_part)}[^}}]*\}}.*?```\s*\n\s*)([^:]+?)((?:\s*:::))'
            
            def replace_caption(match):
                before = match.group(1)
                after = match.group(3)
                return f"{before}{new_caption}{after}"
            
            updated_content = re.sub(pattern, replace_caption, content, flags=re.MULTILINE | re.DOTALL)
            
            # If that didn't work, try a simpler approach - replace text at the end of the div
            if updated_content == content:
                # Find the complete div block and replace just the text portion
                div_pattern = rf'(:::\s*\{{[^}}]*#fig-{re.escape(id_part)}[^}}]*\}}.*?```\s*\n)([^:]*?)(:::'
                
                def replace_div_caption(match):
                    before = match.group(1)
                    after = match.group(3)
                    return f"{before}{new_caption}\n{after}"
                
                updated_content = re.sub(div_pattern, replace_div_caption, content, flags=re.MULTILINE | re.DOTALL)
        else:
            # Fallback to original method
            pattern = rf'(!\[)[^\]]*(\]\([^)]+\)\s*\{{[^}}]*#fig-{re.escape(id_part)}[^}}]*\}})'
            replacement = rf'\g<1>{new_caption}\g<2>'
            updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        return updated_content

    def update_table_caption_in_qmd(self, content: str, tbl_id: str, new_caption: str) -> str:
        """Update table caption in QMD content."""
        # Extract just the ID part after 'tbl-'
        id_part = tbl_id.replace('tbl-', '')
        
        # Pattern: `: old_caption {#tbl-id ...}`
        pattern = rf'^(:\s*)(.+?)(\s*\{{[^}}]*#tbl-{re.escape(id_part)}[^}}]*\}})'
        
        replacement = rf'\g<1>{new_caption}\g<3>'
        updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        return updated_content

    def process_qmd_files(self, directories: List[str], content_map: Dict):
        """Phase 3: Process QMD files to update captions using validated content map."""
        print(f"✏️ Phase 3: Updating QMD files...")
        
        # Filter to only items with confirmed QMD locations
        figures_to_update = {
            fig_id: fig_data for fig_id, fig_data in content_map.get('figures', {}).items()
            if fig_data.get('new_caption') and fig_data.get('source_file')
        }
        
        tables_to_update = {
            tbl_id: tbl_data for tbl_id, tbl_data in content_map.get('tables', {}).items()
            if tbl_data.get('new_caption') and tbl_data.get('source_file')
        }
        
        print(f"📋 Items to update:")
        print(f"   Figures: {len(figures_to_update)} (with new captions & QMD locations)")
        print(f"   Tables:  {len(tables_to_update)} (with new captions & QMD locations)")
        
        if not figures_to_update and not tables_to_update:
            print("ℹ️  No items to update. Add new_caption values to content map first.")
            return content_map
        
        # Group updates by file for efficiency
        files_to_update = {}
        
        for fig_id, fig_data in figures_to_update.items():
            qmd_file = fig_data['source_file']
            if qmd_file not in files_to_update:
                files_to_update[qmd_file] = {'figures': [], 'tables': []}
            files_to_update[qmd_file]['figures'].append((fig_id, fig_data))
        
        for tbl_id, tbl_data in tables_to_update.items():
            qmd_file = tbl_data['source_file']
            if qmd_file not in files_to_update:
                files_to_update[qmd_file] = {'figures': [], 'tables': []}
            files_to_update[qmd_file]['tables'].append((tbl_id, tbl_data))
        
        figures_updated = 0
        tables_updated = 0
        
        for qmd_file, file_data in files_to_update.items():
            try:
                print(f"\n📄 Processing: {qmd_file}")
                
                with open(qmd_file, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                updated_content = original_content
                file_updated = False
                
                # Process figures
                for (fig_id, fig_data) in file_data.get('figures', []):
                    fig_def = self.find_figure_definition_in_qmd(updated_content, fig_id)
                    if fig_def['found']:
                        normalized_caption = self.normalize_caption_case(fig_data['new_caption'])
                        normalized_caption = self.normalize_caption_punctuation(normalized_caption)
                        print(f"  📊 Updating figure: {fig_id}")
                        print(f"    📝 {fig_def['current_caption'][:60]}...")
                        print(f"    ✨ {normalized_caption[:60]}...")
                        
                        updated_content = self.update_figure_caption_in_qmd(
                            updated_content, fig_id, normalized_caption
                        )
                        
                        # Update source file in content map
                        content_map['figures'][fig_id]['source_file'] = qmd_file
                        file_updated = True
                        figures_updated += 1
                
                # Process tables
                for (tbl_id, tbl_data) in file_data.get('tables', []):
                    tbl_def = self.find_table_definition_in_qmd(updated_content, tbl_id)
                    if tbl_def['found']:
                        normalized_caption = self.normalize_caption_case(tbl_data['new_caption'])
                        normalized_caption = self.normalize_caption_punctuation(normalized_caption)
                        print(f"  📋 Updating table: {tbl_id}")
                        print(f"    📝 {tbl_def['current_caption'][:60]}...")
                        print(f"    ✨ {normalized_caption[:60]}...")
                        
                        updated_content = self.update_table_caption_in_qmd(
                            updated_content, tbl_id, normalized_caption
                        )
                        
                        # Update source file in content map
                        content_map['tables'][tbl_id]['source_file'] = qmd_file
                        file_updated = True
                        tables_updated += 1
                
                # Write updated content back to file
                if file_updated:
                    with open(qmd_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"  ✅ Updated: {qmd_file}")
                
            except Exception as e:
                error_msg = f"Error processing {qmd_file}: {e}"
                print(f"  ❌ {error_msg}")
                self.stats['errors'].append(error_msg)
        
        print(f"\n🎉 Phase 3 completed!")
        print(f"   📊 Figures updated: {figures_updated}")
        print(f"   📋 Tables updated: {tables_updated}")
        print(f"   📁 Files modified: {len([f for f in files_to_update if any(files_to_update[f].values())])}")
        
        return content_map
    
    def parse_sections(self, content: str) -> List[Dict]:
        """Parse markdown content into sections with their headers and content."""
        lines = content.split('\n')
        sections = []
        current_section = None
        
        for i, line in enumerate(lines):
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if header_match:
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {
                    'level': level,
                    'title': title,
                    'start_line': i,
                    'content_lines': [],
                    'raw_content': []
                }
            elif current_section:
                current_section['content_lines'].append(line)
                current_section['raw_content'].append(line)
        
        # Add the last section
        if current_section:
            sections.append(current_section)
            
        return sections
    
    def find_figure_references(self, content: str) -> List[Dict]:
        """Find all figure references (@fig-*) in the content."""
        references = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines):
            matches = self.figure_pattern.findall(line)
            for match in matches:
                references.append({
                    'figure_id': match,
                    'line_number': line_num,
                    'line_content': line.strip()
                })
        
        return references
    
    def find_figure_definition(self, content: str, figure_id: str) -> Optional[Dict]:
        """Find the figure definition and its current caption."""
        lines = content.split('\n')
        
        # Look for figure definition patterns
        patterns = [
            rf'!\[([^\]]*)\]\(([^)]+)\)\s*\{{[^}}]*#fig-{figure_id}[^}}]*\}}',  # ![caption](image){... #fig-id ...}
            rf'```{{.*#fig-{figure_id}.*}}',  # Code blocks with figure ids
        ]
        
        for line_num, line in enumerate(lines):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    # Try to extract current caption and image path
                    caption_match = re.search(r'!\[([^\]]*)\]', line)
                    current_caption = caption_match.group(1) if caption_match else ""
                    
                    image_match = re.search(r'!\[[^\]]*\]\(([^)]+)\)', line)
                    image_path = image_match.group(1) if image_match else ""
                    
                    return {
                        'line_number': line_num,
                        'line_content': line,
                        'current_caption': current_caption,
                        'image_path': image_path,
                        'figure_id': figure_id
                    }
        
        return None
    
    def get_section_for_line(self, sections: List[Dict], target_line: int) -> Optional[Dict]:
        """Find which section contains the given line number."""
        current_section = None
        
        for section in sections:
            if section['start_line'] <= target_line:
                current_section = section
            else:
                break
                
        return current_section
    
    def extract_section_text(self, section: Dict) -> str:
        """Extract clean text content from a section, removing markdown formatting."""
        if not section:
            return ""
            
        content = '\n'.join(section['content_lines'])
        
        # Remove common markdown formatting but keep the meaning
        # Remove headers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # Remove code blocks
        content = re.sub(r'```[^`]*```', '[CODE BLOCK]', content, flags=re.DOTALL)
        # Remove inline code
        content = re.sub(r'`([^`]+)`', r'\1', content)
        # Remove links but keep text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        # Remove bold/italic
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^*]+)\*', r'\1', content)
        # Remove figure references from context to avoid confusion
        content = re.sub(r'@fig-[a-zA-Z0-9_-]+', '', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content
    


    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 for Ollama API."""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def find_image_file(self, file_path: Path, image_path: str) -> Optional[str]:
        """Find the actual image file based on the path in the markdown."""
        # If it's already an absolute path, check if it exists
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        
        # Try relative to the .qmd file directory
        file_dir = file_path.parent
        candidate_paths = [
            file_dir / image_path,
            file_dir / "images" / image_path,
            file_dir / "images" / "png" / image_path,
            file_dir / "images" / "jpg" / image_path,
            file_dir / "images" / "jpeg" / image_path,
        ]
        
        # Also try common image extensions if no extension provided
        if '.' not in image_path:
            for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                for base_path in candidate_paths:
                    candidate_paths.append(base_path.with_suffix(ext))
        
        for candidate in candidate_paths:
            if candidate.exists():
                return str(candidate)
        
        return None
    
    def find_tikz_figures_with_pypandoc(self, content: str) -> List[Dict]:
        """Find TikZ figures using pypandoc AST parsing."""
        tikz_figures = []
        
        try:
            # Write content to temp file for pypandoc
            with tempfile.NamedTemporaryFile(mode='w', suffix='.qmd', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Convert to pandoc AST
            ast_json = pypandoc.convert_file(tmp_path, 'json', format='markdown')
            doc = json.loads(ast_json)
            
            def walk_ast(element, line_counter={"count": 0}):
                if isinstance(element, dict):
                    # Look for Div elements with fig- IDs
                    if element.get('t') == 'Div':
                        attrs = element.get('c', [None, None])[0]
                        if attrs and isinstance(attrs, list):
                            # Debug: print the actual structure we're seeing
                            # print(f"DEBUG: Div attrs structure: {attrs}")
                            
                            fig_id = None
                            
                            # Pandoc AST format can vary, let's be flexible
                            if len(attrs) >= 3:
                                # Standard format: [identifier, classes, key-value pairs]
                                identifier = attrs[0] if attrs[0] else ""
                                classes = attrs[1] if len(attrs) > 1 else []
                                key_values = attrs[2] if len(attrs) > 2 else []
                                
                                # Check if identifier starts with 'fig-' (without the #)
                                if identifier and identifier.startswith('fig-'):
                                    fig_id = identifier
                                else:
                                    # Also check in key-value pairs for id attribute
                                    if isinstance(key_values, list):
                                        for kv in key_values:
                                            if (isinstance(kv, list) and len(kv) >= 2 and 
                                                kv[0] == 'id' and kv[1].startswith('fig-')):
                                                fig_id = kv[1]
                                                break
                            
                            # Fallback: search through all attributes for any fig- pattern
                            if not fig_id:
                                for attr in attrs:
                                    if isinstance(attr, str) and attr.startswith('fig-'):
                                        fig_id = attr
                                        break
                                    elif isinstance(attr, list):
                                        for item in attr:
                                            if isinstance(item, str) and item.startswith('fig-'):
                                                fig_id = item
                                                break
                            
                            if fig_id:
                                content_blocks = element.get('c', [None, None])[1]
                                tikz_fig = self.parse_tikz_div_content(fig_id, content_blocks)
                                if tikz_fig:
                                    tikz_fig['line_number'] = line_counter["count"]
                                    tikz_figures.append(tikz_fig)
                                    self.stats['tikz_found'] += 1
                    
                    # Recursively walk through all elements
                    for key, value in element.items():
                        if isinstance(value, (list, dict)):
                            walk_ast(value, line_counter)
                elif isinstance(element, list):
                    for item in element:
                        walk_ast(item, line_counter)
                        line_counter["count"] += 1
            
            walk_ast(doc)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
        except Exception as e:
            error_msg = f"Error parsing TikZ figures with pypandoc: {e}"
            self.stats['errors'].append(error_msg)
            print(f"        ⚠️  {error_msg}")
        
        return tikz_figures
    
    def parse_tikz_div_content(self, fig_id: str, content_blocks: List) -> Optional[Dict]:
        """Extract TikZ code and caption from div content blocks."""
        tikz_code = None
        caption_blocks = []
        
        for block in content_blocks:
            if isinstance(block, dict):
                if (block.get('t') == 'CodeBlock' and 
                    len(block.get('c', [])) >= 2 and
                    isinstance(block['c'][0], list) and
                    len(block['c'][0]) >= 2 and
                    'tikz' in block['c'][0][1]):
                    # Found TikZ code block
                    tikz_code = block['c'][1]
                else:
                    # Collect non-code blocks as caption
                    caption_blocks.append(block)
        
        if tikz_code:
            return {
                'figure_id': fig_id,
                'tikz_code': tikz_code,
                'caption_blocks': caption_blocks,
                'raw_caption': self.extract_text_from_blocks(caption_blocks)
            }
        
        return None
    
    def extract_text_from_blocks(self, blocks: List) -> str:
        """Extract plain text from Pandoc AST blocks."""
        text_parts = []
        
        def extract_from_element(element):
            if isinstance(element, dict):
                if element.get('t') == 'Str':
                    return element.get('c', '')
                elif element.get('t') == 'Space':
                    return ' '
                elif element.get('t') == 'Para':
                    content = element.get('c', [])
                    return ''.join(extract_from_element(item) for item in content)
                elif 'c' in element:
                    if isinstance(element['c'], list):
                        return ''.join(extract_from_element(item) for item in element['c'])
                    else:
                        return str(element['c'])
            elif isinstance(element, list):
                return ''.join(extract_from_element(item) for item in element)
            elif isinstance(element, str):
                return element
            return ''
        
        for block in blocks:
            text_parts.append(extract_from_element(block))
        
        return ' '.join(text_parts).strip()
    
    def compile_tikz_to_image(self, tikz_code: str, figure_id: str) -> Optional[str]:
        """Compile TikZ code to PNG image using LaTeX."""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create LaTeX document with TikZ
                latex_content = f"""
\\documentclass[border=2pt]{{standalone}}
\\usepackage{{tikz}}
\\usetikzlibrary{{arrows,shapes,positioning,shadows,trees,calc,backgrounds,fit,decorations.pathreplacing,patterns}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}

% Common colors that might be used
\\definecolor{{GreenLine}}{{RGB}}{{0,128,0}}
\\definecolor{{GreenL}}{{RGB}}{{200,255,200}}
\\definecolor{{RedLine}}{{RGB}}{{128,0,0}}
\\definecolor{{RedL}}{{RGB}}{{255,200,200}}
\\definecolor{{BrownLine}}{{RGB}}{{139,69,19}}
\\definecolor{{OliveLine}}{{RGB}}{{128,128,0}}

\\begin{{document}}
{tikz_code}
\\end{{document}}
"""
                
                tex_file = os.path.join(tmp_dir, f"{figure_id}.tex")
                with open(tex_file, 'w') as f:
                    f.write(latex_content)
                
                # Compile LaTeX to PDF
                result = subprocess.run([
                    'pdflatex', '-interaction=nonstopmode', 
                    '-output-directory', tmp_dir, tex_file
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    print(f"        ❌ LaTeX compilation failed for {figure_id}")
                    print(f"        Error: {result.stderr[:200]}...")
                    self.stats['tikz_failed'] += 1
                    return None
                
                pdf_file = os.path.join(tmp_dir, f"{figure_id}.pdf")
                if not os.path.exists(pdf_file):
                    print(f"        ❌ PDF not generated for {figure_id}")
                    self.stats['tikz_failed'] += 1
                    return None
                
                # Convert PDF to PNG
                png_file = os.path.join(tmp_dir, f"{figure_id}.png")
                convert_result = subprocess.run([
                    'magick', '-density', '300', pdf_file, png_file
                ], capture_output=True, text=True, timeout=15)
                
                if convert_result.returncode != 0 or not os.path.exists(png_file):
                    print(f"        ❌ PDF to PNG conversion failed for {figure_id}")
                    self.stats['tikz_failed'] += 1
                    return None
                
                # Copy PNG to a permanent location
                output_png = f"/tmp/{figure_id}_compiled.png"
                subprocess.run(['cp', png_file, output_png], check=True)
                
                print(f"        ✅ TikZ compiled to {output_png}")
                self.stats['tikz_compiled'] += 1
                return output_png
                
        except subprocess.TimeoutExpired:
            print(f"        ⚠️  TikZ compilation timeout for {figure_id}")
            self.stats['tikz_failed'] += 1
            return None
        except Exception as e:
            print(f"        ❌ TikZ compilation error for {figure_id}: {e}")
            self.stats['tikz_failed'] += 1
            return None
    
    def update_tikz_caption_in_content(self, content: str, tikz_fig: Dict, new_caption: str) -> str:
        """Update TikZ caption in the original content."""
        lines = content.split('\n')
        
        # Find the TikZ figure div by looking for the figure ID
        fig_id = tikz_fig['figure_id']
        old_caption = tikz_fig['raw_caption']
        
        # Look for the div start and caption
        in_tikz_div = False
        after_code_block = False
        
        for i, line in enumerate(lines):
            # Check if we're entering the TikZ div
            if f"#fig-{fig_id}" in line and line.strip().startswith(':::'):
                in_tikz_div = True
                continue
                
            if in_tikz_div:
                # Check if we're past the code block
                if line.strip() == '```' and after_code_block:
                    # Next non-empty line should be the caption
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() and not lines[j].strip().startswith(':::'):
                            # This is the caption line
                            lines[j] = new_caption
                            return '\n'.join(lines)
                        elif lines[j].strip().startswith(':::'):
                            # Reached end of div without finding caption
                            break
                elif line.strip() == '```' and not after_code_block:
                    after_code_block = True
                elif line.strip().startswith(':::') and after_code_block:
                    # End of div
                    break
        
        return content  # Return unchanged if we couldn't find the caption
    
    def generate_caption_with_ollama(self, section_title: str, section_content: str, 
                                   figure_id: str, current_caption: str, 
                                   image_path: str = None) -> Optional[str]:
        """Use Ollama multimodal model to generate an improved caption."""
        
        # Prepare the prompt
        prompt = f"""You are an expert textbook editor specializing in understanding textbook material and examining figures to generate educational captions. You have extensive experience analyzing visual content in academic publications and creating captions that enhance student learning and comprehension. 

SECTION: "{section_title}"

SECTION CONTENT:
{section_content}

FIGURE ID: {figure_id}
CURRENT CAPTION: "{current_caption}"

Using your expertise in textbook editing and figure analysis, examine the provided image and section content to generate an improved, educational figure caption.

Requirements:
1. Analyze the visual elements in the image (diagrams, interfaces, charts, etc.)
2. Create a bold concept title (2-4 words) that captures the key educational concept
3. Write an educational explanation (1-3 sentences) that helps students understand both what they're seeing and why it matters for learning ML systems
4. Connect the visual content to the surrounding textbook material and learning objectives
5. Use clear, pedagogical language appropriate for university-level computer science students

Respond with ONLY valid JSON in this exact format:
{{
  "bold": "Key Concept Title",
  "explanation": "Educational explanation that describes the visual elements and explains their significance for understanding ML systems concepts."
}}

Do not include any other text, markdown, or formatting - just the JSON object."""

        try:
            # Prepare the request data
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Add image if available
            if image_path:
                encoded_image = self.encode_image(image_path)
                if encoded_image:
                    request_data["images"] = [encoded_image]
                    print(f"        🖼️  Using image: {os.path.basename(image_path)}")
                    self.stats['images_found'] += 1
                else:
                    print(f"        ⚠️  Image encoding failed: {os.path.basename(image_path)} - continuing with text-only")
                    self.stats['images_missing'] += 1
            else:
                print(f"        📝 Text-only mode (no image found)")
                self.stats['images_missing'] += 1
            
            # Call ollama API using requests (handles large payloads better than curl)
            try:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json=request_data,
                    timeout=120
                )
                response.raise_for_status()
                
                if response.status_code == 200:
                    ollama_response = response.json()
                    raw_response = ollama_response.get('response', '').strip()
                    
                    # Parse the JSON response from the model
                    try:
                        # Clean up markdown formatting if present
                        json_text = raw_response
                        if json_text.startswith('```json'):
                            json_text = json_text.replace('```json', '').replace('```', '').strip()
                        elif json_text.startswith('```'):
                            json_text = json_text.replace('```', '').strip()
                        
                        caption_data = json.loads(json_text)
                        bold_part = caption_data.get('bold', '').strip()
                        explanation_part = caption_data.get('explanation', '').strip()
                        
                        if bold_part and explanation_part:
                            # Format consistently as **Bold**: explanation
                            caption = f"**{bold_part}**: {explanation_part}"
                            print(f"        ✅ JSON parsed successfully")
                            self.stats['json_success'] += 1
                            return caption
                        else:
                            print(f"        ⚠️  Missing fields in JSON response")
                            self.stats['json_failed'] += 1
                            return None
                            
                    except json.JSONDecodeError as e:
                        print(f"        ⚠️  Invalid JSON response: {e}")
                        print(f"        Raw response: {raw_response[:100]}...")
                        self.stats['json_failed'] += 1
                        return None
                else:
                    print(f"        ❌ Ollama API error: HTTP {response.status_code}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"        ⚠️  Timeout calling ollama for figure {figure_id}")
                return None
            except requests.exceptions.RequestException as e:
                print(f"        ❌ Request error: {e}")
                return None
                
        except Exception as e:
            print(f"        ❌ Error calling ollama: {e}")
            return None
    
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
        """Phase 2: Scan QMD files and validate mapping for all figures/tables."""
        print(f"🔍 Phase 2: Validating QMD mapping...")
        
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
                        if fig_def['found']:
                            found_figures[fig_id] = {
                                'qmd_file': qmd_file,
                                'current_caption': fig_def['current_caption'],
                                'definition': fig_def
                            }
                            print(f"    ✅ Found figure: {fig_id}")
                
                # Check each table from content map  
                for tbl_id in content_map.get('tables', {}):
                    if tbl_id not in found_tables:
                        tbl_def = self.find_table_definition_in_qmd(content, tbl_id)
                        if tbl_def['found']:
                            found_tables[tbl_id] = {
                                'qmd_file': qmd_file,
                                'current_caption': tbl_def['current_caption'],
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
        
        return content_map

    def check_caption_quality(self, directories: List[str]) -> Dict[str, any]:
        """Analyze all captions and generate quality report."""
        print("🔍 Analyzing caption quality...")
        
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


def main():
    parser = argparse.ArgumentParser(
        description="Improve figure and table captions using local Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build content map from .tex file
  python improve_figure_captions.py --build-map
  
  # Check caption quality
  python improve_figure_captions.py --check -d contents/core/
  
  # Repair only broken captions  
  python improve_figure_captions.py --repair -d contents/core/
  
  # Full workflow (validate + update all)
  python improve_figure_captions.py -d contents/core/
        """
    )
    
    # Main action group (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--build-map', '-b', action='store_true',
                             help='Build content map from .tex file')
    action_group.add_argument('--check', '-c', action='store_true',
                             help='Analyze caption quality and show issues')
    action_group.add_argument('--repair', '-r', action='store_true', 
                             help='Repair only captions that need fixing')
    action_group.add_argument('--validate', '-v', action='store_true',
                             help='Validate QMD mapping only')
    action_group.add_argument('--update', '-u', action='store_true',
                             help='Update captions in QMD files')
    
    # File/directory inputs
    parser.add_argument('-f', '--files', action='append',
                       help='Specific .qmd files to process')
    parser.add_argument('-d', '--directories', action='append',
                       help='Directories to search for .qmd files')
    
    # Model selection
    parser.add_argument('-m', '--model', default="llama3.2:3b",
                       help='Ollama model to use (default: llama3.2:3b)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if (args.check or args.repair or args.validate or args.update) and not (args.files or args.directories):
        parser.error("--check, --repair, --validate, and --update require -f or -d")
    
    improver = FigureCaptionImprover(model_name=args.model)
    
    # Determine directories to process
    directories = []
    if args.directories:
        directories.extend(args.directories)
    if args.files:
        # For individual files, add their parent directories
        directories.extend([str(Path(f).parent) for f in args.files])
    
    # Execute based on action
    if args.build_map:
        # Phase 1: Build content map from .tex
        improver.build_content_map_from_tex()
        
    elif args.check:
        # Check caption quality
        report = improver.check_caption_quality(directories)
        improver.print_quality_report(report)
        
    elif args.repair:
        # Repair only broken captions
        improver.repair_captions(directories)
        
    elif args.validate:
        # Phase 2: Validate QMD mapping
        content_map = improver.load_content_map()
        if content_map:
            improver.validate_qmd_mapping(directories, content_map)
        else:
            print("❌ No content map found. Run --build-map first.")
            
    elif args.update:
        # Phase 3: Update QMD files
        content_map = improver.load_content_map()
        if content_map:
            improver.process_qmd_files(directories, content_map)
        else:
            print("❌ No content map found. Run --build-map first.")
            
    else:
        # Default: Phase 2 + 3 (validate and update)
        content_map = improver.load_content_map()
        if content_map:
            improver.validate_qmd_mapping(directories, content_map)
            improver.process_qmd_files(directories, content_map)
        else:
            print("❌ No content map found. Run --build-map first.")

if __name__ == "__main__":
    main() 