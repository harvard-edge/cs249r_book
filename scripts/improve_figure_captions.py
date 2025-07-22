#!/usr/bin/env python3
"""
Figure Caption Improvement Script (Multimodal Version)

This script uses a local Ollama multimodal model to generate improved, contextually-aware
figure captions for a textbook by analyzing both the image content and section context.

Usage:
    python improve_figure_captions.py -f file1.qmd file2.qmd
    python improve_figure_captions.py -d contents/core/
"""

import argparse
import re
import os
import json
import subprocess
import base64
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class FigureCaptionImprover:
    def __init__(self, model_name: str = "llava:7b"):
        self.model_name = model_name
        self.figure_pattern = re.compile(r'@fig-([a-zA-Z0-9_-]+)')
        self.stats = {
            'files_processed': 0,
            'figures_found': 0,
            'figures_improved': 0,
            'images_found': 0,
            'images_missing': 0,
            'json_success': 0,
            'json_failed': 0,
            'errors': []
        }
        
    def find_qmd_files(self, directory: str) -> List[Path]:
        """Find all .qmd files in a directory recursively."""
        directory_path = Path(directory)
        return list(directory_path.rglob("*.qmd"))
    
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
            
            if not figure_refs:
                print(f"   ℹ️  No figure references found")
                return
            
            print(f"   📊 Found {len(figure_refs)} figure reference(s)")
            self.stats['figures_found'] += len(figure_refs)
            
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
            
        print(f"{'='*60}")

    def run(self, files: List[str] = None, directory: str = None) -> None:
        """Run the caption improvement process."""
        if files:
            file_paths = [Path(f) for f in files]
        elif directory:
            file_paths = self.find_qmd_files(directory)
        else:
            raise ValueError("Must specify either files or directory")
        
        print(f"🚀 Starting caption improvement with model: {self.model_name}")
        print(f"📁 Found {len(file_paths)} .qmd files to process")
        
        for file_path in file_paths:
            self.process_file(file_path)
        
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Improve figure captions using Ollama multimodal LLM"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--files', nargs='+', 
                      help='Specific .qmd files to process')
    group.add_argument('-d', '--directory', 
                      help='Directory to scan for .qmd files')
    
    parser.add_argument('--model', default='llava:7b',
                       help='Ollama multimodal model to use (default: llava:7b)')
    
    args = parser.parse_args()
    
    # Check if ollama is available
    try:
        subprocess.run(['ollama', 'list'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Ollama is not available. Please install and start Ollama first.")
        return 1
    
    # Check if the specified model is available
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        if args.model not in result.stdout:
            print(f"Error: Model '{args.model}' not found. Available models:")
            print(result.stdout)
            print(f"\nInstall the model with: ollama pull {args.model}")
            return 1
    except Exception as e:
        print(f"Error checking model availability: {e}")
        return 1
    
    improver = FigureCaptionImprover(model_name=args.model)
    
    try:
        improver.run(files=args.files, directory=args.directory)
        print("🎉 Caption improvement completed!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 