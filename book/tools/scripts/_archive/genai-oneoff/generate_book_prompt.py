#!/usr/bin/env python3
import os
import yaml
import argparse
import sys

def find_section_by_id(contents, target_id):
    """Recursively finds a section with a specific ID in the sidebar contents."""
    for item in contents:
        if isinstance(item, dict):
            if item.get('id') == target_id:
                return item
            if 'contents' in item:
                result = find_section_by_id(item['contents'], target_id)
                if result:
                    return result
    return None

def extract_hrefs(section_contents):
    """Recursively extracts hrefs from a section's contents."""
    hrefs = []
    for item in section_contents:
        if isinstance(item, dict):
            if 'href' in item:
                hrefs.append(item['href'])
            if 'contents' in item:
                hrefs.extend(extract_hrefs(item['contents']))
        elif isinstance(item, str):
             # In case it's a direct string path (unlikely in this config but possible in Quarto)
             # But usually simple strings in contents are headers or files without labels
             # We'll treat it as a potential path if it ends in .qmd
             if item.endswith('.qmd'):
                 hrefs.append(item)
    return hrefs

def get_chapter_order(volume, config_path="quarto/config/_quarto-html.yml"):
    """
    Parses the _quarto-html.yml file to extract the chapter order for a specific volume.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    chapters = []
    
    if 'website' in config and 'sidebar' in config['website']:
        sidebar = config['website']['sidebar']
        
        # The sidebar is a list of items. We need to search through it.
        # It seems the top level might be a single item or multiple.
        # We pass the whole sidebar list to the search function.
        
        volume_id = f'volume-{volume}'
        volume_section = find_section_by_id(sidebar, volume_id)
        
        if volume_section and 'contents' in volume_section:
            chapters = extract_hrefs(volume_section['contents'])
            
    return chapters

def read_file_content(filepath):
    """Reads the content of a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"[Error reading {filepath}: {e}]"

def generate_prompt(volume, output_file):
    """Generates the prompt file for the specified volume."""
    
    base_dir = "quarto"
    chapters = get_chapter_order(volume)
    
    if not chapters:
        print(f"No chapters found for Volume {volume}. Check your config parsing logic.")
        return

    print(f"Found {len(chapters)} chapters for Volume {volume}.")

    header_vol1 = """You are an expert Technical Editor and Computer Science Professor reviewing a textbook titled "Machine Learning Systems".
The book is divided into two volumes. You are currently reviewing the full text of **Volume 1**.

Your Goal: Perform a holistic review of the entire volume to ensure narrative coherence, technical accuracy, and pedagogical flow.

Please analyze the following text and provide a report addressing:

1. **Narrative Arc & Coherence:**
   - Does the book follow a logical progression (Build -> Optimize -> Deploy)?
   - Are there disconnects between chapters? (e.g., Chapter 5 assumes knowledge only introduced in Chapter 8).
   - Does the Introduction properly set the stage for what follows?
   - Does the Conclusion effectively tie everything together?

2. **Missing Concepts & Gaps:**
   - Are there critical ML Systems concepts missing that should be in a foundational volume?
   - Are there "orphan" concepts introduced but never fully explained or used?

3. **Repetition vs. Reinforcement:**
   - Identify instances of unnecessary repetition (copy-paste style) versus good pedagogical reinforcement.

4. **Tone & Style:**
   - Is the voice consistent across chapters?

Here is the full text of Volume 1, in order:
========================================="""

    header_vol2 = """You are an expert Technical Editor reviewing **Volume 2** of the "Machine Learning Systems" textbook.
Volume 2 focuses on "Scale, Distribute, Govern".

Your Goal: Ensure chapters are standalone where appropriate but consistent with the overall theme, and check for redundancy.

Please analyze the following text:

1. **Independence & Dependencies:**
   - Can these chapters be read somewhat independently, or is there a strict dependency chain?
   - Are there missing prerequisites that should have been covered in Volume 1 or earlier in Volume 2?

2. **Repetition Check:**
   - CRITICAL: Identify major overlaps with Volume 1 (if you have context of it, otherwise focus on internal overlap).
   - Identify overlaps between Volume 2 chapters (e.g., "Distributed Training" vs "Infrastructure" covering the same hardware details).

3. **Thematic Consistency:**
   - Do these chapters collectively tell the story of "Scaling" and "Governing" ML systems?

Here is the text of Volume 2 chapters:
========================================="""

    header = header_vol1 if volume == 1 else header_vol2
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(header)
        outfile.write("\n\n")
        
        for chapter_rel_path in chapters:
            # chapter_rel_path is like 'contents/vol1/intro.qmd' relative to 'quarto/' ?? 
            # Check config: 'href: contents/vol1/introduction/introduction.qmd'
            # The config paths seem to be relative to the project root or quarto root?
            # The file structure is book/quarto/contents/...
            # Config is in book/quarto/config/_quarto-html.yml
            # If I run script from book/, the file path should be quarto/ + path if path starts with contents
            
            # Let's verify path construction.
            # Config says: contents/vol1/introduction/introduction.qmd
            # Real path: quarto/contents/vol1/introduction/introduction.qmd
            
            full_path = os.path.join("quarto", chapter_rel_path)
            
            if not os.path.exists(full_path):
                # Try without 'quarto' prefix if it fails, or maybe it's relative to config?
                # Actually, usually quarto config paths are relative to the project root defined in quarto.
                # Here the quarto root is likely 'quarto/'
                
                # Let's try just the path if the first one failed, just in case.
                if os.path.exists(chapter_rel_path):
                     full_path = chapter_rel_path
                else:
                    print(f"Warning: File not found: {full_path}")
                    outfile.write(f"\n\n[MISSING FILE: {chapter_rel_path}]\n\n")
                    continue

            print(f"Processing: {full_path}")
            content = read_file_content(full_path)
            
            outfile.write(f"\n\n--- START OF CHAPTER: {chapter_rel_path} ---\\n\n")
            outfile.write(content)
            outfile.write(f"\n\n--- END OF CHAPTER: {chapter_rel_path} ---\\n\n")

    print(f"\nSuccess! Prompt file generated at: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM prompts for book review.")
    parser.add_argument("volume", type=int, choices=[1, 2], help="Volume number to process (1 or 2)")
    
    args = parser.parse_args()
    
    output_filename = f"prompt_vol{args.volume}.txt"
    generate_prompt(args.volume, output_filename)
