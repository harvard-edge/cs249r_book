import json
import re
from pathlib import Path

def parse_md_to_questions(file_path, track, scope):
    content = Path(file_path).read_text(encoding='utf-8')
    # Use a robust split that handles potential variations in the header
    blocks = content.split('<details>\n<summary><b><img')
    
    questions = []
    for i, block in enumerate(blocks):
        if i == 0: continue # Header material before first question
        
        try:
            # Re-add the split delimiter for consistent regex/parsing
            full_block = '<summary><b><img' + block
            
            # Extract Level
            level = "Unknown"
            if 'Level-L' in full_block:
                level = full_block.split('Level-')[1].split('_')[0]
            elif 'Level ' in full_block:
                # Handle Level 1, Level 2 format if it exists
                level_part = full_block.split('Level ')[1].split('"')[0]
                level = f"L{level_part}"
            
            # Extract Title
            title = "Unknown"
            if '</b>' in full_block:
                title_part = full_block.split('</b>')[0].split('>')[-1].strip()
                title = title_part
            
            # Extract Topic
            topic = "Unknown"
            if '<code>' in full_block:
                topic = full_block.split('<code>')[1].split('</code>')[0]
            
            # Extract Scenario
            scenario = ""
            if '**Interviewer:**' in full_block:
                scenario_part = full_block.split('**Interviewer:**')[1]
                # End at the next <details> or end of block
                if '<details>' in scenario_part:
                    scenario = scenario_part.split('<details>')[0].strip(' "-:').split('\n\n')[0]
                else:
                    scenario = scenario_part.strip(' "-:').split('\n\n')[0]
            
            # Extract Answer Details
            details = {}
            if '**Common Mistake:**' in full_block:
                details['common_mistake'] = full_block.split('**Common Mistake:**')[1].split('\n')[0].strip()
            
            if '**Realistic Solution:**' in full_block:
                # Get everything until the next double newline or specific marker
                sol_part = full_block.split('**Realistic Solution:**')[1].strip()
                # Split by either double newline or next block like > **Napkin Math:**
                if '\n\n' in sol_part:
                    details['realistic_solution'] = sol_part.split('\n\n')[0].strip()
                elif '> **' in sol_part:
                    details['realistic_solution'] = sol_part.split('> **')[0].strip()
                else:
                    details['realistic_solution'] = sol_part
                
            if '**Napkin Math:**' in full_block:
                details['napkin_math'] = full_block.split('**Napkin Math:**')[1].split('\n')[0].strip()

            if '📖 **Deep Dive:**' in full_block:
                link_part = full_block.split('📖 **Deep Dive:**')[1].split('\n')[0]
                if '[' in link_part and '](' in link_part:
                    details['deep_dive_title'] = link_part.split('[')[1].split(']')[0]
                    details['deep_dive_url'] = link_part.split('(')[1].split(')')[0]

            questions.append({
                "id": f"{track}-{topic}-{title.lower().replace(' ', '-')[:20]}",
                "track": track,
                "scope": scope,
                "level": level,
                "title": title,
                "topic": topic,
                "scenario": scenario,
                "details": details
            })
        except Exception as e:
            # print(f"Error parsing block in {file_path}: {e}")
            continue
            
    return questions

def run():
    base_path = Path("/Users/VJ/GitHub/MLSysBook/interviews")
    tracks = ["cloud", "edge", "mobile", "tinyml"]
    corpus = []
    
    print("🚀 Building MLSys Interview Corpus...")
    
    # Process Foundations first (Global track)
    foundations_file = base_path / "foundations.md"
    if foundations_file.exists():
        qs = parse_md_to_questions(foundations_file, "global", "Foundations")
        print(f"  - foundations.md: {len(qs)} questions")
        corpus.extend(qs)

    for track in tracks:
        track_dir = base_path / track
        if not track_dir.exists(): continue
        
        for md_file in track_dir.glob("*.md"):
            if md_file.name == "README.md": continue
            
            scope_name = md_file.stem.replace('_', ' ').title()
            scope_name = re.sub(r'^\d+\s+', '', scope_name)
            
            qs = parse_md_to_questions(md_file, track, scope_name)
            print(f"  - {track}/{md_file.name}: {len(qs)} questions")
            corpus.extend(qs)
            
    # Normalize Levels
    for q in corpus:
        if q['level'] == 'L6%2B' or q['level'] == 'L6':
            q['level'] = 'L6+'
            
    output_file = base_path / "corpus.json"
    output_file.write_text(json.dumps(corpus, indent=2), encoding='utf-8')
    print(f"\n✅ Done! Found {len(corpus)} questions. Saved to {output_file}")

if __name__ == "__main__":
    run()
