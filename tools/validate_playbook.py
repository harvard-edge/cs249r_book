import re
import sys
from pathlib import Path

# Strict Schema Components
PATTERNS = {
    "Level/Badge": r'<summary><b><img src=".*?" alt="Level.*?" align="center">',
    "Realistic Solution": r'\*\*Realistic Solution:\*\*',
    "Deep Dive": r'📖\s*\*\*Deep Dive:\*\*\s*\[.*?\]\(.*?\)'
}

FLASHCARD_PATTERNS = {
    "Interviewer Prompt": r'-\s*\*\*Interviewer:\*\*',
    "Reveal Answer": r'<details>\s*<summary><b>🔍 Reveal Answer</b></summary>'
}

OPTIONAL_PATTERNS = {
    "Common Mistake": r'\*\*Common Mistake:\*\*',
    "Napkin Math": r'>\s*\*\*Napkin Math:\*\*'
}

VISUAL_PATTERNS = {
    "Mermaid/Image": r'(```mermaid|!\[.*?\]\(.*?\))'
}

def extract_blocks(content):
    starts = [m.start() for m in re.finditer(r'<details>\s*<summary><b><img src="https://img\.shields\.io/badge/Level-', content)]
    blocks = []
    for i in range(len(starts)):
        start = starts[i]
        end = starts[i+1] if i+1 < len(starts) else len(content)
        block_raw = content[start:end]
        last_details = block_raw.rfind('</details>')
        if last_details != -1:
            blocks.append(block_raw[:last_details + 10])
        else:
            blocks.append(block_raw)
    return blocks

def validate_file(file_path):
    content = file_path.read_text(encoding='utf-8')
    question_blocks = extract_blocks(content)
    
    errors = []
    question_index = 0
    
    for block in question_blocks:
        question_index += 1
        title = get_title(block)
        is_visual = "visual_debugging" in str(file_path) or "Reveal the Bottleneck" in block
        
        # Required Patterns
        for name, pattern in PATTERNS.items():
            if not re.search(pattern, block, re.DOTALL):
                errors.append(f"Q{question_index} ('{title}'): Missing {name}")
        
        if is_visual:
            for name, pattern in VISUAL_PATTERNS.items():
                if not re.search(pattern, block, re.DOTALL):
                    errors.append(f"Q{question_index} ('{title}'): Missing {name}")
        else:
            for name, pattern in FLASHCARD_PATTERNS.items():
                if not re.search(pattern, block, re.DOTALL):
                    errors.append(f"Q{question_index} ('{title}'): Missing {name}")
                
    return errors, len(question_blocks)

def get_title(block):
    match = re.search(r'align="center"> (.*?)</b>', block)
    if not match:
        match = re.search(r'</b> (.*?) ·', block)
    return match.group(1).strip() if match else "Unknown Title"

def main():
    base_path = Path("/Users/VJ/GitHub/MLSysBook/interviews")
    tracks = ["cloud", "edge", "mobile", "tinyml"]
    
    all_errors = {}
    total_questions = 0
    
    print("🛡️  Validating MLSys Interview Playbook Schema...")
    
    files_to_check = list((base_path).glob("*.md"))
    for track in tracks:
        track_dir = base_path / track
        if track_dir.exists():
            files_to_check.extend(list(track_dir.glob("*.md")))

    for md_file in files_to_check:
        if md_file.name == "README.md": continue
        
        errs, q_count = validate_file(md_file)
        total_questions += q_count
        if errs:
            relative_path = md_file.relative_to(base_path)
            all_errors[str(relative_path)] = errs
                
    if all_errors:
        print(f"\n❌ Validation Failed! Found issues in {len(all_errors)} files.")
        for file, errs in all_errors.items():
            print(f"\n📄 {file}:")
            for e in errs[:10]: 
                print(f"  - {e}")
            if len(errs) > 10:
                print(f"  - ... and {len(errs)-10} more issues.")
    else:
        print(f"\n✅ Validation Passed! All {total_questions} questions follow the strict schema.")

if __name__ == "__main__":
    main()
