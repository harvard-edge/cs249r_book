
import os
import re

def scan_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.readlines()

    issues = []
    
    # Regex patterns
    bold_pattern = re.compile(r" \*\*(.*?)\*\*")
    rhetorical_pattern = re.compile(r"([.?!]|^)\s+(Why|How|What|When|Where)\s+.*?\?")
    bridge_pattern = re.compile(r"(we call|termed|defined as|known as)\s+(the\s+)?\*\*(.*?)\*\*")

    for i, line in enumerate(content):
        # Check for rhetorical questions (heuristic: starts with Why/How/etc and ends with ?)
        # This is a bit loose, but helps identifying candidates.
        match_rhet = rhetorical_pattern.search(line)
        if match_rhet:
            # Check if the word is italicized
            word = match_rhet.group(2)
            if not f"_{word}" in line and not f"*{word}" in line:
                 # Check if it is a header
                if not line.strip().startswith("#"):
                    issues.append(f"Line {i+1}: Possible non-italicized rhetorical question: '{match_rhet.group(0).strip()}'")

        # Check for bridge sentence violations (bolding concept instead of italics)
        match_bridge = bridge_pattern.search(line)
        if match_bridge:
            issues.append(f"Line {i+1}: Bridge sentence bolding violation: '{match_bridge.group(0)}'")

    return issues

def main():
    vol1_path = "quarto/contents/vol1"
    for root, dirs, files in os.walk(vol1_path):
        for file in files:
            if file.endswith(".qmd"):
                filepath = os.path.join(root, file)
                issues = scan_file(filepath)
                if issues:
                    print(f"\nFile: {filepath}")
                    for issue in issues:
                        print(f"  - {issue}")

if __name__ == "__main__":
    main()
