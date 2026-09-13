import os
import re
import sys

def extract_section(body, header):
    pattern = rf"### {header}\n+(.*?)(?=\n### |$)"
    match = re.search(pattern, body, re.DOTALL)
    if not match:
        return None
    res = match.group(1).strip()
    if res == "_No response_":
        return None
    return res

def get_badge(difficulty):
    if not difficulty:
        return ''
    if "L3" in difficulty:
        return '<img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center">'
    elif "L4" in difficulty:
        return '<img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center">'
    elif "L5" in difficulty:
        return '<img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center">'
    elif "L6" in difficulty:
        return '<img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 6" align="center">'
    return ''

def main():
    body = os.environ.get('ISSUE_BODY', '')

    if not body:
        print("Error: No issue body found in environment variables.")
        sys.exit(1)

    track_raw = extract_section(body, "Track")
    difficulty = extract_section(body, "Difficulty Level")
    topic = extract_section(body, "Topic Tag")
    title = extract_section(body, "Question Title")
    prompt = extract_section(body, "The Interviewer Prompt")
    mistake = extract_section(body, "Common Mistake")
    solution = extract_section(body, "Realistic Solution")
    math = extract_section(body, "Napkin Math")
    equation = extract_section(body, "Key Equation")
    link = extract_section(body, "Deep Dive Link")

    if not all([track_raw, difficulty, topic, title, prompt, mistake, solution]):
        print("Error: Missing required fields in the issue body.")
        sys.exit(1)

    track = track_raw.lower().replace("/", "_")
    if track == "data_ops":
        track_folder = "cloud" # Temporary home until a dedicated data/ops folder is made
    else:
        track_folder = track

    badge = get_badge(difficulty)

    flashcard = f"""
<details>
<summary><b>{badge} {title}</b> · <code>{topic}</code></summary>

- **Interviewer:** "{prompt}"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "{mistake}"

  **Realistic Solution:** {solution}
"""
    if math:
        flashcard += f"\n  > **Napkin Math:** {math}\n"
    if equation:
        flashcard += f"\n  > **Key Equation:** {equation}\n"
    if link:
        flashcard += f"\n  📖 **Deep Dive:** [Textbook Reference]({link})\n"

    flashcard += """
  </details>

</details>
"""

    # We append this to a COMMUNITY.md file in the appropriate track directory
    file_path = f"staffml/{track_folder}/COMMUNITY.md"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Community Contributions ({track_raw.capitalize()})\n\n")
            f.write("Questions submitted by the community. Maintainers will periodically migrate these into the main rounds.\n\n---\n")

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(flashcard)
        f.write("\n---\n")

    print(f"Successfully appended flashcard to {file_path}")

if __name__ == "__main__":
    main()
