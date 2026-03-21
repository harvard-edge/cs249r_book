import os
import json
import random
import glob
import re
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("MLSysBook Assistant")

# Define the root of the repository
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INTERVIEWS_DIR = os.path.join(REPO_ROOT, "interviews")

def parse_markdown_questions(file_path: str) -> list[dict]:
    """Parse the specific markdown structure of the interview flashcards."""
    questions = []
    if not os.path.exists(file_path):
        return questions
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find all top-level question blocks
    # Looking for <details>\n<summary><b>...
    import re
    # Match the whole block from the first <details> to the final </details>
    # We use a heuristic: a question starts with <summary><b>[LEVEL BADGE]
    pattern = re.compile(r'<summary><b><img src=.*?alt="([^"]+)".*?>([^<]+)</b>.*?</summary>\s*-\s*\*\*Interviewer:\*\*\s*"(.*?)"\s*<details>\s*<summary><b>🔍 Reveal Answer</b></summary>\s*(.*?)\s*</details>\s*</details>', re.DOTALL)
    
    matches = pattern.finditer(content)
    
    for match in matches:
        level = match.group(1).strip()
        title = match.group(2).replace('·', '').strip()
        prompt = match.group(3).strip()
        answer = match.group(4).strip()
        
        questions.append({
            "level": level,
            "title": title,
            "prompt": prompt,
            "answer": answer
        })
        
    return questions

@mcp.tool()
def draw_interview_question(track: str) -> str:
    """
    Draw a random ML Systems interview question from the specified track.
    
    Args:
        track: The deployment track (e.g., 'cloud', 'edge', 'mobile', 'tinyml').
    """
    track_dir = os.path.join(INTERVIEWS_DIR, track.lower())
    if not os.path.exists(track_dir):
        return f"Error: Track '{track}' not found."
        
    # Find all markdown files in the track directory
    md_files = glob.glob(os.path.join(track_dir, "*.md"))
    # Filter out README
    md_files = [f for f in md_files if not f.endswith("README.md")]
    
    if not md_files:
        return f"Error: No question files found for track '{track}'."
        
    # Pick a random file and parse questions
    target_file = random.choice(md_files)
    questions = parse_markdown_questions(target_file)
    
    if not questions:
         return f"Could not parse questions from {os.path.basename(target_file)}."
         
    selected = random.choice(questions)
    
    response = (
        f"**Scenario ({os.path.basename(target_file)}):** {selected['title']}\n\n"
        f"**Interviewer Prompt:**\n{selected['prompt']}\n\n"
        f"**System Note for AI (DO NOT SHOW TO USER IMMEDIATELY):**\n"
        f"Here is the 'Reveal Answer' context. Use this to grade the user using the Socratic method.\n"
        f"{selected['answer']}"
    )
    return response

@mcp.resource("book://NUMBERS.md")
def get_numbers_cheat_sheet() -> str:
    """Get the 'Numbers Every ML Systems Engineer Should Know' cheat sheet for physics constraints."""
    numbers_file = os.path.join(INTERVIEWS_DIR, "NUMBERS.md")
    try:
        with open(numbers_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: NUMBERS.md not found."

if __name__ == "__main__":
    # Start the server using stdin/stdout
    mcp.run()