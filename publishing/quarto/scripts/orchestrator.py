import asyncio
import argparse
import os
from pathlib import Path

from google.antigravity import Agent, LocalAgentConfig

CHAPTERS_DIR = Path("publishing/quarto/contents/vol4/chapters")
CHAPTERS = sorted(list(CHAPTERS_DIR.glob("**/*.qmd")))

async def audit_chapter(chapter_path: Path, pass_prompt: str):
    config = LocalAgentConfig(
        model="gemini-3.5-pro",
        system_instructions=(
            "You are an expert textbook editor and AI engineering researcher. "
            "Your task is to audit the provided chapter for the given criteria."
        )
    )
    
    abs_path = chapter_path.resolve()
    
    prompt = f"""
You are auditing {abs_path.name}. The file is located at {abs_path}.

{pass_prompt}

Use your tools to read the file, make the necessary edits directly to the file, and then finish your turn.
Ensure you actually use the file editing tools to make changes where needed.
Explain the changes you made in your final response.
"""
    
    async with Agent(config) as agent:
        try:
            response = await agent.chat(prompt)
            text = await response.text()
            print(f"=== Report for {abs_path.name} ===\n{text}\n")
        except Exception as e:
            print(f"=== Error in {abs_path.name} ===\n{e}\n")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass1", action="store_true", help="Run Pass 1 (Systems Principles)")
    parser.add_argument("--pass2", action="store_true", help="Run Pass 2 (Progressive Disclosure & Indexing)")
    parser.add_argument("--pass3", action="store_true", help="Run Pass 3 (Footnotes)")
    parser.add_argument("--pass4", action="store_true", help="Run Pass 4 (Cross-References)")
    args = parser.parse_args()

    pass_prompt = ""
    if args.pass1:
        pass_prompt = """
Pass 1 (Systems Principles):
- Ensure the text has textbook-style definitions of "physical AI systems" (or related physical AI terms) where appropriate.
- Focus on physical AI first principles.
- Ensure students are not inundated with math in the main text. The first principles should contain math, but heavy derivations or extra math should be moved to footnotes or backmatter to complement it. Ensure good progressive disclosure.
"""
    elif args.pass2:
        pass_prompt = """
Pass 2 (Progressive Disclosure & Indexing):
- Ensure concepts are built progressively.
- Audit for bold terms (index terms) ensuring they are sentence case (or lowercase) inside the main neuroprose, matching the style from Volume I (e.g., `**model compression**\\index{Model compression!definition}`).
- Do not capitalize bold terms unless they are proper nouns or abbreviations.
"""
    elif args.pass3:
        pass_prompt = """
Pass 3 (Footnotes):
- Audit footnotes for proper progressive disclosure vs. main text.
- Base footnote categorizations and style on a reference chapter (Chapter 1).
"""
    elif args.pass4:
        pass_prompt = """
Pass 4 (Cross-References):
- Validate and insert strict cross-references.
"""
    else:
        print("Please specify a pass to run (e.g., --pass1)")
        return

    print(f"Running orchestrator on {len(CHAPTERS)} chapters...")
    
    tasks = []
    for chapter_path in CHAPTERS:
        tasks.append(audit_chapter(chapter_path, pass_prompt))
        
    await asyncio.gather(*tasks)
    print("All agents finished.")

if __name__ == "__main__":
    asyncio.run(main())
