import subprocess
import os

log_file = "periodic-table/refinement-log.md"

with open(log_file, "w") as f:
    f.write("# 10-Round Paper Refinement Log\n\n")

for i in range(1, 11):
    print(f"Running Round {i}...")
    
    with open("periodic-table/paper/paper.tex", "r") as f:
        paper_content = f.read()
        
    with open("periodic-table/index.html", "r") as f:
        html_content = f.read()
        
    feedback_prompt = f"""Act as an expert committee (Patterson, Dean, Lattner, etc.) reviewing a systems research paper 'The Periodic Table of ML Systems'. 
Critically evaluate: 
1. Who's going to read it? 
2. Why are they going to read it? 
3. What is the value add? 
4. Does it have robust pedagogical examples at the end for teaching ML systems canonically? 

Provide brutal, actionable feedback on how to rewrite specific sections to improve depth, flow, and educational impact. 

Here is the current paper:
{paper_content}

Here is the website structure (for context):
{html_content[:5000]}
"""
    
    print("Getting Gemini Feedback...")
    result = subprocess.run(
        ["gemini", "-m", "gemini-3.1-pro-preview", "-y", "-p", feedback_prompt],
        capture_output=True, text=True
    )
    feedback = result.stdout
    
    with open(log_file, "a") as f:
        f.write(f"---\n## Round {i}\n### Gemini Feedback\n{feedback}\n")
        
    claude_prompt = f"""You are an expert academic writer refining 'periodic-table/paper/paper.tex'.
We want to ensure it follows the polished style of 'interviews/paper/paper.tex' and 'mlsysim/paper/paper.tex' (which you can read for reference).
Here is critical feedback from our expert panel:

{feedback}

Please deeply rewrite and refine 'periodic-table/paper/paper.tex' to address all of this feedback. 
- Ensure the paper clarifies: Who is reading it? Why are they reading it? What is the value add?
- Add strong pedagogical examples at the end for teaching.
- Update the LaTeX file directly. Maintain the Makefile buildability. DO NOT break LaTeX syntax.
"""
    
    print("Applying Feedback with Claude...")
    claude_result = subprocess.run(
        ["claude", "-p", claude_prompt, "--dangerously-skip-permissions"],
        capture_output=True, text=True
    )
    claude_output = claude_result.stdout
    
    with open(log_file, "a") as f:
        f.write(f"### Claude Execution\n{claude_output}\n")
        
    print("Building paper...")
    build_result = subprocess.run(["make"], cwd="periodic-table/paper", capture_output=True, text=True)
    if build_result.returncode != 0:
        print("Build failed! Attempting to fix...")
        with open(log_file, "a") as f:
            f.write(f"### Build Failed\n```\n{build_result.stderr}\n```\n")
            
        # Try to fix it once with Claude
        fix_prompt = f"The LaTeX build failed with the following error. Please fix periodic-table/paper/paper.tex immediately so it compiles.\n\n{build_result.stdout}\n{build_result.stderr}"
        subprocess.run(["claude", "-p", fix_prompt, "--dangerously-skip-permissions"], capture_output=True, text=True)
        subprocess.run(["make"], cwd="periodic-table/paper", capture_output=True, text=True)
