import os
import re
import json
import time
import requests

API_KEY = os.environ.get("GEMINI_API_KEY")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

def generate_questions(track, filename, count, current_content):
    prompt = f"""
You are an expert ML Systems Engineering interviewer. You are writing questions for the "{track}" track of an ML Systems Interview Playbook.
The current file is {filename}.

I need you to generate exactly {count} NEW, highly technical, realistic interview questions that fit into this specific file's domain.
DO NOT duplicate any existing questions.

The questions MUST follow this EXACT Markdown format, including the nested <details> tags and spacing:

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The [Catchy Title]</b> · <code>[topic-tag]</code></summary>

- **Interviewer:** "[The scenario or crisis. Make it realistic to a top-tier tech company.]"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "[What most candidates say wrong — creates the 'aha' moment.]"

  **Realistic Solution:** "[The physics/logic/engineering behind the correct answer. Be highly specific and technical.]"

  > **Napkin Math:** [Quick back-of-envelope calculation with real numbers. MUST use realistic hardware specs.]

  > **Key Equation:** $[The formula to memorize]$

  📖 **Deep Dive:** [Volume I: Chapter Name](https://mlsysbook.ai/vol1/...)

  </details>

</details>

Levels to use (distribute them evenly):
- Level-L3_Junior-brightgreen (alt="Level 1")
- Level-L4_Mid-blue (alt="Level 2")
- Level-L5_Senior-yellow (alt="Level 3")
- Level-L6+_Principal-red (alt="Level 4")

Here is the current content of the file so you know what already exists and what topics to target:
---
{current_content[:2000]}... (truncated)
---

Generate ONLY the Markdown for the {count} new questions. Do not include any introductory text, just the concatenated <details> blocks.
"""
    
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
        }
    }
    
    for attempt in range(3):
        try:
            response = requests.post(URL, json=data)
            response.raise_for_status()
            text = response.json()['candidates'][0]['content']['parts'][0]['text']
            # Clean up markdown code blocks if present
            text = re.sub(r'^```markdown\n', '', text)
            text = re.sub(r'^```\n', '', text)
            text = re.sub(r'```$', '', text)
            return text.strip()
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(e.response.text)
            time.sleep(5)
    return ""

targets = {
    "cloud": {
        "01_compute_and_memory.md": 8,
        "02_network_and_distributed.md": 8,
        "03_inference_and_serving.md": 8,
        "04_data_and_mlops.md": 9
    },
    "edge": {
        "01_systems_and_real_time.md": 15,
        "02_compute_and_memory.md": 15,
        "03_data_and_deployment.md": 15,
        "05_heterogeneous_and_advanced.md": 16
    },
    "mobile": {
        "01_systems_and_soc.md": 19,
        "02_compute_and_memory.md": 19,
        "03_data_and_deployment.md": 19,
        "05_advanced_systems.md": 19
    },
    "tinyml": {
        "01_micro_architectures.md": 20,
        "02_compute_and_memory.md": 20,
        "03_data_and_deployment.md": 21,
        "05_advanced_systems.md": 21
    }
}

for track, files in targets.items():
    for filename, count in files.items():
        filepath = f"interviews/{track}/{filename}"
        print(f"Generating {count} questions for {filepath}...")
        
        with open(filepath, 'r') as f:
            content = f.read()
            
        new_content = ""
        remaining = count
        while remaining > 0:
            batch_size = min(10, remaining)
            print(f"  Requesting batch of {batch_size}...")
            batch_text = generate_questions(track, filename, batch_size, content)
            if batch_text:
                new_content += "\n\n" + batch_text
            remaining -= batch_size
            time.sleep(2) # rate limiting
            
        if new_content:
            with open(filepath, 'a') as f:
                f.write(new_content + "\n")
            print(f"Successfully appended to {filepath}")

