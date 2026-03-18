import os
import re

def remove_question_block(filepath, title_substring):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the start of the block
    # Pattern: <details>\s*<summary>.*title_substring.*?</summary>.*?</details>
    # We need to handle nested <details> tags.
    # A simpler way: find the line with the title. Then search backwards for the nearest <details>.
    # Then search forwards for the matching </details>.
    
    lines = content.split('\n')
    start_idx = -1
    for i, line in enumerate(lines):
        if title_substring in line and '<summary>' in line:
            # Found the summary line
            # Go backwards to find <details>
            for j in range(i, -1, -1):
                if '<details>' in lines[j]:
                    start_idx = j
                    break
            break
            
    if start_idx == -1:
        print(f"Could not find '{title_substring}' in {filepath}")
        return False
        
    # Now find the matching </details>
    end_idx = -1
    depth = 0
    for i in range(start_idx, len(lines)):
        if '<details>' in lines[i]:
            depth += 1
        if '</details>' in lines[i]:
            depth -= 1
            if depth == 0:
                end_idx = i
                break
                
    if end_idx == -1:
        print(f"Could not find matching </details> for '{title_substring}' in {filepath}")
        return False
        
    # Remove the block and any trailing blank lines
    while end_idx + 1 < len(lines) and lines[end_idx + 1].strip() == '':
        end_idx += 1
        
    del lines[start_idx:end_idx + 1]
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
        
    print(f"Removed '{title_substring}' from {filepath}")
    return True

# Cloud cuts
cloud_04 = "interviews/cloud/04_data_and_mlops.md"
remove_question_block(cloud_04, "The Model Registry Versioning Mess")
remove_question_block(cloud_04, "The A/B Test Power Failure")
remove_question_block(cloud_04, "The Model Card Gap")
remove_question_block(cloud_04, "The Labeling Quality Cliff")
remove_question_block(cloud_04, "The Build vs Buy Trap")

cloud_06 = "interviews/cloud/06_advanced_systems.md"
remove_question_block(cloud_06, "The Prompt Injection Paradox")
remove_question_block(cloud_06, "The Fairness Gerrymandering Problem")

# Edge cuts
edge_02 = "interviews/edge/02_compute_and_memory.md"
remove_question_block(edge_02, "The Phantom USB Disconnect")

edge_03 = "interviews/edge/03_data_and_deployment.md"
remove_question_block(edge_03, "The Multi-Region Edge Deployment")
remove_question_block(edge_03, "The Edge Security Hardening Checklist")
remove_question_block(edge_03, "The eMMC Log Bomb")
remove_question_block(edge_03, "The USB Camera Dropout")
remove_question_block(edge_03, "The Failed OTA Over Cellular")
remove_question_block(edge_03, "The Corrupted Model Boot Loop")

# TinyML cuts
tinyml_03 = "interviews/tinyml/03_data_and_deployment.md"
remove_question_block(tinyml_03, "Device Provisioning for ML Fleet")

