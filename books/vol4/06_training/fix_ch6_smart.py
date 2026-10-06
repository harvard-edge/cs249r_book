import re

file_path = "/private/tmp/MLSysBook-dev-notation/books/vol4/06_training/06_training.qmd"
with open(file_path, "r") as f:
    lines = f.readlines()

# 1. Collect section definitions
sec_defs = {}
for i, line in enumerate(lines):
    match = re.search(r'\{#(sec-training-[a-zA-Z0-9_-]+)\}', line)
    if match:
        sec_defs[match.group(1)] = i

# 2. Process line by line
out_lines = []
acronyms = {
    r'\bODD\b': 'operational design domain (ODD)',
    r'\bMCU\b': 'microcontroller unit (MCU)',
    r'\bODE\b': 'ordinary differential equation (ODE)',
    r'\bback-EMF\b': 'back electromotive force (back-EMF)',
    r'\bIMU\b': 'inertial measurement unit (IMU)',
    r'\bRMS\b': 'root mean square (RMS)',
    r'\bDoF\b': 'degrees of freedom (DoF)',
    r'\bRL\b': 'reinforcement learning (RL)',
    r'\bBC\b': 'behavioral cloning (BC)',
}
concepts = {
    r'\bPolicy Manifest\b': "policy manifest (a formal declaration of the policy's supported operating domain)",
    r'\bPrivileged State Distillation\b': "privileged state distillation (where a student model learns to replicate a teacher using only deployable sensors)",
    r'\bContact Yield Deadline\b': "contact yield deadline (the strict time limit before mechanical damage occurs)"
}

replaced = set()

# Regexes for inter-chapter references
inter_chap_re = re.compile(r'@sec-(evaluation|perception|memory|intent|planning|enforcement|placement|intervention|verification|release|frontier)[^\s,.\)]*')
inter_chap_text_re = re.compile(r'Chapters?\s+[789]|Chapters?\s+1[0-7]')
margin_ptr_re = re.compile(r'^.*?↩.*?@sec-(evaluation|perception|memory|intent|planning|enforcement|placement|intervention|verification|release|frontier).*?$')

for i, line in enumerate(lines):
    # Remove chapter wrap-up paragraph
    if line.startswith("How much statistical confidence can twenty successful consecutive trials provide for a policy checkpoint"):
        continue
    if "It analyzes an illustrative benchmark campaign beginning with twenty nominal door trials" in line:
        continue
    if "It then evaluates `latch-bc-recovery-02`" in line:
        continue

    # Fix acronyms
    for pattern, repl in acronyms.items():
        if pattern not in replaced and re.search(pattern, line):
            # Only replace if not already lowercase or defined
            line = re.sub(pattern, repl, line, count=1)
            replaced.add(pattern)
            
    # Fix Concepts
    for pattern, repl in concepts.items():
        # Case insensitive match for concepts
        if pattern not in replaced and re.search(pattern, line, re.IGNORECASE):
            line = re.sub(pattern, repl, line, count=1, flags=re.IGNORECASE)
            replaced.add(pattern)
            
    # Remove margin pointers
    if margin_ptr_re.match(line):
        continue
        
    # Replace inter-chapter references
    line = re.sub(r' \(@sec-(evaluation|perception|memory|intent|planning|enforcement|placement|intervention|verification|release|frontier)[^\)]*\)', '', line)
    line = inter_chap_re.sub('a later chapter', line)
    line = inter_chap_text_re.sub('subsequent chapters', line)
    
    # Fix intra-chapter forward references
    # Find all @sec-training-...
    for match in re.finditer(r'@(sec-training-[a-zA-Z0-9_-]+)', line):
        sec = match.group(1)
        if sec in sec_defs and sec_defs[sec] > i:
            # It's a forward reference!
            # Replace in line
            line = line.replace(f" (@{sec})", "")
            line = line.replace(f"(@{sec})", "(detailed later)")
            line = line.replace(f"@{sec}", "a later section")
            
    out_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(out_lines)
