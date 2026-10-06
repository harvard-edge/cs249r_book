import re
import sys

with open("/private/tmp/MLSysBook-dev-notation/books/vol4/06_training/06_training.qmd", "r") as f:
    lines = f.readlines()

def print_matches(pattern_str, name):
    pattern = re.compile(pattern_str, re.IGNORECASE)
    print(f"\n--- {name} ---")
    for i, line in enumerate(lines):
        if pattern.search(line):
            print(f"{i+1}: {line.strip()}")

print_matches(r'\b(ODD|MCU|ODE|back-EMF|IMU|RMS|DoF|RL|BC)\b', "Acronyms")
print_matches(r'(Policy Manifest|Privileged State Distillation|Contact Yield Deadline)', "Concepts")
print_matches(r'(@sec-|\\ref\{|↪)', "Forward Connectors")

