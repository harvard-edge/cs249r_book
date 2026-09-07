import re

with open('/Users/VJ/GitHub/MLSysBook-vol4-deep-audit/publishing/quarto/contents/vol4/chapters/04-nervous/04-nervous.qmd', 'r') as f:
    lines = f.readlines()

targets = [
    "Thermal and Energy ($\mathcal{E}$):",
    "Temporal Determinism ($\sigma_t$):",
    "Fault Domain Isolation ($\mathcal{F}$):",
    "Practical Engineering Question",
    "Recovery Handshake",
    "The Deliberative Brain",
    "Proposal Is Never Permission:",
    "Cadence is Dictated by Physics, Not Software:",
    "Lock-Free Sequence Locks Guarantee Zero-Wait Memory Transport:",
    "Structural Timing Guarantees Outrank Empirical Distributions:",
    "Four Dimensions of Hardware Isolation:",
    "Stopping Dynamics Dimension Watchdog Horizons:",
    "The 5x5 Synthesis Matrix:"
]

for i, line in enumerate(lines):
    for t in targets:
        # escape regex special characters for search, except math ones we just do simple string find
        if t in line:
            print(f"Line {i+1}: {line.strip()}")
