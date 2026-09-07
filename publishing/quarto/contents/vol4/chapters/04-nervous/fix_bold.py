import re

file_path = '/Users/VJ/GitHub/MLSysBook-vol4-deep-audit/publishing/quarto/contents/vol4/chapters/04-nervous/04-nervous.qmd'
with open(file_path, 'r') as f:
    text = f.read()

replacements = {
    r'\*\*Thermal and Energy \(\$\\mathcal\{E\}\$\):\*\*': r'**Thermal and energy ($\mathcal{E}$):**',
    r'\*\*Temporal Determinism \(\$\\sigma_t\$\):\*\*': r'**Temporal determinism ($\sigma_t$):**',
    r'\*\*Fault Domain Isolation \(\$\\mathcal\{F\}\$\):\*\*': r'**Fault domain isolation ($\mathcal{F}$):**',
    r'\*\*Practical Engineering Question\*\*': r'**Practical engineering question**',
    r'\*\*Recovery Handshake\*\*': r'**recovery handshake**',
    r'\*\*The Deliberative Brain\*\*': r'**the deliberative brain**',
    r'\*\*Proposal Is Never Permission:\*\*': r'**Proposal is never permission:**',
    r'\*\*Cadence is Dictated by Physics, Not Software:\*\*': r'**Cadence is dictated by physics, not software:**',
    r'\*\*Lock-Free Sequence Locks Guarantee Zero-Wait Memory Transport:\*\*': r'**Lock-free sequence locks guarantee zero-wait memory transport:**',
    r'\*\*Structural Timing Guarantees Outrank Empirical Distributions:\*\*': r'**Structural timing guarantees outrank empirical distributions:**',
    r'\*\*Four Dimensions of Hardware Isolation:\*\*': r'**Four dimensions of hardware isolation:**',
    r'\*\*Stopping Dynamics Dimension Watchdog Horizons:\*\*': r'**Stopping dynamics dimension watchdog horizons:**',
    r'\*\*The 5x5 Synthesis Matrix:\*\*': r'**The 5x5 synthesis matrix:**'
}

for old, new in replacements.items():
    text = re.sub(old, new, text)

with open(file_path, 'w') as f:
    f.write(text)

