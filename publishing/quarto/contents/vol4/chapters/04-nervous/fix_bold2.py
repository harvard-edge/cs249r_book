file_path = '/Users/VJ/GitHub/MLSysBook-vol4-deep-audit/publishing/quarto/contents/vol4/chapters/04-nervous/04-nervous.qmd'
with open(file_path, 'r') as f:
    text = f.read()

replacements = {
    "**Thermal and Energy ($\mathcal{E}$):**": "**Thermal and energy ($\mathcal{E}$):**",
    "**Temporal Determinism ($\sigma_t$):**": "**Temporal determinism ($\sigma_t$):**",
    "**Fault Domain Isolation ($\mathcal{F}$):**": "**Fault domain isolation ($\mathcal{F}$):**",
    "**Practical Engineering Question**:": "**Practical engineering question**:",
    "**Recovery Handshake**": "**recovery handshake**",
    "**The Deliberative Brain**": "**the deliberative brain**",
    "**Proposal Is Never Permission:**": "**Proposal is never permission:**",
    "**Cadence is Dictated by Physics, Not Software:**": "**Cadence is dictated by physics, not software:**",
    "**Lock-Free Sequence Locks Guarantee Zero-Wait Memory Transport:**": "**Lock-free sequence locks guarantee zero-wait memory transport:**",
    "**Structural Timing Guarantees Outrank Empirical Distributions:**": "**Structural timing guarantees outrank empirical distributions:**",
    "**Four Dimensions of Hardware Isolation:**": "**Four dimensions of hardware isolation:**",
    "**Stopping Dynamics Dimension Watchdog Horizons:**": "**Stopping dynamics dimension watchdog horizons:**",
    "**The 5x5 Synthesis Matrix:**": "**The 5x5 synthesis matrix:**"
}

for old, new in replacements.items():
    text = text.replace(old, new)

with open(file_path, 'w') as f:
    f.write(text)
