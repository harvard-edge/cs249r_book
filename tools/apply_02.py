with open("books/vol1/02_ml_systems/02_ml_systems.qmd", "r") as f:
    content = f.read()

replacements = [
    (r"D_{\text{vol}}", r"\DataVolume"),
    (r"\text{BW}", r"\Bandwidth"),
    (r"R_{\text{peak}}", r"\PeakRate"),
    (r"\eta_{\text{hw}}", r"\EfficiencyHw"),
    (r"L_{\text{lat}}", r"\Latency"),
    (r"T_{\text{deadline}}", r"\Deadline"),
    (r"M_{\text{weights}}", r"\WeightMemoryWeights"),
    (r"M_{\text{activations}}", r"\ActivationMemoryActivations"),
    (r"P_t", r"\CurrentDistribution"),
    (r"P_0", r"\TrainingDistributionZero"),
    (r"\mathcal{D}(P_t \lVert P_0)", r"\StatisticalDivergence"),
    (r"\text{Accuracy}_0", r"\InitialAccuracyZero"),
]

for old, new in replacements:
    content = content.replace(old, new)

# Care with single letters
import re
content = re.sub(r'\bO\b(?!/)', r'\\Operations', content) # O but not O/R
content = content.replace(r"\\Operations/R", r"\\Operations / R")
# We'll just do manual for single letters:
content = re.sub(r'([ \(\$])O([ \)\$/])', r'\1\\Operations\2', content)
content = re.sub(r'([ \(\$])I([ \)\$/])', r'\1\\ArithmeticIntensity\2', content)
content = re.sub(r'([ \(\$])B([ \)\$/])', r'\1\\BatchSize\2', content)

with open("books/vol1/02_ml_systems/02_ml_systems.qmd", "w") as f:
    f.write(content)
