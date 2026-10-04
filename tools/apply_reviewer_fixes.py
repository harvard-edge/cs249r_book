import re

def process_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()

    # We need to skip python cells.
    # We can split the file into chunks of python blocks vs markdown.
    chunks = re.split(r'(```{python}.*?```)', content, flags=re.DOTALL)
    
    for i in range(0, len(chunks), 2): # Even indices are markdown
        text = chunks[i]
        for old, new in replacements:
            text = re.sub(old, new, text)
        chunks[i] = text
        
    new_content = "".join(chunks)
    with open(filepath, 'w') as f:
        f.write(new_content)


# 01_introduction.qmd replacements
replacements_01 = [
    (r"\\text\{Accuracy\}_0", r"\\InitialAccuracyZero"),
    (r"\\mathcal\{D\}\(P_t \\lVert P_0\)", r"\\StatisticalDivergence"),
    (r"(?<!\\)P_t\b", r"\\CurrentDistribution"),
    (r"(?<!\\)P_0\b", r"\\TrainingDistributionZero"),
    (r"\\lambda\b", r"\\FittedSensitivity"),
    (r"\\text\{Accuracy\}\(t\)", r"\\AccuracyAtTime"),
    (r"\\tau\b", r"\\ResponseThreshold"),
    (r"p\(y \\mid x\)", r"\\ConditionalDistribution"),
    (r"\bP(?=[ \)\$])", r"\\Parameters"), # P when referring to parameters
    (r"\bD(?=[ \)\$])", r"\\DatasetSize"),
    (r"\bT(?=[ \)\$=])", r"\\Time"),
    (r"D_\{\\text\{vol\}\}", r"\\DataVolume"),
    (r"\\text\{BW\}", r"\\Bandwidth"),
    (r"\bO(?=[ \)\$/])", r"\\Operations"),
    (r"R_\{\\text\{peak\}\}", r"\\PeakRate"),
    (r"\\eta_\{\\text\{hw\}\}", r"\\EfficiencyHw"),
    (r"L_\{\\text\{lat\}\}", r"\\Latency"),
    (r"N_\{\\text\{GPU\}\}", r"\\AcceleratorCountGpu"),
    (r"E_\{\\text\{total\}\}", r"\\TotalEnergy"),
    (r"E_\{\\text\{move\}\}", r"\\EnergyPerByteMoved"),
    (r"E_\{\\text\{compute\}\}", r"\\EnergyPerOperationCompute")
]

# 05_nn_computation.qmd replacements
replacements_05 = [
    (r"\\theta\b", r"\\ModelParameters"),
    (r"\\eta\b", r"\\LearningRate"),
    (r"\\mathcal\{L\}\b", r"\\LossFunction"),
    (r"\\hat\{\\mathbf\{y\}\}", r"\\PredictedOutput"),
    (r"\\ell\b", r"\\LayerIndex"),
    (r"N_L\b", r"\\NumberOfLayers"),
    (r"(?<=[ \(\$])B(?=[ \)\$])", r"\\BatchSize"),
    (r"(?<=[ \(\$0-9])O(?=[ \)\$/])", r"\\Operations"),
    (r"(?<=[ \(\$])P(?=[ \)\$])", r"\\Parameters"),
    (r"(?<=[ \(\$])T(?=[ \)\$=])", r"\\Time"),
    (r"(?<=[ \(\$])t(?=[ \)\$,])", r"\\ResponseThreshold")
]

# 04_data_engineering.qmd replacements
replacements_04 = [
    (r"(?<!\\)P_0\b", r"\\TrainingDistributionZero"),
    (r"(?<!\\)P_t\b", r"\\CurrentDistribution"),
    (r"p\(x\)", r"\\Distribution"),
    (r"p\(y \\mid x\)", r"\\ConditionalDistribution"),
    (r"(?<=[ \(\$])O(?=[ \)\$/])", r"\\Operations"),
    (r"\\alpha\b", r"\\SignificanceLevel"),
    (r"\\kappa\b", r"\\CohensKappa"),
    (r"(?<=[ \(\$])D(?=[ \)\$])", r"\\DatasetSize")
]

process_file("books/vol1/01_introduction/01_introduction.qmd", replacements_01)
process_file("books/vol1/05_nn_computation/05_nn_computation.qmd", replacements_05)
process_file("books/vol1/04_data_engineering/04_data_engineering.qmd", replacements_04)

print("Applied subagent fixes.")
