import sys
import os

# Configuration: Ordered list of chapters
CHAPTERS = [
    "book/quarto/contents/vol1/introduction/introduction.qmd",
    "book/quarto/contents/vol1/ml_systems/ml_systems.qmd",
    "book/quarto/contents/vol1/workflow/workflow.qmd",
    "book/quarto/contents/vol1/data_engineering/data_engineering.qmd",
    "book/quarto/contents/vol1/dl_primer/dl_primer.qmd",
    "book/quarto/contents/vol1/dnn_architectures/dnn_architectures.qmd",
    "book/quarto/contents/vol1/frameworks/frameworks.qmd",
    "book/quarto/contents/vol1/training/training.qmd",
    "book/quarto/contents/vol1/optimizations/model_compression.qmd",
    "book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd",
    "book/quarto/contents/vol1/data_efficiency/data_efficiency.qmd",
    "book/quarto/contents/vol1/benchmarking/benchmarking.qmd",
    "book/quarto/contents/vol1/serving/serving.qmd",
    "book/quarto/contents/vol1/ops/ops.qmd",
    "book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd",
    "book/quarto/contents/vol1/conclusion/conclusion.qmd"
]

STRUCTURAL_PREFIXES = ("sec-", "fig-", "tbl-", "eq-", "ch-", "part-")

def extract_labels(line):
    # Find all occurrences of {#label}
    labels = []
    # Simple state machine or finding start/end indices
    # Assumption: labels don't span lines
    start_token = "{#"
    end_token = "}"
    
    current_pos = 0
    while True:
        try:
            start_idx = line.index(start_token, current_pos)
            end_idx = line.index(end_token, start_idx)
            label = line[start_idx+2:end_idx]
            labels.append(label)
            current_pos = end_idx + 1
        except ValueError:
            break
    return labels

def scan_files():
    defined_labels = set()
    references = [] 

    print("Scanning for labels...")
    for filepath in CHAPTERS:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                labels = extract_labels(line)
                for label in labels:
                    if label in defined_labels:
                        print(f"Warning: Duplicate label '{label}' found in {filepath}:{i}")
                    defined_labels.add(label)

    print(f"Found {len(defined_labels)} unique labels.")

    print("\nScanning for references...")
    for filepath in CHAPTERS:
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                # Simple tokenization to find @ref
                clean_line = line.replace(']', ' ').replace('[', ' ').replace('(', ' ').replace(')', ' ').replace('.', ' ').replace(',', ' ').replace(';', ' ')
                words = clean_line.split()
                for word in words:
                    if word.startswith('@'):
                        ref = word[1:]
                        if ref.startswith(STRUCTURAL_PREFIXES):
                            references.append((filepath, i, ref))

    print("\nValidating references...")
    broken_links = []
    
    for source, line, ref in references:
        if ref not in defined_labels:
            broken_links.append(f"{source}:{line} -> Reference @{ref} not found.")

    if broken_links:
        print(f"\n❌ Found {len(broken_links)} broken references:")
        for error in broken_links:
            print(error)
        sys.exit(1)
    else:
        print(f"\n✅ All {len(references)} structural cross-references are valid.")
        sys.exit(0)

if __name__ == "__main__":
    scan_files()