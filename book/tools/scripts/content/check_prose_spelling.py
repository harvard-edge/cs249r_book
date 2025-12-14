#!/usr/bin/env python3
r"""
Spell check prose content in QMD files using aspell.

Intelligently parses QMD file structure to only check actual prose text,
excluding YAML frontmatter, code blocks, TikZ diagrams, inline code, URLs, etc.

Usage:
    python3 tools/scripts/content/check_prose_spelling.py [directory]

Requirements:
    - aspell must be installed (brew install aspell)
    - No Python dependencies beyond standard library

Checks:
    - Paragraph text
    - Headings
    - List items
    - Callout content

Ignores:
    - YAML frontmatter
    - Code blocks (```...```)
    - Inline code (`...`)
    - TikZ diagrams
    - URLs and links
    - LaTeX math ($...$, $$...$$)
    - Special Quarto syntax
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Set


def extract_yaml_frontmatter(content: str) -> Tuple[int, int]:
    """
    Find the start and end positions of YAML frontmatter.

    Returns:
        Tuple of (start_pos, end_pos) or (0, 0) if no frontmatter
    """
    if not content.startswith('---'):
        return (0, 0)

    # Find the closing ---
    lines = content.split('\n')
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == '---':
            # Return character positions
            start = 0
            end = sum(len(lines[j]) + 1 for j in range(i + 1))
            return (start, end)

    return (0, 0)


def extract_code_blocks(content: str) -> List[Tuple[int, int]]:
    """
    Find all code blocks (```...``` and TikZ blocks).

    Returns:
        List of (start_pos, end_pos) tuples
    """
    blocks = []

    # Find ``` code blocks
    pattern = r'```.*?```'
    for match in re.finditer(pattern, content, re.DOTALL):
        blocks.append((match.start(), match.end()))

    # Find TikZ blocks specifically (in case they're not in ```)
    tikz_pattern = r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}'
    for match in re.finditer(tikz_pattern, content, re.DOTALL):
        blocks.append((match.start(), match.end()))

    return blocks


def extract_inline_code(content: str) -> List[Tuple[int, int]]:
    """
    Find all inline code spans (`...`).

    Returns:
        List of (start_pos, end_pos) tuples
    """
    spans = []
    pattern = r'`[^`]+?`'
    for match in re.finditer(pattern, content):
        spans.append((match.start(), match.end()))
    return spans


def extract_math_blocks(content: str) -> List[Tuple[int, int]]:
    """
    Find all LaTeX math blocks ($...$, $$...$$).

    Returns:
        List of (start_pos, end_pos) tuples
    """
    blocks = []

    # Display math $$...$$
    pattern = r'\$\$.*?\$\$'
    for match in re.finditer(pattern, content, re.DOTALL):
        blocks.append((match.start(), match.end()))

    # Inline math $...$
    pattern = r'(?<!\$)\$(?!\$)[^\$]+?\$(?!\$)'
    for match in re.finditer(pattern, content):
        blocks.append((match.start(), match.end()))

    return blocks


def extract_links_and_urls(content: str) -> List[Tuple[int, int]]:
    """
    Find all markdown links and URLs.

    Returns:
        List of (start_pos, end_pos) tuples
    """
    spans = []

    # Markdown links [text](url)
    pattern = r'\[([^\]]+)\]\([^\)]+\)'
    for match in re.finditer(pattern, content):
        # Only exclude the URL part, keep the link text
        url_start = match.group(0).find('](') + match.start() + 1
        url_end = match.end() - 1
        spans.append((url_start, url_end))

    # Reference-style links [@ref], {#id}, @sec-name
    pattern = r'(\[@[^\]]+\]|\{#[^\}]+\}|@[a-z]+-[a-z0-9-]+)'
    for match in re.finditer(pattern, content):
        spans.append((match.start(), match.end()))

    # Plain URLs
    pattern = r'https?://[^\s\)>]+'
    for match in re.finditer(pattern, content):
        spans.append((match.start(), match.end()))

    return spans


def extract_quarto_syntax(content: str) -> List[Tuple[int, int]]:
    """
    Find Quarto-specific syntax to exclude.

    Returns:
        List of (start_pos, end_pos) tuples
    """
    spans = []

    # Quarto divs ::: {.classname}
    pattern = r':::\s*\{[^\}]+\}'
    for match in re.finditer(pattern, content):
        spans.append((match.start(), match.end()))

    # Quarto shortcodes {{< ... >}}
    pattern = r'\{\{<.*?>\}\}'
    for match in re.finditer(pattern, content, re.DOTALL):
        spans.append((match.start(), match.end()))

    return spans


def should_exclude_position(pos: int, exclude_ranges: List[Tuple[int, int]]) -> bool:
    """Check if a position falls within any exclude range."""
    for start, end in exclude_ranges:
        if start <= pos < end:
            return True
    return False


def extract_prose_text(content: str) -> List[Tuple[str, int]]:
    """
    Extract only prose text from QMD content.

    Returns:
        List of (text, line_number) tuples
    """
    # Build exclude ranges
    exclude_ranges = []

    yaml_start, yaml_end = extract_yaml_frontmatter(content)
    if yaml_end > 0:
        exclude_ranges.append((yaml_start, yaml_end))

    exclude_ranges.extend(extract_code_blocks(content))
    exclude_ranges.extend(extract_inline_code(content))
    exclude_ranges.extend(extract_math_blocks(content))
    exclude_ranges.extend(extract_links_and_urls(content))
    exclude_ranges.extend(extract_quarto_syntax(content))

    # Sort and merge overlapping ranges
    exclude_ranges.sort()
    merged = []
    for start, end in exclude_ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Extract prose text
    prose_segments = []
    lines = content.split('\n')
    pos = 0

    for line_num, line in enumerate(lines, 1):
        line_start = pos
        line_end = pos + len(line)

        # Check if any part of this line is prose
        if not should_exclude_position(line_start, merged):
            # Extract prose parts from this line
            prose_text = ""
            for i, char in enumerate(line):
                char_pos = line_start + i
                if not should_exclude_position(char_pos, merged):
                    prose_text += char
                else:
                    if prose_text.strip():
                        prose_segments.append((prose_text.strip(), line_num))
                        prose_text = ""

            if prose_text.strip():
                prose_segments.append((prose_text.strip(), line_num))

        pos = line_end + 1  # +1 for newline

    return prose_segments


def clean_prose_text(text: str) -> str:
    """
    Clean prose text of markdown formatting while keeping words.

    Args:
        text: Raw prose text with markdown

    Returns:
        Cleaned text for spell checking
    """
    # Remove markdown formatting
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # Italic
    text = re.sub(r'_([^_]+)_', r'\1', text)          # Italic
    text = re.sub(r'~~([^~]+)~~', r'\1', text)        # Strikethrough

    # Remove remaining markdown symbols
    text = re.sub(r'[#\*_~]', '', text)

    # Remove special characters but keep apostrophes in words
    text = re.sub(r'[^\w\s\'-]', ' ', text)

    return text.strip()


def check_with_aspell(text: str, ignore_terms: Set[str]) -> List[str]:
    """
    Check text with aspell.

    Returns:
        List of misspelled words
    """
    try:
        result = subprocess.run(
            ['aspell', 'list', '--lang=en'],
            input=text,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            words = [w for w in result.stdout.strip().split('\n') if w]
            # Filter ignore terms
            filtered = [w for w in words if w.lower() not in ignore_terms]
            return filtered
        return []
    except Exception as e:
        print(f"Error running aspell: {e}", file=sys.stderr)
        return []


def check_file(filepath: Path) -> List[dict]:
    """
    Check a single QMD file for spelling errors.

    Returns:
        List of error dictionaries
    """
    # Common technical terms to ignore
    ignore_terms = {
        # File formats and common abbreviations
        'qmd', 'yml', 'json', 'png', 'jpg', 'svg', 'pdf',
        'tikz', 'quarto', 'pandoc', 'latex', 'tensorflow', 'pytorch',
        'gpu', 'cpu', 'tpu', 'ram', 'api', 'ui', 'ux', 'cli', 'sdk',
        'yaml', 'toml', 'html', 'css', 'javascript', 'typescript',
        'numpy', 'pandas', 'matplotlib', 'jupyter', 'colab',
        'github', 'gitlab', 'bitbucket',
        'ai', 'ml', 'dl', 'cv', 'nlp', 'iot', 'rl', 'gan',
        'lstm', 'gru', 'rnn', 'cnn', 'vgg', 'resnet', 'bert',

        # ML systems and techniques
        'tinyml', 'microcontroller', 'microcontrollers', 'preprocessing',
        'convolutional', 'latencies', 'dns', 'dennard', 'triadic',
        'benchmarking', 'gdpr', 'hipaa', 'backpropagation', 'quantized',
        'autoregressive', 'overfitting', 'checkpointing', 'hyperparameters',
        'embeddings', 'spectrograms', 'mfcc', 'kws', 'activations',
        'mnist', 'feedforward', 'softmax', 'relu', 'sigmoid', 'thresholding',
        'postprocessing', 'suboptimal', 'multilayer', 'perceptrons',
        'cnns', 'rnns', 'mlps', 'dnn', 'translational', 'invariance',
        'parallelizable', 'uat', 'discriminative', 'fpgas', 'asics',
        'topologies', 'reconceptualization', 'orchestrators', 'bfloat',

        # Product and project names
        'plantvillage', 'nuru', 'farmbeats', 'respira', 'colabs', 'edgeml',
        'mlperf', 'linpack', 'specpowerssj', 'datahub', 'kubeflow',
        'mobilenets', 'efficientnets', 'gpt', 'palm',

        # Company and organization names
        'mckinsey', 'espressif', 'hortonworks', 'linkedin', 'uber', 'cloudtrail',

        # Acronyms and abbreviations
        'cmd', 'cbsd', 'mw', 'sram', 'sox', 'sdg', 'sdgs', 'agi', 'tco',
        'gpus', 'mlops', 'gigaflops', 'eniac', 'cpus', 'tpus', 'fp', 'nist',

        # Legitimate English words often flagged
        'underserved', 'sociotechnical', 'ebola', 'forecasted', 'unmonitored',
        'transformative', 'microclimates', 'microclimate', 'responders',
        'scalable', 'aspirational', 'lifecycle', 'lifecycles',
        'representativeness', 'reproducibility', 'milliwatt', 'milliwatts',
        'decomposable', 'interpretability', 'modularity', 'architecting',
        'instantiations', 'crowdsourcing', 'crowdsourced', 'interdependencies',
        'degradations', 'natively', 'detections', 'observability', 'exfiltration',
        'auditable', 'cryptographic', 'curation', 'engineerable', 'subfield',
        'misrouted', 'tradeoff', 'tradeoffs', 'pre',

        # People names (for attributions)
        'vijay', 'janapa', 'reddi', 'yann', 'lecun', 'corinna', 'burges',
        'cybenko', 'hornik', 'augereau',

        # Image filename patterns (without extensions)
        'covermlsystems', 'coveraigood', 'coveraibenchmarking',
        'coverconclusion', 'coverdataengineering', 'covernnprimer',
        'coverdlarch',

        # LaTeX commands
        'noindent',

        # AI tools
        'dall', 'dalle',

        # Short codes/patterns
        'fn',

        # Additional comprehensive technical terms (auto-generated from book content)
        'accelerometers', 'acm', 'adamw', 'additionality', 'adreno', 'aes', 'agentic', 'aiops',
        'airbnb', 'aitraining', 'akida', 'al', 'alexa', 'alexnet', 'algorithmically', 'alphafold',
        'ambri', 'amodei', 'anonymization', 'anonymized', 'anthropic', 'asilomar', 'auditability',
        'autocorrect', 'autocorrection', 'autocorrections', 'automatable', 'automl', 'avr', 'axonal',
        'backdoored', 'backdoors', 'backend', 'backends', 'balancers', 'batchsize', 'bibliometric',
        'binarization', 'biometric', 'bist', 'blas', 'bostrom', 'bottlenecked', 'brominated', 'carlini',
        'cfe', 'channelwise', 'chatbot', 'chatbots', 'chatgpt', 'checkmark', 'chiplet', 'chiplets',
        'clinaiops', 'cloudlets', 'cmsis', 'codecarbon', 'compas', 'conda', 'contestability', 'coprocessor',
        'coprocessors', 'coveraihardware', 'coveraiworkflow', 'coverefficientai', 'coverfrontiers',
        'coverintroduction', 'covermlframeworks', 'covermlops', 'covermodeloptimizations',
        'coverondevicelearning', 'coverresponsibleai', 'coverrobustai', 'coversecurityprivacy',
        'coversustainableai', 'cublas', 'cuda', 'customizations', 'cybersecurity', 'cyberweapon',
        'de', 'debois', 'debuggable', 'deepsparse', 'deepspeed', 'devops', 'distilbert', 'dma', 'dp',
        'dsp', 'dsps', 'dvfs', 'dwork', 'dx', 'eacs', 'electrodermal', 'electromechanical',
        'epistemologically', 'esg', 'esrs', 'et', 'ethnicities', 'ets', 'ewc', 'exaflops',
        'explainability', 'explanations', 'expressivity', 'externality', 'facto', 'failover', 'fairlearn',
        'fairscale', 'fe', 'fedavgm', 'fedprox', 'fi', 'flops', 'forrester', 'fpu', 'frac', 'freertos',
        'fx', 'gapped', 'gboard', 'gemm', 'gflops', 'giga', 'goertzel', 'gradcam', 'greenwashing',
        'groupwise', 'handlin', 'hbm', 'hd', 'hdfs', 'hitl', 'homomorphic', 'hsms', 'huggingface',
        'hwacc', 'hyperscale', 'iid', 'imagenet', 'imbalancing', 'incentivized', 'incentivizing',
        'instantiation', 'intentioned', 'interdependency', 'intra', 'jax', 'jenkins', 'jpeg', 'kaggle',
        'kanies', 'kawaguchi', 'kdd', 'keras', 'kinetis', 'kleinberg', 'kohsuke', 'kolmogorov', 'krum',
        'kryo', 'kubernetes', 'lapack', 'lca', 'leaderboards', 'lidar', 'llms', 'ln', 'loihi', 'lora',
        'lpddr', 'mah', 'maml', 'mance', 'mapa', 'mbed', 'mbps', 'mcus', 'medskip', 'metux', 'metuxs',
        'micronpu', 'microservices', 'microsystems', 'millijoules', 'misalignments', 'misclassification',
        'misclassifies', 'misclassify', 'misconfigured', 'mitigations', 'mj', 'mlcommons', 'mlflow',
        'mlir', 'mlp', 'mobilenetv', 'modelscaling', 'moores', 'msqe', 'multimodal', 'multiphase',
        'mwh', 'nas', 'natanz', 'nbsp', 'netron', 'neurosymbolic', 'ngo', 'nm', 'nn', 'npu', 'npus',
        'npv', 'nsight', 'numenta', 'numerics', 'nvlink', 'nwp', 'nxp', 'oecd', 'onnx', 'ons', 'openai',
        'opencl', 'openvino', 'openwebtext', 'operationalization', 'operationalize', 'operationalizing',
        'optum', 'ota', 'overcorrecting', 'overfit', 'overreliance', 'parallelizes', 'pcie', 'perceptron',
        'performant', 'personalization', 'pes', 'picojoules', 'pipelining', 'pj', 'plcs', 'ppv',
        'prefetched', 'prefetching', 'pretrained', 'programmability', 'proliferative', 'proprioception',
        'propublica', 'ptq', 'pufs', 'pypi', 'qat', 'qos', 'quadratically', 'quant', 'rbac', 'recalibrate',
        'recalibrating', 'recommender', 'reconceptualizes', 'recyclability', 'reframing', 'reimagined',
        'reimagining', 'reimplement', 'reimplementing', 'renewables', 'repairability', 'rescoring',
        'reskilling', 'retinopathy', 'reusability', 'ridesharing', 'rlhf', 'roadmap', 'rollout', 'rollouts',
        'rss', 'runtimes', 'sagemaker', 'sanitization', 'scipy', 'scopus', 'sdt', 'sgd', 'shader', 'shaders',
        'shap', 'shapley', 'simd', 'siri', 'situationally', 'slas', 'smi', 'smirnov', 'snns', 'snpe',
        'soc', 'socs', 'sparseml', 'sparsification', 'spinoff', 'sprase', 'spss', 'stationarity', 'stm',
        'stuxnet', 'swappable', 'synergistically', 'tcp', 'tdsp', 'tees', 'tensorboard', 'tensorrt',
        'tera', 'ternarization', 'tflite', 'tfx', 'thresholded', 'timm', 'titration', 'tls', 'tokenization',
        'toolchains', 'torchscript', 'torchserve', 'tpr', 'tpuv', 'tradeable', 'trojan', 'truenorth',
        'tvm', 'ultrapure', 'unbundled', 'underutilization', 'unexplainable', 'unimodal', 'unoptimized',
        'untrusted', 'upgradable', 'upgradeable', 'upskilling', 'uptime', 'usb', 'utensor', 'utopian',
        'vectornet', 'virusbokbok', 'vitis', 'von', 'vr', 'vtune', 'vulkan', 'waymo', 'wearables',
        'wellbeing', 'wi', 'xla', 'zero',
    }

    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return []

    prose_segments = extract_prose_text(content)
    errors = []

    for text, line_num in prose_segments:
        cleaned = clean_prose_text(text)
        if not cleaned:
            continue

        misspelled = check_with_aspell(cleaned, ignore_terms)
        if misspelled:
            errors.append({
                'file': filepath.resolve(),  # Store absolute path
                'line': line_num,
                'text': text[:100] + ('...' if len(text) > 100 else ''),
                'misspelled': misspelled
            })

    return errors


def main():
    """Main function."""
    # Check if aspell is available
    try:
        subprocess.run(['aspell', '--version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: aspell not found. Install it with: brew install aspell", file=sys.stderr)
        return 1

    # Get directory to check
    repo_root = Path(__file__).resolve().parents[3]

    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = repo_root / 'quarto' / 'contents' / 'core'

    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}", file=sys.stderr)
        return 1

    # Find all QMD files
    qmd_files = list(target_dir.rglob('*.qmd'))
    print(f"Checking {len(qmd_files)} .qmd files for prose spelling errors...\n")

    all_errors = []
    files_with_errors = 0

    for qmd_file in sorted(qmd_files):
        errors = check_file(qmd_file)
        if errors:
            files_with_errors += 1
            all_errors.extend(errors)

    # Print results
    if all_errors:
        print(f"Found {len(all_errors)} potential spelling errors in {files_with_errors} files:\n")

        current_file = None
        for error in sorted(all_errors, key=lambda e: (str(e['file']), e['line'])):
            if error['file'] != current_file:
                current_file = error['file']
                try:
                    rel_path = error['file'].relative_to(repo_root)
                except ValueError:
                    rel_path = error['file']
                print(f"\n{rel_path}")
                print("=" * len(str(rel_path)))

            print(f"  Line {error['line']}: {error['text']}")
            print(f"    → Misspelled: {', '.join(error['misspelled'])}")

        print(f"\n\nSummary: {len(all_errors)} potential errors in {files_with_errors} files")
        return 1
    else:
        print("✓ No spelling errors found in prose text!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
