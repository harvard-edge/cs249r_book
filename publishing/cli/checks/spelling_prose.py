"""
Spell check prose content in QMD files using aspell.

Intelligently parses QMD file structure to only check actual prose text,
excluding YAML frontmatter, code blocks, TikZ diagrams, inline code, URLs, etc.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Set, Tuple


def extract_yaml_frontmatter(content: str) -> Tuple[int, int]:
    if not content.startswith('---'):
        return (0, 0)
    lines = content.split('\n')
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == '---':
            start = 0
            end = sum(len(lines[j]) + 1 for j in range(i + 1))
            return (start, end)
    return (0, 0)


def extract_code_blocks(content: str) -> List[Tuple[int, int]]:
    blocks = []
    pattern = r'```.*?```'
    for match in re.finditer(pattern, content, re.DOTALL):
        blocks.append((match.start(), match.end()))
    tikz_pattern = r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}'
    for match in re.finditer(tikz_pattern, content, re.DOTALL):
        blocks.append((match.start(), match.end()))
    return blocks


def extract_inline_code(content: str) -> List[Tuple[int, int]]:
    spans = []
    pattern = r'`[^`]+?`'
    for match in re.finditer(pattern, content):
        spans.append((match.start(), match.end()))
    return spans


def extract_math_blocks(content: str) -> List[Tuple[int, int]]:
    blocks = []
    pattern = r'\$\$.*?\$\$'
    for match in re.finditer(pattern, content, re.DOTALL):
        blocks.append((match.start(), match.end()))
    pattern = r'(?<!\$)\$(?!\$)[^\$]+?\$(?!\$)'
    for match in re.finditer(pattern, content):
        blocks.append((match.start(), match.end()))
    return blocks


def extract_links_and_urls(content: str) -> List[Tuple[int, int]]:
    spans = []
    pattern = r'\[([^\]]+)\]\([^\)]+\)'
    for match in re.finditer(pattern, content):
        url_start = match.group(0).find('](') + match.start() + 1
        url_end = match.end() - 1
        spans.append((url_start, url_end))
    pattern = r'(\[@[^\]]+\]|\{#[^\}]+\}|@[a-z]+-[a-z0-9-]+)'
    for match in re.finditer(pattern, content):
        spans.append((match.start(), match.end()))
    pattern = r'https?://[^\s\)>]+'
    for match in re.finditer(pattern, content):
        spans.append((match.start(), match.end()))
    return spans


def extract_quarto_syntax(content: str) -> List[Tuple[int, int]]:
    spans = []
    pattern = r':::\s*\{[^\}]+\}'
    for match in re.finditer(pattern, content):
        spans.append((match.start(), match.end()))
    pattern = r'\{\{<.*?>\}\}'
    for match in re.finditer(pattern, content, re.DOTALL):
        spans.append((match.start(), match.end()))
    return spans


def should_exclude_position(pos: int, exclude_ranges: List[Tuple[int, int]]) -> bool:
    for start, end in exclude_ranges:
        if start <= pos < end:
            return True
    return False


def extract_prose_text(content: str) -> List[Tuple[str, int]]:
    exclude_ranges = []
    yaml_start, yaml_end = extract_yaml_frontmatter(content)
    if yaml_end > 0:
        exclude_ranges.append((yaml_start, yaml_end))

    exclude_ranges.extend(extract_code_blocks(content))
    exclude_ranges.extend(extract_inline_code(content))
    exclude_ranges.extend(extract_math_blocks(content))
    exclude_ranges.extend(extract_links_and_urls(content))
    exclude_ranges.extend(extract_quarto_syntax(content))

    exclude_ranges.sort()
    merged = []
    for start, end in exclude_ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    prose_segments = []
    lines = content.split('\n')
    pos = 0

    for line_num, line in enumerate(lines, 1):
        line_start = pos
        if not should_exclude_position(line_start, merged):
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
        pos = line_start + len(line) + 1

    return prose_segments


def clean_prose_text(text: str) -> str:
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    text = re.sub(r'[#\*_~]', '', text)
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    return text.strip()


def check_with_aspell(text: str, ignore_terms: Set[str]) -> List[str]:
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
            return [w for w in words if w.lower() not in ignore_terms]
        return []
    except Exception:
        return []


def check_file(filepath: Path) -> List[dict]:
    ignore_terms = {
        'qmd', 'yml', 'json', 'png', 'jpg', 'svg', 'pdf',
        'tikz', 'quarto', 'pandoc', 'latex', 'tensorflow', 'pytorch',
        'gpu', 'cpu', 'tpu', 'ram', 'api', 'ui', 'ux', 'cli', 'sdk',
        'yaml', 'toml', 'html', 'css', 'javascript', 'typescript',
        'numpy', 'pandas', 'matplotlib', 'jupyter', 'colab',
        'github', 'gitlab', 'bitbucket',
        'ai', 'ml', 'dl', 'cv', 'nlp', 'iot', 'rl', 'gan',
        'lstm', 'gru', 'rnn', 'cnn', 'vgg', 'resnet', 'bert',
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
        'plantvillage', 'nuru', 'farmbeats', 'respira', 'colabs', 'edgeml',
        'mlperf', 'linpack', 'specpowerssj', 'datahub', 'kubeflow',
        'mobilenets', 'efficientnets', 'gpt', 'palm',
        'mckinsey', 'espressif', 'hortonworks', 'linkedin', 'uber', 'cloudtrail',
        'cmd', 'cbsd', 'mw', 'sram', 'sox', 'sdg', 'sdgs', 'agi', 'tco',
        'gpus', 'mlops', 'gigaflops', 'eniac', 'cpus', 'tpus', 'fp', 'nist',
        'underserved', 'sociotechnical', 'ebola', 'forecasted', 'unmonitored',
        'transformative', 'microclimates', 'microclimate', 'responders',
        'scalable', 'aspirational', 'lifecycle', 'lifecycles',
        'representativeness', 'reproducibility', 'milliwatt', 'milliwatts',
        'decomposable', 'interpretability', 'modularity', 'architecting',
        'instantiations', 'crowdsourcing', 'crowdsourced', 'interdependencies',
        'degradations', 'natively', 'detections', 'observability', 'exfiltration',
        'auditable', 'cryptographic', 'curation', 'engineerable', 'subfield',
        'misrouted', 'tradeoff', 'tradeoffs', 'pre',
        'vijay', 'janapa', 'reddi', 'yann', 'lecun', 'corinna', 'burges',
        'cybenko', 'hornik', 'augereau',
        'covermlsystems', 'coveraigood', 'coveraibenchmarking',
        'coverconclusion', 'coverdataengineering', 'covernnprimer',
        'coverdlarch', 'noindent', 'dall', 'dalle', 'fn',
    }

    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception:
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
                'file': filepath.resolve(),
                'line': line_num,
                'text': text[:100] + ('...' if len(text) > 100 else ''),
                'misspelled': misspelled
            })

    return errors
