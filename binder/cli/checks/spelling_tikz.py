"""
Spell check text content within TikZ diagrams.

Extracts and validates all visible text from TikZ diagrams in .qmd files,
including node labels, inline annotations, formatted text, and comments.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Set, Tuple


def extract_tikz_blocks(content: str, filepath: str) -> List[Tuple[str, int]]:
    blocks = []
    lines = content.split('\n')
    in_tikz = False
    current_block = []
    start_line = 0

    for i, line in enumerate(lines, 1):
        if r'\begin{tikzpicture}' in line:
            in_tikz = True
            start_line = i
            current_block = [line]
        elif r'\end{tikzpicture}' in line and in_tikz:
            current_block.append(line)
            blocks.append(('\n'.join(current_block), start_line))
            in_tikz = False
            current_block = []
        elif in_tikz:
            current_block.append(line)

    return blocks


def clean_latex_text(text: str) -> str:
    text = text.replace('\\\\', ' ')
    text = re.sub(r'\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)\s+', ' ', text)
    text = re.sub(r'\\usefont\{[^}]*\}\{[^}]*\}\{[^}]*\}\{[^}]*\}', ' ', text)
    text = re.sub(r'\\fontsize\{[^}]*\}\{[^}]*\}\\selectfont', ' ', text)
    text = re.sub(r'\\bfseries\s*', ' ', text)

    for _ in range(3):
        text = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\textit\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\emph\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\textsubscript\{([^}]+)\}', r'_\1', text)
        text = re.sub(r'\\textsuperscript\{([^}]+)\}', r'^\1', text)
        text = re.sub(r'\\textcolor\{[^}]*\}\{([^}]+)\}', r'\1', text)

    text = text.replace('$', '')
    text = re.sub(r'\\[a-zA-Z]+\s*', ' ', text)
    text = ' '.join(text.split())
    return text.strip()


def extract_all_curly_brace_text(tikz_content: str) -> List[Tuple[str, str, int]]:
    texts = []
    node_standalone = r'\\node\s*(?:\[[^\]]*\])?\s*(?:\([^)]*\))?\s*(?:at\s*\([^)]*\))?\s*\{([^}]+)\}'
    for match in re.finditer(node_standalone, tikz_content):
        texts.append((match.group(1), '\\node{...}', match.start()))

    node_inline = r'(?<!\\)node\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}'
    for match in re.finditer(node_inline, tikz_content):
        texts.append((match.group(1), 'node{...} in draw/path/fill', match.start()))

    text_commands = [
        (r'\\textbf\{([^}]+)\}', '\\textbf{...}'),
        (r'\\textit\{([^}]+)\}', '\\textit{...}'),
        (r'\\emph\{([^}]+)\}', '\\emph{...}'),
        (r'\\text\{([^}]+)\}', '\\text{...}'),
    ]
    for pattern, context in text_commands:
        for match in re.finditer(pattern, tikz_content):
            texts.append((match.group(1), context, match.start()))

    label_pattern = r'(?:label|pin|xlabel|ylabel)\s*=\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}'
    for match in re.finditer(label_pattern, tikz_content):
        texts.append((match.group(1), 'label={...}', match.start()))

    legend_pattern = r'\\legend\s*\{([^}]+)\}'
    for match in re.finditer(legend_pattern, tikz_content):
        texts.append((match.group(1), '\\legend{...}', match.start()))

    return texts


def extract_text_from_foreach(tikz_content: str) -> List[Tuple[str, str]]:
    texts = []
    foreach_pattern = r'\\foreach[^{]+in\s*\{([^}]+)\}'
    for match in re.finditer(foreach_pattern, tikz_content, re.DOTALL):
        content = match.group(1)
        text_in_braces = re.findall(r'/\{([^}]+)\}/', content)
        for text in text_in_braces:
            cleaned = clean_latex_text(text)
            if cleaned and len(cleaned) > 2 and not re.match(r'^[\d\s\.,\-\+]+$', cleaned):
                texts.append((cleaned, f'\\foreach loop: /{{{text}}}/'))
    return texts


def extract_text_from_tikz(tikz_content: str) -> List[Tuple[str, str]]:
    texts = []
    seen_texts = set()

    for raw_text, context, pos in extract_all_curly_brace_text(tikz_content):
        cleaned = clean_latex_text(raw_text)
        if not cleaned or re.match(r'^[\d\s\.,\-\+\*\/\(\)_\^]+$', cleaned) or re.match(r'^[a-z]+!?\d*$', cleaned) or len(cleaned) < 2:
            continue
        key = (cleaned.lower(), context)
        if key not in seen_texts:
            seen_texts.add(key)
            texts.append((cleaned, f'{context}: "{raw_text}"'))

    for text, context in extract_text_from_foreach(tikz_content):
        key = (text.lower(), 'foreach')
        if key not in seen_texts:
            seen_texts.add(key)
            texts.append((text, context))

    pic_name_pattern = r'pics/([a-zA-Z_]+)/'
    for match in re.finditer(pic_name_pattern, tikz_content):
        name = match.group(1)
        if len(name) > 2:
            key = (name.lower(), 'pics')
            if key not in seen_texts:
                seen_texts.add(key)
                texts.append((name, f'pics/{name}/'))

    pic_usage_pattern = r'\\pic\s*(?:\[[^\]]*\])?\s*(?:at\s*\([^)]*\))?\s*\{([^}]+)\}'
    for match in re.finditer(pic_usage_pattern, tikz_content):
        name = match.group(1)
        if len(name) > 2 and not re.match(r'^[\d\s]+$', name):
            key = (name.lower(), 'pic_usage')
            if key not in seen_texts:
                seen_texts.add(key)
                texts.append((name, f'\\pic{{...}}{{{name}}}'))

    comment_pattern = r'%\s*(.+?)(?:\n|$)'
    for match in re.finditer(comment_pattern, tikz_content):
        comment = match.group(1).strip()
        if comment and not re.match(r'^[\-\=\*\s]+$', comment):
            key = (comment.lower(), 'comment')
            if key not in seen_texts:
                seen_texts.add(key)
                texts.append((comment, f'% {comment}'))

    def_pattern = r'\\def\\([a-zA-Z]+)\{'
    for match in re.finditer(def_pattern, tikz_content):
        name = match.group(1)
        if len(name) > 3 and not name.isupper() and not name.startswith('r'):
            key = (name.lower(), 'def')
            if key not in seen_texts:
                seen_texts.add(key)
                texts.append((name, f'\\def\\{name}'))

    return texts


def check_spelling_with_aspell(text: str) -> List[str]:
    ignore_terms = {
        'scalefac', 'picname', 'filllcolor', 'drawcolor', 'linewidth',
        'filllcirclecolor', 'drawcircle', 'bodycolor', 'tiecolor', 'stetcolor',
        'drawchannelcolor', 'channelcolor',
        'brownline', 'redline', 'blueline', 'violetline', 'greenline', 'orangeline',
        'violetl', 'greenl', 'bluel', 'redl', 'orangel', 'greend',
        'tikzset', 'foreach', 'tikz', 'usefont', 'phv', 'bfseries', 'textbf',
        'pgfmathparse', 'addplot', 'sqrt',
        'cellsize', 'cellheight', 'xmax', 'ymin', 'newx', 'pos', 'sep',
        'mycylinder', 'mycycle', 'myline', 'rgpoly', 'zerofill',
        'displaye', 'autotext',
        'zgl', 'zgd', 'da', 'dcd', 'dcl', 'dsc', 'ggb', 'lca', 'sre',
        'ui', 'kpis', 'oss', 'rtx', 'tpus', 'bg', 'eniac', 'fp',
        'preprocessing', 'backprop', 'weightgradient', 'davit', 'tokenize',
        'multimodality', 'microarchitecture', 'hypercomputing', 'curation',
        'transformative', 'helvetica', 'geeksforgeeks', 'lightgray', 'gaussian', 'yshift',
        'ack', 'zz', 'yy',
    }

    try:
        result = subprocess.run(
            ['aspell', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return []
        result = subprocess.run(
            ['aspell', 'list', '--lang=en'],
            input=text,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            words = [word for word in result.stdout.strip().split('\n') if word]
            return [w for w in words if w.lower() not in ignore_terms]
    except Exception:
        pass
    return []


def simple_spell_check(text: str) -> List[str]:
    common_typos = {
        'teh': 'the', 'htat': 'that', 'taht': 'that', 'adn': 'and', 'nad': 'and',
        'gatewey': 'gateway', 'poihnts': 'points', 'poitns': 'points',
        'recieve': 'receive', 'seperate': 'separate', 'occured': 'occurred',
        'occurance': 'occurrence', 'begining': 'beginning', 'lenght': 'length',
        'widht': 'width', 'heigth': 'height', 'coordiante': 'coordinate',
        'cooridate': 'coordinate', 'paramter': 'parameter', 'paramters': 'parameters',
        'intellignet': 'intelligent',
    }
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    typos = []
    for word in words:
        if word in common_typos:
            typos.append(f'{word} (suggest: {common_typos[word]})')
    return typos


def check_file(filepath: Path, use_aspell: bool = True) -> List[dict]:
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception:
        return []

    tikz_blocks = extract_tikz_blocks(content, str(filepath))
    errors = []

    for tikz_content, start_line in tikz_blocks:
        texts = extract_text_from_tikz(tikz_content)
        for text, context in texts:
            simple_errors = simple_spell_check(text)
            if simple_errors:
                errors.append({
                    'file': str(filepath),
                    'line': start_line,
                    'text': text,
                    'context': context,
                    'suggestions': simple_errors
                })
            if use_aspell:
                aspell_errors = check_spelling_with_aspell(text)
                if aspell_errors:
                    errors.append({
                        'file': str(filepath),
                        'line': start_line,
                        'text': text,
                        'context': context,
                        'suggestions': aspell_errors
                    })
    return errors
