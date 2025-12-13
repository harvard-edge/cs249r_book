#!/usr/bin/env python3

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


TIKZ_BLOCK_PATTERN = re.compile(r"```\{\.tikz\}[\s\S]*?```", re.MULTILINE)
TIKZSET_BLOCK_PATTERN = re.compile(r"\\tikzset\s*\{([\s\S]*?)\}")
STYLE_DEF_PATTERN = re.compile(r"(^|[,\s])([A-Za-z][\w-]*)\s*/\\.style\b")

# Commands that take option lists in square brackets
OPTIONED_COMMANDS = (
    "draw",
    "node",
    "path",
    "filldraw",
    "shade",
    "clip",
    "coordinate",
    "begin\\{scope\}",  # \begin{scope}[...]
)

OPTION_CAPTURE_PATTERNS = [
    re.compile(rf"\\{cmd}\s*\[([^\]]+)\]") for cmd in OPTIONED_COMMANDS
]


# Heuristic tokens we will ignore when seen as standalone options, to reduce false positives
KNOWN_TOKENS: Set[str] = {
    # thickness
    "ultra thin",
    "very thin",
    "thin",
    "semithick",
    "thick",
    "very thick",
    "ultra thick",
    # dashing
    "solid",
    "dashed",
    "densely dashed",
    "loosely dashed",
    "dotted",
    "densely dotted",
    "loosely dotted",
    "dashdotted",
    # common shape keywords
    "circle",
    "rectangle",
    # positioning
    "left",
    "right",
    "above",
    "below",
    # path arrows sometimes appear as tokens, but usually expressed via -{...}
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(errors="ignore")


def find_tikz_blocks(text: str) -> List[Tuple[int, int, str]]:
    blocks: List[Tuple[int, int, str]] = []
    for m in TIKZ_BLOCK_PATTERN.finditer(text):
        start, end = m.span()
        blocks.append((start, end, m.group(0)))
    return blocks


def collect_defined_styles(text: str) -> Set[str]:
    styles: Set[str] = set()
    for m in TIKZSET_BLOCK_PATTERN.finditer(text):
        body = m.group(1)
        for s in STYLE_DEF_PATTERN.finditer(body):
            styles.add(s.group(2))
    return styles


def extract_option_tokens(option_text: str) -> List[str]:
    # naive split by comma, ignore content inside braces or brackets
    tokens: List[str] = []
    buf: List[str] = []
    depth_curly = 0
    depth_brack = 0
    for ch in option_text:
        if ch == '{':
            depth_curly += 1
        elif ch == '}':
            depth_curly = max(0, depth_curly - 1)
        elif ch == '[':
            depth_brack += 1
        elif ch == ']':
            depth_brack = max(0, depth_brack - 1)
        if ch == ',' and depth_curly == 0 and depth_brack == 0:
            token = ''.join(buf).strip()
            if token:
                tokens.append(token)
            buf = []
        else:
            buf.append(ch)
    last = ''.join(buf).strip()
    if last:
        tokens.append(last)
    return tokens


def looks_like_style_token(token: str) -> bool:
    # Exclude key=value and tokens with spaces that are clearly compound phrases unless uppercase start
    if '=' in token:
        return False
    # Ignore shorten <=, shorten >=, etc.
    if 'shorten <=' in token or 'shorten >=' in token:
        return False
    # Arrow tip specifications or path operators are not styles
    if token.startswith('-') or token.endswith('-') or '->' in token:
        return False
    # Common known tokens
    if token in KNOWN_TOKENS:
        return False
    # If token contains spaces and starts lowercase, likely a built-in keyword like very thick, dashed, etc.
    if ' ' in token and not token[0].isupper():
        return False
    # Only consider tokens that start with an uppercase letter as potential custom styles
    if not token or not token[0].isupper():
        return False
    return bool(re.match(r"^[A-Za-z][\w-]*$", token))


def find_undefined_styles_in_block(block_text: str, defined_styles: Set[str]) -> Set[str]:
    used_unknown: Set[str] = set()
    for pat in OPTION_CAPTURE_PATTERNS:
        for m in pat.finditer(block_text):
            options_text = m.group(1)
            for token in extract_option_tokens(options_text):
                token = token.strip()
                if not looks_like_style_token(token):
                    continue
                # Accept styles explicitly defined
                if token in defined_styles:
                    continue
                used_unknown.add(token)
    return used_unknown


def build_line_index(text: str) -> List[int]:
    # returns start index of each line
    idxs = [0]
    for m in re.finditer(r"\n", text):
        idxs.append(m.end())
    return idxs


def offset_to_line(line_starts: List[int], offset: int) -> int:
    # binary search for line number from offset
    lo, hi = 0, len(line_starts) - 1
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if line_starts[mid] <= offset:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans + 1  # 1-based


def scan_quarto_root(root: Path) -> int:
    # Collect global styles from header-includes.tex if present
    global_styles: Set[str] = set()
    header_includes = root / "quarto" / "tex" / "header-includes.tex"
    if header_includes.exists():
        global_styles |= collect_defined_styles(read_text(header_includes))

    qmd_files = list((root / "quarto").rglob("*.qmd"))
    total_issues = 0
    for qmd in qmd_files:
        text = read_text(qmd)
        line_index = build_line_index(text)
        blocks = find_tikz_blocks(text)
        if not blocks:
            continue
        for bstart, bend, btext in blocks:
            local_defs = collect_defined_styles(btext)
            defined = set(global_styles) | set(local_defs)
            unknown = find_undefined_styles_in_block(btext, defined)
            if not unknown:
                continue
            total_issues += len(unknown)
            # Try to locate first reference lines for each unknown token
            print(f"File: {qmd}")
            for token in sorted(unknown):
                # find usage position within block (simplified matcher for robustness)
                usage_pattern = (
                    r"\\(draw|node|path|filldraw|shade|clip|coordinate|begin\{scope\})\s*\["
                    + r"[^\]]*" + re.escape(token) + r"[^\]]*\]"
                )
                m = re.search(usage_pattern, btext)
                if m:
                    pos_in_block = m.start()
                    line_no = offset_to_line(line_index, bstart + pos_in_block)
                    context_line = text.splitlines()[line_no - 1].rstrip()
                    print(f"  line {line_no}: uses undefined style '{token}'")
                    print(f"    {context_line}")
                else:
                    print(f"  uses undefined style '{token}' (exact line not found)")
            print()
    return total_issues


def main(argv: List[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else Path(__file__).resolve().parents[3]
    if not root.exists():
        print(f"Root path does not exist: {root}", file=sys.stderr)
        return 2
    issues = scan_quarto_root(root)
    if issues:
        print(f"Found {issues} undefined style references.")
        return 1
    print("No undefined TikZ style references found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
