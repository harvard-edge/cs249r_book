#!/usr/bin/env python3
"""
Transform flat PICO Python cells in QMD files to class-based namespace isolation.

Usage:
    python3 book/tools/scripts/transform_pico_cells.py <path_to_qmd>
"""

import re
import sys
from pathlib import Path


def label_to_classname(label: str) -> str:
    """Convert label like 'nn-ops-calc' to 'NnOpsCalc'."""
    parts = re.split(r'[-_]', label)
    return ''.join(p.capitalize() for p in parts)


def extract_exports_from_header(header_text: str) -> list:
    """Extract export variable names from the PICO header comment."""
    exports = []
    in_exports = False
    for line in header_text.split('\n'):
        if re.search(r'#\s*│\s*[Ee]xports?:', line):
            in_exports = True
            after_colon = re.split(r'[Ee]xports?:', line, maxsplit=1)[-1].strip()
            vars_raw = re.sub(r'\(.*?\)', '', after_colon)
            for var in re.split(r'[,\s]+', vars_raw):
                var = var.strip().rstrip(',').strip()
                if var and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var):
                    exports.append(var)
        elif in_exports:
            if re.search(r'#\s*│', line):
                content = re.split(r'#\s*│', line, maxsplit=1)[-1].strip()
                # Stop if we hit a new section keyword
                if re.match(r'[A-Z][a-z]+:', content):
                    in_exports = False
                elif content == '' or content == '└─' * 5:
                    in_exports = False
                else:
                    vars_raw = re.sub(r'\(.*?\)', '', content)
                    for var in re.split(r'[,\s]+', vars_raw):
                        var = var.strip().rstrip(',').strip()
                        if var and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var):
                            exports.append(var)
            else:
                in_exports = False
    return exports


def wrap_cell_in_class(cell_body: str, label: str) -> str:
    """
    Takes the full content between ``` marks and wraps flat code in a class.
    Returns the new cell body (without the ``` delimiters).
    """
    class_name = label_to_classname(label)
    lines = cell_body.split('\n')

    quarto_opts = []
    header_lines = []
    import_lines = []
    body_lines = []

    state = 'start'

    for line in lines:
        stripped = line.strip()

        if state == 'start':
            if stripped.startswith('#|'):
                quarto_opts.append(line)
            elif stripped == '':
                pass  # skip leading blank lines
            elif (stripped.startswith('# ┌') or stripped.startswith('# │') or
                  stripped.startswith('# ├') or stripped.startswith('# └')):
                state = 'header'
                header_lines.append(line)
            elif stripped.startswith('from ') or stripped.startswith('import '):
                state = 'imports'
                import_lines.append(line)
            else:
                state = 'body'
                body_lines.append(line)

        elif state == 'header':
            if (stripped.startswith('# ┌') or stripped.startswith('# │') or
                    stripped.startswith('# ├') or stripped.startswith('# └')):
                header_lines.append(line)
            elif stripped == '':
                state = 'post_header'
            elif stripped.startswith('from ') or stripped.startswith('import '):
                state = 'imports'
                import_lines.append(line)
            else:
                state = 'body'
                body_lines.append(line)

        elif state == 'post_header':
            if stripped == '':
                pass
            elif stripped.startswith('from ') or stripped.startswith('import '):
                state = 'imports'
                import_lines.append(line)
            else:
                state = 'body'
                body_lines.append(line)

        elif state == 'imports':
            if stripped.startswith('from ') or stripped.startswith('import '):
                import_lines.append(line)
            elif stripped == '':
                state = 'post_imports'
            else:
                state = 'body'
                body_lines.append(line)

        elif state == 'post_imports':
            if stripped == '':
                pass
            else:
                state = 'body'
                body_lines.append(line)

        elif state == 'body':
            body_lines.append(line)

    # Remove trailing empty lines from body
    while body_lines and body_lines[-1].strip() == '':
        body_lines.pop()

    # Extract export vars from header
    header_text = '\n'.join(header_lines)
    export_vars = extract_exports_from_header(header_text)

    # Get actual vars assigned at the top level of body
    actual_vars = set()
    for line in body_lines:
        m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=', line)
        if m:
            actual_vars.add(m.group(1))

    # Filter to exports that actually exist in body
    final_exports = [v for v in export_vars if v in actual_vars]
    if not final_exports:
        # Fallback: export all _str vars found in body
        final_exports = sorted(v for v in actual_vars if v.endswith('_str'))

    # Build output
    out = []

    # Quarto options
    for opt in quarto_opts:
        out.append(opt)
    out.append('')

    # PICO header (module-level)
    for h in header_lines:
        out.append(h)

    # Imports (module-level)
    if import_lines:
        out.append('')
        for imp in import_lines:
            out.append(imp)

    # Class
    out.append('')
    out.append('# ┌── P.I.C.O. ISOLATED SCENARIO ───────────────────────────────────────────────')
    out.append(f'class {class_name}:')
    readable = label.replace('-', ' ').title()
    out.append(f'    """Namespace for {readable}."""')
    out.append('')

    # Check if body has structured sections already
    has_sections = any(
        '# --- Inputs' in l or '# --- Process' in l or '# --- Outputs' in l or
        '# --- Derived' in l
        for l in body_lines
    )

    if has_sections:
        # Remap section headers to PICO style and indent
        skip_next_blank = False
        for line in body_lines:
            stripped = line.strip()
            if '# --- Inputs' in stripped:
                out.append('    # ┌── 1. PARAMETERS (Inputs) ──────────────────────────────────────────────')
            elif '# --- Derived' in stripped:
                out.append('    # ┌── 2. CALCULATION (The Physics) ────────────────────────────────────────')
            elif '# --- Process' in stripped:
                out.append('    # ┌── 2. CALCULATION (The Physics) ────────────────────────────────────────')
            elif '# --- Outputs' in stripped:
                out.append('    # ┌── 4. OUTPUTS (Formatting) ─────────────────────────────────────────────')
            elif stripped == '':
                out.append('')
            else:
                out.append('    ' + line)
    else:
        # Just indent everything
        for line in body_lines:
            if line.strip():
                out.append('    ' + line)
            else:
                out.append('')

    # Remove trailing blank lines before exports
    while out and out[-1] == '':
        out.pop()

    # Exports
    if final_exports:
        out.append('')
        out.append('# ┌── EXPORTS (Bridge to Text) ─────────────────────────────────────────────────')
        for var in final_exports:
            out.append(f'{var} = {class_name}.{var}')

    return '\n'.join(out)


def process_file(filepath: str) -> tuple:
    """Process a QMD file, transforming flat PICO cells. Returns (count, issues)."""
    path = Path(filepath)
    content = path.read_text()

    # Find all python code blocks
    pattern = r'(```\{python\})(.*?)(```)'
    matches = list(re.finditer(pattern, content, re.DOTALL))

    transformations = 0
    issues = []
    labels_transformed = []

    # Process in reverse order to preserve positions
    for match in reversed(matches):
        cell_body = match.group(2)

        # Check criteria
        has_pico = '# │' in cell_body
        has_class = bool(re.search(r'^class [A-Z]', cell_body, re.MULTILINE))
        is_fig = bool(re.search(r'#\| label: fig-', cell_body))

        if not has_pico or has_class or is_fig:
            continue

        label_match = re.search(r'#\| label: (.+)', cell_body)
        if not label_match:
            issues.append(f"No label found in cell at position {match.start()}")
            continue
        label = label_match.group(1).strip()

        # Skip chapter-start and other non-compute cells
        if label in ('chapter-start',):
            continue

        try:
            transformed_body = wrap_cell_in_class(cell_body, label)
            new_cell = match.group(1) + '\n' + transformed_body + '\n' + match.group(3)
            content = content[:match.start()] + new_cell + content[match.end():]
            transformations += 1
            labels_transformed.append(label)
        except Exception as e:
            issues.append(f"Error transforming cell '{label}': {e}")

    path.write_text(content)
    return transformations, labels_transformed, issues


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 transform_pico_cells.py <file.qmd>")
        sys.exit(1)

    filepath = sys.argv[1]
    count, labels, issues = process_file(filepath)
    print(f"Transformed {count} cells in {filepath}")
    for label in labels:
        print(f"  + {label}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
