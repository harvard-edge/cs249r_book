#!/usr/bin/env python3
"""
Audit QMD files for hardcoded numbers that should use {python} inline references.

Key insight: search for variable values ONLY in prose near the code block that defines them
(within ~200 lines after the block). This dramatically reduces false positives from
coincidental number matches.
"""

import re, sys, os, math

COMMON_NUMBERS = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '12', '15', '16', '20', '24', '25', '30', '32', '40', '50',
    '60', '64', '80', '100', '128', '200', '256', '500', '512',
    '1000', '1024', '2048', '4096',
    '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025',
}


def parse_file(filepath):
    """Parse a QMD file, returning blocks, prose lines, and raw lines."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    python_blocks = []  # (start_line, end_line, code_text)
    prose_lines = []    # (line_num, text)
    
    in_python_block = False
    in_any_code_block = False
    current_block = []
    block_start = 0
    yaml_end = 0
    yaml_count = 0
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        if stripped == '---':
            yaml_count += 1
            if yaml_count == 2:
                yaml_end = i
            continue
        
        if i <= yaml_end:
            continue
        
        if re.match(r'^```\{python\}', stripped):
            in_python_block = True
            in_any_code_block = True
            block_start = i
            current_block = []
        elif in_python_block and stripped == '```':
            python_blocks.append((block_start, i, ''.join(current_block)))
            in_python_block = False
            in_any_code_block = False
        elif not in_python_block and re.match(r'^```', stripped) and len(stripped) > 3:
            in_any_code_block = True
        elif not in_python_block and in_any_code_block and stripped == '```':
            in_any_code_block = False
        elif in_python_block:
            current_block.append(line)
        elif not in_any_code_block:
            prose_lines.append((i, line))
    
    return python_blocks, prose_lines, lines


def execute_blocks_incrementally(python_blocks):
    """Execute blocks incrementally, tracking which variables are new after each block."""
    exec_globals = {'__builtins__': __builtins__}
    block_vars = []  # For each block: dict of NEW _str variables defined in that block
    
    for bstart, bend, code in python_blocks:
        # Snapshot current _str vars
        before = {k: v for k, v in exec_globals.items() if k.endswith('_str') and isinstance(v, str)}
        
        try:
            exec(code, exec_globals)
        except Exception as e:
            pass
        
        # Find new _str vars
        after = {k: v for k, v in exec_globals.items() if k.endswith('_str') and isinstance(v, str)}
        new_vars = {}
        for k, v in after.items():
            if k not in before or before[k] != v:
                new_vars[k] = v
        
        block_vars.append(new_vars)
    
    return block_vars, exec_globals


def is_specific_enough(core_num):
    """Is this number specific enough to avoid false positives?"""
    if not core_num:
        return False
    
    no_comma = core_num.replace(',', '')
    
    if no_comma in COMMON_NUMBERS:
        return False
    
    try:
        num = float(no_comma)
        
        # Decimals with non-zero fractional part
        if '.' in no_comma:
            frac = num - int(num)
            if abs(frac) > 0.001:
                return True
            if num >= 100:
                return True
        
        # Numbers with commas (formatted large numbers)
        if ',' in core_num:
            return True
        
        # 3+ digit non-round numbers
        if num >= 100 and num not in {100, 128, 200, 256, 500, 512, 1000, 1024, 2048, 4096}:
            return True
        
        # 2-digit non-round numbers (like 23, 39, 14, etc.)
        if 10 < num < 100 and num % 5 != 0 and num % 10 != 0:
            return True
        
        # Special: numbers > 1000 that are round but still specific
        if num >= 2000 and num not in COMMON_NUMBERS:
            return True
        
    except (ValueError, OverflowError):
        pass
    
    return False


def extract_core_number(value_str):
    """Extract the core numeric part from a formatted string like '$150K', '3,500 ms', etc."""
    s = value_str.strip()
    s = re.sub(r'^[~$≈<>]', '', s)
    s = re.sub(r'\s*(ms|GB|TB|MB|KB|GHz|MHz|pJ|mJ|kWh|tons|days|hours|minutes|seconds|K|M|B|T|%|mJ|s)\b.*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'[%]$', '', s)
    return s.strip()


def find_hardcoded_in_prose(prose_lines, var_name, core_num, block_end, search_range=300):
    """Search for a specific core number in prose lines near a block."""
    findings = []
    
    escaped = re.escape(core_num)
    pattern = r'(?<![A-Za-z0-9`\._])' + escaped + r'(?![A-Za-z0-9`_])'
    
    for line_num, line_text in prose_lines:
        # Only search within range after the block
        if line_num < block_end:
            continue
        if line_num > block_end + search_range:
            break
        
        stripped = line_text.strip()
        if not stripped:
            continue
        if stripped.startswith('#|') or stripped.startswith('<!--'):
            continue
        if re.match(r'^:?\s*#?(fig|tbl|sec|eq|lst)-', stripped):
            continue
        if stripped.startswith('![') or stripped.startswith('{{< include'):
            continue
        
        # Remove inline {python} references
        cleaned = re.sub(r'`\{python\}\s+[^`]+`', '___PYREF___', line_text)
        # Remove code spans
        cleaned = re.sub(r'`[^`]+`', '___CODE___', cleaned)
        
        matches = list(re.finditer(pattern, cleaned))
        for match in matches:
            # Check it's not inside a placeholder
            in_placeholder = False
            for ph in re.finditer(r'___(?:PYREF|CODE)___', cleaned):
                if ph.start() <= match.start() < ph.end():
                    in_placeholder = True
                    break
            if in_placeholder:
                continue
            
            # Get context from original line
            ctx_start = max(0, match.start() - 50)
            ctx_end = min(len(line_text), match.end() + 50)
            context = line_text[ctx_start:ctx_end].strip().replace('\n', ' ')
            
            findings.append({
                'line': line_num,
                'var_name': var_name,
                'hardcoded': core_num,
                'var_display': '',
                'context': context,
            })
    
    return findings


def audit_file(filepath):
    """Full audit of a single QMD file."""
    print(f"\n{'='*140}")
    short_path = filepath.split('contents/')[-1] if 'contents/' in filepath else filepath
    print(f"FILE: {short_path}")
    print(f"{'='*140}")
    
    python_blocks, prose_lines, all_lines = parse_file(filepath)
    print(f"  {len(python_blocks)} Python blocks, {len(prose_lines)} prose lines")
    
    block_vars, exec_globals = execute_blocks_incrementally(python_blocks)
    
    total_str = sum(len(bv) for bv in block_vars)
    print(f"  {total_str} _str variables total across all blocks")
    
    all_findings = []
    
    for idx, ((bstart, bend, code), new_vars) in enumerate(zip(python_blocks, block_vars)):
        if not new_vars:
            continue
        
        # Determine search range: until the next Python block (or +300 lines)
        if idx + 1 < len(python_blocks):
            next_block_start = python_blocks[idx + 1][0]
            search_range = next_block_start - bend
        else:
            search_range = 300
        
        for var_name, var_value in new_vars.items():
            core = extract_core_number(var_value)
            if not is_specific_enough(core):
                continue
            
            findings = find_hardcoded_in_prose(prose_lines, var_name, core, bend, search_range)
            for f in findings:
                f['var_display'] = var_value
                f['block_range'] = f"L{bstart}-{bend}"
            all_findings.extend(findings)
    
    # Deduplicate
    seen = set()
    unique = []
    for f in all_findings:
        key = (f['line'], f['var_name'])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    
    unique.sort(key=lambda x: (x['line'], x['var_name']))
    
    if unique:
        print(f"\n  FINDINGS ({len(unique)}):")
        print(f"  {'Line':>6} | {'Variable':40s} | {'Value':12s} | {'Defined':12s} | Context")
        print(f"  {'-'*6}-+-{'-'*40}-+-{'-'*12}-+-{'-'*12}-+-{'-'*70}")
        for f in unique:
            ctx = f['context'][:70]
            print(f"  {f['line']:6d} | {f['var_name']:40s} | {f['hardcoded']:12s} | {f['block_range']:12s} | {ctx}")
    else:
        print(f"\n  No findings (all values properly use {{python}} references)")
    
    return unique


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    quarto_dir = os.path.join(script_dir, '..')
    sys.path.insert(0, quarto_dir)
    os.chdir(quarto_dir)
    
    files = [
        'contents/vol1/data_engineering/data_engineering.qmd',
        'contents/vol1/data_selection/data_selection.qmd',
        'contents/vol1/training/training.qmd',
        'contents/vol1/serving/serving.qmd',
        'contents/vol1/workflow/workflow.qmd',
    ]
    
    all_findings = {}
    for fpath in files:
        if os.path.exists(fpath):
            findings = audit_file(fpath)
            all_findings[fpath] = findings
        else:
            print(f"File not found: {fpath}")
    
    print(f"\n\n{'='*140}")
    print("SUMMARY")
    print(f"{'='*140}")
    total = 0
    for fpath, findings in all_findings.items():
        short = fpath.split('contents/')[-1]
        print(f"  {short}: {len(findings)} findings")
        total += len(findings)
    print(f"  TOTAL: {total} findings")
