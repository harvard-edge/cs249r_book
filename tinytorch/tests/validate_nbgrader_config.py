#!/usr/bin/env python3
"""
NBGrader Configuration Validation Script
Validates all TinyTorch modules for NBGrader compatibility
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

class NBGraderValidator:
    """Validates NBGrader configuration in Jupytext Python files"""

    def __init__(self, module_path: Path):
        self.module_path = module_path
        self.module_name = module_path.stem
        self.content = module_path.read_text()
        self.lines = self.content.split('\n')
        self.issues = []
        self.grade_ids = []
        self.cells = self._parse_cells()

    def _parse_cells(self) -> List[Dict]:
        """Parse Jupytext file into cells"""
        cells = []
        current_cell = None
        in_metadata = False
        metadata_lines = []

        for i, line in enumerate(self.lines, 1):
            # Detect cell boundaries
            if line.startswith('# %%'):
                # Save previous cell
                if current_cell:
                    cells.append(current_cell)

                # Start new cell
                is_markdown = '[markdown]' in line
                current_cell = {
                    'line_start': i,
                    'type': 'markdown' if is_markdown else 'code',
                    'content': [],
                    'metadata': {},
                    'raw_line': line
                }

                # Check for inline metadata
                if 'nbgrader=' in line:
                    try:
                        # Extract JSON from cell marker
                        match = re.search(r'nbgrader=({[^}]+})', line)
                        if match:
                            metadata_str = match.group(1)
                            # Clean up the string for JSON parsing
                            metadata_str = metadata_str.replace("'", '"')
                            current_cell['metadata'] = {'nbgrader': json.loads(metadata_str)}
                    except:
                        pass

            elif current_cell:
                # Check for metadata block at start of cell
                if line.strip().startswith('# metadata='):
                    in_metadata = True
                    metadata_lines = [line]
                elif in_metadata:
                    metadata_lines.append(line)
                    if line.strip() == '# ---':
                        in_metadata = False
                        # Parse metadata
                        try:
                            metadata_text = '\n'.join(metadata_lines)
                            # Extract the dictionary part
                            match = re.search(r'metadata=({.*?})\s*# ---', metadata_text, re.DOTALL)
                            if match:
                                metadata_str = match.group(1).replace("'", '"')
                                current_cell['metadata'] = json.loads(metadata_str)
                        except:
                            pass
                        metadata_lines = []
                else:
                    current_cell['content'].append(line)

        # Don't forget last cell
        if current_cell:
            cells.append(current_cell)

        return cells

    def validate_jupytext_header(self) -> bool:
        """Check for proper Jupytext header in first 15 lines"""
        header_found = False
        jupytext_marker = False

        for i, line in enumerate(self.lines[:15]):
            if line.startswith('# ---'):
                header_found = True
            if 'jupytext:' in line or 'text_representation:' in line:
                jupytext_marker = True

        if not header_found:
            self.issues.append({
                'severity': 'P0-BLOCKER',
                'category': 'Jupytext Header',
                'line': 1,
                'issue': 'Missing Jupytext YAML header (lines 1-13)',
                'detail': 'File must start with # --- header containing jupytext metadata'
            })
            return False

        if not jupytext_marker:
            self.issues.append({
                'severity': 'P0-BLOCKER',
                'category': 'Jupytext Header',
                'line': 1,
                'issue': 'Jupytext header missing required fields',
                'detail': 'Header must contain jupytext: and text_representation: fields'
            })
            return False

        return True

    def validate_solution_blocks(self) -> bool:
        """Check for proper BEGIN/END SOLUTION pairing"""
        begin_count = 0
        end_count = 0
        stack = []

        for i, line in enumerate(self.lines, 1):
            if '### BEGIN SOLUTION' in line:
                begin_count += 1
                stack.append(i)
            elif '### END SOLUTION' in line:
                end_count += 1
                if not stack:
                    self.issues.append({
                        'severity': 'P0-BLOCKER',
                        'category': 'Solution Blocks',
                        'line': i,
                        'issue': 'END SOLUTION without matching BEGIN',
                        'detail': f'Found ### END SOLUTION at line {i} without prior ### BEGIN SOLUTION'
                    })
                else:
                    stack.pop()

        # Check for unmatched BEGINs
        if stack:
            for line_num in stack:
                self.issues.append({
                    'severity': 'P0-BLOCKER',
                    'category': 'Solution Blocks',
                    'line': line_num,
                    'issue': 'BEGIN SOLUTION without matching END',
                    'detail': f'Found ### BEGIN SOLUTION at line {line_num} without matching ### END SOLUTION'
                })

        if begin_count != end_count:
            self.issues.append({
                'severity': 'P0-BLOCKER',
                'category': 'Solution Blocks',
                'line': 0,
                'issue': f'Mismatched solution blocks: {begin_count} BEGIN vs {end_count} END',
                'detail': 'Every BEGIN SOLUTION must have exactly one END SOLUTION'
            })
            return False

        return len(stack) == 0

    def validate_cell_metadata(self) -> bool:
        """Check cell metadata for NBGrader requirements"""
        all_valid = True
        grade_ids_seen = set()

        for cell in self.cells:
            if 'nbgrader' not in cell['metadata']:
                # Check if this is a cell that should have metadata
                content_str = '\n'.join(cell['content'])

                # Solution cells should have metadata
                if '### BEGIN SOLUTION' in content_str:
                    self.issues.append({
                        'severity': 'P0-BLOCKER',
                        'category': 'Cell Metadata',
                        'line': cell['line_start'],
                        'issue': 'Solution cell missing NBGrader metadata',
                        'detail': 'Cell contains BEGIN SOLUTION but no nbgrader metadata'
                    })
                    all_valid = False

                # Test cells should have metadata
                if re.search(r'def test_unit_', content_str):
                    self.issues.append({
                        'severity': 'P0-BLOCKER',
                        'category': 'Cell Metadata',
                        'line': cell['line_start'],
                        'issue': 'Test cell missing NBGrader metadata',
                        'detail': 'Cell contains test function but no nbgrader metadata'
                    })
                    all_valid = False

                continue

            nbgrader = cell['metadata']['nbgrader']

            # Check for required fields
            if 'grade_id' in nbgrader:
                grade_id = nbgrader['grade_id']
                self.grade_ids.append(grade_id)

                # Check for duplicates
                if grade_id in grade_ids_seen:
                    self.issues.append({
                        'severity': 'P0-BLOCKER',
                        'category': 'Grade IDs',
                        'line': cell['line_start'],
                        'issue': f'Duplicate grade_id: {grade_id}',
                        'detail': 'Every grade_id must be unique within the module'
                    })
                    all_valid = False
                else:
                    grade_ids_seen.add(grade_id)

            # Validate test cells
            if nbgrader.get('grade') == True:
                if not nbgrader.get('locked', False):
                    self.issues.append({
                        'severity': 'P1-IMPORTANT',
                        'category': 'Test Cell',
                        'line': cell['line_start'],
                        'issue': 'Test cell not locked',
                        'detail': f'grade_id={nbgrader.get("grade_id")}: Test cells must have locked=true'
                    })
                    all_valid = False

                if 'points' not in nbgrader:
                    self.issues.append({
                        'severity': 'P0-BLOCKER',
                        'category': 'Test Cell',
                        'line': cell['line_start'],
                        'issue': 'Test cell missing points',
                        'detail': f'grade_id={nbgrader.get("grade_id")}: Graded cells must have points assigned'
                    })
                    all_valid = False

                if nbgrader.get('solution', False):
                    self.issues.append({
                        'severity': 'P1-IMPORTANT',
                        'category': 'Test Cell',
                        'line': cell['line_start'],
                        'issue': 'Test cell marked as solution',
                        'detail': f'grade_id={nbgrader.get("grade_id")}: Test cells should have solution=false'
                    })
                    all_valid = False

            # Validate solution cells
            if nbgrader.get('solution') == True:
                if nbgrader.get('grade', False):
                    self.issues.append({
                        'severity': 'P2-ADVISORY',
                        'category': 'Solution Cell',
                        'line': cell['line_start'],
                        'issue': 'Solution cell marked for grading',
                        'detail': f'grade_id={nbgrader.get("grade_id")}: Solution cells typically have grade=false'
                    })

                if nbgrader.get('locked', False):
                    self.issues.append({
                        'severity': 'P1-IMPORTANT',
                        'category': 'Solution Cell',
                        'line': cell['line_start'],
                        'issue': 'Solution cell is locked',
                        'detail': f'grade_id={nbgrader.get("grade_id")}: Solution cells should have locked=false'
                    })
                    all_valid = False

        return all_valid

    def validate_cell_types(self) -> bool:
        """Verify proper cell type markers"""
        all_valid = True

        for i, line in enumerate(self.lines, 1):
            if line.startswith('# %%'):
                # Check for invalid cell markers
                if line.startswith('# %%%') or line.startswith('#%%') and not line.startswith('# %%'):
                    self.issues.append({
                        'severity': 'P1-IMPORTANT',
                        'category': 'Cell Type',
                        'line': i,
                        'issue': 'Invalid cell marker syntax',
                        'detail': f'Cell marker must be "# %%" or "# %% [markdown]", found: {line[:30]}'
                    })
                    all_valid = False

        return all_valid

    def check_schema_version(self) -> bool:
        """Check for nbgrader schema version"""
        all_valid = True

        for cell in self.cells:
            if 'nbgrader' in cell['metadata']:
                schema_version = cell['metadata']['nbgrader'].get('schema_version')
                if schema_version != 3:
                    self.issues.append({
                        'severity': 'P2-ADVISORY',
                        'category': 'Schema Version',
                        'line': cell['line_start'],
                        'issue': f'NBGrader schema version is {schema_version}, expected 3',
                        'detail': 'Schema version 3 is current standard'
                    })
                    all_valid = False

        return all_valid

    def run_all_validations(self) -> Dict:
        """Run all validation checks"""
        results = {
            'module': self.module_name,
            'path': str(self.module_path),
            'checks': {
                'jupytext_header': self.validate_jupytext_header(),
                'solution_blocks': self.validate_solution_blocks(),
                'cell_metadata': self.validate_cell_metadata(),
                'cell_types': self.validate_cell_types(),
                'schema_version': self.check_schema_version(),
            },
            'issues': self.issues,
            'grade_ids': self.grade_ids,
            'cell_count': len(self.cells),
            'status': 'PASS' if not self.issues else 'FAIL'
        }

        # Count by severity
        results['issue_count'] = {
            'P0-BLOCKER': len([i for i in self.issues if i['severity'] == 'P0-BLOCKER']),
            'P1-IMPORTANT': len([i for i in self.issues if i['severity'] == 'P1-IMPORTANT']),
            'P2-ADVISORY': len([i for i in self.issues if i['severity'] == 'P2-ADVISORY']),
        }

        return results


def validate_all_modules(modules_dir: Path) -> Dict:
    """Validate all modules in the directory"""
    results = {}

    # Find all module Python files
    module_files = sorted(modules_dir.glob('*/[0-9][0-9]_*.py'))

    # Also check for named files like tensor.py, activations.py, etc.
    for module_dir in sorted(modules_dir.glob('[0-9][0-9]_*')):
        module_py_files = list(module_dir.glob('*.py'))
        # Filter out test and validation files
        module_py_files = [f for f in module_py_files if not any(
            exclude in f.name for exclude in ['test_', 'validate_', 'analysis', '__']
        )]
        if module_py_files:
            # Use the first non-test Python file found
            module_file = module_py_files[0]
            validator = NBGraderValidator(module_file)
            result = validator.run_all_validations()
            results[module_dir.name] = result

    return results


def print_validation_report(results: Dict):
    """Print comprehensive validation report"""

    print("=" * 100)
    print("NBGrader Configuration Validation Report")
    print("=" * 100)
    print()

    # Summary statistics
    total_modules = len(results)
    passed_modules = sum(1 for r in results.values() if r['status'] == 'PASS')
    failed_modules = total_modules - passed_modules

    total_blockers = sum(r['issue_count']['P0-BLOCKER'] for r in results.values())
    total_important = sum(r['issue_count']['P1-IMPORTANT'] for r in results.values())
    total_advisory = sum(r['issue_count']['P2-ADVISORY'] for r in results.values())

    print(f"SUMMARY:")
    print(f"  Total Modules: {total_modules}")
    print(f"  Passed: {passed_modules}")
    print(f"  Failed: {failed_modules}")
    print(f"  Overall Status: {'PASS' if failed_modules == 0 else 'FAIL'}")
    print()
    print(f"ISSUE BREAKDOWN:")
    print(f"  P0-BLOCKER (Critical): {total_blockers}")
    print(f"  P1-IMPORTANT: {total_important}")
    print(f"  P2-ADVISORY: {total_advisory}")
    print(f"  Total Issues: {total_blockers + total_important + total_advisory}")
    print()

    # Per-module status matrix
    print("=" * 100)
    print("MODULE VALIDATION MATRIX")
    print("=" * 100)
    print(f"{'Module':<25} {'Status':<8} {'Cells':<7} {'P0':<5} {'P1':<5} {'P2':<5} {'Grade IDs':<12}")
    print("-" * 100)

    for module_name, result in sorted(results.items()):
        status_icon = "PASS" if result['status'] == 'PASS' else "FAIL"
        print(f"{module_name:<25} {status_icon:<8} {result['cell_count']:<7} "
              f"{result['issue_count']['P0-BLOCKER']:<5} "
              f"{result['issue_count']['P1-IMPORTANT']:<5} "
              f"{result['issue_count']['P2-ADVISORY']:<5} "
              f"{len(result['grade_ids']):<12}")

    print()

    # Detailed issues by module
    print("=" * 100)
    print("DETAILED ISSUES BY MODULE")
    print("=" * 100)

    for module_name, result in sorted(results.items()):
        if result['issues']:
            print()
            print(f"MODULE: {module_name}")
            print(f"Path: {result['path']}")
            print(f"Status: {result['status']}")
            print("-" * 100)

            # Group by severity
            for severity in ['P0-BLOCKER', 'P1-IMPORTANT', 'P2-ADVISORY']:
                severity_issues = [i for i in result['issues'] if i['severity'] == severity]
                if severity_issues:
                    print(f"\n  {severity}:")
                    for issue in severity_issues:
                        print(f"    Line {issue['line']:4d} | {issue['category']:<20} | {issue['issue']}")
                        print(f"               {issue['detail']}")

    # Check summary
    print()
    print("=" * 100)
    print("VALIDATION CHECK SUMMARY")
    print("=" * 100)

    check_names = ['jupytext_header', 'solution_blocks', 'cell_metadata', 'cell_types', 'schema_version']

    for check in check_names:
        passed = sum(1 for r in results.values() if r['checks'][check])
        failed = total_modules - passed
        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {check.replace('_', ' ').title():<30} {status:<8} ({passed}/{total_modules} modules)")

    print()
    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    if total_blockers > 0:
        print("\nCRITICAL BLOCKERS (P0) - Must fix before NBGrader deployment:")
        print("  These issues will prevent NBGrader from functioning correctly.")
        print("  Priority: Fix immediately")

    if total_important > 0:
        print("\nIMPORTANT ISSUES (P1) - Should fix soon:")
        print("  These issues may cause NBGrader to behave unexpectedly.")
        print("  Priority: Fix before student deployment")

    if total_advisory > 0:
        print("\nADVISORY ISSUES (P2) - Consider fixing:")
        print("  These issues are minor but should be addressed for consistency.")
        print("  Priority: Fix when convenient")

    print()


if __name__ == "__main__":
    modules_dir = Path("/Users/VJ/GitHub/TinyTorch/modules")
    results = validate_all_modules(modules_dir)
    print_validation_report(results)

    # Save results to JSON
    import json
    output_file = Path("/Users/VJ/GitHub/TinyTorch/nbgrader_validation_results.json")
    with output_file.open('w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
