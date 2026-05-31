import os
import ast

docs = {
    'mlsysim/show.py': {
        '_align': 'Aligns numeric and string metrics for clean CLI output.'
    },
    'mlsysim/fmt.py': {
        '_repr_markdown_': 'Jupyter notebook hook to render the object as Markdown.'
    },
    'mlsysim/tools/audit_provenance.py': {
        '_registry_nodes': 'Yields all Sourced AST nodes found in the target registry file.',
        '_validate_provenance_record': 'Validates that a Provenance record meets the required traceability constraints.',
        '_check_node': 'Inspects an AST node to verify its provenance lineage.'
    },
    'mlsysim/core/optimization/registry.py': {
        '_load_scipy_backend': 'Lazy-loads the SciPy optimization backend.',
        '_load_ortools_backend': 'Lazy-loads the Google OR-Tools optimization backend.',
        '_load_exhaustive_backend': 'Lazy-loads the Exhaustive grid-search optimization backend.'
    },
    'mlsysim/core/registry/plugin_manager.py': {
        '_load_plugins': 'Dynamically discovers and loads entry points for third-party extensions.'
    },
    'mlsysim/cli/renderers.py': {
        '_format_metric': 'Formats a physical quantity or scalar into a human-readable string for CLI rendering.'
    },
    'mlsysim/cli/commands/optimize.py': {
        '_load_plan_schema': 'Loads the optimization plan schema from the specified YAML configuration.'
    },
    'mlsysim/cli/commands/eval.py': {
        '_parse_val': 'Parses a command-line value into a typed physical Quantity or standard scalar.'
    }
}

def insert_docstring(filepath, node_name, docstring):
    if not os.path.exists(filepath): return False
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == node_name:
            if not ast.get_docstring(node):
                line_idx = node.lineno - 1
                while line_idx < len(lines):
                    if ':' in lines[line_idx].split('#')[0]:
                        break
                    line_idx += 1
                
                indent = ' ' * (node.col_offset + 4)
                doc = f'{indent}\"\"\"{docstring}\"\"\"\n'
                lines.insert(line_idx + 1, doc)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                # Re-parse to catch multiple nodes with same name
                return insert_docstring(filepath, node_name, docstring)
    return False

for fp, node_dict in docs.items():
    for node_name, docstr in node_dict.items():
        insert_docstring(fp, node_name, docstr)

print("Done inserting private docstrings")
