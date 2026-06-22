import re
import sys

def audit_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    blocks = re.findall(r'# ┌── LEGO ───────────────────────────────────────────────\nclass (.*?):(.*?)```', content, re.DOTALL)
    
    print(f"=== AUDIT: {filepath} ===")
    print(f"Found {len(blocks)} LEGO blocks.\n")
    
    for class_name, block_content in blocks:
        print(f"--- Block: {class_name} ---")
        
        # Check if it uses mlsysim solvers or scenarios
        uses_solver = 'Engine.solve' in block_content or 'calc_' in block_content or 'Model' in block_content
        print(f"Uses MLSysIM Core Solvers/Formulas: {uses_solver}")
        
        # Check if it has suspicious manual math (+, -, *, /) that should be a solver
        math_lines = [line.strip() for line in block_content.split('\n') if re.search(r'[\+\-\*/]', line) and 'fmt' not in line and 'check(' not in line and 'import' not in line and line.strip() and not line.strip().startswith('#')]
        
        print("Raw Math Lines (Check for manual physics):")
        for m in math_lines:
            print(f"  {m}")
            
        print("")

if __name__ == '__main__':
    audit_file(sys.argv[1])
