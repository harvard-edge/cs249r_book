import sys
import traceback

def test_file(qmd_file):
    with open(qmd_file, 'r') as f:
        content = f.read()
    
    import re
    blocks = re.findall(r'```\{python\}(.*?)```', content, re.DOTALL)
    if not blocks:
        print(f"✅ {qmd_file} (No Python blocks)")
        return True
        
    combined_code = "\n".join(blocks)
    
    # Save combined code to temp file to get accurate line numbers
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp:
        temp.write(combined_code)
        temp_name = temp.name
        
    namespace = {}
    try:
        exec(compile(combined_code, temp_name, 'exec'), namespace)
        print(f"✅ {qmd_file} (Passed)")
        import os
        os.remove(temp_name)
        return True
    except Exception as e:
        print(f"❌ Error in {qmd_file}:")
        traceback.print_exc(file=sys.stdout)
        import os
        os.remove(temp_name)
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        success = test_file(sys.argv[1])
        sys.exit(0 if success else 1)
    else:
        print("Usage: python3 scripts/exec_single.py <path/to/file.qmd>")
        sys.exit(1)
