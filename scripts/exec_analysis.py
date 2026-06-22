import glob
import traceback
import sys

qmd_files = sorted(glob.glob('book/quarto/contents/**/*.qmd', recursive=True))

failed = False
for qmd in qmd_files:
    with open(qmd, 'r') as f:
        content = f.read()
    
    import re
    blocks = re.findall(r'```\{python\}(.*?)```', content, re.DOTALL)
    if not blocks:
        continue
        
    combined_code = "\n".join(blocks)
    
    # We execute it in a clean dictionary to act as the global namespace
    namespace = {}
    try:
        exec(combined_code, namespace)
    except Exception as e:
        failed = True
        print(f"❌ Error in {qmd}:")
        traceback.print_exc(limit=0, file=sys.stdout)
        print("-" * 40)

if not failed:
    print("✅ All Python blocks executed successfully across the entire book!")
    sys.exit(0)
else:
    sys.exit(1)
