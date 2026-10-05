import re
import sys

def check_orphans(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Match {#tbl-something} or {#fig-something}, possibly with other attributes (case-sensitive definition)
    defined_raw = re.findall(r'\{#(tbl-[^\s}]+|fig-[^\s}]+)', content)
    defined = set(defined_raw)
    
    chunk_raw = re.findall(r'label:\s*(tbl-[^\s]+|fig-[^\s]+)', content)
    defined = defined.union(set(chunk_raw))

    # Match @tbl-something or @fig-something case-insensitively
    refs_raw = re.findall(r'@(tbl-[^\s.,;)]+|fig-[^\s.,;)]+)', content, re.IGNORECASE)
    
    # Lowercase the refs to match the definition labels which are lowercased
    refs = {r.split('[')[0].lower() for r in refs_raw}
    defined = {d.lower() for d in defined}

    print("Defined but not referenced (Orphans):")
    orphans = defined - refs
    for o in sorted(orphans):
        print(f" - {o}")

    print("\nReferenced but not defined (Broken refs):")
    broken = refs - defined
    for b in sorted(broken):
        print(f" - {b}")

check_orphans('15_responsible_engr.qmd')
