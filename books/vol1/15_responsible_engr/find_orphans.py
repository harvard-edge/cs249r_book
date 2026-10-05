import re
import sys

def check_orphans(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find all table and figure declarations
    # Assuming markdown format: {#tbl-something} or {#fig-something}
    # and quarto chunks with label: tbl-something or label: fig-something
    
    defined_tbls = set(re.findall(r'\{#tbl-[^}]+\}', content))
    defined_figs = set(re.findall(r'\{#fig-[^}]+\}', content))
    
    chunk_tbls = set(re.findall(r'label:\s*(tbl-[^\s]+)', content))
    chunk_figs = set(re.findall(r'label:\s*(fig-[^\s]+)', content))

    all_defined = defined_tbls.union(defined_figs).union(
        {f"{{#{x}}}" for x in chunk_tbls}
    ).union(
        {f"{{#{x}}}" for x in chunk_figs}
    )

    all_defined_labels = {x.strip('{#}').strip() for x in all_defined}

    # Find all references: @tbl-something or @fig-something
    refs = set(re.findall(r'@(tbl-[^\s.,;)]+|fig-[^\s.,;)]+)', content))

    print("Defined but not referenced (Orphans):")
    orphans = all_defined_labels - refs
    for o in orphans:
        print(f" - {o}")

    print("\nReferenced but not defined (Broken refs):")
    broken = refs - all_defined_labels
    for b in broken:
        print(f" - {b}")

check_orphans('15_responsible_engr.qmd')
