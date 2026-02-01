import glob
import re
import os

def find_definitions_and_references(root_dir):
    definitions = {}  # id -> filepath
    references = {}   # id -> list of filepaths
    
    # Regex for definitions: {#tbl-id}, {#fig-id}, {#lst-id}
    # Handling potential extra attributes like {#fig-id .class width=50%}
    def_pattern = re.compile(r'\{#((tbl|fig|lst)-[a-zA-Z0-9_-]+)')
    
    # Regex for references: @tbl-id, @fig-id, @lst-id
    ref_pattern = re.compile(r'@((tbl|fig|lst)-[a-zA-Z0-9_-]+)')

    for filepath in glob.glob(os.path.join(root_dir, '**/*.qmd'), recursive=True):
        if 'backmatter' in filepath: continue # Skip backmatter for now or keep it? user said "all other chapters". Let's include everything.
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Find definitions
                for match in def_pattern.finditer(content):
                    def_id = match.group(1)
                    definitions[def_id] = filepath
                    
                # Find references
                for match in ref_pattern.finditer(content):
                    ref_id = match.group(1)
                    if ref_id not in references:
                        references[ref_id] = []
                    references[ref_id].append(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return definitions, references

def main():
    root_dir = 'quarto/contents/vol1'
    definitions, references = find_definitions_and_references(root_dir)
    
    unreferenced = []
    for def_id, filepath in definitions.items():
        if def_id not in references:
            unreferenced.append((def_id, filepath))
            
    if unreferenced:
        print("Found unreferenced items:")
        for def_id, filepath in sorted(unreferenced):
            print(f"{def_id} in {filepath}")
    else:
        print("All items are referenced!")

if __name__ == '__main__':
    main()
