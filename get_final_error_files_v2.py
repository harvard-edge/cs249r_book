import json
import os

with open('final_check_results_v2.json', 'r') as f:
    data = json.load(f)

error_files = set()
for error in data.get('errors', []):
    uri = error.get('uri')
    if uri and (uri.startswith('interviews/vault/') or uri.startswith('../vault/')):
        path = uri.replace('../vault/', 'interviews/vault/')
        if os.path.exists(path):
            error_files.add(path)

with open('files_to_repair_final_v2.txt', 'w') as f:
    for path in sorted(list(error_files)):
        f.write(path + '\n')

print(f"Found {len(error_files)} unique files for Final Repair Pass V2.")
