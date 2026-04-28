import json
import os

with open('interviews/vault-cli/check_results.json', 'r') as f:
    data = json.load(f)

error_files = set()
for error in data.get('errors', []):
    uri = error.get('uri')
    if uri and uri.startswith('../vault/'):
        # Convert relative path to absolute or project-relative
        # uri is like "../vault/questions/cloud/cloud-0050.yaml"
        # Project root is /Users/VJ/GitHub/MLSysBook-yaml-audit/
        # vault dir is interviews/vault/
        path = uri.replace('../vault/', 'interviews/vault/')
        error_files.add(path)

with open('files_to_repair.txt', 'w') as f:
    for path in sorted(list(error_files)):
        f.write(path + '\n')

print(f"Found {len(error_files)} unique files with errors.")
