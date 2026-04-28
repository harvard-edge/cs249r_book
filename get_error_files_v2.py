import json

with open('interviews/vault-cli/check_results_after_repair.json', 'r') as f:
    data = json.load(f)

error_files = set()
for error in data.get('errors', []):
    uri = error.get('uri')
    if uri and uri.startswith('../vault/'):
        path = uri.replace('../vault/', 'interviews/vault/')
        error_files.add(path)

with open('files_to_repair_v2.txt', 'w') as f:
    for path in sorted(list(error_files)):
        f.write(path + '\n')

print(f"Found {len(error_files)} unique files for Repair Pass V2.")
