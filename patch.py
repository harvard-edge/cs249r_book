import yaml

with open('/Users/VJ/GitHub/MLSysBook/.pre-commit-config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Find the local repo hook
for repo in config['repos']:
    if repo['repo'] == 'local':
        hooks = repo['hooks']
        # Check if already exists
        if not any(h['id'] == 'book-check-lego-dead-code' for h in hooks):
            hooks.append({
                'id': 'book-check-lego-dead-code',
                'name': 'Book: Check LEGO Dead Code',
                'entry': './scripts/check_lego_vars.py',
                'language': 'system',
                'pass_filenames': True,
                'files': '^book/quarto/contents/.*\\.qmd$'
            })
            break

with open('/Users/VJ/GitHub/MLSysBook/.pre-commit-config.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)
