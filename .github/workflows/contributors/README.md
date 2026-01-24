# Contributor Management Scripts

This folder contains scripts for managing contributor recognition across the repository.

## Overview

The contributor system tracks contributions to four projects:
- **book/** - ML Systems textbook
- **tinytorch/** - Educational ML framework
- **kits/** - Hardware kits
- **labs/** - Lab exercises

Each project has its own `.all-contributorsrc` file, and the main `README.md` displays all contributors in organized sections.

## Scripts

### `update_contributors.py`

Updates the root `.all-contributorsrc` from GitHub API.

```bash
# Requires GITHUB_TOKEN environment variable
python update_contributors.py
```

**What it does:**
- Queries GitHub API for all repository contributors
- Resolves git emails to GitHub usernames
- Generates gravatar URLs for non-GitHub contributors
- Merges new contributors with existing entries

### `generate_main_readme.py`

Generates the sectioned contributor table in the main `README.md`.

```bash
python generate_main_readme.py [--dry-run]
```

**What it does:**
- Reads all four project `.all-contributorsrc` files
- Generates HTML tables with contributor avatars and badges
- Updates the Contributors section in `README.md`
- Creates sections: Book, TinyTorch, Kits, Labs

### `generate_readme_tables.py`

Updates per-project README files with contributor tables.

```bash
python generate_readme_tables.py [--project PROJECT] [--update]
```

**Options:**
- `--project`: Process only one project (book, tinytorch, kits, labs)
- `--update`: Actually update the README files (without this, just prints)

**What it does:**
- Reads each project's `.all-contributorsrc`
- Generates HTML contributor tables
- Updates the `<!-- ALL-CONTRIBUTORS-LIST -->` section in each project's README

### `scan_contributors.py`

Scans git history to discover contributors (manual/one-time use).

```bash
python scan_contributors.py [--project PROJECT] [--output FORMAT] [--update]
```

**Options:**
- `--project`: Scan only one project
- `--output`: Output format (table, json, rc)
- `--update`: Update `.all-contributorsrc` files directly
- `--dry-run`: Preview changes without writing

**What it does:**
- Analyzes git commit history per project folder
- Categorizes contributions (code, doc, bug, etc.) from commit messages
- Maps git emails to GitHub usernames
- Filters out bots and AI tools

## Workflow Integration

There are two workflows that manage contributors:

### 1. `all-contributors-add.yml` - Comment-Triggered (Recommended)

Automatically adds contributors when you comment on any issue or PR:

```
@all-contributors please add @username for bug, code, doc
```

**How it works:**
1. Parses the comment to extract username and contribution types
2. Detects which project (book, tinytorch, kits, labs) from labels/title
3. Updates the project's `.all-contributorsrc` file
4. Regenerates README tables
5. Commits and pushes directly (no PR needed!)
6. Replies to confirm the addition

**Project Detection:**
- Add `tinytorch` label OR mention "tinytorch" in issue title → tinytorch project
- Add `kits` label OR mention "kits" in issue title → kits project
- Add `labs` label OR mention "labs" in issue title → labs project
- Otherwise → book project (default)

### 2. `update-contributors.yml` - Push-Triggered

Runs when `.all-contributorsrc` files are manually edited and pushed:

```
Trigger: Push to dev/main with .all-contributorsrc changes
         OR manual dispatch

Steps:
1. update_contributors.py    → Update root config from GitHub API
2. generate_main_readme.py   → Rebuild main README sections
3. generate_readme_tables.py → Update per-project READMEs
4. Commit and push changes
```

## Adding Contributors

### Method 1: Comment Command (Recommended)

Comment on any issue or PR:
```
@all-contributors please add @username for bug, code, doc
```

The workflow will automatically:
- Look up the user's GitHub profile
- Add them to the correct project's contributor list
- Update all README files
- Reply with confirmation

### Method 2: Manual Edit

1. Edit the appropriate `.all-contributorsrc` file
2. Add entry with: login, name, avatar_url, contributions
3. Push to dev/main to trigger the update workflow

## Contribution Types

We use the standard [All Contributors emoji key](https://allcontributors.org/docs/en/emoji-key).

Common types: `bug`, `code`, `doc`, `design`, `ideas`, `review`, `test`, `tool`, `tutorial`, `maintenance`, `infra`, `research`

## File Structure

```
.github/workflows/
├── all-contributors-add.yml     # Comment-triggered workflow (main)
├── update-contributors.yml      # Push-triggered workflow
└── contributors/
    ├── README.md                 # This file
    ├── requirements.txt          # Python dependencies
    ├── update_contributors.py    # GitHub API updater
    ├── generate_main_readme.py   # Main README generator
    ├── generate_readme_tables.py # Per-project README generator
    └── scan_contributors.py      # Git history scanner

Project configs:
├── .all-contributorsrc           # Root config (legacy)
├── book/.all-contributorsrc      # Book contributors
├── tinytorch/.all-contributorsrc # TinyTorch contributors
├── kits/.all-contributorsrc      # Kits contributors
└── labs/.all-contributorsrc      # Labs contributors
```
