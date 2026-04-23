# Contributor Management Scripts

This folder contains scripts for managing contributor recognition across the repository.

## Overview

The contributor system tracks contributions across each top-level Quarto site
or sub-project. Every consumer — the two workflows, both generators, the bot
reply text — reads from a single config file:

> **`projects.json`** is the **single source of truth**. Add, rename, reorder
> a project there and nowhere else.

`projects.py` loads `projects.json` and exposes:

- a small Python API (`projects()`, `keys()`, `dirs()`, `alias_pairs()`) used
  by the generator scripts via `from projects import …`
- a CLI used by the workflows that need shell-friendly strings:
  ```bash
  python3 projects.py keys             # → book,tinytorch,…,instructors
  python3 projects.py aliases          # → tito:tinytorch,interviews:staffml,…
  python3 projects.py dirs-overrides   # → staffml:interviews
  python3 projects.py update-files     # → newline-separated file list
  ```

### Each project entry has

- **`key`** — canonical name used in commit messages, bot replies, and
  section markers (e.g. `staffml`).
- **`dir`** — on-disk directory holding `.all-contributorsrc` and
  `README.md` (e.g. `interviews`). May differ from the key.
- **`aliases`** — alternate strings recognised in the trigger comment
  (substring-matched, so prefer non-ambiguous tokens).
- **`section`** — `{emoji, title, marker}` rendered into the root README.

### Where `projects.json` is consumed

| Consumer | How it reads |
|---|---|
| `generate_main_readme.py` | imports `projects.projects()` for section list |
| `generate_readme_tables.py` | imports `projects.dirs()` for the path map |
| `all-contributors-add.yml` | sparse-checkouts the file, runs `projects.py keys/aliases/dirs-overrides`, exports to `$GITHUB_ENV` |
| `update-contributors.yml` | runs `projects.py update-files` to derive the change-detection and `git add` lists |

The push trigger in `update-contributors.yml` uses the glob
`*/.all-contributorsrc`, so adding a new project's config doesn't even need
a workflow edit to start triggering.

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
- Reads each project's `.all-contributorsrc` (project list comes from
  `projects.json`).
- Generates HTML tables with contributor avatars and badges.
- Updates the Contributors section in the root `README.md`, in the order
  declared in `projects.json`.

### `generate_readme_tables.py`

Updates per-project README files with contributor tables.

```bash
python generate_readme_tables.py [--project PROJECT] [--update]
```

**Options:**
- `--project`: Process only one project. Valid values come from
  `projects.json` (currently: `book`, `tinytorch`, `mlsysim`, `staffml`,
  `kits`, `labs`, `slides`, `instructors`).
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
1. Project name (or alias) explicitly mentioned in the trigger comment, e.g.
   `@all-contributors please add @user for code in TinyTorch, Slides`. Multiple
   projects in one comment are supported.
2. PR file paths — if the PR only touches files under one top-level project
   directory (`tinytorch/...`, `labs/...`, etc.), that project is used.
3. Issue labels or title (matched against project names and aliases).

If detection fails or is ambiguous, the workflow replies asking the user to
specify the project explicitly.

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
2. Add entry with: login, name, avatar_url, profile, contributions
3. Push to dev/main to trigger the update workflow

**Avatar URLs:** For anyone with a GitHub account, set `avatar_url` to `https://avatars.githubusercontent.com/{login}` (and `profile` to `https://github.com/{login}`). Do **not** use `gravatar.com/...?d=identicon` for GitHub users—that shows a generic pattern, not their GitHub profile photo. Gravatar fallbacks are only for contributors who have no GitHub username.

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
    ├── projects.json             # ⭐ Single source of truth
    ├── projects.py               # Loader + CLI for projects.json
    ├── requirements.txt          # Python dependencies
    ├── update_contributors.py    # GitHub API updater
    ├── generate_main_readme.py   # Main README generator (reads projects.json)
    ├── generate_readme_tables.py # Per-project README generator (reads projects.json)
    └── scan_contributors.py      # Git history scanner

Project configs (path = on-disk directory, project key in parens):
├── .all-contributorsrc              # Root config (legacy / aggregate)
├── book/.all-contributorsrc         # book
├── tinytorch/.all-contributorsrc    # tinytorch
├── kits/.all-contributorsrc         # kits
├── labs/.all-contributorsrc         # labs
├── mlsysim/.all-contributorsrc      # mlsysim
├── interviews/.all-contributorsrc   # staffml  (directory ≠ project key)
├── slides/.all-contributorsrc       # slides
└── instructors/.all-contributorsrc  # instructors
```
