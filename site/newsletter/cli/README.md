# news — Newsletter CLI

Command-line interface for authoring and publishing the ML Systems newsletter. Do everything from the terminal: draft in the repo, push to Buttondown for preview, send from the Buttondown UI, pull the sent version back into the repo.

## Install

```bash
# Dependencies (two options)
pip install -r site/newsletter/requirements.txt      # quick
pip install -e site/newsletter                        # editable, registers `news` on PATH

# API key
cp site/newsletter/cli/.env.example site/newsletter/cli/.env
# edit .env with your key from buttondown.com/settings/programming
```

Optional alias if you did not do the editable install (run from your clone root; adjust the path if your repo lives elsewhere):

```bash
echo "alias news=$(pwd)/site/newsletter/bin/news" >> ~/.zshrc
```

## Commands

```
news                            # welcome screen, commands grouped by category
news <command> --help           # per-command help

# Drafting
news new <slug>                 # scaffold a new draft from the template
                                # --category essay|community|hands-on

# Inspection
news list                       # drafts and recently published posts in the repo
news check <slug>               # preflight: frontmatter, figures, deps, auth
news status                     # Buttondown-side: drafts and recently sent
news open [slug]                # open Buttondown in the browser (--archive for public)
news diff <slug>                # diff a local draft against its Buttondown version

# Publishing
news push <slug>                # upload draft + figures, create Buttondown draft
                                # --dry-run | --skip-checks
news pull [email-id]            # sync sent emails from Buttondown into posts/YYYY/
                                # --since YYYY-MM-DD | --dry-run | --force
                                # --category essay|community|hands-on|update

# Archiving
news archive <slug>             # fallback: manually move a draft to posts/YYYY/
                                # --date YYYY-MM-DD | --slug | --dry-run
```

## The Workflow

```bash
# 1. Start a new essay
news new essay-04-fleet-scale

# 2. Write and iterate in site/newsletter/drafts/ — commit freely to a
#    feature branch. `draft: true` keeps it invisible on the public site.

# 3. Preflight
news check essay-04-fleet-scale

# 4. Push to Buttondown
news push essay-04-fleet-scale

# 5. Preview in Buttondown (news open will take you there)
news open essay-04-fleet-scale

# 6. Send from the Buttondown UI. Keeps a human in the loop for the
#    actual send.

# 7. Pull the sent version back into the repo
news pull

# 8. Commit the new post and merge the feature branch to dev
git add site/newsletter/posts/ && git commit -m "publish: fleet scale"
```

`news archive` remains as a manual fallback for when you prefer to move a draft yourself rather than pull the Buttondown version. With `pull`, the archive flow is optional.

## Design

Architecture mirrors the Tito CLI:

- `cli/main.py` — command registry in one dict (single source of truth)
- `cli/commands/` — one file per subcommand, each a `BaseCommand` subclass
- `cli/core/` — shared utilities: `theme.py`, `console.py`, `config.py`, `buttondown.py`, `validate.py`
- `bin/news` — wrapper script that runs without pip install

Rich handles all colored output via the `Theme` class in `cli/core/theme.py`. Global options `-v/--verbose`, `-q/--quiet`, `--version`, and the `NO_COLOR` environment variable work across every subcommand.
