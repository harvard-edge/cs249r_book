# news — Newsletter CLI

Command-line interface for authoring and publishing the ML Systems newsletter.

## Install

```bash
# Dependencies
pip install rich requests python-frontmatter

# API key for Buttondown
cp site/newsletter/cli/.env.example site/newsletter/cli/.env
# then edit .env with your key from buttondown.com/settings/programming
```

Optional: add to PATH so `news` works from anywhere.

```bash
echo 'alias news=/Users/VJ/GitHub/MLSysBook-newsletter/site/newsletter/bin/news' >> ~/.zshrc
```

## Usage

```
news                            # welcome screen, shows all commands by category
news <command> --help           # per-command help

# Drafting
news new <slug>                 # scaffold a new draft from the template
                                # options: --category essay|community|hands-on
news list                       # show all drafts and recent published posts

# Publishing
news push <slug>                # upload draft and figures to Buttondown as a draft email
                                # prints preview URL for you to finalize and send

# Archiving
news archive <slug>             # move a sent draft to posts/YYYY/ with date stamp
                                # options: --date YYYY-MM-DD, --slug, --dry-run

# Inspection
news status                     # show Buttondown drafts and recently sent emails
```

## The Full Workflow

```bash
# 1. Start a new essay
news new essay-04-fleet-scale --category essay

# 2. Write and iterate in site/newsletter/drafts/
#    Commit freely to a feature branch as you go.
#    The `draft: true` frontmatter keeps it invisible on the site.

# 3. When ready, push to Buttondown for preview
news push essay-04-fleet-scale

# 4. Open the printed URL. Preview in Buttondown. Tweak if needed.
#    Send from the Buttondown UI.

# 5. Archive back to the repo
news archive essay-04-fleet-scale

# 6. Commit and merge the feature branch to dev
```

## Design

The architecture mirrors the Tito CLI:

- `cli/main.py` registers commands in a single dict (single source of truth)
- `cli/commands/` — one file per subcommand, each a `BaseCommand` subclass
- `cli/core/` — shared utilities (console, theme, config, Buttondown client)
- `bin/news` — wrapper script that adds the newsletter dir to `sys.path`

Rich handles all colored output. The `Theme` class in `cli/core/theme.py` centralizes semantic colors so a future palette change is one file.
