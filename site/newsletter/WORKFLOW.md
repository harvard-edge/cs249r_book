# Newsletter Workflow

How to write, review, publish, and archive a newsletter issue.

## Where things live

```
site/newsletter/
├── index.qmd              # Landing page (tabs: All, Essays, Community, Hands On)
├── newsletter.css         # Landing page styles
├── WORKFLOW.md            # This file
├── posts/                 # Published issues (in git, rendered on site)
│   ├── 2025/
│   └── 2026/
├── drafts/                # Work-in-progress (in git on feature branches only)
│   ├── _template.md
│   ├── figures/           # SVG and PNG sources for drafts
│   └── ESSAY_ROADMAP.md   # Series planning doc
└── images/                # Shared reusable figures
```

## The four stages

### 1. Draft

Create a new file in `drafts/` named `essay-NN-slug.md` (e.g. `essay-03-builders-gap.md`). Use `_template.md` as a starting point.

Frontmatter must include:

```yaml
---
title: "Your Title"
date: "YYYY-MM-DD"          # placeholder OK, use YYYY-MM-DD when ready
author: "Vijay Janapa Reddi"
categories: ["essay"]        # essay, community, or hands-on
description: "One or two sentences for the listing card."
draft: true                  # critical: hides from listing until you remove it
image: "figures/your-hero.svg"
---
```

Put figures for this draft in `drafts/figures/`. Name them to match the essay, e.g. `essay-03-stack-overlay.svg`.

### 2. Review

Work on a feature branch named `feat/newsletter-essay-NN-slug`. Commit drafts to that branch. Because `draft: true` is in the frontmatter, Quarto will not render the draft to the public listing even if the branch accidentally gets merged.

Iterate openly. Use feedback agents, human reviewers, whatever you need. The feature branch is your backup and your iteration history.

### 3. Publish

When the issue is ready to send:

1. **Finalize content.** Remove `draft: true` from frontmatter. Set the real date.

2. **Move the file** from `drafts/` to `posts/YYYY/` with a date-stamped name:
   ```
   git mv site/newsletter/drafts/essay-03-builders-gap.md \
          site/newsletter/posts/2026/2026-04-15_the-builders-gap.md
   ```

3. **Move figures.** Published issues currently host figures externally on Buttondown's CDN. Two options:
   - **Option A (current practice):** Upload SVGs/PNGs to the Buttondown asset manager, copy the CDN URL (e.g. `https://assets.buttondown.email/images/xxx.png`), and replace the relative image paths in the markdown.
   - **Option B (future):** Move figure sources into `site/newsletter/images/` or per-post `posts/YYYY/figures/` and reference them with relative paths. Cleaner, but requires the site build to serve them.

4. **Send via Buttondown.** Either paste the rendered HTML into the Buttondown editor, or use the Buttondown API (see section below).

5. **Merge the feature branch to dev.** The `git mv` commit is what graduates the draft to a published post. Once merged, the post renders on the site listing.

### 4. Archive

The published post lives in `posts/YYYY/`. It is rendered by Quarto into the site at `mlsysbook.ai/newsletter/`. Buttondown hosts the email archive at `buttondown.com/mlsysbook/archive/`.

## Directory discipline

| Directory | Git-tracked? | Visible on site? |
|-----------|-------------|-----------------|
| `posts/YYYY/` | Yes, on `dev`/`main` | Yes |
| `drafts/` (except drafts marked `draft: true`) | Yes, on feature branches only | No (hidden by `draft: true`) |
| `drafts/figures/` | Yes, on feature branches only | No |
| `images/` | Yes | Referenced by posts |

## Buttondown as the "send" layer

Buttondown is the email delivery platform. The repository is the source of truth for the newsletter *content*. Buttondown is the source of truth for the email *delivery* — who got it, when, analytics, unsubscribe state.

The clean division of labor:
- **Repo owns**: content, figures, edit history, public archive on the website
- **Buttondown owns**: sending, subscriber list, email analytics, asset CDN

## Is there a better flow?

**Current flow (manual):**
1. Write markdown in `drafts/`
2. Render or copy to Buttondown composer
3. Upload images to Buttondown, swap URLs
4. Send
5. `git mv` the markdown to `posts/YYYY/` with the CDN URLs embedded

**Friction points:**
- Double-handling between the markdown and the Buttondown composer
- Manual image upload and URL swap
- Published markdown diverges from draft markdown (CDN URLs replace local paths)

**Proposed improvement (future):** use the [Buttondown API](https://docs.buttondown.email/api-reference/introduction) to push drafts directly. A small script could:
- Read the markdown + frontmatter
- Upload local figures and get back CDN URLs
- Rewrite image paths in the markdown
- Create a Buttondown draft via API
- On send confirmation, `git mv` the markdown to `posts/YYYY/`

This removes the double-handling and makes publishing a one-command operation. Worth building if the cadence picks up.

**Recommended for now:** stick with the manual flow and focus on writing. Automate only when the cadence justifies it.

## Using `draft: true` correctly

Quarto respects `draft: true` in frontmatter: the file is rendered for you to preview locally but is excluded from the published listing. This means:

- A draft can sit in `posts/YYYY/` with `draft: true` and stay hidden
- A draft can sit in `drafts/` with `draft: true` and stay hidden
- Either works — the convention is `drafts/` for unfinished work and `posts/` for published work

Always remove `draft: true` only when you are ready to send. That flag is the gate.
