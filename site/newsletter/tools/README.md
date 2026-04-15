# Newsletter Publishing Tools

Small Python tools that bridge the GitHub-based drafting workflow with Buttondown, the email delivery platform.

## One-time setup

1. Get a Buttondown API key from [buttondown.com/settings/programming](https://buttondown.com/settings/programming).

2. Copy `.env.example` to `.env` in this directory and fill in your key:

   ```bash
   cp site/newsletter/tools/.env.example site/newsletter/tools/.env
   # then edit .env to add your key
   ```

   `.env` is gitignored. Your key never gets committed.

3. Install the Python dependencies (one-time):

   ```bash
   pip install requests python-frontmatter
   ```

## `publish.py push` — upload a draft to Buttondown

The core command. Take a draft markdown file, upload all its local figures to Buttondown's CDN, and create a draft email in Buttondown that renders with all images in place.

```bash
python3 site/newsletter/tools/publish.py push \
    site/newsletter/drafts/essay-03-builders-gap.md
```

What happens:

1. The script reads the markdown and frontmatter.
2. Every local image reference (e.g. `figures/essay-03-stack-overlay.svg`) is uploaded to Buttondown's image endpoint. The returned CDN URL replaces the local path in the body.
3. A Buttondown email is created with `status: draft`, the frontmatter `title` as subject, and the rewritten body.
4. The script prints the URL you can open to preview, finalize, and send from the Buttondown UI.

You never leave your normal git workflow to compose. Buttondown becomes the preview and send layer, not the authoring layer.

## After the send

Once you have sent the newsletter from Buttondown:

1. The images referenced in your local markdown are now pointing at local files. Update them to the Buttondown CDN URLs that were printed during `push` (or keep a second copy of the post where the URLs are CDN-hosted).

2. Move the markdown from `drafts/` to `posts/YYYY/` with a date-stamped filename:

   ```bash
   git mv site/newsletter/drafts/essay-03-builders-gap.md \
          site/newsletter/posts/2026/2026-04-15_the-builders-gap.md
   ```

3. Remove `draft: true` from the frontmatter. Set the real publish date.

4. Commit and merge the feature branch to `dev`. The post now appears on the site listing.

A future `publish.py archive` subcommand could automate step 2 and 3 by fetching the sent version back from Buttondown and staging the git move for you. Not built yet — straightforward add when the cadence justifies it.

## Troubleshooting

**`BUTTONDOWN_API_KEY is not set`** — check your `.env` file exists in `site/newsletter/tools/` and has the key on a single line.

**`Draft creation failed: 401`** — your API key is wrong or has been rotated. Regenerate at buttondown.com/settings/programming and update `.env`.

**`Image upload failed: 413`** — file too large. Buttondown's image endpoint has a size limit; compress the image and retry.

**SVG not rendering in the Buttondown preview** — some email clients do not render SVG inline. Consider exporting the SVG to PNG for the published version, while keeping the SVG as the source in the repo.
