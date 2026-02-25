# Book contents layout

**Do not assume content lives only under `vol1/` and `vol2/`.** This directory has shared and front matter that both volumes depend on. Deleting or moving them will break builds.

## Directory structure

| Path | Purpose |
|------|---------|
| **`shared/`** | Content used by **both** Volume I and Volume II (e.g. notation, conventions). |
| **`frontmatter/`** | Site-level front matter (about, acknowledgements, Socratiq, etc.). |
| **`index.qmd`** | Root book index (under `frontmatter/` in some configs). |
| **`vol1/`** | Volume I: Introduction to Machine Learning Systems. |
| **`vol2/`** | Volume II: Machine Learning Systems at Scale. |

## Required non-volume paths

- **`shared/`** — Must exist. Contains `notation.qmd` and any other shared reference material.
- **`frontmatter/`** — Must exist. Contains about, acknowledgements, and other front matter.

Scripts and tooling that iterate over “all content” should include `shared/` and `frontmatter/` (and the root `index.qmd` where applicable), not only `vol1/` and `vol2/`.
