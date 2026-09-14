# Book Configurations

Quarto configurations for every volume and output format.

| Files | Purpose |
|---|---|
| `_quarto-html-vol<N>.yml` | Website build for each volume |
| `_quarto-pdf-vol<N>.yml` | PDF build for each volume; its chapter list is the canonical reading order |
| `_quarto-epub-vol<N>.yml` | EPUB build for each volume |
| `_quarto-pdf-vol<N>-copyedit.yml` | PDF variants used for publisher copyedit rounds (Volumes I and II) |
| `shared/` | Fragments the per-volume configurations include |

Quarto only reads `books/_quarto.yml`. The binder copies the configuration for the requested volume and format there before each build, so make changes here rather than in that copy. Paths inside these files are relative to `books/`.
