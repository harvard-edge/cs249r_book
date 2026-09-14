# Books

The sources for the *Machine Learning Systems* textbook series. This folder is the Quarto project root for all four volumes.

| Volume | Title | Status |
|---|---|---|
| [`vol1/`](vol1/) | Introduction to Machine Learning Systems | Released ([read online](https://mlsysbook.ai/vol1/)) |
| [`vol2/`](vol2/) | Machine Learning Systems at Scale | Preview ([read online](https://mlsysbook.ai/vol2/)) |
| [`vol3/`](vol3/) | Agentic Machine Learning Systems | In development, changing quickly |
| [`vol4/`](vol4/) | Physical AI Systems | In development, changing quickly |

## What lives here

```
books/
├── vol1/ … vol4/          one folder per volume, one NN_chapter/ folder per chapter
├── shared/                material more than one volume uses (front and back matter, TeX, filters, assets)
├── config/                Quarto configurations for each volume and output format
├── _extensions/           Quarto extensions (they must sit at the project root)
├── index-vol1.qmd …       per-volume landing pages for the website
├── references*.bib        bibliographies (Volumes I and II share references.bib)
└── _build/                render output; gitignored and safe to delete
```

A chapter lives at `books/vol<N>/<NN_chapter>/<NN_chapter>.qmd`, and there is no other copy of it.

## Building

Builds and checks run through the binder CLI from the repository root:

```bash
./binder/binder build html --vol1
./binder/binder build pdf --vol2
./binder/binder check refs
```

The binder copies the chosen file from `config/` to `books/_quarto.yml` before each build, so edit the files in `config/`, not the copy.

See [`docs/REPO_LAYOUT.md`](../docs/REPO_LAYOUT.md) for how this folder fits into the rest of the repository.
