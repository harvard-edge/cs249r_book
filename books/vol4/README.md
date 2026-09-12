# Volume IV: Physical AI Systems

*Machine learning systems that sense and act in the physical world.*

> [!NOTE]
> **In development.** This volume is being written now and changes quickly as I iterate. Chapters are added, reorganized, and rewritten often, so please do not cite or teach from it yet. Feedback is welcome through the [book feedback issue forms](https://github.com/harvard-edge/cs249r_book/issues/new/choose).

---

## About This Volume

Volume IV covers machine learning systems whose outputs are physical actions. It begins at the causal boundary, the point where a command becomes current in a winding and mass begins to accelerate, and follows the consequences of that boundary through the design of the body, the brain, and the nervous system that connects them.

The governing constraint is irreversibility. A mispredicted token costs a retry. A mistimed motor command delivers kinetic energy into an environment that will not roll back. Every design decision in this volume is shaped by that asymmetry.

## Structure

This volume follows the content layout shared by all four volumes:

```
vol4/
├── index.qmd              volume home page
├── frontmatter/           author note, about, prerequisites, syllabus, notation
├── parts/                 part openers and part summaries
├── NN_chapter/            one directory per chapter: NN_chapter.qmd plus its images and data
└── backmatter/            references, appendices, glossary
```

Chapter order comes from `books/config/_quarto-pdf-vol4.yml`, and `books/shared/STRUCTURE.md` lists it for every volume.
