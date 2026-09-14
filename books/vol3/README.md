# Volume III: Agentic Machine Learning Systems

*The systems engineering of inference-time compute and autonomous control loops.*

> [!NOTE]
> **In development.** This volume is being written now and changes quickly as I iterate. Chapters are added, reorganized, and rewritten often, so please do not cite or teach from it yet. Feedback is welcome through the [book feedback issue forms](https://github.com/harvard-edge/cs249r_book/issues/new/choose).

---

## About This Volume

Volume III covers machine learning systems that spend inference-time compute in a loop with state and tools. When a model plans, acts, observes the result, and tries again, the unit of engineering stops being a single served request and becomes the trajectory: the whole sense, decide, act, and observe arc a system runs to reach a goal.

The volume treats that loop as a computer system. It covers the model as a stochastic processor, context and memory as a hierarchy with locality and eviction, checkpointing and scheduling of long-running work, the actuation boundary and its isolation, the data and training pipelines that improve agents over time, and the observability and economics of running them.

## Structure

This volume follows the content layout shared by all four volumes:

```
vol3/
├── index.qmd              volume home page
├── frontmatter/           author note, about, notation
├── parts/                 part openers and part summaries
├── NN_chapter/            one directory per chapter: NN_chapter.qmd plus images/, data/, scripts/
├── appendices/            reference appendices
└── backmatter/            references, math appendix, glossary
```

Chapter order comes from `books/config/_quarto-pdf-vol3.yml`, and `books/shared/STRUCTURE.md` lists it for every volume.
