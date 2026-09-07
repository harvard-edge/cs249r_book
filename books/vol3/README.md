# Volume III: Agentic Machine Learning Systems

*Reasoning and acting loops, memory, tools, planning, search, orchestration, evaluation, security, and multi-agent systems.*

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Development-d73a49?style=for-the-badge" alt="Status: In Development">
  <a href="https://github.com/harvard-edge/cs249r_book/issues/new?template=book-vol3.yml"><img src="https://img.shields.io/badge/Feedback-Open%20an%20Issue-0969da?style=for-the-badge&logo=github" alt="Give feedback"></a>
</p>

> [!CAUTION]
> **This volume is not finished. I am still writing it.**
> Chapters are incomplete, the structure will change, and the technical claims have not yet been through the review that Volumes I and II received. Please do not cite it, teach from it, or redistribute excerpts as if it were a released book. You are welcome to read along as it takes shape, and if you spot something wrong, I want to hear about it.

---

## What this book is about

Volumes I and II are about producing a good answer to a single request, first on one machine and then across a fleet. Volume III asks what changes when the model stops answering and starts acting: when it runs in a loop, keeps state across steps, calls tools, and pursues a goal over many steps before anyone checks the result.

The goal is to teach the systems engineering behind those loops. That means the memory hierarchy that holds an agent's context, the caching and scheduling that make long trajectories affordable, the interfaces and sandboxes that let an agent act without causing harm, the verification that catches errors before they compound, and the coordination and telemetry needed to run many agents at once. The emphasis is on durable principles and physical trade-offs, not on any particular framework or prompting recipe.

A few things this book is deliberately not:

- A guide to any particular agent framework or orchestration library. Those change every quarter; the constraints underneath them do not.
- A collection of prompt recipes or templates.
- A book about agent psychology. Reflection, planning, and memory are treated as runtime mechanisms with costs, not as metaphors.

## Who it is for

Readers who have worked through Volume I and Volume II, or who are comfortable with single-machine and distributed ML systems at that level. The book assumes you know how inference serving, KV caching, and accelerator memory hierarchies work, and builds the agent runtime on top of that.

## Where things stand

The working outline has five parts. Expect names, order, and scope to change.

1. **Foundations.** What a trajectory is, how it is described, and how control flows through it.
2. **Training and adaptation.** Fine-tuning on trajectories, reinforcement learning, and test-time search.
3. **Serving and memory.** Context as a working set, prefix caching and paging, and trajectory scheduling.
4. **Security and isolation.** Tool interfaces, sandboxing, and verification and recovery.
5. **Scale and operations.** Multi-agent coordination, telemetry, and evaluation.

The chapter drafts live in `publishing/quarto/contents/vol3/`. The detailed chapter-by-chapter plan is in [`OUTLINE.md`](OUTLINE.md), and it tracks the same chapter list as the build.

## Follow along

I write this volume in the open, so every commit and every editorial decision is visible. If something looks rough, that is because you are watching the book being written. To keep up:

- **Watch or star** [the repository](https://github.com/harvard-edge/cs249r_book) to see commits as they land.
- **Subscribe** to the [newsletter](https://buttondown.email/mlsysbook) for occasional updates across the whole series.
- **Build the draft locally** from the `publishing/` directory with `./binder build html --vol3`. Setup instructions are in the [publishing CLI guide](../../publishing/cli/README.md).

## Feedback and contact

I would rather hear about a problem early than after it is in print.

- **Found an error or something unclear?** [Open an issue](https://github.com/harvard-edge/cs249r_book/issues/new?template=book-vol3.yml). The form asks for the chapter and lets you describe what you found.
- **Have a broader question or suggestion?** Start a thread in [Discussions](https://github.com/harvard-edge/cs249r_book/discussions).
- **Want to reach me directly?** I am [@profvjreddi](https://github.com/profvjreddi) on GitHub.

## License and citation

Like the rest of the series, this volume is released under [CC BY-NC-SA 4.0](../../LICENSE.md). You may read, share, and adapt it for non-commercial use with attribution. Because the text is still changing, please do not cite it yet. A citation entry will be added when the volume reaches a stable release.

## Related

- [Machine Learning Systems series overview](../../README.md)
- [Volume I: Introduction to Machine Learning Systems](../vol1/README.md) (released)
- [Volume II: Scaling Machine Learning Systems](../vol2/README.md) (preview)
- [Volume IV: Physical AI Systems](../vol4/README.md) (in development)
