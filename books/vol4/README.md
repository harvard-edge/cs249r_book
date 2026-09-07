# Volume IV: Physical AI Systems

*Sensing, perception, world models, control, robotics, embodiment, real-time constraints, safety, and hardware.*

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Development-d73a49?style=for-the-badge" alt="Status: In Development">
  <a href="https://github.com/harvard-edge/cs249r_book/issues/new?template=book-vol4.yml"><img src="https://img.shields.io/badge/Feedback-Open%20an%20Issue-0969da?style=for-the-badge&logo=github" alt="Give feedback"></a>
</p>

> [!CAUTION]
> **This volume is not finished. I am still writing it.**
> Treat everything here as a working notebook, not a preview release or a settled plan. The scope, terminology, chapter order, and technical claims may all change substantially. Please do not cite it, teach from it, or redistribute excerpts as if it were a released book. Comments and suggestions are welcome, with the understanding that this is early-stage work.

---

## What this book is about

When a cloud server makes a mistake, it drops a packet or returns an error. When a physical machine makes a mistake, it collides. Volume IV is about what happens when machine learning crosses that boundary: when the model's output moves a motor, and computational delay becomes physical distance.

The goal is to teach the systems engineering of machines that sense and act. That means the sensors and perception that build a picture of the world, the world models and memory that keep it current, the planning and control that turn intent into motion under hard real-time deadlines, the safety mechanisms that bound what the machine can do, and the hardware that all of this has to fit on. Physics sets the constraints, so the book works from physical invariants rather than from any particular robot or software stack.

A few things this book is deliberately not:

- A robotics textbook. Kinematics, dynamics, and control theory appear where they constrain the ML system, and the book points to the standard references for the rest.
- A survey of particular robots, simulators, or software stacks.
- A finished argument. Several chapters are still deciding what they claim.

## Who it is for

Readers who have worked through Volume I and Volume II, or who are comfortable with single-machine and distributed ML systems at that level. Some familiarity with basic physics and control (forces, inertia, feedback loops) helps, and the appendices review the notation the book uses.

## Where things stand

The working outline has four parts and seventeen chapters. Expect it to change.

1. **The machine anatomy.** The causal boundary, the physical body, the cognitive brain, and the real-time nervous system.
2. **Teaching the machine.** Physical data collection, policy training, and closed-loop evaluation.
3. **Running the machine.** Perception, spatial memory, grounded intent, trajectory planning, safety enforcement, and silicon placement.
4. **Governing the machine.** Supervisory intervention, adversarial verification, deployment release, and the epistemic frontier.

The chapter drafts live in `publishing/quarto/contents/vol4/`.

## Follow along

I write this volume in the open, so every commit and every editorial decision is visible. If something looks rough, that is because you are watching the book being written. To keep up:

- **Watch or star** [the repository](https://github.com/harvard-edge/cs249r_book) to see commits as they land.
- **Subscribe** to the [newsletter](https://buttondown.email/mlsysbook) for occasional updates across the whole series.
- **Build the draft locally** from the `publishing/` directory with `./binder build html --vol4`. Setup instructions are in the [publishing CLI guide](../../publishing/cli/README.md).

## Feedback and contact

I would rather hear about a problem early than after it is in print.

- **Found an error or something unclear?** [Open an issue](https://github.com/harvard-edge/cs249r_book/issues/new?template=book-vol4.yml). The form asks for the chapter and lets you describe what you found.
- **Have a broader question or suggestion?** Start a thread in [Discussions](https://github.com/harvard-edge/cs249r_book/discussions).
- **Want to reach me directly?** I am [@profvjreddi](https://github.com/profvjreddi) on GitHub.

## License and citation

Like the rest of the series, this volume is released under [CC BY-NC-SA 4.0](../../LICENSE.md). You may read, share, and adapt it for non-commercial use with attribution. Because the text is still changing, please do not cite it yet. A citation entry will be added when the volume reaches a stable release.

## Related

- [Machine Learning Systems series overview](../../README.md)
- [Volume I: Introduction to Machine Learning Systems](../vol1/README.md) (released)
- [Volume II: Scaling Machine Learning Systems](../vol2/README.md) (preview)
- [Volume III: Agentic Machine Learning Systems](../vol3/README.md) (in development)
