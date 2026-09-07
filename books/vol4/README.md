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

## Where things stand

The working outline has four parts and seventeen chapters. Expect it to change.

1. **The machine anatomy.** The causal boundary, the physical body, the cognitive brain, and the real-time nervous system.
2. **Teaching the machine.** Physical data collection, policy training, and closed-loop evaluation.
3. **Running the machine.** Perception, spatial memory, grounded intent, trajectory planning, safety enforcement, and silicon placement.
4. **Governing the machine.** Supervisory intervention, adversarial verification, deployment release, and the epistemic frontier.

The chapter drafts live in `publishing/quarto/contents/vol4/`.

## Feedback and contact

I would rather hear about a problem early than after it is in print.

- **Found an error or something unclear?** [Open an issue](https://github.com/harvard-edge/cs249r_book/issues/new?template=book-vol4.yml). The form asks for the chapter and lets you describe what you found.
- **Have a broader question or suggestion?** Start a thread in [Discussions](https://github.com/harvard-edge/cs249r_book/discussions).
- **Want to reach me directly?** I am [@profvjreddi](https://github.com/profvjreddi) on GitHub.

## Related

- [Machine Learning Systems series overview](../../README.md)
- [Volume I: Introduction to Machine Learning Systems](../vol1/README.md) (released)
- [Volume II: Scaling Machine Learning Systems](../vol2/README.md) (preview)
- [Volume III: Agentic Machine Learning Systems](../vol3/README.md) (in development)
