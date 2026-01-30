# Labs

**Understanding the Interplay Between Algorithms and Systems**

> **Status:** Coming Summer 2026

---

## What Are Labs?

Labs are hands-on interactive notebooks that bridge the gap between **reading about ML systems** (the textbook) and **building them from scratch** (TinyTorch).

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│    Textbook     │────▶│      Labs       │────▶│    TinyTorch    │
│                 │     │                 │     │                 │
│  Concepts &     │     │  Experiment &   │     │  Build from     │
│  Theory         │     │  Explore        │     │  Scratch        │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
      READ                    EXPLORE                  BUILD
```

## The Learning Journey

| Phase | Resource | What You Do |
|-------|----------|-------------|
| **Understand** | [Textbook](https://mlsysbook.ai) | Learn concepts, theory, and system design principles |
| **Experiment** | Labs | Explore tradeoffs, tweak parameters, see how decisions ripple through systems |
| **Build** | [TinyTorch](https://mlsysbook.ai/tinytorch) | Implement everything from scratch, own every line of code |

## Why Labs?

ML systems are where algorithms meet hardware. A model that works perfectly in theory can fail in practice due to memory limits, latency constraints, or numerical precision. Labs help you develop intuition for these algorithm-system interactions.

- **See the tradeoffs** — How does batch size affect memory? How does quantization affect accuracy?
- **Explore interactively** — Adjust parameters and watch how changes ripple through the system
- **Build intuition** — Understand *why* systems behave the way they do, not just *what* they do
- **Zero setup** — Run directly in your browser via Google Colab

## Example Topics (Planned)

- **Memory vs. Compute Tradeoffs** — Watch how batch size affects memory footprint and training speed
- **Quantization Effects** — See accuracy degradation as you reduce precision from FP32 → INT8 → INT4
- **Attention Visualization** — Explore what transformer attention heads actually learn
- **Optimization Landscapes** — Navigate loss surfaces with different optimizers
- **Pruning Strategies** — Compare structured vs. unstructured pruning on real models

## Stay Updated

Labs are under active development. To be notified when they launch:

- [Subscribe to updates](https://buttondown.email/mlsysbook)
- [Star the repo](https://github.com/harvard-edge/cs249r_book)
- [Join discussions](https://github.com/harvard-edge/cs249r_book/discussions)

---

## Related Resources

| Resource | Description |
|----------|-------------|
| [Textbook](https://mlsysbook.ai) | ML Systems principles and practices |
| [TinyTorch](https://mlsysbook.ai/tinytorch) | Build your own ML framework from scratch |
| [Discussions](https://github.com/harvard-edge/cs249r_book/discussions) | Ask questions, share feedback |

---

## Contributors

Thanks to these wonderful people who helped build the labs!

**Legend:** 🪲 Bug Hunter · ⚡ Code Warrior · 📚 Documentation Hero · 🎨 Design Artist · 🧠 Idea Generator · 🔎 Code Reviewer · 🧪 Test Engineer · 🛠️ Tool Builder

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on any issue or PR:
```
@all-contributors please add @username for code, tutorial, test, or doc
```

---

<div align="center">

**Read. Explore. Build.** *(Labs coming soon)*

</div>
