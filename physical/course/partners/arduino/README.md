# Physical AI Kit — Partner Brief (Arduino)

**From:** Prof. Vijay Janapa Reddi (Harvard / ETH Zurich)
**Contact:** [vj@eecs.harvard.edu](mailto:vj@eecs.harvard.edu)
**Curriculum:** [physical.mlsysbook.ai](https://physical.mlsysbook.ai)
**Repo:** [github.com/harvard-edge/physical-ai](https://github.com/harvard-edge/physical-ai)

> **PDF for partners:** [`Physical-AI-Kit-Arduino-Brief.pdf`](Physical-AI-Kit-Arduino-Brief.pdf)
> Rebuild: `typst compile brief.typ Physical-AI-Kit-Arduino-Brief.pdf`
> Contact on this brief is Harvard only (`vj@eecs.harvard.edu`). Andrea is intentionally omitted.


## Where this sits for Arduino

| Then | Now |
| --- | --- |
| **TinyML Kit** | Deploy a model on a small board |
| **Physical AI Kit** | Stand behind a machine whose learned proposals may **act**—only when independently **permitted** |

UNO Q’s dual-brain (Linux MPU + real-time MCU) is the teaching instrument: **MPU proposes · MCU permits**.


## The ask

Co-create a **Physical AI Kit** on **Arduino UNO Q**—successor story to the TinyML Kit—backed by an open book and a university project course so anyone can follow along.

We are **not** asking Arduino to invent pedagogy. We want a **kit that embodies pedagogy that already exists**.


## What learners practice on the kit

| Primitive | Question |
| --- | --- |
| **Loop** | Does action change the next observation? |
| **Time** | How old may belief be when the move happens? |
| **Budget** | What is shared among sensing, inference, link, actuation? |
| **Permission** | What may refuse a capable proposal? |
| **Evidence** | What measurement supports ship / condition / refuse? |


## Pedagogy already built

| Layer | What it is |
| --- | --- |
| **Book** | *Physical AI: Machine Learning Systems That Sense and Act* — 11 chapters + capstone |
| **Course** | Project seminar / studio · dossier + midterm + defense · [syllabus](../../README.md) |
| **Labs** | Chapter-aligned kit contracts (bring-up → **MCU enforcer** → ship gate) |
| **Baseline** | Quantize / serve → [mlsysbook.ai](https://mlsysbook.ai) (not re-taught here) |

**Signature experience:** hang the MPU; the MCU still holds the physical boundary.


## Kit sketch (TinyML Kit, next generation)

| Include | Why |
| --- | --- |
| Arduino UNO Q (or equivalent dual-brain) | Proposal ≠ permission on one board |
| Camera + motion / actuation path | Real causal loop on the desk |
| Sensors as needed | Freshness, state, fault demos |
| Safe-idle defaults + pin map | Studio safety |
| Starter firmware checkpoints | Failed week ≠ stranded student |
| Link to course + book | Box → `physical.mlsysbook.ai` |

**Box line (suggested)**

```text
PHYSICAL AI KIT
Proposal ≠ permission · Dual-brain UNO Q
Follow the open course: physical.mlsysbook.ai
```


## Partnership shape

| Arduino | Academic / open curriculum |
| --- | --- |
| Kit SKU, store, global access | Book + labs + syllabus |
| Board bring-up quality | Method ownership (proposal–permission spine) |
| Co-marketing: TinyML → Physical AI | Co-marketing: AI engineering sequence |


## Next conversation

1. Kit contents (studio MVP vs retail SKU)
2. Naming: Physical AI Kit · optional TinyAgents flavor
3. Timeline: pilot studio → open follow-along → store
4. Narrative: *TinyML deployed the model; Physical AI stands behind the machine.*

**Forwardable paragraph**

> Harvard/ETH Physical AI Systems is building the educational successor to TinyML: an open book and project course where students engineer dual-brain agents on Arduino UNO Q—Linux intelligence proposes; the real-time MCU independently permits or refuses action. We want to co-design a **Physical AI Kit** so the same pedagogy can ship worldwide the way the TinyML Kit did—bring-up, measurement, signature enforcer lab, and an evidence-backed deploy / condition / refuse decision.

**Next step:** 30-minute working session on kit BOM + out-of-box lab path.
**Email:** [vj@eecs.harvard.edu](mailto:vj@eecs.harvard.edu)
