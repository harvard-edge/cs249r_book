# Essay Series — Road Ahead

The essay series builds one cumulative argument: AI engineering is a real discipline,
and the people who understand how AI systems actually work — from silicon to serving —
will define the next era of technology.

Each essay stands alone. A new reader can pick up any one and get value. But for a
subscriber reading in order, there is a deliberate arc.

---

## The Arc

| # | Title | Core shift | Forwarding sentence |
|---|-------|-----------|-------------------|
| 1 | **The Shift Toward AI Engineering** (Jan 2026) | The field needs to exist. Enrollment data + $1T infrastructure + the gap between using AI and building AI systems. | "This is the data behind what we've both been feeling." |
| 2 | **The Model Is Not the Product** (Mar 2026) | See the system, not just the model. D-A-M taxonomy. DeepSeek + Meta as proof that constraints drive architecture. | "This is why we keep breaking things in production." |
| 3 | **The Builder's Gap** (target: Apr 2026) | The deepest divide in technology: builders vs. users. The stack is one co-designed system. Everyone can use PyTorch; almost nobody can build it. | "This explains the hiring problem we keep talking about." |
| 4 | **When One Machine Isn't Enough** | Single-node to fleet scale. Everything from Essay 3 multiplied by 10,000 machines. Failure as statistical certainty. Network as the new bus. Vol 2 territory. | "Now I understand why distributed training is a different field." |
| 5 | **Benchmarks Are Lying to You** | Accuracy-only evaluation → systems-aware metrics. Latency, throughput, cost-per-token, energy. D-A-M says measure all three axes, not just one. | "We've been measuring the wrong thing this whole time." |
| 6 | **The Hardware Lottery Returns** | GPU monoculture → accelerator diversity. TPUs, Trainium, Groq, edge NPUs. Why AI engineers need architectural literacy, not just CUDA skills. | "The next bottleneck isn't software — it's which chip you pick." |
| 7 | **Inference Is Eating the World** | Training-centric → serving-centric. Models are commoditizing; serving them at scale is the hard (and expensive) problem now. | "Training got all the glory. Inference is where the money goes." |

---

## How the Three Tracks Reinforce Each Other

The essays don't live in isolation. They form a triangle with Community Spotlights
and Updates:

**Essays** provide the IDEAS:
- "AI Engineering is about builders, not users"

**Community Spotlights** provide the PROOF:
- Colombian students building TinyML rehab systems = real builders
- Educators in the Global South teaching with $30 kits under real constraints

**Updates** provide the TOOLS:
- New TinyTorch module = a way to close your own builder's gap
- New textbook chapter = depth on one of the layers the essay introduced
- Hardware kit program = the physical entry point

A subscriber getting all three sees a coherent story: this is the field, these are
the people building it, and here's how you can join.

---

## Design Principles for Every Essay

1. **Universal entry, ML proof.** The thesis must work for a non-ML reader.
   The ML examples are the sharpest available evidence, not the entry point.

2. **One surprising number.** Every essay needs at least one number that
   makes the reader stop. Essay 1: $1.15 trillion. Essay 2: 5% (Google).
   Essay 3: ~15 layers between loss.backward() and the GPU.

3. **One callback.** Each essay references a concept from a previous essay,
   rewarding the subscriber who reads in order. Essay 2 called back to
   Essay 1's "AI Engineering" framing. Essay 3 calls back to Essay 2's
   D-A-M taxonomy and DeepSeek.

4. **One bridge to the book.** Not a pitch — a natural "if you want to go
   deeper, this is where the knowledge lives" moment. Essay 1: the book
   fills the gap. Essay 2: the D-A-M taxonomy comes from the book.
   Essay 3: TinyTorch lets you build the framework yourself.

5. **One forward tease.** The closing paragraph sets up the next essay
   so the reader wants to come back.

### The Drift Warning (from reviewer feedback)

Essays 1-3 have real momentum because they are personal and opinionated.
The risk is essays 4-7. If they drift into survey territory ("here are the
types of distributed training"), subscribers will notice and leave. Every
essay must have a THESIS that the author is willing to defend, not just a
TOPIC to cover. The "one surprising number" discipline is the canary: if
you cannot name the number before drafting, the essay is not ready.

Test before drafting any essay: "If I sent this to 100 ML engineers, would
at least 10 of them disagree with my thesis?" If no one would disagree,
the thesis is too safe to be interesting.

---

## What Success Looks Like

Someone reads Essay 3 and forwards it to:
- Their engineering manager ("this is why we can't hire")
- A colleague in a different field ("this applies to us too")
- A student deciding what to learn ("this is the gap to close")

The person who receives it subscribes because the essay made them feel
like they understood something they couldn't see before.

Over 6-12 months, this subscriber reads 3-4 essays → sees community
spotlights of real people using the materials → sees an update about
a new chapter or TinyTorch module → clicks through to mlsysbook.ai →
becomes a reader, adopter, or contributor.

That's the funnel. The essays are the top.
