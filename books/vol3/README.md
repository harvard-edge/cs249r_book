# Volume III: Agentic Machine Learning Systems

*How much machine a delegated task needs around a foundation model, and how to build, measure, improve, and scale it.*

> [!NOTE]
> **In development.** This volume is being written now and changes quickly as I iterate. Chapters are added, reorganized, and rewritten often, so please do not cite or teach from it yet. Feedback is welcome through the [book feedback issue forms](https://github.com/harvard-edge/cs249r_book/issues/new/choose).

---

## About This Volume

Volume III covers agentic machine learning systems, which put a foundation model in a loop with context, memory, tools, and a runtime, then learn from the trajectories they produce. When a model proposes an action, the runtime carries it out, and the result feeds the next call, the unit of engineering stops being a single served request and becomes the trajectory: the whole record of one delegated task, turn by turn, from goal to verified result.

An agent adds three exposures beyond a single model call, horizon (how many turns it runs), state (what it carries between turns), and authority (what its tools may do). The runtime closes them with mechanical checks below the model and with evidence the model cannot edit. The volume builds that machine in dependency order (model, memory, tools, runtime, measure, learn, scale) and keeps to software agents that act through tools on digital systems.

## Book Map

| Part | Chapters |
|:-----|:---------|
| Introduction | Foundations of Agentic Systems |
| I. The Model | The Foundation Model; Test-Time Compute |
| II. Agent Memory | Context Engineering; KV Cache Management; Long-Term Memory |
| III. Tool Use | Tool Calling; Agent Sandboxes |
| IV. The Agent Runtime | The Agent Harness; Durable Execution; Failure Recovery; Agent Evaluation |
| V. Learning from Trajectories | Trajectory Curation; Trajectory Fine-Tuning; Reinforcement Learning from Verifiable Rewards |
| VI. Agents at Scale | Multi-Agent Coordination; Agent Economics |
| VII. Synthesis | Conclusion |

Evaluation closes Part IV so that nothing is trained before it can be measured. If this table and `books/config/_quarto-pdf-vol3.yml` ever disagree, the config is authoritative.

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
