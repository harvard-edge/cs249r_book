# ML Systems Taxonomy changelog

Versions follow `MAJOR.MINOR-draft` until the first reviewed release.

**Identifier policy.** Concept identities (`mlsys.kv-cache`) are permanent: they are never renamed or reused. A concept that is merged or withdrawn is listed under `retired` in `shared/concepts/ml-systems-field.yml` with the identity that replaces it, and links to it redirect. Readable codes (`P4.5`) are stable within a minor version and may be renumbered in a new minor or major version; tools should store identities, not codes. Relation identities (`limits`, `reduces`) follow the same rule as concepts.

## 0.2-draft (October 2026)

- New organizing frame: five groups (Foundations, Stack, Lifecycle, Qualities, System types), 20 areas, 99 subjects, each with a nested readable code.
- Concept vocabulary grown from 175 to 692 concepts, with background concepts from adjacent fields tagged.
- Relationships rebuilt for concept maps: 1,443 relationships, each using one of 17 controlled relations with forward and inverse phrasing; every concept has at least two; the graph is connected. The relation `bounds` was renamed `limits` before release.
- Coverage is measured with `binder concepts coverage` against recorded checklists in `shared/concepts/coverage/`, separating design inputs from held-out tests.
- First held-out measurement, before closing the gaps it found (concept coverage / subject home): MLSys 2025–2026 papers 67.2% / 97.2%; MLSysBook Volume I–II primary concepts 85.1% / 99.5%; Volume III–IV 69.6% / 100%. After adding the field-level concepts those tests surfaced, the same sets read 87.6%, 88.7%, and 83.0% concept coverage and now serve as design inputs; book-specific framing devices were left to the books. The next held-out test is the MLSys 2027 program.
- Cross-model review (one GPT and one Claude reviewer, both "yes with fixes") led to: CPU execution, dynamic shapes, graph rewriting, sparse and ragged tensors, streaming and serverless inference, LLM request scheduling, GNN partitioning, licensing and provenance, robotics middleware, and a fleet data engine (24 concepts); the serving subject split into Serving architectures and Compound AI & routing; diffusion inference moved under generative inference, streaming speech under serving, in-context learning under adaptation, and neighbor sampling under parallelism; Lifecycle codes changed from L to P so codes never read as StaffML depth levels; tagging rules published; `mlsys.llm-kernel-generation` retired in favor of `mlsys.autotuning` (using ML to optimize systems is out of scope).
- Public downloads: subjects and concepts CSV, JSON, SKOS Turtle, GraphML, BibTeX.

## 0.1-draft

- Initial candidate framework: 12 areas, 56 rows, 175 concepts, 195 relationships.
