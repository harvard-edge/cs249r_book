# Review: mechanism boundaries

Status: sparse-attention and interrupt-latency corrections applied through `accepted-sparse-attention.json` and `accepted-interrupt-latency.json`. The memory-entry scope decision remains deferred. This targeted review preserves the existing structure and identifiers. `decisions.json` contains the exact baseline records, proposed field changes, and every incident relationship. The source hash prevents silent reuse against a different vocabulary.

## Evidence and decisions

**Sparse and sliding-window attention.** The existing definition says fixed sparse patterns make attention linear. Child et al., [Sparse Transformers, abstract](https://arxiv.org/abs/1904.10509v1), explicitly gives a sparse construction with complexity proportional to sequence length times its square root. The proposed definition and context-length edge distinguish reduced attention cost from linear cost. Bounded attended positions per query is the analytical condition for the latter, assuming fixed per-pair work. The hierarchy edge remains valid. The cache edge already qualifies its claim to windowed decoding; the alternative-method edges remain provisional.

**Interrupt latency.** Arm's [definition and nested-interrupt discussion](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/beginner-guide-on-interrupt-latency-and-interrupt-latency-of-the-arm-cortex-m-processors) distinguishes request-to-handler latency from handler execution. The proposed definition removes execution time and removes `interrupt service routine budget` as an equivalent alias. The scheduling edge is rewritten around an interrupt-triggered task's event-response budget; this is an analytical consequence, not a timing measurement. The interrupt-driven feedback-delay edge remains valid under its stated condition.

**Memory coalescing and bank conflicts.** NVIDIA's [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/), sections “Coalesced Access to Global Memory” and “Shared Memory and Memory Banks,” treats transaction coalescing and shared-memory bank serialization separately. The current entry deliberately names both, so silently changing its definition to coalescing alone would narrow its extension. Defer that change pending a split/retention decision and review of existing assignments. Its incident edges must then distinguish global-memory transaction efficiency from bank serialization. The exact baseline is retained in the packet; no new concept has been admitted.

Sources were inspected on November 4, 2026. Documentation URLs are mutable; accepted provenance must preserve the relevant source evidence. These sources support the identified distinctions, not all claims in the surrounding subjects.

## Repeating this review

Run `python3 taxonomy/reviews/mechanism-boundaries/verify.py` from the repository root. It verifies the source hash, exact selected records, complete incident-edge inventory, and proposed field names. It does not approve the proposed meanings, establish reviewer independence, or apply changes.

Read each original record alongside its proposed fields, inspect the cited source, and assess all incident edges. Record any disagreement before the governed application step. A further reviewer should also assess the memory-entry scope decision and downstream assignments. The initial screening and this editorial pass used the same model family; they do not establish independent or human validation. The correction policy no longer requires a model quorum.
