**TL;DR:** **NEEDS_WORK.** Most of the technical repair held up. Every worked number I recomputed is right, and the H100 and B200 specs match the MLSysIM registry. The chapter still isn't ready to sign off, for three reasons. It breaks the book's rule that every number comes from MLSysIM. It has at least two defects that will break or garble the PDF build. And it has drifted from its own blueprint: the status envelope grew from four outcomes to seven, the KV-cache footprint is worked out here even though the blueprint gives it to Chapter 05, and Section 2.8 claims measured results it never shows.

┌──────────────────────────────────────────────────────────────────────────────┐
│ 🛑 NEEDS_WORK: the numbers are right, but the chapter hardcodes them, won't  │
│    build cleanly, and states benchmark "results" that were never measured.   │
└──────────────────────────────────────────────────────────────────────────────┘

---

**What I checked against the repo (not just the pasted text)**

```diff
+ H100 FP8 1979 TFLOP/s, 3.35 TB/s, 50 MiB L2   → match mlsysim/hardware/data/cloud/H100.yaml
+ B200 FP8 4500 TFLOP/s, 8.0 TB/s                 → match cloud/B200.yaml
+ KV math: 327,680 B/token; 41.94 GB; 39.06 GiB; 5.37 GB @32k FP8     → all correct
+ Decode floors: 20.9 / 26.1 ms (H100), 8.75 / 10.9 ms (B200), +2.0 / +3.3 ms KV → correct
+ MoE: 1-(0.75)^16 = 98.998%; all 8 experts selected (top-2 without replacement) = 92.11%; >98% at B≥21 → correct
+ Bitmask 16,032 B = 15.66 KiB → correct
+ All 13 bib keys exist; all 6 forward @sec- anchors resolve; all 7 SVGs exist on disk
- M4 Max (40 TFLOP/s, 546 GB/s): not in the registry, and the 40 TFLOP/s figure has no source
- Registry lists capacities as 80 GiB / 180 GiB; the chapter says "80 GB" / "180 GB" (conclusion survives: 91 GB > 85.9 GB)
- "513 KiB" logit payload is actually 513 KB = 501 KiB; "over 16 MiB" at B=32 is actually 15.66 MiB
```

I couldn't re-verify the `cl100k_base` token IDs because `tiktoken` isn't installed in this environment. They look plausible, but you should run a one-off check before printing them.

---

**Your four questions**

1. **Were the technical, ABI, and hardware issues resolved?** Mostly. The roofline, weight-traffic bound, MoE batching collapse, and deadline propagation sections are correct and well argued. The problems left are consistency across sections, not physics.
2. **Are the contracts, envelopes, and equations publication-grade?** The equations are. The status envelope isn't yet (details below).
3. **Were false analogies removed?** Largely yes. "BPE as hardware encoding" is gone. Two framing errors are still there: one about who drives the decode loop, and one where the chapter contradicts itself on whether reliability "emerges" (both below).
4. **Sign-off:** NEEDS_WORK.

---

**Blockers**

- `[P0][BUILD]` **The seven-class set equation in §2.5 will break the PDF.** It uses `\texttt{TRANSPORT_FAILURE}` with an unescaped `_` inside display math, which is a LaTeX "Missing $" error. §2.2 escapes the same token correctly (`TRANSPORT\_FAILURE`), so this is a missed spot, not a policy choice.
- `[P0][BUILD]` **Two "figures" are ASCII art in code fences.** `@fig-verification-closure` and `@fig-grammar-constrained-decoding` are fenced code blocks followed by `: caption {#fig-…}`. Quarto reads that as table-caption syntax, not a figure, so the cross-references won't resolve. The box-drawing glyphs also risk missing-font problems in the PDF. `grammar_constrained_decoding_fsm.svg` already exists on disk; use it. The verification figure needs an SVG.
- `[P0][BUILD]` **The inline stop-delimiter code in §2.5 is malformed.** The spans ``` ``\n\n``` `` ``` (in the terminal-delimiter bullet and the suffix-matching paragraph) will render as garbage. The SentencePiece marker `' '` in §2.2 has also lost its U+2581 `▁` glyph.
- `[P0][MLSYSIM]` **Every physical number is hardcoded.** That includes 3,350 GB/s, 26.1 ms, 70 GB, 16,032 B, and the whole "MLSysIM LEGO Roofline Cell". That cell is a plain ```` ```python ```` block with literal specs, not an executable LEGO cell importing from the registry. CLAUDE.md makes this a hard rule. H100 and B200 already exist in the registry. M4 Max would need a registry entry with provenance first, or should be replaced with `MacBookM3Max`, which is already there.
- `[P0][EVIDENCE]` **§2.8 claims results it never measured.** It pins a full protocol (Llama-3-70B, vLLM 0.7.2, TP=2, τ=0), then asserts outcomes such as "achieves the lowest downstream verified task success" and "establishes the Pareto frontier". The table is qualitative. Either run the benchmark or recast §2.8 as a protocol plus hypotheses. The blueprint asks for an *empirical* comparison. A metric is not a verdict, and a finding that was never measured can't ship.

---

**Major (contract and scope coherence)**

- `[P1][BLUEPRINT]` **Seven outcomes instead of four.** The added classes (`REJECTED`/`CANCELLED`/`FAULTED`) are well justified: admission, client abort, and engine fault really are distinct fault domains. But the blueprint's LO and §2.5 still say four. Update the blueprint so it stays the source of truth. Don't let the draft silently diverge.
- `[P1][CONTRACT]` **The status envelope isn't mutually exclusive yet:**
  - `DEADLINE_EXPIRED` (TRUNCATED) overlaps `STREAM_TIMEOUT` (TRANSPORT_FAILURE). The Pitfalls section also mentions a third code, `DEADLINE_EXCEEDED`, filed under CANCELLED. Pick one owner for deadline expiry.
  - The overflow boundary conflicts: §2.2 rejects when `M ≥ S_max`, §2.5 when `M > S_max`. §2.2 also *clamps* `K_max ≤ S_max − M`, while §2.5 *rejects* when `M + K_max > S_max`. That's two different policies.
  - The Pitfalls section routes HTTP 429/503 overload to `REJECTED`, but there's no reason code for it (e.g. `CAPACITY_EXHAUSTED`). There's also no code for a Θ_id or grammar-hash mismatch, even though both digests are in the request.
  - The table drops `UNSUPPORTED_ABI_VERSION`, which the prose lists.
  - `π_overflow = TRUNCATE_HEAD/TAIL` is allowed, but a `COMPLETED` envelope has no field recording that truncation happened.
- `[P1][CONTRACT]` **The release gate (eq-release-gate-predicate) is internally inconsistent:**
  - `EscrowIsolated(y)` is vacuous: under A=0 it is always true.
  - The five-layer verification figure includes tests (Layer 4), but the predicate leaves them out.
  - The text calls the four conditions "jointly sufficient", which overclaims given the matrix's own coverage limits.
- `[P1][SCOPE]` **The KV footprint is worked out in §2.2.** The blueprint's §2.2 says physical KV footprint "belong[s] to Chapter 05 (§5.1)". (§2.5's "Covered in Section 2.2" contradicts that, so the blueprint is inconsistent too.) Recommendation: keep only the per-token traffic term in §2.7, where the decode bound needs it. Cut the 128k capacity and fit discussion from §2.2 and defer it to Chapter 05.
- `[P1][SCOPE]` **§2.8 contradicts §2.7.** §2.7 says "our execution model here remains strictly bounded by the single core" and defers multi-GPU to Chapters 15/17. Then the §2.8 benchmark runs TP=2 on two H100s. Either use FP8 on one H100, or say explicitly that TP is an opaque serving detail.
- `[P1][CORRECTNESS]` **Wrong owner of the decode loop.** §2.3 ends with "the host runtime must externally drive each forward pass". Under your own three-tier model, the *inference service* drives the loop.
- `[P1][CORRECTNESS]` **The NaN-propagation claim is wrong.** The pitfall says NaNs "propagate into subsequent autoregressive steps, corrupting the KV-cache". The sampled token is an integer ID, and the next step's KV entries come from its embedding, so NaN probabilities don't flow into KV. What actually happens is a device-side assert or an arbitrary index (for example, `torch.multinomial` rejects NaN input). The "GPU memory access violations" wording in §2.6 overclaims the same way.
- `[P1][CORRECTNESS]` **The truncated-patch example doesn't support its conclusion:**
  - Its hunk header `@@ -1042,6 +1042,7` has only 5 old lines.
  - Hunk 1 already has a balanced lock/unlock, so "leaving critical locks unreleased / deadlocks" doesn't follow from what's shown.
  - `git apply` is atomic, so a corrupt second hunk rejects the whole patch. The partial-apply hazard is real for GNU `patch` or file-by-file application only. Say that, and label the diff as illustrative.
- `[P1][EDITORIAL]` **Hardcoded chapter names.** "Chapter 5", "Chapter 04", "Chapters 15 and 17", and "Chapter 03 ('The Trajectory Loop')" appear in the text. Chapter 3 is actually *Inference-Time Deliberation*. All of these must be `@sec-` refs.
- `[P1][CONSISTENCY]` **§2.5 contradicts itself on deadlines.** Its closing says "absolute deadline propagation", but the section argues *against* absolute timestamps and for relative remaining-budget propagation.

---

**Minor, and the length problem**

- `[P2]` **The Purpose section and the Summary contradict each other.** The Purpose says reliability "is not an emergent property"; the Summary says "it is an emergent guarantee." Pick one; I'd say *enforced*.
- `[P2]` **Duplicate footnote and term definitions.** Zero Ambient Authority (×2), quarantine invariant (four names: Quarantining / DRAM Quarantine / DRAM Quarantining / Escrow Quarantine), grammar dead-end/NaN (×3), and fail-plausible (×2). Keep one name, one equation label, and one footnote each.
- `[P2]` **Vocabulary range contradiction.** §2.2 says |V| ∈ [32k, 131k]; §2.6 cites `o200k_base` at 200k.
- `[P2]` **Self-assessment claim overstated.** "A distribution cannot independently verify its own samples" is stronger than the blueprint's hedged wording. "Attentional confirmation bias" is presented as a mechanism with no citation. Soften it and cite the self-correction literature.
- `[P2]` **§2.2's table is irreproducible.** The payloads aren't shown, and the 15–40% range comes from a single measurement.
- `[P2]` **Uncited straw man.** "Early analyses suggested…seamlessly" in §2.6 has no citation.
- `[P2]` **Proxy-hop detail crosses into Ch 17.** §2.5's deadline propagation discusses proxy hops; the blueprint defers gateways and proxies to Ch 17.
- `[P2][LENGTH]` **Too long.** At 23k words with ten fallacy/pitfall entries (the blueprint asks for four), much of §2.5 reads like an operations manual: the 13-field ABI, W3C Trace Context, `grpc-timeout` header formats. That fails the book's LLM test. The Pitfalls section also repeats §2.6's material (dead-end, BPE/UTF-8, syntactic divide) almost word for word. Target: cut Fallacies to about 5 and give the ABI one table instead of 13 prose bullets.

---

**What's strong (keep all of this)**

- §2.1 has no subsections, as the blueprint requires, and the three-tier boundary is crisp.
- The §2.7 roofline, KV-traffic, and MoE batching-collapse chain is correct and recomputable. Once it's wired into LEGO it's the chapter's best section.
- Schema forcing, with its renormalization equation and escape-channel schema, is a real contribution.
- The recovery dispatcher that sends each outcome to its own recovery path is the right answer to "Pitfall 2".

`★ Insight ─────────────────────────────────────`
- The MLSysIM check caught me before it caught the chapter. I was about to flag the H100 FP8 figure of 1979 TFLOP/s as the sparsity number, but FP8 *dense* on H100 really is 1979 (FP16 dense is 989). Check against the registry, not memory; that's why the book's rule that every number comes from MLSysIM exists.
- The repair loop fixed errors *inside* sections but introduced drift *between* them: two overflow policies, three deadline codes, four names for the quarantine invariant. When sections are repaired one at a time, the certification pass should diff definitions across sections rather than re-read each section alone.
- A good regression check to add: a binder scope that flags `\texttt{…_…}` inside `$$` blocks and `: caption {#fig-` after a code fence. Both are recurring build breakers that a lint catches for free.
`─────────────────────────────────────────────────`

---

🎯 **Next action:** Clear the P0s first: escape the underscore, swap in the two SVGs, replace the Roofline cell with a registry-backed LEGO cell (H100/B200 plus `MacBookM3Max`), and either run the §2.8 benchmark or reword it as hypotheses. Then update the blueprint to the seven-class envelope and make each deadline and overflow case map to exactly one status code.
