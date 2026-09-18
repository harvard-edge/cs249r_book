# Certification

**Final sign-off: NEEDS_WORK**

The repaired chapter now has a strong conceptual spine, but it is not publication-ready. It directly contradicts the supplied master blueprint, crosses several explicit scope boundaries, contains material technical overclaims, presents unexecuted comparisons as empirical conclusions, and currently fails the PDF build.

## Assessment

| Certification question | Result | Assessment |
|---|---|---|
| Technical, ABI, and hardware repairs | **Partial** | The three-tier architecture, autoregressive loop, verification perimeter, and latency decomposition are substantially improved. Several equations and systems claims remain incorrect or insufficiently qualified. |
| Contracts, envelopes, and equations | **Not yet publication-grade** | The chapter replaces the required four-outcome contract with seven primary outcomes, conflates completion with structural validity, and hardcodes hardware data outside MLSysIM. |
| Elimination of false analogies | **Partial** | The model is consistently treated as an unprivileged proposal generator. The CPU, Ring 0, OS, ABI, and fail-stop comparisons are still sometimes presented as literal equivalences. |
| Final sign-off | **NEEDS_WORK** | The remaining issues affect architectural consistency, technical correctness, reproducibility, scope, and buildability. |

## Publication blockers

### 1. The invocation contract contradicts the master blueprint

The blueprint requires four normalized outcomes:

- `COMPLETED`
- `TRUNCATED`
- `REFUSED`
- `TRANSPORT_FAILURE`

The repaired chapter changes this into seven primary outcomes by adding `REJECTED`, `CANCELLED`, and `FAULTED`, beginning in the [learning objectives](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:43) and propagating throughout the chapter.

The extra distinctions are operationally useful, but they cannot silently replace the architectural contract being certified. The clean repair is to retain the four primary outcomes and represent admission, cancellation, and execution faults through reason codes, or formally revise the master blueprint before certifying the chapter.

Additional contract defects:

- `COMPLETED` is described as having structurally well-formed content. Normal termination establishes only that a stop condition was reached; unconstrained output may still be malformed.
- A TLS handshake failure cannot be returned by the failed server as an envelope. The caller must synthesize that status.
- The runtime build digest does not guarantee reproducible floating-point behavior.
- Deadline expiration and transport timeout are not cleanly separated.
- The chapter says cancellation frames immediately halt GPU launches and free KV allocations. gRPC cancellation only signals cancellation; the server application must observe it and stop its work. [Official gRPC deadline guidance](https://grpc.io/docs/guides/deadlines/).

The 13-field “production ABI” in [the contract section](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:527) also obscures the blueprint’s compact architectural tuple. Several fields belong in an implementation profile rather than the processor contract taught here.

### 2. Explicit negative scope boundaries are crossed

The most substantial violation is the full KV-cache derivation in [Section 2.2](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:197). It derives the per-token footprint, computes 128K allocations, analyzes H100/B200 capacity, invokes tensor parallelism, and motivates paged attention. The blueprint assigns physical KV footprint and allocation to Chapter 5.

Other scope crossings include:

- Two-invocation decomposed probes in Section 2.8, despite the atomic \(H=1\) evaluation boundary.
- Continuous batching and chunked prefill implementation guidance.
- Sandbox targets and execution environments.
- Prompt compaction, replica failover, circuit breakers, retries, and task decomposition.
- Later-chapter material embedded directly in the invocation, autoregressive, and Roofline figures.

These are explanations of downstream subsystems, rather than brief forward references.

### 3. Several technical claims require correction

The most consequential examples are:

- The chapter says the host runtime drives every forward pass. Under its own three-tier architecture, the inference service drives the decode loop.
- The top-\(p\) equation minimizes over arbitrary subsets. Nucleus sampling selects the shortest prefix after sorting tokens by descending probability.
- Temperature ranges such as “balanced sampling” at \(0.2\)–\(0.8\) are model and task dependent, not architectural regimes.
- The temperature figure labels greedy decoding deterministic despite the text acknowledging kernel and floating-point nondeterminism.
- An all-\(-\infty\) mask may produce invalid or NaN probabilities, but NaNs do not inherently corrupt GPU memory or the KV cache. Sampling APIs require finite, nonnegative probabilities with nonzero sum and should reject this state. [PyTorch multinomial contract](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html).
- Schema renormalization makes probabilities sum to one; it does not create “100% confidence” unless exactly one legal outcome remains.
- Tests do not close invariants with probability \(1.0\). They establish the tested predicates under the observed execution and can themselves be incomplete, flaky, or unsound.
- The MoE calculation is numerically reasonable under uniform independent routing, but that assumption is unstated. The recommendation to budget total expert traffic whenever \(B>1\) is too strong.
- The latency stages are called non-overlapping even though streaming serving systems can overlap host, network, and accelerator work.
- “Low-batch decode is strictly memory-bandwidth bound” should use the blueprint’s qualified “often becomes memory-bandwidth bound.”

The BPE examples were reproducible with `tiktoken` 0.9.0 and are among the chapter’s strongest quantitative material.

### 4. The hardware analysis violates the project’s source-of-truth rule

The block labeled “MLSysIM LEGO Roofline Cell” is a plain Python fence that:

- does not execute,
- imports nothing from MLSysIM,
- hardcodes all accelerator and model data,
- produces no rendered result.

See the [purported LEGO cell](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:980).

Most of the simple weight-traffic arithmetic is internally consistent, but publication requires registry-backed inputs and explicit assumptions. Examples needing repair include:

- 70.0B hardcoded instead of the model registry value.
- Decimal GB and binary GiB used interchangeably in capacity arguments.
- An unsourced 40 TFLOP/s value for M4 Max. Apple publishes 546 GB/s memory bandwidth and capacity information, but not that compute figure. [Apple M4 Max specifications](https://www.apple.com/ca/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/).
- Cache reuse, tensor parallel communication, attention work, quantization metadata, and runtime overhead omitted while the resulting estimate is sometimes described as realized latency.

The Llama 3.1 assumptions of 128K context and grouped-query attention are supportable when the model is named precisely. [Meta Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md).

### 5. Section 2.8 reports conclusions without benchmark evidence

The interface evaluation defines a useful protocol, but it supplies no:

- sample size,
- fixture list,
- trial results,
- variance or uncertainty,
- raw \(R_{\text{syntax}}\), \(R_{\text{trunc}}\), or \(R_{\text{task}}\) measurements.

It nevertheless declares interfaces “highest,” “lowest,” “maximal,” “catastrophic,” and on a “Pareto frontier.” Those are empirical verdicts. The section must either report an executable benchmark with results or recast these statements as hypotheses and expected failure modes.

Claims that grammar-constrained JSON has \(R_{\text{syntax}}\equiv1\) also need conditioning on successful, completed constrained generation. Truncation, grammar dead ends, and implementation failure prevent an unconditional guarantee.

### 6. The chapter fails publication gates

The targeted PDF build fails on malformed stop-delimiter markup near the contract discussion:

```text
Undefined control sequence ... \textbackslash{}n
```

Other detected production defects include:

- ASCII diagrams labeled as figures are not valid Quarto figure structures, leaving local references unresolved.
- Multiple figures lack valid captions or alt text under the project checks.
- The latency table lacks a proper caption.
- Five footnote definitions near the end are not separated correctly and may not parse as footnotes.
- The numbers check rejects six prose percentages and reports unit-format violations.
- `git diff --check` finds trailing whitespace.
- Several figure captions and house-style structures fail binder checks.

Visual inspection found no major clipping or overlap in the rendered SVGs. Their technical content still needs correction, particularly the temperature and Roofline figures.

## Analogy assessment

The central analogy now works at the functional level: the neural core consumes staged token IDs and emits an unprivileged candidate, while the host controls budgets, validation, and authority.

Several comparisons remain overextended:

- CPUs are described as intrinsically deterministic and fail-stop. ISA-visible execution and architected exception behavior are the defensible comparison.
- A user-space runtime is repeatedly equated with Ring 0.
- An RPC request contract is called an ABI in the literal binary-interface sense.
- Token IDs are called “gather addresses”; they are embedding-table indices.
- The inference service is sometimes called an accelerator driver even though it operates above the actual device driver.

These should be qualified as teaching correspondences rather than asserted equivalences.

## What the repair accomplished

The chapter now successfully establishes:

- the host, serving daemon, and neural-core boundary;
- the distinction between raw logits and sampled tokens;
- autoregressive serialization along one realized output;
- the separation of likelihood, syntax, semantic correctness, and authority;
- quarantine of incomplete output;
- the need for explicit resource limits and machine-readable termination status;
- the distinction between per-trajectory latency and service throughput;
- grammar escape channels for uncertainty and failure.

Those improvements are substantial. The remaining work is primarily disciplined reduction and correction, rather than a conceptual rewrite.

🎯 **Next action:** Restore blueprint compliance first—four primary outcomes, atomic \(H=1\) scope, and Chapter 5 KV deferral—then correct the sampling and grammar claims, replace the hardcoded Roofline block with an executable MLSysIM cell, and rerun the binder PDF and chapter checks.
