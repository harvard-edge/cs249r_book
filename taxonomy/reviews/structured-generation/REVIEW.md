# Worked review: decoding, sampling, and structured output

Date: November 4, 2026. Status: boundary corrections and sampling conditions applied through `accepted-boundaries.json` and `accepted-sampling-conditions.json`; remaining proposals are deferred. No independent approval is claimed. Scope: subject `structured-generation` (P4.6). This is a retrospective review of an existing subject using the construction protocol in `taxonomy/METHOD.md`, not an independent reconstruction or held-out evaluation. These are local model-assisted editorial corrections; publication approval remains separate.

## Baseline and method

`baseline.json` records the source revision and byte hash, the subject, every homed concept, and every incident relationship. The inventory was computed from the canonical YAML: 5 concepts and 13 incident relationships. These are inventory counts, not performance measurements. Repeating the extraction is specified in `verify.py`.

This subject was selected because it contains distinct kinds of engineering ideas that can easily be conflated. Review questions were: Are the entries distinct? Do aliases preserve meaning? Do definitions separate a requirement from its enforcement and checking? Do relationship conditions support the strength of the claim?

Sources were selected purposively for these questions from original research and official documentation. They are construction/review evidence, not a representative sample of the entire field. No completeness, human agreement, or independent-source admission threshold is claimed. New concepts are deferred until the full source-lineage and admission procedure is satisfied.

## Evidence register

Sources below were opened on the review date. Locators identify the inspected material; summaries are paraphrases. Live documentation can change and should be archived before an accepted patch uses it as immutable provenance.

| Key | Primary source and locator | Evidence used |
|---|---|---|
| G | [Willard and Louf, Efficient Guided Generation, version 2](https://arxiv.org/html/2307.09702v2), introduction and section headings for iterative parsing | Token filtering enforces admissible continuations. The paper distinguishes finite-state treatment of regular expressions from parsing for context-free grammars. It does not establish semantic truth of generated content. |
| H | [Transformers generation documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation), generation strategy, logits manipulation, and stopping parameters | Temperature, top-k, top-p, and repetition penalties have different functions. Greedy selection differs from sampling. Token limits, stop strings, and elapsed-time controls are separate settings. Even the elapsed-time setting allows the current pass to finish. |
| J | [JSON Schema object reference](https://json-schema.org/understanding-json-schema/reference/object), Required Properties and Additional Properties | Schema validation checks stated structural constraints. Validation itself does not prescribe repair or regeneration. |
| N | [Holtzman et al., The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), abstract | Nucleus sampling selects from a probability-dependent subset of candidate tokens; it is a particular sampling method. |
| S | [Wang et al., Self-Consistency, version 4](https://arxiv.org/abs/2203.11171v4), abstract | The described method samples diverse reasoning paths and aggregates their answers. |
| W | [Kirchenbauer et al., A Watermark for Large Language Models, version 4](https://arxiv.org/abs/2301.10226v4), abstract | The described watermark promotes selected tokens during sampling. |
| C | [Chen et al., Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374), bibliographic record | Identified as the existing pass-at-k evidence lead. The relevant temperature analysis was not inspected in this pass; it does not support an accepted correction yet. |

## Standards application

The review applies the term distinction and evidence principles described in the project's ANSI/NISO Z39.19 baseline, concept-centered organization informed by ISO 25964-1, and the SKOS distinction between labels, collections, and semantic relationships. It does not constitute a clause-level ISO audit. ISO 25964-2 mapping work is outside this subject review; no external exact-match assertions are proposed.

Sources: [NISO Z39.19](https://www.niso.org/publications/ansiniso-z3919-2005-r2010), [ISO 25964-1](https://www.iso.org/standard/53657.html), [ISO 25964-2](https://www.iso.org/standard/53658.html), [SKOS Reference](https://www.w3.org/TR/skos-reference/).

## Subject and concept decisions

Retain the subject and its browsing home provisionally. The distinction among selection, structural enforcement, checking, and stopping is useful for teaching. Add an exclusion pointing to test-time-compute for candidate search and answer selection across completed generations. This boundary proposal needs neighbor review before application.

| Identity | Proposed disposition | Reason and candidate wording |
|---|---|---|
| `mlsys.grammar` | Retain | The definition distinguishes the output contract from the mechanisms that enforce or check it. Preserve the stable ID even though its display label is broader than grammar. Do not imply that structural validity establishes factual correctness or tool authorization. Evidence: G, J. |
| `mlsys.sampling-controls` | Retain umbrella; revise alias treatment | Temperature, nucleus sampling, and greedy decoding are not interchangeable names for the umbrella. Classify each as a narrower entry term or candidate distinct concept, with explicit scope and warrant. Do not automatically split them into new nodes. N and H establish meaningful distinctions; the existing upward-posting policy still requires item-level justification. |
| `mlsys.constrained-decoding` | Retain; qualify definition | Candidate: “A generation mechanism that restricts token choices to continuations permitted by an output constraint, so a successfully completed output satisfies the supported syntax or schema constraints.” Avoid implying that a truncated prefix is a complete valid object. Keep parsing implementations broader than finite-state machines. G supports the mechanism; completion qualification is an editorial boundary inference. The alias `grammar-guided decoding` remains deferred pending exact-term attestation under the existing alias policy. |
| `mlsys.output-validation` | Revise definition | Candidate: “Checking a generated output against declared structural or application rules and reporting whether it satisfies them.” Separate validation from the caller's optional repair, retry, or rejection policy. J supports structural checking; the broader application-rule scope needs a direct source before acceptance. |
| `mlsys.generation-stopping` | Retain; narrow resource claims | Candidate: “Rules that terminate autoregressive generation when a stopping criterion is met, including an end token, a stop sequence, or an output-token limit.” Keep latency, cache behavior, and API completion reporting in qualified relationships or implementation notes. H supports the stopping distinction. Its existing aliases require narrower-term treatment rather than blanket synonym claims. |

## Incident relationship decisions

The entries below cover every incident relationship in the snapshot. “Retain” means provisionally retain after this editorial reading, not independent approval. “Revise” describes a proposed change; final relation-enum selection and cross-check are still required.

| Relationship ID | Disposition | Reason or proposed correction |
|---|---|---|
| `rel.constrained-decoding.grammar` | Revise condition | Require a supported constraint representation, tokenizer-compatible enforcement, and successful completion. Retain implements. G; completion boundary is editorial inference. |
| `rel.output-validation.constrained-decoding` | Revise | They can be complementary. Restrict alternative-to to the choice of enforcing the same expressible structural constraint during generation versus checking it afterward; application checks can still follow constrained decoding. G, J, and editorial synthesis. |
| `rel.sampling-controls.self-consistency` | Retain with narrower condition | S supports diverse-path sampling for the described self-consistency procedure. Qualify the identical-greedy-path claim to repeated deterministic generation with the same model and prompt. |
| `rel.best-of-n.sampling-controls` | Defer | Candidate diversity can arise from different prompts or other generators, not necessarily these controls. Inspect the best-of-n source and neighboring definition before replacing requires with uses. This counterexample is an editorial inference, not a measured result. |
| `rel.sampling-controls.pass-at-k` | Revise; evidence pending | The claim that the optimal temperature rises with k is too general for an unconditional field relationship. Candidate predicate: “affects the candidate distribution used to estimate”; condition: “For a fixed model, task set, sampling procedure, and correctness criterion.” Inspect C's relevant analysis before acceptance; no universal monotonic rule is established here. |
| `rel.constrained-decoding.function-calling` | Revise condition | Limit the guarantee to supported argument constraints on successfully completed outputs. Tool authorization and argument truth remain separate. G, J, and editorial inference about completion. |
| `rel.function-calling.grammar` | Retain | Its schema-bound condition distinguishes a declared argument contract from execution authority. J supports the schema interpretation; tool-specific evidence remains a follow-up. |
| `rel.output-validation.grammar` | Revise condition | Replace “instead of masked during it” with “whether or not constrained decoding was used.” Post-generation checks can follow either generation strategy. G, J, editorial synthesis. |
| `rel.output-validation.tail-latency` | Revise condition | Restrict to retries on the response's critical path relative to a comparable run without those retries. Validation alone does not entail regeneration. This latency implication is analytical, not a measured latency result. |
| `rel.generated-content-watermarking.sampling-controls` | Retain; improve warrant | W directly supports biased token selection for sampling-time watermarking. Add this source through the governed source procedure; the current generic references are less specific. |
| `rel.generation-stopping.kv-cache` | Revise condition | Qualify to engines whose retained autoregressive cache grows with generated tokens; an output cap limits additional token positions, not all allocated cache memory. Cache allocation can be static. H supports the implementation distinction; no memory measurement is claimed. |
| `rel.generation-stopping.tail-latency` | Revise | Remove the assertion that output length alone bounds worst-case latency. Candidate predicate: “limits the output-length contribution to”; condition: “When decode-step time is bounded and queueing, stalls, and retries are accounted for separately.” H distinguishes length and time limits; the counterexample involving stalls is analytical. |
| `rel.generation-stopping.resource-exhaustion-attack` | Retain with qualification | State that the cap limits generated-token work per request; it does not by itself cap prompt processing, request rate, or aggregate client consumption. Additional admission and rate policies remain necessary. Editorial threat-model reasoning; direct security warrant remains pending. |

## Constructed teaching checks

These are author-constructed examples for inspecting boundaries, not a held-out test or evidence of measured usefulness.

- A JSON object lacks a required field: tag output validation when discussing detection; tag the structured output constraint when discussing the contract.
- A decoder masks inadmissible continuations: tag constrained decoding.
- A generation changes from top-k to nucleus sampling: the umbrella can currently retrieve the topic, but does not distinguish the methods. This is the witness for evaluating finer granularity.
- A request reaches its token cap before closing a JSON object: stopping succeeded; complete-output validity was not established.
- A short output waits indefinitely behind a stalled worker: output length alone has not bounded response latency.

## Reusable outcome

The subject can be retained while its term relationships and overbroad guarantees are improved. The worked example demonstrates why syntactically valid records still require semantic review. It also shows that documenting a narrower entry term can be preferable to creating an unsupported new concept.

The next editorial pass should obtain exact-term evidence for deferred aliases, inspect the best-of-n and pass-at-k sources, and review the proposed text independently. Applying accepted changes requires the existing provenance workflow and a working validation baseline. The accepted boundary corrections are now recorded in the canonical vocabulary and provenance chain; this original findings register also retains the deferred proposals. The validator/evidence-ledger mismatch has been repaired with conservative historical recovery. This packet still makes no release or human-evaluation claim.

## Follow-up disposition

The temperature analysis in Chen et al., version 2, Section 3.3 and Figure 5 was inspected on November 4, 2026. Its observed temperature trend is specific to the reported evaluation, so the accepted condition does not universalize it. Best-of-n and self-consistency dependencies now state the fixed-input stochastic-generation setup. Alias granularity, exact-term attestation, the neighboring subject exclusion, and additional watermark-source metadata remain deferred for a later scope/evidence revision. These deferrals do not add new concepts or assert new synonym equivalences.
