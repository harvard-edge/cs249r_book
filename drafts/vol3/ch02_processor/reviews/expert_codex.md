**TL;DR:** Chapter 02 has the right architecture and a strong Zero Ambient Authority thesis, but it needs major technical revision before publication. The principal blockers are an incomplete invocation ABI, an internally inconsistent four-status state machine, an insufficient quarantine invariant, overgeneralized grammar-decoder implementation claims, and several failure traces or benchmark numbers that are not mechanically reproducible.

## Overall verdict

The chapter executes its assigned “Here is the Processor” mission well and mostly respects the atomic scope \(H=1, A=0\). It does not need search, memory management, sandbox construction, tool dispatch, or Agent OS internals.

The strongest material is:

- The three-tier Host Runtime → Inference Service → Neural Core boundary.
- The separation between generation completion and task completion.
- The syntactic-validity versus semantic-authority divide.
- The recognition that streaming prefixes remain provisional.
- The requirement for explicit uncertainty/error paths in constrained schemas.

The draft is not yet production-grade in four areas:

1. The purportedly “fully qualified” request does not identify the tokenizer or decoding policy.
2. The four statuses are not exhaustive under cancellation, constraint dead-end, request rejection, and server failures.
3. The quarantine formula supplies only one necessary condition and conflates validation with actuation.
4. Several claimed empirical results and example traces are unsupported or do not behave as described.

---

## 1. Invocation contract and status envelope

### Request tuple

The tuple at [02_processor.qmd:435](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:435) is a good pedagogical starting point, but it is not fully qualified.

At minimum, it lacks:

- A request/trace identifier and contract version.
- The tokenizer identity or digest. Token IDs are meaningless without the exact vocabulary, special-token map, and encoding rules.
- The decoding policy: temperature, top-\(p\), top-\(k\), seed, repetition penalties, and logit biases.
- An input-overflow policy. Section 2.2 requires explicit over-limit behavior, but the ABI does not say `REJECT`, left-truncate, or right-truncate.
- Grammar/schema version or digest.
- A precise deadline representation.
- Stream-finalization metadata.

The chapter also assigns tokenization to the inference service in the architecture figure, then defines \(\mathbf{x}\) as pre-tokenized IDs and says the host must tokenize. Choose one interface:

- Text/bytes cross the RPC boundary and the service owns tokenization; or
- Token IDs cross the boundary and the service validates a pinned tokenizer digest.

The model digest at [line 439](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:439) provides artifact traceability, not reproducibility. Fixed weights do not pin tokenizer configuration, quantization, runtime kernels, sampling state, or parallel reduction order.

A minimal production contract could be shown as:

```proto
message InvocationRequest {
  string request_id = 1;
  uint32 contract_version = 2;

  oneof input {
    string prompt_utf8 = 3;
    TokenIds input_token_ids = 4;
  }

  bytes tokenizer_digest = 5;
  bytes model_artifact_digest = 6;
  string runtime_build_id = 7;

  DecodingPolicy decoding = 8;
  uint32 max_output_tokens = 9;
  google.protobuf.Duration timeout = 10;
  repeated bytes stop_sequences = 11;

  GrammarSpec grammar = 12;
  InputOverflowPolicy overflow_policy = 13; // REJECT by default
}
```

The deadline is currently described as an “absolute timeout in milliseconds,” which mixes two different concepts. Use either an absolute deadline or a relative timeout. gRPC explicitly distinguishes deadline propagation and converts deadlines into remaining time to avoid clock-skew errors. [gRPC deadline semantics](https://grpc.io/docs/guides/deadlines/)

### Stop sequences

The decode predicate at [line 232](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:232) treats \(\mathcal{S}_{\text{stop}}\) as individual tokens, while the contract later defines it as token sequences.

Replace:

\[
y_t \in \mathcal{S}_{\text{stop}}
\]

with:

\[
\exists s \in \mathcal{S}_{\text{stop}}
\;:\;
s \text{ is a suffix of } y_{1:t}.
\]

The contract should also define whether the matched delimiter is included in the returned payload.

### Four-outcome envelope

The four categories are useful, but they are not exhaustive as defined at [lines 459–462](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:459).

Missing cases include:

- Caller cancellation.
- Host prefix rejection.
- Grammar compilation failure.
- Constraint dead-end or empty valid-token set.
- Context overflow or malformed request.
- Admission rejection due to overload.
- A server error returned over an otherwise healthy transport.

The cleanest resolution while preserving the blueprint’s four top-level outcomes is:

1. Treat invalid requests and admission failures as pre-invocation RPC errors, outside the four-outcome generation envelope.
2. Add a mandatory structured `reason_code`.
3. Broaden `TRUNCATED` to mean “admitted generation ended without normal completion,” not only token/deadline exhaustion.
4. Define `TRANSPORT_FAILURE` from the caller’s observation point: no trusted final envelope was received.

```proto
enum Outcome {
  COMPLETED = 0;
  TRUNCATED = 1;
  REFUSED = 2;
  TRANSPORT_FAILURE = 3;
}

enum FinishReason {
  EOS = 0;
  STOP_SEQUENCE = 1;
  MAX_OUTPUT_TOKENS = 2;
  DEADLINE_EXCEEDED = 3;
  CLIENT_CANCELLED = 4;
  PREFIX_REJECTED = 5;
  CONSTRAINT_DEAD_END = 6;
  POLICY_BLOCK = 7;
  CONNECTION_LOST = 8;
  WORKER_FAILURE = 9;
}

message FinalEnvelope {
  string request_id = 1;
  Outcome outcome = 2;
  FinishReason reason = 3;
  bytes output = 4;
  uint64 final_chunk_sequence = 5;
  Usage usage = 6;
}
```

`COMPLETED` must not mean “well-formed candidate,” as stated in the table at [line 468](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:468). It means only that generation reached a configured normal terminator. An unconstrained `COMPLETED` result may still contain malformed JSON, incomplete code, or an invalid patch.

Likewise, `REFUSED` should be assigned only from trusted service metadata. A model-generated sentence resembling a refusal is still ordinary candidate text.

### Streaming and cancellation

The transport description at [lines 445–447](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:445) needs three corrections:

- SSE is a server-to-client event stream after an HTTP request, not a bidirectional protocol. [WHATWG SSE specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- `RST_STREAM` requests HTTP/2 stream termination; gRPC `CANCELLED` is an observed status code, not the equivalent wire command. [RFC 9113, §6.4](https://www.rfc-editor.org/rfc/rfc9113.html)
- Cancellation does not guarantee immediate KV-cache reclamation. The serving handler must observe cancellation and coordinate cleanup; active device kernels may not be preemptible. [gRPC cancellation guide](https://grpc.io/docs/guides/cancellation/)

Replace “immediately freeing GPU KV cache memory” with “making the request eligible for scheduler cleanup once cancellation propagates.”

An HTTP `200 OK` also does not establish that the whole stream arrived without reset. It can be emitted before the streaming response body completes.

### Quarantine invariant

The intent is correct, but the formula at [line 497](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:497) is incomplete.

Problems:

- It does not state that `COMPLETED` is insufficient for actuation.
- `ActuationPipeline` incorrectly includes parsers and compilers, which are verification mechanisms.
- “Host DRAM only” is too physical. Quarantined data may legitimately enter restricted diagnostic storage or telemetry.
- Streaming prefixes need protection before any final envelope exists.

A stronger invariant is:

\[
\operatorname{Commit}(\mathbf y)
\implies
\operatorname{FinalEnvelopeReceived}
\land S=\texttt{COMPLETED}
\land \operatorname{Verify}(\mathbf y)=\texttt{PASS}
\land \operatorname{Authorize}(\mathbf y)=\texttt{ALLOW}.
\]

Add the streaming rule:

\[
\neg\operatorname{FinalEnvelopeReceived}
\implies
\mathbf y_{\text{prefix}}\notin\operatorname{ActuationPipeline}.
\]

Define `VerificationPipeline` separately from `ActuationPipeline`. The former may parse or compile an output under bounded conditions; the latter can mutate authoritative state.

---

## 2. Grammar-constrained logit masking

### What is accurate

The core explanation is sound:

- The decoder maintains grammar/parser state.
- It computes a tokenizer-aware valid-token mask.
- Invalid logits are assigned \(-\infty\).
- Sampling occurs over the remaining support.
- Grammar validity establishes no semantic correctness or authority.

The chapter’s Syntactic Divide is particularly effective.

### Valid-prefix versus valid-output guarantee

The definition at [line 531](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:531) is too weak:

\[
\delta^*(q,s(y))\ne\text{ERROR}
\]

only establishes that a token does not immediately violate the parser. A production matcher should admit a token only if the resulting configuration remains capable of reaching an accepting state:

\[
\mathcal V_{\mathrm{valid}}(c)=
\left\{
v\in\mathcal V
\;\middle|\;
c'=\operatorname{Advance}(c,\operatorname{bytes}(v)),
\;c'\ne\bot,
\;\operatorname{ReachAccept}(c')
\right\}.
\]

Here \(c\) includes the parser stack for a context-free grammar. EOS must be masked unless \(c\) is accepting.

The publication-grade guarantee should be conditional:

\[
\texttt{COMPLETED}
\land \text{supported grammar}
\land \text{correct matcher}
\land \text{accepting termination}
\implies
\operatorname{detokenize}(\mathbf y)\in L(\mathcal G).
\]

A mask guarantees that intermediate output remains an extendable prefix. It guarantees a complete parse only if generation reaches an accepting state rather than truncation, cancellation, or backend failure.

### DFA/PDA explanation

The DFA/PDA distinction is appropriate as a conceptual model, but “JSON Schema” must not be treated as one uniformly supported grammar. Production backends support subsets; for example, vLLM explicitly checks for unsupported schema features such as `uniqueItems`, `contains`, and some numeric constraints. [vLLM XGrammar backend](https://docs.vllm.ai/en/stable/api/vllm/v1/structured_output/backend_xgrammar/)

Thus “valid JSON matching these types” should become:

> valid output for the subset of the declared schema compiled by the selected backend.

Unsupported schema features must cause request rejection, never silent constraint weakening.

### Bitmask execution topology

The packed-mask calculation is correct: a 131,072-entry one-bit mask occupies 16 KiB. The universal topology asserted at [lines 576–604](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:576) is not.

Real engines commonly:

1. Maintain parser/matcher state on the CPU.
2. Fill a packed bitmask in CPU or pinned memory.
3. Transfer the mask to the logits device.
4. Apply it with a GPU kernel.

XGrammar’s own integration documentation explicitly fills masks using CPU logic and then copies them to the logits device. [XGrammar workflow](https://github.com/mlc-ai/xgrammar/blob/main/docs/start/workflow_of_xgrammar.md), [engine integration](https://github.com/mlc-ai/xgrammar/blob/main/docs/using_xgrammar/engine_integration.md)

For CFGs, complete masks cannot generally be precomputed for every possible pushdown-stack configuration. XGrammar instead precomputes context-independent portions and evaluates context-dependent tokens at runtime using persistent parser stacks. [XGrammar paper](https://arxiv.org/abs/2411.15100)

Therefore:

- Change “static lookup tables yield the valid set directly from \(q_t\)” to an implementation-dependent optimization.
- Change “bitmasks reside in GPU L2” to “the applied mask may be transferred to or maintained on-device; its small footprint can improve cache locality.”
- Change “preserving static graph execution” to “fixed-shape mask buffers can be designed to remain compatible with captured execution graphs.”
- Remove “eliminating” cache/HBM traffic. L2 residence is not guaranteed because the cache is shared and replacement-driven.

The ASCII figure also labels the bitmask as `[1,0,…]` but adds it arithmetically to logits. A packed eligibility bitmap must first be interpreted by a kernel that writes \(0\) or \(-\infty\); the raw bits themselves are not the additive mask.

### Constraint dead-end

The pseudocode needs an explicit empty-mask branch:

```python
mask = matcher.next_token_mask()

if mask.none():
    finish(outcome=TRUNCATED, reason=CONSTRAINT_DEAD_END)

logits.masked_fill_(~mask, float("-inf"))
token = sample(logits)
matcher.accept(token)
```

Without it, masking every logit to \(-\infty\) can produce undefined softmax behavior or NaNs.

---

## 3. Production failure-mode realism

The chapter identifies authentic failure classes, but several concrete examples need correction.

### Tokenization evidence

The hard-coded token IDs at [lines 131–132](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:131), compression table, and JSON-overhead percentages are not tied to a named tokenizer/version or executable artifact.

The blueprint explicitly requires measured tokens from the chosen tokenizer. Use a reproducible cell:

```python
import json
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

def inspect(label: str, text: str) -> None:
    ids = enc.encode(text)
    pieces = [
        enc.decode_single_token_bytes(i).decode("utf-8", "backslashreplace")
        for i in ids
    ]
    print(label, len(ids), list(zip(ids, pieces)))

source = 'def get_user_id(user_uuid):\n    return user_uuid\n'
payload = json.dumps({"patch": source}, separators=(",", ":"))

inspect("source", source)
inspect("json", payload)
```

Byte-level BPE is reversible and lossless; it does not alter indentation or convert tabs to spaces. The model may select the wrong whitespace-bearing token, but the tokenizer itself does not corrupt the bytes. [tiktoken README](https://github.com/openai/tiktoken/blob/main/README.md)

The claim that BPE uses “greedy longest matching” also needs correction. BPE applies ranked merge operations; it is not generally longest-prefix tokenization. [tiktoken implementation](https://github.com/openai/tiktoken/blob/main/src/lib.rs)

### Markdown fence trace

The regex example at [lines 876–889](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:876) does not fracture at `sentinel = "```"` because the supplied regex requires the closing fence to begin after a newline. Use a payload containing a literal line beginning with three backticks, or change the parser to a genuinely naive “first three-backtick substring” implementation.

### Truncated patch trace

The trace at [lines 480–494](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:480) does not establish that the deletion half of that single malformed hunk will be applied.

A mechanically accurate distinction is:

- `git apply` is atomic by default and does not modify the working tree if a hunk fails.
- GNU `patch` applies hunks independently and can leave earlier successful hunks applied while later hunks fail.

See the [git apply atomicity documentation](https://git-scm.com/docs/git-apply) and [GNU patch behavior](https://www.gnu.org/software/diffutils/manual/html_node/Merging-with-patch.html).

Either:

- Change the trace to show two hunks, where the first applies and the second is truncated; or
- Use `git apply --check` and show a fail-closed “corrupt patch” diagnostic without claiming mutation.

### Invocation status versus verification status

The search/replace trace at [line 936](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:936) conflates model completion with patch application and uses exit code 127, conventionally associated with command-not-found behavior.

Use separate records:

```json
{
  "invocation": {
    "outcome": "COMPLETED",
    "reason": "STOP_SEQUENCE"
  },
  "verification": {
    "check": "EXACT_ANCHOR_MATCH",
    "status": "FAILED",
    "diagnostic_code": "ANCHOR_NOT_FOUND",
    "exit_code": 1
  },
  "authorization": "DENIED"
}
```

### Search/replace ABI

A production patch proposal should bind itself to the exact repository state:

```json
{
  "status": "OK",
  "target_path": "src/core/parser.py",
  "base_blob_sha256": "…",
  "expected_old": "    if not line.startswith(\"#\"):\n        return None\n",
  "replacement": "    if not line.startswith((\"#\", \"//\")):\n        return None\n",
  "error": null
}
```

Host checks should require:

- A capability-scoped relative path.
- Matching base digest.
- Exactly one exact anchor match.
- No ambiguous fuzzy match.
- Verification before authorization.

The recommendation at [line 963](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:963) that Levenshtein matching raises success past 80% is both unsupported and dangerous. Fuzzy matching can silently modify the wrong repeated block. It should fail closed unless additional location and digest invariants disambiguate the target.

### Empirical table

The purported \(N=500\) experiment at [lines 947–955](/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/02_processor/02_processor.qmd:947) is a publication blocker. No supporting harness, fixture manifest, raw results, model identifier, sampling configuration, tokenizer, random seeds, or result artifact is cited or present in the searched repository paths.

The exact percentages, TTFT delta, latency baseline, and “past 80%” fuzzy-match result must be either:

- Replaced with a qualitative comparison table;
- Clearly labeled as a hypothetical worked example; or
- Generated by a committed/reproducible benchmark with confidence intervals and failure classification rules.

---

## Actionable Issue Ledger

| ID | Priority | Section | Issue | Required correction |
|---|---|---|---|---|
| ABI-01 | P0 | §2.5 | Four outcomes do not cover cancellation or grammar dead-end | Add structured reason codes; define admitted-request scope; broaden `TRUNCATED` or add an explicit abort outcome |
| ABI-02 | P0 | §2.5 | `COMPLETED` is called “well-formed” | Define it only as normal generation termination |
| ABI-03 | P0 | §2.5 | Quarantine invariant is necessary but insufficient | Require final envelope, `COMPLETED`, verification pass, and authorization before commit |
| ABI-04 | P1 | §2.5 | Request tuple omits tokenizer and decoding policy | Add tokenizer digest, decoding policy, overflow policy, request ID, and contract version |
| ABI-05 | P1 | §2.5 | Model digest is said to guarantee reproducibility | Recast it as artifact identity/traceability |
| ABI-06 | P1 | §2.3/§2.5 | Stop predicate handles only single-token delimiters | Match suffixes against token sequences and define delimiter inclusion |
| ABI-07 | P1 | §2.5 | SSE/gRPC cancellation semantics are inaccurate | Describe HTTP+SSE as server streaming; cancellation as propagation with implementation-dependent cleanup |
| GCD-01 | P0 | §2.6 | Non-error transitions are equated with valid completions | Require viable prefixes, accepting termination, and EOS masking |
| GCD-02 | P0 | §2.6 | GPU-resident static masks are presented as universal | Present CPU matcher + device-mask transfer and GPU-resident approaches as implementation alternatives |
| GCD-03 | P1 | §2.6 | No empty-mask failure path | Add `CONSTRAINT_DEAD_END` handling before softmax |
| GCD-04 | P1 | §2.6/§2.8 | Full JSON Schema conformance is implied | State backend-supported subset and reject unsupported features |
| FAIL-01 | P0 | §2.8 | Empirical \(N=500\) numbers lack provenance | Remove, mark hypothetical, or ship the benchmark and raw results |
| FAIL-02 | P1 | §2.2/§2.8 | Token IDs and overhead percentages are unmeasured | Generate them from a pinned tokenizer in an executable cell |
| FAIL-03 | P1 | §2.8 | Tokenizer is blamed for changing whitespace | Attribute wrong whitespace to model token selection; tokenization remains lossless |
| FAIL-04 | P1 | §2.5 | Single truncated hunk does not support claimed partial deletion | Use two GNU `patch` hunks or show atomic `git apply --check` rejection |
| FAIL-05 | P1 | §2.8 | Markdown fence example does not trigger the stated regex failure | Replace with a payload containing a line-level internal fence |
| FAIL-06 | P1 | §2.8 | `COMPLETED` and exit code 127 are conflated | Separate invocation, verification, and authorization records |
| FAIL-07 | P1 | §2.8 | Fuzzy anchor matching is recommended without safety conditions | Require base digest, unique exact anchor, and fail-closed ambiguity handling |
| VERIFY-01 | P2 | §2.4 | Passing tests are described as probability-1 closure | Scope the claim to deterministic checks under a fixed fixture; acknowledge flaky/nondeterministic checks |

> **Publication recommendation:** preserve the chapter’s architecture and Syntactic Divide, but block copyedit until all P0 items and the trace inaccuracies are resolved.

No repository files were modified.

---

🎯 **Next action:** Revise §2.5’s state machine and quarantine invariant first; they are the contract on which §2.6 and §2.8 depend.
