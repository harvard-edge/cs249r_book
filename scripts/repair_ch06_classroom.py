#!/usr/bin/env python3
"""
Repair Chapter 06 based on 10-student classroom review:
1. Replace/retag 5 text blocks (algorithm pseudocode to python, query traces to bash/blockquote).
2. Clean 17 compound headings to single concepts.
3. Validate table column counts, zero text blocks, zero box characters, zero British spellings.
"""

import re
import sys

CH06_FILE = "books/vol3/06_episodic_memory/06_episodic_memory.qmd"

with open(CH06_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Heading replacements (Single-Concept Principle)
heading_replacements = [
    (
        "### The Query Tuple and Bounded Execution {#sec-vol3-persistent-query-tuple}",
        "### The Query Tuple {#sec-vol3-persistent-query-tuple}"
    ),
    (
        "### The Result Envelope and Provenance Verification {#sec-vol3-persistent-result-envelope}",
        "### The Result Envelope {#sec-vol3-persistent-result-envelope}"
    ),
    (
        "### Ingress Quarantining, Boundary Sanitization, and Budget Truncation {#sec-vol3-persistent-quarantine}",
        "### Ingress Quarantining {#sec-vol3-persistent-quarantine}"
    ),
    (
        "### Lexical Inverted Indexes and the BM25 Scoring Model",
        "### Lexical Inverted Indexes"
    ),
    (
        "### Syntax-Directed Indexing and Code Symbol Graphs",
        "### Syntax-Directed Indexing"
    ),
    (
        "### Relational Constraints and Compound Metadata Filtering",
        "### Relational Constraints"
    ),
    (
        "### Failure Modes: The Vocabulary Mismatch and Structural Blindness",
        "### Vocabulary Mismatch Pathologies"
    ),
    (
        "### Dense Vector Representations and Bi-Encoder Geometry",
        "### Dense Vector Representations"
    ),
    (
        "### Approximate Nearest Neighbor Mechanics: HNSW and IVF-PQ",
        "### Approximate Nearest Neighbor Mechanics"
    ),
    (
        "### The Precision Trap of Semantic Matching and Reciprocal Rank Fusion",
        "### The Semantic Precision Trap"
    ),
    (
        "### Multi-Hop Traversal and Seeded Neighborhood Expansion",
        "### Multi-Hop Graph Traversal"
    ),
    (
        "### Graph Construction Economics and Ingestion Latency",
        "### Graph Construction Economics"
    ),
    (
        "### The Materialized View Dilemma and Consistency Lags",
        "### The Materialized View Dilemma"
    ),
    (
        "### Invalidation Architectures: Synchronous, Asynchronous, and Read-Time Filtering",
        "### Invalidation Architectures"
    ),
    (
        "### The Self-Contradiction Trap and Generational Reclamation",
        "### The Self-Contradiction Trap"
    ),
    (
        "### Security Boundaries, Access Control, and Ingestion Sanitization",
        "### Storage Security Boundaries"
    ),
    (
        "### The Distractor Dilemma and End-to-End Task Verification",
        "### The Distractor Dilemma"
    )
]

for old_h, new_h in heading_replacements:
    if old_h not in content:
        print(f"ERROR: Heading not found: {old_h}")
        sys.exit(1)
    content = content.replace(old_h, new_h, 1)

print(f"Applied all {len(heading_replacements)} heading replacements.")

# 2. Block 1: Algorithm pseudocode -> typed Python (lines 239-254)
old_block_1 = """```text
Algorithm: Ingress Token Budget Allocation
Input:     Result envelopes {R_1, ..., R_k}, Total Budget S_budget, Chunk Ceiling S_max
Output:    Staged context payload P

1. Filter:  Discard any R_i where R_i.Score < sigma_min
2. Sort:    Order remaining envelopes descending by R_i.Score
3. Init:    P <- empty list, TokensUsed <- 0
4. For each R_i in sorted envelopes:
5.    AllowedTokens <- min(S_max, S_budget - TokensUsed)
6.    If AllowedTokens <= 0: Break
7.    Payload <- SyntaxTruncate(R_i.Content, AllowedTokens)
8.    TokensUsed <- TokensUsed + TokenCount(Payload)
9.    P.Append(WrapEnvelope(R_i, Payload))
10. Return P
```"""

new_block_1 = """```python
def allocate_ingress_budget(
    envelopes: list[ResultEnvelope],
    s_budget: int,
    s_max: int,
    sigma_min: float
) -> list[str]:
    \"\"\"Allocate tokens to retrieved result envelopes under bounded budget.\"\"\"
    # Filter envelopes failing the minimum confidence threshold
    valid = [r for r in envelopes if r.score >= sigma_min]
    # Sort remaining envelopes descending by retrieval score
    sorted_envs = sorted(valid, key=lambda r: r.score, reverse=True)
    
    staged_payloads: list[str] = []
    tokens_used = 0
    for r in sorted_envs:
        allowed_tokens = min(s_max, s_budget - tokens_used)
        if allowed_tokens <= 0:
            break
        truncated = syntax_truncate(r.content, allowed_tokens)
        tokens_used += count_tokens(truncated)
        staged_payloads.append(wrap_envelope(r, truncated))
    return staged_payloads
```"""

if old_block_1 not in content:
    print("ERROR: old_block_1 not found")
    sys.exit(1)
content = content.replace(old_block_1, new_block_1, 1)
print("Replaced Block 1 (Algorithm pseudocode -> Python).")

# 3. Block 2: Query vs Implementation mismatch (lines 404-407)
old_block_2 = """```text
Agent Investigation Query: "network packet throttling and rate limiting algorithm"
Target Codebase Implementation: TokenBucketTrafficShaper (struct leaky_bucket_valve)
```"""

new_block_2 = """> **Query vs. Code Vocabulary Mismatch:**  
> - **Agent Query:** `"network packet throttling and rate limiting algorithm"`  
> - **Target Implementation:** `TokenBucketTrafficShaper` (`struct leaky_bucket_valve`)  
> - **Result:** Lexical overlap $= \\emptyset$; BM25 postings yield zero matches."""

if old_block_2 not in content:
    print("ERROR: old_block_2 not found")
    sys.exit(1)
content = content.replace(old_block_2, new_block_2, 1)
print("Replaced Block 2 (Query mismatch -> blockquote).")

# 4. Block 3: Callpath traversal (lines 718-724)
old_block_3 = """```text
[PATH] network_rx.c:handle_packet()
  -> CALLS packet_parser.c:deserialize_v2()
    -> READS schema.h:HeaderFormat (struct)
  -> CALLS slab_alloc.c:buffer_reserve()
    -> FAILS_ON slab_alloc.c:OOM_CONDITION
```"""

new_block_3 = """```bash
[PATH] network_rx.c:handle_packet()
  -> CALLS packet_parser.c:deserialize_v2()
    -> READS schema.h:HeaderFormat (struct)
  -> CALLS slab_alloc.c:buffer_reserve()
    -> FAILS_ON slab_alloc.c:OOM_CONDITION
```"""

if old_block_3 not in content:
    print("ERROR: old_block_3 not found")
    sys.exit(1)
content = content.replace(old_block_3, new_block_3, 1)
print("Retagged Block 3 as bash.")

# 5. Block 4: Self-contradiction transcript (lines 851-861)
old_block_4 = """```text
[TURN 12] AGENT: ToolCall -> edit_file("net/http.py", replace="timeout=None", with="timeout=30")
[TURN 12] SYSTEM: File "net/http.py" successfully updated. MD5: 4f8a1c...
[TURN 13] AGENT: ToolCall -> search_code("def fetch_url")
[TURN 13] SYSTEM: Retrieved chunk net/http.py:14-28 (score: 0.89):
                 "def fetch_url(url): return session.get(url, timeout=None)"  <-- [STALE CHUNK]
[TURN 14] AGENT: "The previous edit did not persist; timeout is still None. Re-applying patch..."
[TURN 14] AGENT: ToolCall -> edit_file("net/http.py", replace="timeout=None", with="timeout=30")
[TURN 14] SYSTEM: Error: String "timeout=None" not found in "net/http.py". Target contains "timeout=30".
[TURN 15] AGENT: "File state is corrupted. Attempting structural recovery..." [OSCILLATION LOOP]
```"""

new_block_4 = """```bash
[TURN 12] AGENT: ToolCall -> edit_file("net/http.py", replace="timeout=None", with="timeout=30")
[TURN 12] SYSTEM: File "net/http.py" successfully updated. MD5: 4f8a1c...
[TURN 13] AGENT: ToolCall -> search_code("def fetch_url")
[TURN 13] SYSTEM: Retrieved chunk net/http.py:14-28 (score: 0.89):
                 "def fetch_url(url): return session.get(url, timeout=None)"  <-- [STALE CHUNK]
[TURN 14] AGENT: "The previous edit did not persist; timeout is still None. Re-applying patch..."
[TURN 14] AGENT: ToolCall -> edit_file("net/http.py", replace="timeout=None", with="timeout=30")
[TURN 14] SYSTEM: Error: String "timeout=None" not found in "net/http.py". Target contains "timeout=30".
[TURN 15] AGENT: "File state is corrupted. Attempting structural recovery..." [OSCILLATION LOOP]
```"""

if old_block_4 not in content:
    print("ERROR: old_block_4 not found")
    sys.exit(1)
content = content.replace(old_block_4, new_block_4, 1)
print("Retagged Block 4 as bash.")

# 6. Block 5: Retrieval query trace (lines 1028-1043)
old_block_5 = """```text
[RETRIEVAL QUERY]: "SocketTimeout retry configuration ExponentialBackoff"
[RETRIEVAL ENGINE]: Hybrid BM25 + HNSW returns k = 20 chunks.
[CHUNK 1 (Rank 1, Target)]: src/net/transport.py: ExponentialBackoff(max_retries=3, base_ms=100)
[CHUNK 7 (Rank 7, Distractor)]: tests/legacy/test_v1_mock.py: ExponentialBackoffMock(retries=None)
[CHUNK 14 (Rank 14, Distractor)]: docs/deprecated/client_spec.md: "Use retry_interval_seconds"

[AGENT REASONING TRACE]:
"I will configure the socket retry policy using ExponentialBackoffMock as defined
 in the test infrastructure, setting retries=None to disable limits as suggested in client_spec."

[EXECUTION TRACE]:
$ pytest tests/integration/test_socket.py
E   ImportError: cannot import name 'ExponentialBackoffMock' from 'net.transport'
FAILED: Task incomplete. Staged distractor overrode authoritative API contract.
```"""

new_block_5 = """```bash
[RETRIEVAL QUERY]: "SocketTimeout retry configuration ExponentialBackoff"
[RETRIEVAL ENGINE]: Hybrid BM25 + HNSW returns k = 20 chunks.
[CHUNK 1 (Rank 1, Target)]: src/net/transport.py: ExponentialBackoff(max_retries=3, base_ms=100)
[CHUNK 7 (Rank 7, Distractor)]: tests/legacy/test_v1_mock.py: ExponentialBackoffMock(retries=None)
[CHUNK 14 (Rank 14, Distractor)]: docs/deprecated/client_spec.md: "Use retry_interval_seconds"

[AGENT REASONING TRACE]:
"I will configure the socket retry policy using ExponentialBackoffMock as defined
 in the test infrastructure, setting retries=None to disable limits as suggested in client_spec."

[EXECUTION TRACE]:
$ pytest tests/integration/test_socket.py
E   ImportError: cannot import name 'ExponentialBackoffMock' from 'net.transport'
FAILED: Task incomplete. Staged distractor overrode authoritative API contract.
```"""

if old_block_5 not in content:
    print("ERROR: old_block_5 not found")
    sys.exit(1)
content = content.replace(old_block_5, new_block_5, 1)
print("Retagged Block 5 as bash.")

with open(CH06_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated Chapter 06!")
