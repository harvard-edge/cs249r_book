# ML Systems Design Grammar

The primitive catalog supplies the parts; the design grammar supplies the method.
The deeper teaching idea is that fast-moving ML systems are built from a small
set of stable primitives and rewrite rules.

## Core Claim

New ML systems usually look novel because the scale, hardware, or product
constraint changed. Underneath, many of them are built by recombining the same
building blocks:

```text
naive system + binding constraint -> rewrite rule -> feasible system
```

The primitive catalog is the parts list. The design grammar is the method.

## Five Pieces

We write the grammar as `G=(P,O,T,C,R)`.

1. **Primitive vocabulary (`P`)** lives in `grammar.yml`.
   They include mathematical objects, algorithmic blocks, runtime mechanisms,
   hardware resources, production controls, and measurements.

2. **Composition operators (`O`)** describe how primitives assemble.
   System Assembly Notation uses operators for sequence, adjacency,
   concurrency, repetition, residency, constraints, and transfer.

3. **Typing and validity rules (`T`)** decide whether an assembly is
   well-formed. Role, layer, residency, dependency, and rewrite preconditions
   make the notation more than a diagram.

4. **Cost semantics (`C`)** explain why the naive system fails.
   Common constraints are capacity, bandwidth, latency, utilization,
   fragmentation, energy, reliability, and cost.

5. **Rewrite rules (`R`)** live in `rewrite-rules.yml`.
   They are transformations such as tiling, fusion, sharding, pipelining,
   batching, caching, prefetching, quantization, virtualization, scheduling,
   routing, and replication.

## Teaching Loop

Every worked example should follow the same protocol:

1. Write the naive system in primitives.
2. Estimate the first constraint that fails.
3. Choose the smallest rewrite rule that relieves that constraint.
4. Recompute the cost and check for the next constraint.

This is the transferable skill. Students do not need to memorize every named
system if they can derive the move from the constraint.

## Example

```text
naive attention
  -> materializes an O(n^2) intermediate in HBM
  -> violates capacity and bandwidth
  -> apply tiling + fusion
  -> FlashAttention-style execution
```

The point is not that the method automatically invents FlashAttention. The
point is that it makes the rewrite feel inevitable once the constraint is
visible.
