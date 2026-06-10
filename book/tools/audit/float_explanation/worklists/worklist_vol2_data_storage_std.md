# Float Exposition Worklist — `data_storage.qmd` (vol2)

Graded against FLOAT_EXPOSITION_STANDARD.md. Caption, fig-alt, in-figure labels, code
comments, and callout interiors do NOT count toward the prose's job.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|------|-------|--------|----|----|----|
| eq   | 🔴    | 3      | 3  | 0  | 0  |
| fig  | 🟠    | 6      | 4  | 2  | 0  |
| tbl  | 🟠    | 5      | 2  | 3  | 0  |
| lst  | 🟡    | 0      | —  | —  | —  |
| **Total** | | **14** | **9** | **5** | **0** |

---

## Findings (⚠️ only)

---

### Finding 1 — `fig-storage-compute-chasm` (fig 🟠) — def L231

**Verbatim ref sentence (L229):**
> @Fig-storage-compute-chasm makes this widening gap visually precise by tracking GPU
> throughput and storage bandwidth side by side on the same timescale.

**Missing move:** Lead-out / Interpret. The cite sentence names what the figure tracks
but delivers no takeaway. The only interpret move is at L353, which is separated from the
first citation by the entire figure and ~120 lines of Python. On first encounter the reader
gets a pointer only; the "so what" does not arrive until after an intervening section
boundary. The lead-in content (why the gap matters) lives inside the perspective callout
at L223-227, which does not count as body prose.

**Where the takeaway currently lives:** Body prose at L353 (second citation context),
after the figure has long since been placed — plus partially in the callout body (L225).

**Rule-compliant diff rewrite (L229 replacement):**

```diff
- @Fig-storage-compute-chasm makes this widening gap visually precise by tracking GPU
- throughput and storage bandwidth side by side on the same timescale.
+ The gap between compute and storage has grown roughly 60-fold since 2016: GPU
+ throughput has risen 236 times while NVMe bandwidth has risen only 4 times.
+ @Fig-storage-compute-chasm plots both trends on the same log scale, making the
+ divergence concrete. Every new GPU generation therefore increases pressure on the
+ data pipeline rather than relieving it.
```

---

### Finding 2 — `fig-storage-pyramid` (fig 🟠) — def L509

**Verbatim ref sentence (cite at L507, embedded in a Python LEGO output line):**
> @Fig-storage-pyramid maps these six tiers into a spatial hierarchy, showing how
> bandwidth decreases and capacity increases at each step away from the accelerator.

**Missing move:** Lead-out / Interpret at the citation site. The sentence names the
figure's content ("maps…") and repeats the bandwidth-vs.-capacity axis already established
in the preceding sentence ("each tier in @tbl-storage-hierarchy-merged drops bandwidth
by 10--100× while increasing capacity by 10--100×"). It adds nothing beyond pointing.
The actual insight (the trade-off reflects the physics of data proximity, from millimeters
to kilometers) lives in the payoff at L515 after the float.

**Where the takeaway currently lives:** Body prose at L515 (payoff paragraph after the
figure).

**Rule-compliant diff rewrite (L507 replacement):**

```diff
- @Fig-storage-pyramid maps these six tiers into a spatial hierarchy, showing how
- bandwidth decreases and capacity increases at each step away from the accelerator.
+ Each step down that hierarchy reflects a physical reality: HBM sits millimeters from
+ the accelerator, while object storage may span kilometers of fiber. @Fig-storage-pyramid
+ arranges the six tiers spatially so that the bandwidth-vs.-capacity trade-off reads
+ directly as distance from the accelerator.
```

---

### Finding 3 — `tbl-storage-assumptions` (tbl 🟠) — def L408

**Verbatim ref sentence (L410):**
> As @tbl-storage-assumptions shows, these inversions have a fourth, subtler dimension:
> the read/write ratio shifts dramatically by lifecycle phase.

**Missing move:** The cite sentence does not interpret the table. It introduces a *new*
dimension (read/write ratio) not represented in the table's columns, effectively pivoting
away from the table's content. The table contrasts ML training access patterns against
database access patterns across three dimensions (access pattern, working set size, I/O
size, read/write ratio). The body prose before the table (L355-407) describes access
patterns and shuffling in detail but never draws the synthesizing "the key result is..."
conclusion the table encodes. The table's core claim — that ML workloads systematically
invert every assumption that traditional database storage was optimized for — is stated
only in the caption, not in body prose.

**Where the takeaway currently lives:** Caption only.

**Rule-compliant diff rewrite — add a brief interpret sentence before the pivot (replace
the L410 opening):**

```diff
- As @tbl-storage-assumptions shows, these inversions have a fourth, subtler dimension:
+ @Tbl-storage-assumptions collects these inversions and shows that ML workloads
+ contradict database storage assumptions on every axis simultaneously: sequential
+ replaces random, cache-overflowing datasets replace cacheable working sets, and
+ large bulk reads replace small transactional I/Os. The table also reveals a fourth,
+ subtler inversion:
  the read/write ratio shifts dramatically by lifecycle phase.
```

---

### Finding 4 — `tbl-data-formats` (tbl 🟠) — def L1302

**Verbatim ref sentence (L1304):**
> The comparison in @tbl-data-formats sets the data-volume term in the pipeline equation,
> turning format selection into a bandwidth-sizing problem.

**Missing move:** Lead-out / Interpret. The cite names the table's *function* in the
argument but does not state the specific conclusion the table encodes. Which format wins
for large-scale training and why? The 500+ words of body prose before the table (L1280-1292)
describe every format's design rationale but make no synthesizing recommendation. The
per-format overhead figures (0 to ~50 µs) live only in the cells. The prose never draws
the decision from the table: that WebDataset and raw binary achieve near-zero per-sample
overhead while individual files impose a catastrophic metadata penalty, which is the
specific result that drives the shard-based format choice in the running example.

**Where the takeaway currently lives:** Cell values only.

**Rule-compliant diff rewrite (replace or extend L1304):**

```diff
- The comparison in @tbl-data-formats sets the data-volume term in the pipeline equation,
- turning format selection into a bandwidth-sizing problem.
+ @Tbl-data-formats makes the decision concrete: individual files impose roughly 50 µs
+ of metadata overhead per sample, while sequential formats (WebDataset, raw binary)
+ reduce that to zero by amortizing file-open cost across thousands of samples in a shard.
+ At the throughputs required by large-scale training, that overhead difference determines
+ whether the storage hardware's bandwidth is realized or consumed by metadata operations.
+ Format selection is therefore a bandwidth-sizing decision as much as a schema choice.
```

---

### Finding 5 — `tbl-data-storage-175b-footprint` (tbl 🟠) — def L2503

**Verbatim ref sentence (L2491, inside `.callout-notebook`):**
> @Tbl-data-storage-175b-footprint assembles the complete storage picture for our running
> example, a 30-day training run of a 175B-parameter model on
> `{python} StorageFootprintTable.training_nodes_str`.

**Missing move:** All three prose moves (lead-in, cite, lead-out) live inside the
`.callout-notebook` interior, which does not count toward the prose's job per the standard.
No body prose outside the callout sets up, cites, or interprets the table. The body
paragraph immediately following the callout (L2509) moves directly to feature stores and
model registries without ever naming the key result the table encodes: that checkpoint I/O
dominates training-data I/O by a large factor for single-epoch language model runs.

**Where the takeaway currently lives:** Callout interior only (L2505).

**Rule-compliant diff rewrite — add a body-prose bridge before the callout (insert
before L2489):**

```diff
+ The full storage footprint for the running example integrates every tier discussed in
+ this chapter. Across a 30-day training run of a 175B-parameter model, checkpoint I/O
+ dominates training-data I/O by a substantial factor, inverting the naive expectation
+ that data loading drives the storage budget. The notebook below assembles those numbers
+ tier by tier.
+
  ::: {#nbk-data-storage-175b-models-storage-footprint .callout-notebook ...}
```

---

## Dangling reference

`@fig-fleet-stack` at L132 has no matching definition in this chapter (no `#fig-fleet-stack` anchor). Likely defined in a parent chapter or shared figure file. Not a float-exposition finding (out of scope for this audit), but recorded here for cross-reference completeness.
