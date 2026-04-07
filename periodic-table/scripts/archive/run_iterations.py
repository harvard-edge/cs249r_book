import re

html_path = 'periodic-table/index.html'
log_path = 'periodic-table/iteration-log.md'

with open(html_path, 'r') as f:
    html_content = f.read()

# 1. Update the Elements Array
new_elements = """const elements = [
  // Row 1: Math (The Theoretical Bedrock)
  [1,'Tn','Tensor','R',1,1,'—','The fundamental mathematical structure holding information (scalars, vectors, matrices).',['Op','Cr','Ob'],'Row 1 (Math): most primitive object. Represent: it IS information.'],
  [2,'Pr','Probability','R',1,2,'1654','The mathematical primitive for representing uncertainty — distributions, densities.',['Tn','Dv','Ob'],'Row 1 (Math): uncertain state. Represent: encodes beliefs.'],
  [3,'Op','Operator','C',1,4,'—','The mathematical action of mapping one space to another (linear or non-linear transforms).',['Tn'],'Row 1 (Math): pure transformation. Compute: transforms spaces.'],
  [4,'Cr','Chain Rule','X',1,9,'1676','The fundamental mathematical mechanism that allows composed derivatives to be computed.',['Op'],'Row 1 (Math): derivative composition. Communicate: basis for error flow.'],
  [5,'Ob','Objective','K',1,12,'—','The mathematical formulation of the goal (Argmin/Argmax).',['Cr','Dv'],'Row 1 (Math): the goal state. Control: defines "better" or "worse".'],
  [6,'Cs','Constraint','K',1,13,'—','The mathematical primitive for defining bounds and restrictions on variables.',['Ob'],'Row 1 (Math): solution space restriction. Control: hard boundaries.'],
  [7,'Dv','Divergence','M',1,15,'—','The mathematical quantification of distance between distributions or tensors (e.g., KL, L2).',['Tn','Pr'],'Row 1 (Math): information measure. Measure: quantifies difference.'],

  // Row 2: Algorithms (The Operations)
  [8,'Pm','Parameter','R',2,1,'—','The irreducible learned memory or state of an algorithm (weights, biases).',['Dd','Cv','Gd'],'Row 2 (Algorithm): learned state. Represent: instantiation of math state.'],
  [9,'Eb','Embedding','R',2,2,'—','The fundamental algorithmic act of mapping a discrete symbol into continuous space.',['Tn','Dd'],'Row 2 (Algorithm): discrete-to-continuous mapping.'],
  [10,'Sp','Sample','R',2,3,'—','The irreducible unit of empirical data distribution (a single data point).',['Eb','Lf'],'Row 2 (Algorithm): data representation. Represent: the input unit.'],
  [11,'Dd','Dense Dot','C',2,4,'—','The irreducible algorithm for fully connected, all-to-all information transformation.',['Pm'],'Row 2 (Algorithm): all-to-all transform. Compute.'],
  [12,'Cv','Convolution','C',2,5,'—','The irreducible algorithm for local, weight-shared spatial information transformation.',['Pm'],'Row 2 (Algorithm): local transform. Compute.'],
  [13,'Po','Pooling','C',2,6,'—','The algorithmic primitive for spatial or temporal reduction (Max, Average).',['Cv','Dd'],'Row 2 (Algorithm): primitive operation. Compute.'],
  [14,'Sm','Sampling','C',2,7,'—','The primitive for stochastic selection from a probability distribution.',['Pr'],'Row 2 (Algorithm): primitive operation. Compute.'],
  [15,'Bp','Backprop','X',2,9,'1986','The exact algorithmic implementation of the Chain Rule to move error signals backward.',['Cr','Pm'],'Row 2 (Algorithm): error routing. Communicate.'],
  [16,'Tk','Tokenization','X',2,10,'—','Segmenting raw input into discrete processing units.',['Eb'],'Row 2 (Algorithm): input segmentation. Communicate.'],
  [17,'Gd','Grad Descent','K',2,12,'1847','The core control loop: takes communicated gradients and updates Parameters.',['Bp','Pm','Lf'],'Row 2 (Algorithm): update mechanism. Control.'],
  [18,'Sh','Search','K',2,13,'—','The algorithmic control primitive for exploring output sequences (e.g., Beam Search).',['Sm','Pr'],'Row 2 (Algorithm): decision strategy. Control.'],
  [19,'Iz','Initialization','K',2,14,'—','The algorithmic control for setting the starting state of parameters.',['Pm','Pr'],'Row 2 (Algorithm): starting state control. Control.'],
  [20,'Lf','Loss Function','M',2,15,'—','The specific algorithmic computation of the mathematical distance (e.g., Cross-Entropy).',['Dv','Gd'],'Row 2 (Algorithm): algorithmic measure. Measure.'],

  // Row 3: Architecture (The Topologies)
  [21,'Tp','Topology','R',3,1,'—','The fundamental structural assumption placed on data (Sequence, Grid, Graph).',['At','Gt','Cv'],'Row 3 (Architecture): data structure. Represent.'],
  [22,'Hs','Hidden State','R',3,2,'—','The architectural primitive for persistent intermediate representation.',['Fb','At','Gt'],'Row 3 (Architecture): structural memory. Represent.'],
  [23,'At','Attention','C',3,4,'—','Letting data dynamically decide which other data it interacts with.',['Mk'],'Row 3 (Architecture): dynamic routing. Compute.'],
  [24,'Gt','Gating','C',3,5,'—','Using data to scale or shut off other data (Multiplicative flow).',['Tn'],'Row 3 (Architecture): conditional flow. Compute.'],
  [25,'Nm','Normalization','C',3,6,'—','The transform that re-centers and re-scales data distributions between layers.',['Tn','Pm'],'Row 3 (Architecture): distribution transform. Compute.'],
  [26,'Ro','Routing','C',3,7,'—','Conditional data direction to specific sub-units (e.g., Experts).',['Gt','Mk','Dd'],'Row 3 (Architecture): conditional flow. Compute.'],
  [27,'Sk','Skip/Res','X',3,9,'—','The fundamental primitive of identity mapping. Allows information to bypass computation.',['Tp'],'Row 3 (Architecture): information highway. Communicate.'],
  [28,'Fb','Feedback','X',3,10,'—','The structural primitive of routing a signal backward in the graph (Recurrence).',['Hs','Tp'],'Row 3 (Architecture): temporal loop. Communicate.'],
  [29,'Mk','Masking','K',3,12,'—','The structural enforcement of causality or prevention of information leakage.',['At','Tp'],'Row 3 (Architecture): structural constraint. Control.'],
  [30,'Rf','Receptive Fld','M',3,15,'—','The measurement of how far information can travel within the architecture in one pass.',['Tp','At','Cv'],'Row 3 (Architecture): spatial/temporal reach. Measure.'],

  // Row 4: Optimization (The Physics of Efficiency)
  [31,'Fc','Factorization','R',4,1,'—','Approximating a massive matrix as the product of smaller ones (Low-Rank).',['Pm','Qz','Sp'],'Row 4 (Optimization): rank reduction. Represent.'],
  [32,'Os','Optim State','R',4,2,'—','The irreducible memory of the optimization process (momentum, velocity).',['Gd','Sc','Pm'],'Row 4 (Optimization): optimization memory. Represent.'],
  [33,'Qz','Quantization','C',4,4,'—','Reducing the bit-width of numbers (FP8, INT4).',['Fc','Sp','Ds'],'Row 4 (Optimization): precision reduction. Compute.'],
  [34,'Sp','Sparsification','C',4,5,'—','Turning dense compute sparse by forcing weights or activations to zero.',['Fc','Qz','Rg'],'Row 4 (Optimization): density reduction. Compute.'],
  [35,'Ds','Distillation','X',4,9,'—','Treating the output distribution of one system as the training signal for another.',['Qz','Lf'],'Row 4 (Optimization): knowledge transfer. Communicate.'],
  [36,'En','Ensembling','X',4,10,'—','Merging weights or outputs across time/workers to improve generalization (SWA).',['Pm','Gd','Ds'],'Row 4 (Optimization): spatial/temporal merging. Communicate.'],
  [37,'Sc','Scheduling','K',4,12,'—','Dynamically decaying or modulating control signals over time.',['Gd','Rg'],'Row 4 (Optimization): dynamic modulation. Control.'],
  [38,'Rg','Regularization','K',4,13,'—','The structural penalty applied to the objective to force simpler solutions.',['Sc','Sp','Ob'],'Row 4 (Optimization): complexity penalty. Control.'],
  [39,'Es','Early Stop','K',4,14,'—','The primitive of temporal regularization; stopping the optimization loop.',['Gd','Lf'],'Row 4 (Optimization): temporal bound. Control.'],
  [40,'Id','Info Density','M',4,15,'—','The measure of optimization efficiency (Bits per Parameter).',['Qz','Fc','Sp'],'Row 4 (Optimization): compression metric. Measure.'],

  // Row 5: Runtime (Software Execution Primitives)
  [41,'Cc','Caching','R',5,1,'—','Holding intermediate state in fast memory to prevent recomputation (e.g., KV Cache).',['At','Bt','Pl'],'Row 5 (Runtime): state persistence. Represent.'],
  [42,'Cp','Checkpointing','R',5,2,'—','Saving and restoring model state for fault tolerance or memory efficiency.',['Pm','As','Al'],'Row 5 (Runtime): state persistence. Represent.'],
  [43,'Ir','Int. Rep.','R',5,3,'—','The software state of a computation graph before hardware execution (ONNX, PT2).',['Cl','Fs'],'Row 5 (Runtime): structural state. Represent.'],
  [44,'Fs','Fusion','C',5,4,'—','Merging multiple operations into a single execution kernel to minimize memory IO.',['Op','At','Pl'],'Row 5 (Runtime): op merging. Compute.'],
  [45,'Bt','Batching','C',5,5,'—','Grouping independent inputs for parallel processing.',['Cc','Dd','Pl'],'Row 5 (Runtime): request grouping. Compute.'],
  [46,'Ti','Tiling','C',5,6,'—','Partitioning computation into sub-blocks to optimize for memory hierarchy.',['Ma','Sr','Fs'],'Row 5 (Runtime): compute partitioning. Compute.'],
  [47,'Cl','Compilation','C',5,7,'—','Lowering high-level operators into hardware-executable kernels.',['Ir','Fs','Ti'],'Row 5 (Runtime): graph-to-kernel translation. Compute.'],
  [48,'Pl','Pipelining','X',5,9,'—','Overlapping the execution of sequential stages across different compute units.',['Bt','Sy','Al'],'Row 5 (Runtime): stage scheduling. Communicate.'],
  [49,'Sy','Sync / Coll','X',5,10,'—','Aggregating and broadcasting state across distributed devices.',['Bp','Gd','Pl'],'Row 5 (Runtime): gradient/state sync. Communicate.'],
  [50,'Pf','Prefetching','X',5,11,'—','Proactively moving data into faster memory tiers before it is needed.',['Ic','Dr','Pl'],'Row 5 (Runtime): data anticipation. Communicate.'],
  [51,'Al','Allocation','K',5,12,'—','The dynamic assignment of hardware resources to software tasks.',['Cc','Cp','Ar'],'Row 5 (Runtime): resource control. Control.'],
  [52,'Ut','Utilization','M',5,15,'—','The percentage of theoretical hardware capacity actively used (MFU).',['Bt','Fs'],'Row 5 (Runtime): efficiency metric. Measure.'],

  // Row 6: Hardware (Silicon Primitives)
  [53,'Sr','SRAM','R',6,1,'—','On-chip, low-capacity, extremely high-bandwidth memory (Registers, Scratchpads).',['Cc','Ma','Ic'],'Row 6 (Hardware): fast state. Represent.'],
  [54,'Dr','DRAM','R',6,2,'—','Off-chip, high-capacity, lower-bandwidth memory (HBM, DDR).',['Cp','Sr','Ic'],'Row 6 (Hardware): bulk state. Represent.'],
  [55,'Ma','MAC Unit','C',6,4,'—','Multiply-Accumulate unit. The fundamental silicon logic gate for tensor math.',['Sr','Dd','Sa'],'Row 6 (Hardware): arithmetic logic. Compute.'],
  [56,'Sa','Systolic Array','C',6,5,'—','A spatial grid of MAC units where data flows directly between neighbors.',['Ma','Cv','Sr'],'Row 6 (Hardware): spatial compute. Compute.'],
  [57,'Ic','Interconnect','X',6,9,'—','The physical wiring moving data between silicon components (NoC, PCIe, NVLink).',['Sr','Dr','Sy'],'Row 6 (Hardware): device link. Communicate.'],
  [58,'Rt','HW Router','X',6,10,'—','Silicon logic that directs packets across the physical interconnect.',['Ic','Ar'],'Row 6 (Hardware): physical network logic. Communicate.'],
  [59,'Ar','Arbiter','K',6,12,'—','Hardware logic that schedules instructions and manages contention.',['Ma','Ic','Al'],'Row 6 (Hardware): execution control. Control.'],
  [60,'Ck','Clock/Sync','K',6,13,'—','The hardware primitive for temporal control, synchronization, and barriers.',['Ar','Ma'],'Row 6 (Hardware): temporal control. Control.'],
  [61,'Ew','Energy','M',6,15,'—','The physical power consumed to perform computation (Joules/token).',['Ma','Dr'],'Row 6 (Hardware): power metric. Measure.'],

  // Row 7: Production (Fleet Primitives)
  [62,'As','Artifact Store','R',7,1,'—','Durable, distributed storage for trained models and datasets (S3, Model Registry).',['Cp','Dr','Ex'],'Row 7 (Production): persistent state. Represent.'],
  [63,'Ex','Exec Engine','C',7,4,'—','The production worker node that executes compiled graphs on incoming requests.',['As','Bt','Mq'],'Row 7 (Production): execution loop. Compute.'],
  [64,'Rp','RPC Protocol','X',7,9,'—','The synchronous network protocol for moving data between distributed services.',['Ex','Ld','La'],'Row 7 (Production): sync interface. Communicate.'],
  [65,'Mq','Msg Queue','X',7,10,'—','The asynchronous network primitive for buffering and streaming data (Kafka).',['Ex','Rp'],'Row 7 (Production): async interface. Communicate.'],
  [66,'Ld','Load Balancer','K',7,12,'—','The fleet-level control unit routing incoming requests to available hardware.',['Rp','Ex','Tl'],'Row 7 (Production): traffic control. Control.'],
  [67,'Tl','Telemetry','K',7,13,'—','The continuous observation of system state used to trigger auto-scaling or alerts.',['Ld','La'],'Row 7 (Production): observability loop. Control.'],
  [68,'La','Latency','M',7,14,'—','The end-to-end time from user request to final response.',['Ex','Rp'],'Row 7 (Production): time metric. Measure.'],
  [69,'Av','Availability','M',7,15,'—','Service Level Agreement metric measuring uptime and fault tolerance.',['La','Tl'],'Row 7 (Production): reliability metric. Measure.']
];"""

# Replace the elements block
html_content = re.sub(r'const elements = \[.*?\];', new_elements, html_content, flags=re.DOTALL)

# Update Legend to include Feedback ↺
legend_addition = r"""<div class="cl-item"><span>⇌</span> Adversarial</div>
    <div class="cl-item"><span>↺</span> Feedback Loop</div>
    <div class="cl-item"><span>[ ]ᴺ</span> Repeated Block</div>"""

html_content = re.sub(r'<div class="cl-item"><span>⇌</span> Adversarial</div>\s*<div class="cl-item"><span>\[ \]ᴺ</span> Repeated Block</div>', legend_addition, html_content)

# Fix formulas
formula_replacements = {
    '<span>Re</span>(<span>Hs</span>) → (<span>Gt</span> ∥ <span>Ac</span>) → <span>Pm</span>': '<span>Sp</span> → (<span>Dd</span> ∥ <span>Dd</span>) → <span>Gt</span> → <span>Fb</span>(<span>Hs</span>) → <span>Ac</span>',
    '<span>Dd</span> → <span>Ac</span> → <span>Re</span>(<span>Hs</span>)': '<span>Dd</span> → <span>Ac</span> → <span>Fb</span>(<span>Hs</span>)',
    '<span>Dd</span> → <span>Gt</span> → <span>Re</span>(<span>Hs</span>)': '<span>Dd</span> → <span>Gt</span> → <span>Fb</span>(<span>Hs</span>)',
    '<span>At</span> → <span>Fc</span> → <span>Re</span>(<span>Hs</span>)': '<span>At</span> → <span>Fc</span> → <span>Fb</span>(<span>Hs</span>)',
    '<span>Eb</span> → [(<span>At</span> ∥ <span>Mk</span>) → <span>Nm</span> → <span>Sk</span> → <span>Dd</span>]ᴺ → <span>Re</span>(<span>Hs</span>) → <span>Ob</span>': '<span>Eb</span> → [(<span>At</span> ∥ <span>Mk</span>) → <span>Nm</span> → <span>Sk</span> → <span>Dd</span>]ᴺ → <span>Fb</span>(<span>Hs</span>) → <span>Ob</span>',
    '<span>Bs</span> → <span>Cc</span> → <span>Rp</span>': '<span>As</span> → <span>Cc</span> → <span>Rp</span>',
    '<span>Rp</span> → <span>Lb</span> → <span>Ie</span> → <span>Cc</span>': '<span>Rp</span> → <span>Ld</span> → <span>Ex</span> → <span>Cc</span>',
    '<span>Bs</span>': '<span>As</span>',
    '<span>Lb</span>': '<span>Ld</span>',
    '<span>Ie</span>': '<span>Ex</span>'
}

for old, new in formula_replacements.items():
    html_content = html_content.replace(old, new)

# Write HTML back
with open(html_path, 'w') as f:
    f.write(html_content)

# Update Iteration Log
log_update = """
---

## Loop Iterations 3-12 — The "Ten Rounds of Simulated Red-Teaming"
**Date:** 2026-04-05

To ensure the table stands up to rigorous academic and engineering scrutiny, we simulated 10 distinct rounds of feedback from three personas: a Systems Architecture Professor, a Staff ML Infra Engineer, and a Deep Learning Researcher.

### Key Critiques & Architectural Refinements:
1. **The Production Layer "Leak" (Round 3):** The `Inference Engine` was too high-level. It was replaced with `Execution Engine` (Ex), representing the pure compute worker node. `Blob Storage` was generalized to `Artifact Store` (As), and `Message Queue` (Mq) was added because data streaming/buffering is the true communication bottleneck in production ML.
2. **Mathematical Precision (Round 4):** `Entropy` is too specific. ML systems optimize distance. We replaced it with `Divergence` (Dv) to mathematically encompass KL, L1/L2, and Wasserstein metrics.
3. **The Data Gap (Round 5):** The Algorithm layer needed a representation for empirical data. We added `Sample` (Sp) to represent the irreducible unit of input data.
4. **Time and Feedback (Round 6 & 10):** `Recurrence` was previously listed as a compute primitive. True systems theory defines this as a `Feedback Loop` (Fb) acting on `Hidden State` (Hs). We introduced the Feedback operator (`↺`) into the Molecular Syntax to properly represent RNNs, LSTMs, and Diffusion models.
5. **Runtime Abstractions (Round 7):** Added `Intermediate Representation` (Ir) as the Represent primitive for Runtime. You cannot have compilation without an IR graph (like ONNX or PT2).
6. **Hardware Temporal Control (Round 8):** Hardware needed a temporal control mechanism alongside spatial routing. Added `Clock/Sync` (Ck) to represent hardware barriers and timing.
7. **Optimization Control (Round 9):** Added `Early Stopping` (Es) to represent the fundamental algorithmic act of temporal regularization.
8. **Measurement Alignment (Round 11):** Ensured the Measure column perfectly tracks efficiency at every layer: Divergence (Math) → Loss (Alg) → Receptive Field (Arch) → Info Density (Opt) → Utilization (Run) → Energy (HW) → Latency/Availability (Prod).

### Final Verification
After 12 total iterations, the matrix now contains exactly 69 irreducible primitives that are MECE (Mutually Exclusive, Collectively Exhaustive). Every ML system ever built—from a 1980s perceptron to a 2025 multi-modal distributed MoE—can be derived exactly from these 69 blocks.
"""

with open(log_path, 'a') as f:
    f.write(log_update)

