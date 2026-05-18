// @generated — DO NOT EDIT BY HAND
// Source of truth: design-grammar catalog data (design-grammar/grammar.yml)
// Regenerate: npm run sync:design-grammar   (from interviews/staffml)
//
// Edits to this file will be overwritten on the next sync. To change
// any primitive, assembly, color, or label, edit grammar.yml and re-run
// the sync script (or just run `npm run dev` — predev hook handles it).

export type RoleKey = "R" | "C" | "X" | "K" | "M";

export interface Role {
  key: RoleKey;
  name: string;
  sub: string;
  color: string;
  cols: number[];
}

export const roles: Record<RoleKey, Role> = {
  R: { key: "R", name: "Represent", sub: "What holds information", color: "#42a5f5", cols: [1, 2, 3, 4] },
  C: { key: "C", name: "Compute", sub: "What transforms", color: "#ef6c00", cols: [5, 6, 7, 8, 9] },
  X: { key: "X", name: "Communicate", sub: "What moves", color: "#26a69a", cols: [10, 11, 12] },
  K: { key: "K", name: "Control", sub: "What decides", color: "#fdd835", cols: [13, 14, 15, 16] },
  M: { key: "M", name: "Measure", sub: "What observes", color: "#78909c", cols: [17, 18] },
};

export const layerLabels = [
  "Data",
  "Math",
  "Algorithms",
  "Architecture",
  "Optimization",
  "Runtime",
  "Hardware",
  "Production",
];

export interface Primitive {
  num: number;
  sym: string;
  name: string;
  role: RoleKey;
  layer: number;
  col: number;
  year: string;
  description: string;
  composition_links: string[];
  rationale: string;
}

export const primitives: Primitive[] = [
  { num: 1, sym: "Tn", name: "Tensor", role: "R", layer: 2, col: 1, year: "—", description: "The fundamental mathematical structure holding information (scalars, vectors, matrices).", composition_links: ["Op", "Cr", "Ob"], rationale: "Layer 1 (Math): most primitive object. Represent: it IS information." },
  { num: 2, sym: "Pr", name: "Probability", role: "R", layer: 2, col: 2, year: "1654", description: "The mathematical primitive for representing uncertainty — distributions, densities.", composition_links: ["Tn", "Dv", "Ob"], rationale: "Layer 1 (Math): uncertain state. Represent: encodes beliefs." },
  { num: 3, sym: "Op", name: "Operator", role: "C", layer: 2, col: 5, year: "—", description: "The mathematical action of mapping one space to another (linear or non-linear transforms).", composition_links: ["Tn"], rationale: "Layer 1 (Math): pure transformation. Compute: transforms spaces." },
  { num: 4, sym: "Cr", name: "Chain Rule", role: "X", layer: 2, col: 10, year: "1676", description: "The fundamental mathematical mechanism that allows composed derivatives to be computed.", composition_links: ["Op"], rationale: "Layer 1 (Math): derivative composition. Communicate: basis for error flow." },
  { num: 5, sym: "Ob", name: "Objective", role: "K", layer: 2, col: 13, year: "—", description: "The mathematical formulation of the goal (Argmin/Argmax).", composition_links: ["Cr", "Dv"], rationale: "Layer 1 (Math): the goal state. Control: defines \"better\" or \"worse\"." },
  { num: 6, sym: "Cs", name: "Constraint", role: "K", layer: 2, col: 14, year: "—", description: "The mathematical primitive for defining bounds and restrictions on variables.", composition_links: ["Ob"], rationale: "Layer 1 (Math): solution space restriction. Control: hard boundaries." },
  { num: 7, sym: "Dv", name: "Divergence", role: "M", layer: 2, col: 17, year: "—", description: "The mathematical quantification of distance between distributions or tensors (e.g., KL, L2).", composition_links: ["Tn", "Pr"], rationale: "Layer 1 (Math): information measure. Measure: quantifies difference." },
  { num: 8, sym: "Pm", name: "Parameter", role: "R", layer: 3, col: 1, year: "—", description: "The irreducible learned memory or state of an algorithm (weights, biases).", composition_links: ["Dd", "Cv", "Gd"], rationale: "Layer 2 (Algorithm): learned state. Represent: instantiation of math state." },
  { num: 9, sym: "Eb", name: "Embedding", role: "R", layer: 3, col: 2, year: "—", description: "The fundamental algorithmic act of mapping a discrete symbol into continuous space.", composition_links: ["Tn", "Dd"], rationale: "Layer 2 (Algorithm): discrete-to-continuous mapping." },
  { num: 10, sym: "Sp", name: "Sample", role: "R", layer: 3, col: 3, year: "—", description: "The irreducible unit of empirical data distribution (a single data point).", composition_links: ["Eb", "Lf"], rationale: "Layer 2 (Algorithm): data representation. Represent: the input unit." },
  { num: 11, sym: "Dd", name: "Dense Dot", role: "C", layer: 3, col: 5, year: "—", description: "The irreducible algorithm for fully connected, all-to-all information transformation.", composition_links: ["Pm"], rationale: "Layer 2 (Algorithm): all-to-all transform. Compute." },
  { num: 12, sym: "Cv", name: "Convolution", role: "C", layer: 3, col: 6, year: "—", description: "The irreducible algorithm for local, weight-shared spatial information transformation.", composition_links: ["Pm"], rationale: "Layer 2 (Algorithm): local transform. Compute." },
  { num: 13, sym: "Po", name: "Pooling", role: "C", layer: 3, col: 7, year: "—", description: "The algorithmic primitive for spatial or temporal reduction (Max, Average).", composition_links: ["Cv", "Dd"], rationale: "Layer 2 (Algorithm): primitive operation. Compute." },
  { num: 14, sym: "Sm", name: "Sampling", role: "C", layer: 3, col: 8, year: "—", description: "The primitive for stochastic selection from a probability distribution.", composition_links: ["Pr"], rationale: "Layer 2 (Algorithm): primitive operation. Compute." },
  { num: 15, sym: "Ad", name: "Autodiff", role: "X", layer: 3, col: 10, year: "1970", description: "The algorithmic primitive that mechanically computes exact derivatives through arbitrary control flow.", composition_links: ["Cr", "Pm"], rationale: "Layer 2 (Algorithm): error routing. Communicate." },
  { num: 16, sym: "Tk", name: "Tokenization", role: "X", layer: 3, col: 11, year: "—", description: "Segmenting raw input into discrete processing units.", composition_links: ["Eb"], rationale: "Layer 2 (Algorithm): input segmentation. Communicate." },
  { num: 17, sym: "Gd", name: "Grad Descent", role: "K", layer: 3, col: 13, year: "1847", description: "The core control loop: takes communicated gradients and updates Parameters.", composition_links: ["Ad", "Pm", "Lf"], rationale: "Layer 2 (Algorithm): update mechanism. Control." },
  { num: 18, sym: "Rw", name: "Reward", role: "K", layer: 3, col: 14, year: "—", description: "A scalar control signal evaluating the quality of an action (RL).", composition_links: ["Sp", "Gd"], rationale: "Layer 2 (Algorithm): evaluative signal. Control." },
  { num: 19, sym: "Iz", name: "Initialization", role: "K", layer: 3, col: 15, year: "—", description: "The algorithmic control for setting the starting state of parameters.", composition_links: ["Pm", "Pr"], rationale: "Layer 2 (Algorithm): starting state control. Control." },
  { num: 20, sym: "Lf", name: "Loss Function", role: "M", layer: 3, col: 17, year: "—", description: "The specific algorithmic computation of the mathematical distance (e.g., Cross-Entropy).", composition_links: ["Dv", "Gd"], rationale: "Layer 2 (Algorithm): algorithmic measure. Measure." },
  { num: 21, sym: "Tp", name: "Topology", role: "R", layer: 4, col: 1, year: "—", description: "The fundamental structural assumption placed on data (Sequence, Grid, Graph).", composition_links: ["At", "Gt", "Cv"], rationale: "Layer 3 (Architecture): data structure. Represent." },
  { num: 22, sym: "Hs", name: "Hidden State", role: "R", layer: 4, col: 2, year: "—", description: "The architectural primitive for persistent intermediate representation.", composition_links: ["Fb", "At", "Gt"], rationale: "Layer 3 (Architecture): structural memory. Represent." },
  { num: 23, sym: "At", name: "Attention", role: "C", layer: 4, col: 5, year: "—", description: "Letting data dynamically decide which other data it interacts with.", composition_links: ["Mk"], rationale: "Layer 3 (Architecture): dynamic routing. Compute." },
  { num: 24, sym: "Gt", name: "Gating", role: "C", layer: 4, col: 6, year: "—", description: "Using data to scale or shut off other data (Multiplicative flow).", composition_links: ["Tn"], rationale: "Layer 3 (Architecture): conditional flow. Compute." },
  { num: 25, sym: "Nm", name: "Normalization", role: "C", layer: 4, col: 7, year: "—", description: "The transform that re-centers and re-scales data distributions between layers.", composition_links: ["Tn", "Pm"], rationale: "Layer 3 (Architecture): distribution transform. Compute." },
  { num: 26, sym: "Ro", name: "Routing", role: "C", layer: 4, col: 8, year: "—", description: "Conditional data direction to specific sub-units (e.g., Experts).", composition_links: ["Gt", "Mk", "Dd"], rationale: "Layer 3 (Architecture): conditional flow. Compute." },
  { num: 27, sym: "Sk", name: "Skip/Res", role: "X", layer: 4, col: 10, year: "—", description: "The fundamental primitive of identity mapping. Allows information to bypass computation.", composition_links: ["Tp"], rationale: "Layer 3 (Architecture): information highway. Communicate." },
  { num: 28, sym: "Fb", name: "Feedback", role: "X", layer: 4, col: 11, year: "—", description: "The structural primitive of routing a signal backward in the graph (Recurrence).", composition_links: ["Hs", "Tp"], rationale: "Layer 3 (Architecture): temporal loop. Communicate." },
  { num: 29, sym: "Mk", name: "Masking", role: "K", layer: 4, col: 13, year: "—", description: "The structural enforcement of causality or prevention of information leakage.", composition_links: ["At", "Tp"], rationale: "Layer 3 (Architecture): structural constraint. Control." },
  { num: 30, sym: "Rf", name: "Receptive Fld", role: "M", layer: 4, col: 17, year: "—", description: "The measurement of how far information can travel within the architecture in one pass.", composition_links: ["Tp", "At", "Cv"], rationale: "Layer 3 (Architecture): spatial/temporal reach. Measure." },
  { num: 31, sym: "Fc", name: "Factorization", role: "R", layer: 5, col: 1, year: "—", description: "Approximating a massive matrix as the product of smaller ones (Low-Rank).", composition_links: ["Pm", "Qz", "Sp"], rationale: "Layer 4 (Optimization): rank reduction. Represent." },
  { num: 32, sym: "Os", name: "Optim State", role: "R", layer: 5, col: 2, year: "—", description: "The irreducible memory of the optimization process (momentum, velocity).", composition_links: ["Gd", "Sc", "Pm"], rationale: "Layer 4 (Optimization): optimization memory. Represent." },
  { num: 33, sym: "Qz", name: "Quantization", role: "C", layer: 5, col: 5, year: "—", description: "Reducing the bit-width of numbers (FP8, INT4).", composition_links: ["Fc", "Sp", "Ws"], rationale: "Layer 4 (Optimization): precision reduction. Compute." },
  { num: 34, sym: "Sp", name: "Sparsification", role: "C", layer: 5, col: 6, year: "—", description: "Turning dense compute sparse by forcing weights or activations to zero.", composition_links: ["Fc", "Qz", "Rg"], rationale: "Layer 4 (Optimization): density reduction. Compute." },
  { num: 35, sym: "Ws", name: "Weight Sharing", role: "X", layer: 5, col: 10, year: "1980s", description: "The structural optimization of communicating the same learned state across multiple functional paths (e.g., CNNs).", composition_links: ["Pm", "Tp"], rationale: "Layer 4 (Optimization): state reuse. Communicate." },
  { num: 36, sym: "En", name: "Ensembling", role: "X", layer: 5, col: 11, year: "—", description: "Merging weights or outputs across time/workers to improve generalization (SWA).", composition_links: ["Pm", "Gd", "Ws"], rationale: "Layer 4 (Optimization): spatial/temporal merging. Communicate." },
  { num: 37, sym: "Sc", name: "Scheduling", role: "K", layer: 5, col: 13, year: "—", description: "Dynamically decaying or modulating control signals over time.", composition_links: ["Gd", "Rg"], rationale: "Layer 4 (Optimization): dynamic modulation. Control." },
  { num: 38, sym: "Rg", name: "Regularization", role: "K", layer: 5, col: 14, year: "—", description: "The structural penalty applied to the objective to force simpler solutions.", composition_links: ["Sc", "Sp", "Ob"], rationale: "Layer 4 (Optimization): complexity penalty. Control." },
  { num: 39, sym: "Tm", name: "Termination", role: "K", layer: 5, col: 15, year: "—", description: "The control primitive that evaluates conditions to halt an iterative optimization loop.", composition_links: ["Gd", "Lf"], rationale: "Layer 4 (Optimization): temporal bound. Control." },
  { num: 40, sym: "Id", name: "Info Density", role: "M", layer: 5, col: 17, year: "—", description: "The measure of optimization efficiency (Bits per Parameter).", composition_links: ["Qz", "Fc", "Sp"], rationale: "Layer 4 (Optimization): compression metric. Measure." },
  { num: 41, sym: "Cc", name: "Caching", role: "R", layer: 6, col: 1, year: "—", description: "Holding intermediate state in fast memory to prevent recomputation (e.g., KV Cache).", composition_links: ["At", "Bt", "Pl"], rationale: "Layer 5 (Runtime): state persistence. Represent." },
  { num: 42, sym: "Cp", name: "Checkpointing", role: "R", layer: 6, col: 2, year: "—", description: "Saving and restoring model state for fault tolerance or memory efficiency.", composition_links: ["Pm", "As", "Al"], rationale: "Layer 5 (Runtime): state persistence. Represent." },
  { num: 43, sym: "Ir", name: "Int. Rep.", role: "R", layer: 6, col: 3, year: "—", description: "The software state of a computation graph before hardware execution (ONNX, PT2).", composition_links: ["Cl", "Fs"], rationale: "Layer 5 (Runtime): structural state. Represent." },
  { num: 44, sym: "Fs", name: "Fusion", role: "C", layer: 6, col: 5, year: "—", description: "Merging multiple operations into a single execution kernel to minimize memory IO.", composition_links: ["Op", "At", "Pl"], rationale: "Layer 5 (Runtime): op merging. Compute." },
  { num: 45, sym: "Bt", name: "Batching", role: "C", layer: 6, col: 6, year: "—", description: "Grouping independent inputs for parallel processing.", composition_links: ["Cc", "Dd", "Pl"], rationale: "Layer 5 (Runtime): request grouping. Compute." },
  { num: 46, sym: "Ti", name: "Tiling", role: "C", layer: 6, col: 7, year: "—", description: "Partitioning computation into tiles to optimize for memory hierarchy.", composition_links: ["Ma", "Sr", "Fs"], rationale: "Layer 5 (Runtime): compute partitioning. Compute." },
  { num: 47, sym: "Cl", name: "Compilation", role: "C", layer: 6, col: 8, year: "—", description: "Lowering high-level operators into hardware-executable kernels.", composition_links: ["Ir", "Fs", "Ti"], rationale: "Layer 5 (Runtime): graph-to-kernel translation. Compute." },
  { num: 48, sym: "Pl", name: "Pipelining", role: "X", layer: 6, col: 10, year: "—", description: "Overlapping the execution of sequential stages across different compute units.", composition_links: ["Bt", "Sy", "Al"], rationale: "Layer 5 (Runtime): stage scheduling. Communicate." },
  { num: 49, sym: "Sy", name: "Sync / Coll", role: "X", layer: 6, col: 11, year: "—", description: "Aggregating and broadcasting state across distributed devices.", composition_links: ["Ad", "Gd", "Pl"], rationale: "Layer 5 (Runtime): gradient/state sync. Communicate." },
  { num: 50, sym: "Pf", name: "Prefetching", role: "X", layer: 6, col: 12, year: "—", description: "Proactively moving data into faster memory tiers before it is needed.", composition_links: ["Ic", "Dr", "Pl"], rationale: "Layer 5 (Runtime): data anticipation. Communicate." },
  { num: 51, sym: "Al", name: "Allocation", role: "K", layer: 6, col: 13, year: "—", description: "The dynamic assignment of hardware resources to software tasks.", composition_links: ["Cc", "Cp", "Ar"], rationale: "Layer 5 (Runtime): resource control. Control." },
  { num: 52, sym: "Ut", name: "Utilization", role: "M", layer: 6, col: 17, year: "—", description: "The percentage of theoretical hardware capacity actively used (MFU).", composition_links: ["Bt", "Fs"], rationale: "Layer 5 (Runtime): efficiency metric. Measure." },
  { num: 53, sym: "Sr", name: "SRAM", role: "R", layer: 7, col: 1, year: "—", description: "On-chip, low-capacity, extremely high-bandwidth memory (Registers, Scratchpads).", composition_links: ["Cc", "Ma", "Ic"], rationale: "Layer 6 (Hardware): fast state. Represent." },
  { num: 54, sym: "Dr", name: "DRAM", role: "R", layer: 7, col: 2, year: "—", description: "Off-chip, high-capacity, lower-bandwidth memory (HBM, DDR).", composition_links: ["Cp", "Sr", "Ic"], rationale: "Layer 6 (Hardware): bulk state. Represent." },
  { num: 55, sym: "Ma", name: "MAC Unit", role: "C", layer: 7, col: 5, year: "—", description: "Multiply-Accumulate unit. The fundamental silicon logic gate for tensor math.", composition_links: ["Sr", "Dd", "Vu"], rationale: "Layer 6 (Hardware): arithmetic logic. Compute." },
  { num: 56, sym: "Vu", name: "Vector Unit", role: "C", layer: 7, col: 6, year: "—", description: "Single Instruction, Multiple Data (SIMD) ALU. The silicon primitive for parallel arithmetic.", composition_links: ["Ma", "Sr"], rationale: "Layer 6 (Hardware): parallel compute logic. Compute." },
  { num: 57, sym: "Ic", name: "Interconnect", role: "X", layer: 7, col: 10, year: "—", description: "The physical wiring moving data between silicon components (NoC, PCIe, NVLink).", composition_links: ["Sr", "Dr", "Sy"], rationale: "Layer 6 (Hardware): device link. Communicate." },
  { num: 58, sym: "Rt", name: "HW Router", role: "X", layer: 7, col: 11, year: "—", description: "Silicon logic that directs packets across the physical interconnect.", composition_links: ["Ic", "Ar"], rationale: "Layer 6 (Hardware): physical network logic. Communicate." },
  { num: 59, sym: "Ar", name: "Arbiter", role: "K", layer: 7, col: 13, year: "—", description: "Hardware logic that schedules instructions and manages contention.", composition_links: ["Ma", "Ic", "Al"], rationale: "Layer 6 (Hardware): execution control. Control." },
  { num: 60, sym: "Ck", name: "Clock/Sync", role: "K", layer: 7, col: 14, year: "—", description: "The hardware primitive for temporal control, synchronization, and barriers.", composition_links: ["Ar", "Ma"], rationale: "Layer 6 (Hardware): temporal control. Control." },
  { num: 61, sym: "Ew", name: "Energy", role: "M", layer: 7, col: 17, year: "—", description: "The physical power consumed to perform computation (Joules/token).", composition_links: ["Ma", "Dr"], rationale: "Layer 6 (Hardware): power metric. Measure." },
  { num: 62, sym: "As", name: "Artifact Store", role: "R", layer: 8, col: 1, year: "—", description: "Durable, distributed storage for trained models and datasets (S3, Model Registry).", composition_links: ["Cp", "Dr", "Ex"], rationale: "Layer 7 (Production): persistent state. Represent." },
  { num: 63, sym: "Ex", name: "Exec Engine", role: "C", layer: 8, col: 5, year: "—", description: "The production worker node that executes compiled graphs on incoming requests.", composition_links: ["As", "Bt", "Mq"], rationale: "Layer 7 (Production): execution loop. Compute." },
  { num: 64, sym: "Rp", name: "RPC Protocol", role: "X", layer: 8, col: 10, year: "—", description: "The synchronous network protocol for moving data between distributed services.", composition_links: ["Ex", "Ld", "La"], rationale: "Layer 7 (Production): sync interface. Communicate." },
  { num: 65, sym: "Mq", name: "Msg Queue", role: "X", layer: 8, col: 11, year: "—", description: "The asynchronous network primitive for buffering and streaming data (Kafka).", composition_links: ["Ex", "Rp"], rationale: "Layer 7 (Production): async interface. Communicate." },
  { num: 66, sym: "Ld", name: "Load Balancer", role: "K", layer: 8, col: 13, year: "—", description: "The fleet-level control unit routing incoming requests to available hardware.", composition_links: ["Rp", "Ex", "Oc"], rationale: "Layer 7 (Production): traffic control. Control." },
  { num: 67, sym: "Oc", name: "Orchestrator", role: "K", layer: 8, col: 14, year: "—", description: "The fleet-level control plane that scales, restarts, and manages the lifecycle of execution nodes (e.g., K8s).", composition_links: ["Ld", "Av"], rationale: "Layer 7 (Production): fleet control loop. Control." },
  { num: 68, sym: "La", name: "Latency", role: "M", layer: 8, col: 17, year: "—", description: "The end-to-end time from user request to final response.", composition_links: ["Ex", "Rp"], rationale: "Layer 7 (Production): time metric. Measure." },
  { num: 69, sym: "Av", name: "Availability", role: "M", layer: 8, col: 18, year: "—", description: "Service Level Agreement metric measuring uptime and fault tolerance.", composition_links: ["La", "Oc"], rationale: "Layer 7 (Production): reliability metric. Measure." },
  { num: 70, sym: "Rc", name: "Record", role: "R", layer: 1, col: 1, year: "—", description: "The fundamental atomic unit of raw information (a single row, image, or document).", composition_links: [], rationale: "Layer 0 (Data): the raw state. Represent." },
  { num: 71, sym: "Ds", name: "Dataset", role: "R", layer: 1, col: 2, year: "—", description: "A structured collection of records.", composition_links: ["Rc", "Sm"], rationale: "Layer 0 (Data): the collective state. Represent." },
  { num: 72, sym: "Tr", name: "Transform", role: "C", layer: 1, col: 5, year: "—", description: "The deterministic action of altering raw data (cropping, resizing, parsing).", composition_links: ["Rc"], rationale: "Layer 0 (Data): raw manipulation. Compute." },
  { num: 73, sym: "Ag", name: "Aggregate", role: "C", layer: 1, col: 6, year: "—", description: "Combining multiple records into summary statistics.", composition_links: ["Ds"], rationale: "Layer 0 (Data): statistical manipulation. Compute." },
  { num: 74, sym: "Fl", name: "Flow/Stream", role: "X", layer: 1, col: 10, year: "—", description: "The continuous movement of raw data from source to system (ETL, Kafka).", composition_links: ["Rc", "Ds"], rationale: "Layer 0 (Data): data pipeline. Communicate." },
  { num: 75, sym: "Fm", name: "Format", role: "X", layer: 1, col: 11, year: "—", description: "The structural encoding of data for storage or transit (Parquet, TFRecord).", composition_links: ["Rc", "Fl"], rationale: "Layer 0 (Data): serialization. Communicate." },
  { num: 76, sym: "Fi", name: "Filter", role: "K", layer: 1, col: 13, year: "—", description: "The deterministic logic that includes or excludes records based on predicates.", composition_links: ["Rc", "Tr"], rationale: "Layer 0 (Data): data gating. Control." },
  { num: 77, sym: "Sm", name: "Schema", role: "K", layer: 1, col: 14, year: "—", description: "The structural constraint defining the expected types and fields of a record.", composition_links: ["Rc", "Ds"], rationale: "Layer 0 (Data): type constraint. Control." },
  { num: 78, sym: "Vl", name: "Volume", role: "M", layer: 1, col: 17, year: "—", description: "The physical size or cardinality of the dataset (Bytes, Row Count).", composition_links: ["Ds"], rationale: "Layer 0 (Data): scale metric. Measure." },
  { num: 79, sym: "An", name: "Analog ALU", role: "C", layer: 7, col: 7, year: "—", description: "Continuous-voltage compute unit (e.g., memristor, optical) for extremely low-power inference.", composition_links: ["Ma"], rationale: "Layer 6 (Hardware): non-digital compute. Compute." },
  { num: 80, sym: "En", name: "Entropy", role: "M", layer: 1, col: 18, year: "1948", description: "The Shannon information-theoretic limit; the absolute bound on data compressibility.", composition_links: ["Vl"], rationale: "Layer 0 (Data): information limit. Measure." },
  { num: 81, sym: "Ix", name: "Indexing", role: "R", layer: 4, col: 3, year: "—", description: "The high-dimensional partitioning of vector space (e.g., HNSW) for sub-linear retrieval.", composition_links: ["Tp"], rationale: "Layer 3 (Architecture): structured retrieval. Represent." },
  { num: 82, sym: "Ro", name: "Routing", role: "K", layer: 4, col: 14, year: "—", description: "The dynamic, data-dependent dispatch of tensors (e.g., Mixture of Experts).", composition_links: ["Gt"], rationale: "Layer 3 (Architecture): dynamic flow. Control." },
  { num: 83, sym: "Vr", name: "Virtualization", role: "R", layer: 6, col: 4, year: "—", description: "The abstraction of physical memory via page tables (e.g., PagedAttention) to solve fragmentation.", composition_links: ["Cc"], rationale: "Layer 5 (Runtime): memory mapping. Represent." },
  { num: 84, sym: "Td", name: "Thermodynamics", role: "M", layer: 7, col: 18, year: "—", description: "The ultimate physical limitation (Landauer limit, thermal throttling) capping system scale.", composition_links: ["Ew"], rationale: "Layer 6 (Hardware): thermal limit. Measure." },
  { num: 85, sym: "Rs", name: "Resilience", role: "K", layer: 8, col: 15, year: "—", description: "The systemic countermeasures (checkpointing, elastic recovery) for macroscopic hardware decay.", composition_links: ["Oc"], rationale: "Layer 7 (Production): fault tolerance. Control." },
  { num: 86, sym: "Ac", name: "Activation", role: "C", layer: 3, col: 9, year: "—", description: "Non-linear functions (ReLU, GELU) providing expressive power.", composition_links: ["Dd"], rationale: "Layer 2 (Algorithm): non-linear transform. Compute." },
  { num: 87, sym: "St", name: "State", role: "R", layer: 2, col: 3, year: "—", description: "The mathematical representation of an environment or context (RL, SSMs).", composition_links: ["Ob"], rationale: "Layer 1 (Math): contextual state. Represent." },
  { num: 88, sym: "Re", name: "Retrieve", role: "X", layer: 5, col: 12, year: "—", description: "Fetching stored state or external knowledge (e.g., from a KV Cache or Vector DB).", composition_links: ["Hs"], rationale: "Layer 4 (Optimization): state retrieval. Communicate." },
  { num: 89, sym: "Wa", name: "Weight Avg", role: "C", layer: 5, col: 7, year: "—", description: "Averaging model weights across time or distributed workers (e.g., SWA, EMA).", composition_links: ["Pm"], rationale: "Layer 4 (Optimization): parameter smoothing. Compute." },
  { num: 90, sym: "Ct", name: "Critic", role: "K", layer: 3, col: 16, year: "—", description: "The value function evaluating the expected return of a state (Actor-Critic RL).", composition_links: ["St", "Gd"], rationale: "Layer 2 (Algorithm): evaluative model. Control." },
];

// Lookup map by symbol — last write wins (matches the original
// behavior for documented symbol collisions like Sm/Sp/Ro/En).
export const primitiveMap: Record<string, Primitive> = {};
primitives.forEach((e) => { primitiveMap[e.sym] = e; });

// ── Assemblies ────────────────────────────────────────────────────────────
// Each expression is a list of typed tokens parsed from the YAML's expression
// string at sync time, so the React renderer never has to parse anything.
//   sym tokens reference a primitive by symbol, with optional subscript.
//   op tokens are literal connective text (→ ∥ ⇌ ↺ [ ] ( ) ?).
export type ExpressionToken =
  | { kind: "sym"; sym: string; sub?: string }
  | { kind: "op"; text: string };

export interface Assembly {
  name: string;
  expression: ExpressionToken[];
}

export interface AssemblySection {
  title: string;
  hint?: string;
  items: Assembly[];
}

export const assemblies: AssemblySection[] = [
  {
    title: "Core Architectures",
    hint: "Fundamental end-to-end model structures",
    items: [
      { name: "Transformer", expression: [{ kind: "sym", sym: "Eb" }, { kind: "op", text: " → [(" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: "]ᴺ" }] },
      { name: "Encoder-Decoder Transformer", expression: [{ kind: "sym", sym: "Eb" }, { kind: "op", text: " → [(" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: "]ᴺ_enc → [(" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "At", sub: "cross" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: "]ᴺ_dec → " }, { kind: "sym", sym: "Dd" }] },
      { name: "Vision Transformer (ViT)", expression: [{ kind: "sym", sym: "Tk", sub: "patch" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Eb" }, { kind: "op", text: " → [(" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: "]ᴺ → " }, { kind: "sym", sym: "Dd" }] },
      { name: "Multimodal (Whisper)", expression: [{ kind: "sym", sym: "Tk", sub: "audio" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Eb" }, { kind: "op", text: " → [(" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: "]ᴺ" }] },
      { name: "CNN", expression: [{ kind: "op", text: "[" }, { kind: "sym", sym: "Cv" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ac" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Po" }, { kind: "op", text: "]ᴺ → " }, { kind: "sym", sym: "Dd" }] },
      { name: "ResNet", expression: [{ kind: "sym", sym: "Eb" }, { kind: "op", text: " → [" }, { kind: "sym", sym: "Cv" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: "]ᴺ → " }, { kind: "sym", sym: "Po" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }] },
      { name: "LSTM", expression: [{ kind: "sym", sym: "Sp" }, { kind: "op", text: " → (" }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Gt" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Fb" }, { kind: "op", text: "(" }, { kind: "sym", sym: "Hs" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Ac" }] },
      { name: "State Space Model (SSM)", expression: [{ kind: "sym", sym: "Dd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ac" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Fb" }, { kind: "op", text: "(" }, { kind: "sym", sym: "Hs" }, { kind: "op", text: ")" }] },
      { name: "Mamba (Selective SSM)", expression: [{ kind: "sym", sym: "Dd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Gt" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Fb" }, { kind: "op", text: "(" }, { kind: "sym", sym: "Hs" }, { kind: "op", text: ")" }] },
      { name: "GNN (Graph Neural Network)", expression: [{ kind: "sym", sym: "Tp" }, { kind: "op", text: " → " }, { kind: "sym", sym: "At" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Po" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }] },
    ],
  },
  {
    title: "Structural & Training Patterns",
    hint: "Reusable structures and paradigms",
    items: [
      { name: "Linear Attention", expression: [{ kind: "sym", sym: "At" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Fc" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Fb" }, { kind: "op", text: "(" }, { kind: "sym", sym: "Hs" }, { kind: "op", text: ")" }] },
      { name: "Mixture of Experts (MoE)", expression: [{ kind: "sym", sym: "Ro" }, { kind: "op", text: " ? (" }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " ∥ … ∥ " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Gt" }] },
      { name: "Multi-Head Attention", expression: [{ kind: "sym", sym: "Dd" }, { kind: "op", text: " → (" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") ∥ … ∥ (" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Dd" }] },
      { name: "Batch Normalization", expression: [{ kind: "sym", sym: "Bt" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Nm" }] },
      { name: "Contrastive Learning (CLIP)", expression: [{ kind: "op", text: "(" }, { kind: "sym", sym: "Tk", sub: "img" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Tk", sub: "txt" }, { kind: "op", text: ") → (" }, { kind: "sym", sym: "Eb" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Eb" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ob", sub: "contrastive" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
      { name: "Masked Autoencoder (MAE)", expression: [{ kind: "sym", sym: "Mk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Eb" }, { kind: "op", text: " → [(" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: "]ᴺ → " }, { kind: "sym", sym: "Ob" }] },
    ],
  },
  {
    title: "Generative & Latent Models",
    items: [
      { name: "Diffusion Model", expression: [{ kind: "op", text: "[" }, { kind: "sym", sym: "St" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → (" }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ac" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: ")]ᴺ → " }, { kind: "sym", sym: "Ob" }] },
      { name: "Diffusion Transformer (DiT)", expression: [{ kind: "sym", sym: "Tk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Eb" }, { kind: "op", text: " → [(" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: "]ᴺ → " }, { kind: "sym", sym: "St" }] },
      { name: "VAE", expression: [{ kind: "sym", sym: "Eb" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " → (" }, { kind: "sym", sym: "Pr" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "St" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ob" }] },
      { name: "GAN", expression: [{ kind: "op", text: "(" }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ac" }, { kind: "op", text: ") ⇌ (" }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ac" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Ob" }] },
      { name: "World Model (JEPA/Sora)", expression: [{ kind: "sym", sym: "Eb" }, { kind: "op", text: " → [(" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: "]ᴺ → " }, { kind: "sym", sym: "Fb" }, { kind: "op", text: "(" }, { kind: "sym", sym: "Hs" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Ob" }] },
      { name: "Sparse Autoencoder (SAE)", expression: [{ kind: "sym", sym: "Hs" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ac" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sp" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ob" }] },
    ],
  },
  {
    title: "Efficiency & Optimization",
    items: [
      { name: "Knowledge Distillation", expression: [{ kind: "sym", sym: "Tp", sub: "teacher" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dv" }, { kind: "op", text: " ← " }, { kind: "sym", sym: "Tp", sub: "student" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Gd" }] },
      { name: "Systolic Array (TPU Core)", expression: [{ kind: "op", text: "[" }, { kind: "sym", sym: "Ma" }, { kind: "op", text: " ↔ " }, { kind: "sym", sym: "Ic" }, { kind: "op", text: "]ᴺ" }] },
      { name: "Flash Attention", expression: [{ kind: "sym", sym: "At" }, { kind: "op", text: " → (" }, { kind: "sym", sym: "Ti" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Fs" }, { kind: "op", text: ")" }] },
      { name: "LoRA", expression: [{ kind: "sym", sym: "Pm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Fc" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }] },
      { name: "Adam Optimizer", expression: [{ kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Os" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sc" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
      { name: "Weight Averaging (SWA)", expression: [{ kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Wa" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
      { name: "BitNet (1-bit LLM)", expression: [{ kind: "sym", sym: "Qz" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ac" }] },
      { name: "Quantization-Aware Training (QAT)", expression: [{ kind: "sym", sym: "Qz" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
      { name: "Speculative Decoding", expression: [{ kind: "sym", sym: "St", sub: "draft" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Rw" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Bt" }] },
      { name: "Neural Architecture Search (NAS)", expression: [{ kind: "sym", sym: "Rw" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Tp" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ob" }] },
      { name: "Hyperparameter Optimization (HPO)", expression: [{ kind: "sym", sym: "Rw" }, { kind: "op", text: " → (" }, { kind: "sym", sym: "Sc" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Rg" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Ob" }] },
      { name: "DP-SGD (Differential Privacy)", expression: [{ kind: "op", text: "(" }, { kind: "sym", sym: "St" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Ct" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
    ],
  },
  {
    title: "Alignment & Fine-Tuning",
    items: [
      { name: "RLHF", expression: [{ kind: "sym", sym: "St" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ob", sub: "reward" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
      { name: "DPO", expression: [{ kind: "op", text: "(" }, { kind: "sym", sym: "St" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "St" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Ob" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
      { name: "PPO", expression: [{ kind: "sym", sym: "St" }, { kind: "op", text: " → (" }, { kind: "sym", sym: "Ob" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Ct" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
      { name: "Chain-of-Thought (CoT)", expression: [{ kind: "sym", sym: "St" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Re" }, { kind: "op", text: "(" }, { kind: "sym", sym: "Hs" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Rw" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ob" }] },
      { name: "RAFT", expression: [{ kind: "op", text: "(" }, { kind: "sym", sym: "Eb" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Rw" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Cc" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
      { name: "Prompt Tuning", expression: [{ kind: "sym", sym: "Eb", sub: "prompt" }, { kind: "op", text: " → [(" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Nm" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sk" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Dd" }, { kind: "op", text: "]ᴺ → " }, { kind: "sym", sym: "Dd" }] },
    ],
  },
  {
    title: "Distributed & Scaling",
    items: [
      { name: "Data Parallelism (DP)", expression: [{ kind: "sym", sym: "Bt" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sy" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
      { name: "FSDP (Fully Sharded DP)", expression: [{ kind: "sym", sym: "Bt" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Fc" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sy" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Pm" }] },
      { name: "Pipeline Parallelism (PP)", expression: [{ kind: "sym", sym: "Pl" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sy" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Al" }] },
      { name: "Tensor Parallelism (TP)", expression: [{ kind: "sym", sym: "Fc" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sy" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Al" }] },
      { name: "Federated Learning", expression: [{ kind: "sym", sym: "Gd" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Wa" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Sy" }] },
      { name: "Model Merging / Ensembling", expression: [{ kind: "op", text: "(" }, { kind: "sym", sym: "Pm" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Pm" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Wa" }] },
    ],
  },
  {
    title: "System & Production",
    items: [
      { name: "RAG", expression: [{ kind: "sym", sym: "Eb" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Rw" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Cc" }, { kind: "op", text: " → (" }, { kind: "sym", sym: "At" }, { kind: "op", text: " ∥ " }, { kind: "sym", sym: "Mk" }, { kind: "op", text: ") → " }, { kind: "sym", sym: "Dd" }] },
      { name: "Inference Service", expression: [{ kind: "sym", sym: "Rp" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ld" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Ex" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Cc" }] },
      { name: "Feature Store", expression: [{ kind: "sym", sym: "As" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Cc" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Rp" }] },
      { name: "KV Cache", expression: [{ kind: "sym", sym: "At" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Cc" }] },
      { name: "Gradient Checkpointing", expression: [{ kind: "sym", sym: "Ad" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Cp" }, { kind: "op", text: " → " }, { kind: "sym", sym: "Al" }] },
    ],
  },
];
