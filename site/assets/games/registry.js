/* ============================================================
   MLSysBook Playground — game registry
   The 404 randomizer picks an `available: true` game.
   The volume field records the playground's teaching placement.
   ============================================================ */

window.MLSP = window.MLSP || {};

MLSP.registry = [
  {
    id: "lander",
    volume: 2,
    name: "Gradient Lander",
    tagline: "Balance batch size and learning rate to converge safely.",
    url: "/games/lander.html",
    module: "/assets/games/lander.mjs",
    teaches: "Large Batch Training & Convergence.",
    featured404: true,
    inline404: true,
    available: true
  },
  {
    id: "pipeline",
    volume: 2,
    name: "Pipeline Pacer",
    tagline: "Keep the GPUs fed without bubbling.",
    url: "/games/pipeline.html",
    module: "/assets/games/pipeline.mjs",
    teaches: "Pipeline Parallelism.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "oom",
    volume: 1,
    name: "Tensor Tetris",
    tagline: "Pack training memory before you OOM.",
    url: "/games/oom.html",
    module: "/assets/games/oom.mjs",
    teaches: "Training Memory Constraints.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "prune",
    volume: 1,
    name: "Pulse Prune",
    tagline: "Shrink a network without breaking it.",
    url: "/games/prune.html",
    module: "/assets/games/prune.mjs",
    teaches: "Model Compression & Pruning.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "quantization",
    volume: 1,
    name: "Quantization Sharp Shot",
    tagline: "Compress a model before the target blurs.",
    url: "/games/quantization.html",
    module: "/assets/games/quantization.mjs",
    teaches: "Mixed-Precision Quantization.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "batch",
    volume: 1,
    name: "Batch Size Balancer",
    tagline: "Push throughput to the edge of OOM.",
    url: "/games/batch.html",
    module: "/assets/games/batch.mjs",
    teaches: "Throughput vs. Memory.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "moe",
    volume: 2,
    name: "MoE Router",
    tagline: "Route tokens to the right experts before they expire.",
    url: "/games/moe.html",
    module: "/assets/games/moe.mjs",
    teaches: "Mixture of Experts.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "loader",
    volume: 1,
    name: "Data Loader Dash",
    tagline: "The CPU preparing data before the GPU starves.",
    url: "/games/loader.html",
    module: "/assets/games/loader.mjs",
    teaches: "The I/O Bottleneck.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "checkpoint",
    volume: 2,
    name: "Checkpoint Roulette",
    tagline: "Fault tolerance and checkpointing at scale.",
    url: "/games/checkpoint.html",
    module: "/assets/games/checkpoint.mjs",
    teaches: "Fault Tolerance.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "roofline",
    volume: 1,
    name: "Roofline Rider",
    tagline: "Ride from the memory limit to the compute limit.",
    url: "/games/roofline.html",
    module: "/assets/games/roofline.mjs",
    teaches: "Hardware Acceleration.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "allreduce",
    volume: 2,
    name: "All-Reduce Rhythm",
    tagline: "Keep the gradients flowing in a perfect ring.",
    url: "/games/allreduce.html",
    module: "/assets/games/allreduce.mjs",
    teaches: "Collective Communication.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "topology",
    volume: 2,
    name: "Topology Tycoon",
    tagline: "Build the fabric, avoid the bottlenecks.",
    url: "/games/topology.html",
    module: "/assets/games/topology.mjs",
    teaches: "Network Fabrics.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "kvcache",
    volume: 2,
    name: "KV Cache Packer",
    tagline: "Pack the pages. Defrag the cache.",
    url: "/games/kvcache.html",
    module: "/assets/games/kvcache.mjs",
    teaches: "LLM Serving & PagedAttention.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "cluster",
    volume: 2,
    name: "Cluster Commander",
    tagline: "Pack your workloads, avoid fragmentation.",
    url: "/games/cluster.html",
    module: "/assets/games/cluster.mjs",
    teaches: "Fleet Orchestration.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "context-cache",
    volume: 3,
    name: "Context Cache",
    tagline: "Keep the next action's evidence inside a limited working set.",
    url: "/games/context-cache.html",
    module: "/assets/games/context-cache.mjs",
    teaches: "Context Working Sets.",
    featured404: false,
    inline404: false,
    available: true
  },
  {
    id: "tool-trail",
    volume: 3,
    name: "Tool Trail",
    tagline: "Recover a trajectory without duplicating an external effect.",
    url: "/games/tool-trail.html",
    module: "/assets/games/tool-trail.mjs",
    teaches: "Tool Actuation and Recovery.",
    featured404: false,
    inline404: false,
    available: true
  },
  {
    id: "latency-line",
    volume: 4,
    name: "Latency Line",
    tagline: "Get a physical action through the pipeline before its deadline.",
    url: "/games/latency-line.html",
    module: "/assets/games/latency-line.mjs",
    teaches: "Action Latency.",
    featured404: false,
    inline404: false,
    available: true
  },
  {
    id: "safety-gate",
    volume: 4,
    name: "Safety Gate",
    tagline: "Keep the rover inside the safe set.",
    url: "/games/safety-gate.html",
    module: "/assets/games/safety-gate.mjs",
    teaches: "Safety Enforcement.",
    featured404: false,
    inline404: false,
    available: true
  }
];

MLSP.pickRandomGame = function() {
  var avail = MLSP.registry.filter(function(g) { return g.available; });
  if (avail.length === 0) return null;
  return avail[Math.floor(Math.random() * avail.length)];
};
