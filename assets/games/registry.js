/* ============================================================
   MLSysBook Playground — game registry
   Single source of truth. The 404 randomiser picks an
   `available: true` game; the gallery lists them all.
   ============================================================ */

window.MLSP = window.MLSP || {};

MLSP.registry = [
  {
    id: "lander",
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
    name: "MoE Router",
    tagline: "Distribute tokens, balance the experts.",
    url: "/games/moe.html",
    module: "/assets/games/moe.mjs",
    teaches: "Mixture of Experts.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "loader",
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
    name: "Roofline Rider",
    tagline: "The Roofline model: Compute vs Memory bound.",
    url: "/games/roofline.html",
    module: "/assets/games/roofline.mjs",
    teaches: "Hardware Acceleration.",
    featured404: false,
    inline404: true,
    available: true
  },
  {
    id: "allreduce",
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
    name: "Cluster Commander",
    tagline: "Pack your workloads, avoid fragmentation.",
    url: "/games/cluster.html",
    module: "/assets/games/cluster.mjs",
    teaches: "Fleet Orchestration.",
    featured404: false,
    inline404: true,
    available: true
  }
];

MLSP.pickRandomGame = function() {
  var avail = MLSP.registry.filter(function(g) { return g.available; });
  if (avail.length === 0) return null;
  return avail[Math.floor(Math.random() * avail.length)];
};