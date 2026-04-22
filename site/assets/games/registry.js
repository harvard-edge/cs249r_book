/* ============================================================
   MLSys Playground — game registry
   Single source of truth. The 404 randomiser picks an
   `available: true` game; the gallery lists them all.
   ============================================================ */

window.MLSP = window.MLSP || {};

MLSP.registry = [
  {
    id: "prune",
    name: "Pulse Prune",
    tagline: "Click the dim weights. Keep the bright ones.",
    url: "/games/prune.html",
    script: "/assets/games/prune.js",
    teaches: "Magnitude-based pruning.",
    available: true
  },
  {
    id: "roofline",
    name: "Roofline Runner",
    tagline: "Catch kernels under the ceiling.",
    url: "/games/roofline.html",
    script: "/assets/games/roofline.js",
    teaches: "Memory-bound vs compute-bound; the ridge point.",
    available: true
  },
  {
    id: "oom",
    name: "OOM",
    tagline: "Pack tensors into HBM before you crash.",
    url: "/games/oom.html",
    script: "/assets/games/oom.js",
    teaches: "GPU memory management under live allocation.",
    available: true
  },
  {
    id: "quantization",
    name: "Quantization Cliff",
    tagline: "Dial layer precisions. Ship within budget.",
    url: "/games/quantization.html",
    script: "/assets/games/quantization.js",
    teaches: "Mixed-precision quantization and layer sensitivity.",
    available: true
  }
];

MLSP.pickRandomGame = function() {
  var avail = MLSP.registry.filter(function(g) { return g.available; });
  if (avail.length === 0) return null;
  return avail[Math.floor(Math.random() * avail.length)];
};
