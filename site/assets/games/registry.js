/* ============================================================
   MLSysBook Playground — game registry
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
    id: "straggler",
    name: "Straggler",
    tagline: "Don't be the slow GPU.",
    url: "/games/straggler.html",
    script: "/assets/games/straggler.mjs",
    teaches: "Tail latency in distributed training; ring all-reduce.",
    /* ESM-only game; the 404 randomiser uses legacy <script> injection,
       so exclude from random rotation until 404 supports module loading.
       Still listed in the gallery via straggler.qmd. */
    available: false
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
    name: "Sharp Shot",
    tagline: "Shoot a target. Lower precision = cheaper but blurry.",
    url: "/games/quantization.html",
    script: "/assets/games/quantization.js",
    teaches: "Mixed-precision quantization and per-layer sensitivity.",
    available: true
  },
  {
    id: "roofline",
    name: "Roofline Runner",
    tagline: "Catch kernels under the ceiling.",
    url: "/games/roofline.html",
    script: "/assets/games/_archive/roofline.js",
    teaches: "Memory-bound vs compute-bound; the ridge point.",
    available: false
  }
];

MLSP.pickRandomGame = function() {
  var avail = MLSP.registry.filter(function(g) { return g.available; });
  if (avail.length === 0) return null;
  return avail[Math.floor(Math.random() * avail.length)];
};
