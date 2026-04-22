/* ============================================================
   MLSys Playground — game registry
   Single source of truth. The 404 randomiser picks an
   `available: true` game; the gallery lists them all.
   Adding a new game = one entry + flip the flag.
   ============================================================ */

window.MLSP = window.MLSP || {};

MLSP.registry = [
  {
    id: "prune",
    name: "Prune",
    tagline: "Most weights don't matter. Find which ones do.",
    url: "/games/prune.html",
    script: "/assets/games/prune.js",
    teaches: "Magnitude-based pruning and the lottery-ticket hypothesis.",
    available: true
  },
  {
    id: "roofline",
    name: "Roofline Runner",
    tagline: "Catch kernels under the ceiling.",
    url: "/games/roofline.html",
    script: "/assets/games/roofline.js",
    teaches: "Memory-bound vs compute-bound; the ridge point.",
    available: false
  },
  {
    id: "oom",
    name: "OOM",
    tagline: "Survive the forward + backward pass.",
    url: "/games/oom.html",
    script: "/assets/games/oom.js",
    teaches: "Memory hierarchy of training; checkpointing tradeoffs.",
    available: false
  }
];

MLSP.pickRandomGame = function() {
  var avail = MLSP.registry.filter(function(g) { return g.available; });
  if (avail.length === 0) return null;
  return avail[Math.floor(Math.random() * avail.length)];
};
