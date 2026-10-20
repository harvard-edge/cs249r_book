# When does a sampled token acquire KV state?

Follow consumed inputs into one growing request-local KV cache, then release that cache before the next independent evaluation.

Original request-local schematic. Three input tiles and two decode passes illustrate ownership; tile counts are schematic, not tensor sizes or timing.

Guard: Snapshots repeat one growing request cache; sampled output acquires KV only on consumption. No KV survives into an independent control evaluation.

Original artwork is the main mechanism surface. The static source and caption remain unchanged. `check.py` creates a disposable fixture in the actual owning chapter; it does not publish or enroll the prototype.

Verification: `python3 books/_prototypes/vol1-web-animations/original-v4-03-request-local-prefill-decode/check.py` runs four viewport/theme profiles against the owning chapter, then the shared HTTP Diagram/Explore fixture (server serves `books` on port 8882). The scene uses schematic object sizes. Browser time and object pixel dimensions do not represent hardware timings.
