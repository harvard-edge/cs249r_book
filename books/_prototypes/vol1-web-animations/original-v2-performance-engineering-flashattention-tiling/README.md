# Why must online softmax rescale earlier tiles?

Source-art prototype for `vol2:fig-performance-engineering-flashattention-tiling`. Static source and caption stay unchanged.

Declared scalar worked case for one query: score blocks [1],[3], value blocks [2],[6]. The original HBM ↔ registers/SRAM boundary is retained. Values are toy data, not the chapter’s model workload.

Invariant: Invariant: l=Σexp(s−m), a=Σexp(s−m)v over consumed blocks; not a speedup model.

Run `python3 books/_prototypes/vol1-web-animations/original-v2-performance-engineering-flashattention-tiling/check.py` for actual-chapter fixtures and browser checks. Review screenshots are written only under `/tmp`.

## Frozen review assets

`original-v2-performance-engineering-flashattention-tiling.js`: `9ec7209bdffe07e6acb58f8412706281cde2f4bf606047226812575b0e13ad3f`
`original-v2-performance-engineering-flashattention-tiling.css`: `709644667db3366ce1305d10f849766469415b13c56069bb302bd5634e6bbb5a`
`entry.json`: `f6b7057791c8ffab0dc0ad1844fb1e09c1b8ea243fa0d79d5e1609ad0e9ac56f`
