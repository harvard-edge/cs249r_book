# How do block coordinates select reads and retain output sums?

Use each exact-pattern coordinate entry to select a dense activation block and update the correct retained output tile. Empty coordinates schedule no block multiplication.

Scope: Source pattern: a 3 × 3 grid of tiles, each dense tile 3 × 3; six stored tiles at (0,0),(0,1),(1,0),(1,2),(2,1),(2,2). Supporting numeric case: every stored scalar weight is 1; the activation column contains 1–9. Colors in the original do not encode these toy values.

Guard: The 3 × 3 tile size matches the source; toy weights are explicitly separate from source colors. No Tensor Core, alignment, bandwidth, or speedup assertion.

Run `/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-block-sparse-gemm/check.py`. Fixture uses the complete existing chapter; production enrollment and static builds remain orchestration work.

## Original visual scaffold

The verified original SVG is the primary diagram: blue dense 3 × 3 tiles in the sparse grid, packed nonzero mini-matrices and adjacent coordinate index, plus tall input/output matrices. Overlays retain partial sums in the first output column and target contribution motion to actual packed-tile/vector objects. A magnified 3 × 3 by 3 × 1 operation is a view of those same objects, not another allocation. Other columns are schematic context.

The Inspect source geometry menu crops the same verified artwork for readable inspection and never changes state. Checks verify the original scaffold, mapped source-node values where applicable, unchanged state under camera selection, chapter overflow, and the retained playback/numerical invariants. QA fixtures and screenshots are local review artifacts; production enrollment and static builds remain separate.
