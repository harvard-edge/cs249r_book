# Which spatial results can both output channels reuse?

Standard outputs accumulate channel contributions independently. Factorized computation produces three spatial intermediates, retains them through both mixing consumers, then releases them.

Scope: The source has three input channels and two outputs at one spatial position, with depth multiplier one. Declared toy values: each 3 × 3 patch is constant 1, 2, or 3. Standard filters use weights 1 and 2. Depthwise weights are 1; pointwise rows are [1,1,1] and [1,0,−1]. Same shape does not imply equivalent layers.

Guard: The declared parameterizations produce different outputs. Same shape and fewer MACs do not establish arbitrary standard-convolution equivalence or speedup.

Run `/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-depthwise-pointwise-channels/check.py`. Fixture uses the complete existing chapter; production enrollment and static builds remain orchestration work.

## Original visual scaffold

The verified original standard and factorized path panels are the primary diagram. Output/intermediate readouts occupy their original nodes; read/compute payloads follow those nodes. The live allocation detail below magnifies the same values. Intermediate release fades in place after both mixers consume them.

The Inspect source geometry menu crops the same verified artwork for readable inspection and never changes state. Checks verify the original scaffold, mapped source-node values where applicable, unchanged state under camera selection, chapter overflow, and the retained playback/numerical invariants. QA fixtures and screenshots are local review artifacts; production enrollment and static builds remain separate.

The core presents the original diagram and a concise live result. Full calculation detail remains in the closed Inspect the calculation disclosure. On phones, Follow operation magnifies a 128-unit window in the original source coordinates; Whole remains available. The camera follows the pending visual operation independently of the last completed numerical state. SVG packets preserve their source coordinates through paused resizing and Expand.

Final Chromium and WebKit checks cover all four viewport/theme profiles, endpoint calculations, normal-motion freeze/resume, paused cross-breakpoint resizing, camera selection, keyboard controls, reduced motion, Expand, completion/replay, and print restoration. These are fixture checks, not production or human learner evidence.
