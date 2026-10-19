# Why does batch size change the memory peak?

Show actual activation-buffer identities appearing and disappearing in both source batch cases, while each layer’s gradients materialize and persistent model state survives. Compute totals and peak events from the source formulas.

Scope: Source network 784–128–64–10, FP32, batch 32 and 1,024. Weights and two Adam buffers persist. Backward releases saved outputs as gradients accumulate. Meters share one scale across both batch cases; the source plot retains gradients at update.

Guard: Parameter-related state includes weights, gradients, and Adam buffers. Follow the source timeline: input survives through B1 and is released at update; gradients remain resident until a separate clearing action. Exclude transient workspace and transfer/timing claims.

Run `/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-training-step-memory/check.py`. Fixture uses the complete existing chapter; production enrollment and static builds remain orchestration work.

## Original visual scaffold

The verified original two-batch phase-history plot is primary, preserving the different peak shapes and complete source legend. Current-event bands identify the endpoint; the live activation/gradient identities and common-scale meters below explain that endpoint. The update retains parameter-related state, including gradients; clearing gradients is a separate action.

The Inspect source geometry menu crops the same verified artwork for readable inspection and never changes state. Checks verify the original scaffold, mapped source-node values where applicable, unchanged state under camera selection, chapter overflow, and the retained playback/numerical invariants. QA fixtures and screenshots are local review artifacts; production enrollment and static builds remain separate.

The core presents one source diagram and a concise live result. Full arithmetic and any detailed companion inventory are available in the closed Inspect the calculation disclosure; training keeps only its two compact common-scale live meters beside the source plot. DLRM booking opacity, the hypothetical rejected probe, release, and canary admission animate the original plot regions directly.
