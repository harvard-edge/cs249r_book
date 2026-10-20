# Original activation-checkpointing animation

Factory: `original-activation-checkpointing`. It enhances the source's five forward/backward activation positions and retains the two green checkpoint locations. The two rows represent phases of the same chain, not duplicate storage.

The declared deterministic toy chain has four trainable scalar layers, `a_i = tanh(w_i a_(i-1))`, fixed weights `w_i = 1`, input `a0 = 0.5`, and loss `L = a4`. Checkpoints are a0 and a3. Both input and output activations are needed for each parameter derivative, so recomputation serves a real dependency. Values use unrounded arithmetic internally; displayed values are rounded. The derivative worksheet stays collapsed.

Forward outputs remain live until their next forward consumer finishes. Non-checkpoint values are then discarded. At backward, replay a4 from a3; finish B4 before releasing a4. Replay a1 and a2 once from a0; retain both while B3 uses a2/a3, B2 uses a1/a2, and B1 uses a0/a1. Release each after its final consumer. An explicit rightward replay arrow distinguishes recomputation from the original leftward backward arrows. This toy schedule counts activation slots and replayed layer calls, not bytes, total training memory, or speedup. RNG and stateful-layer restoration are outside its deterministic scope.

Run from the repository root:

```sh
python3 books/_prototypes/vol1-web-animations/original-activation-checkpointing/check.py
```

The focused test serves the complete canonical owning chapter found through the local art index and injects the factory/player for review without editing manuscript enrollment. It checks all endpoints and exact menu jumps, activation dependencies and releases, independently computed gradient endpoints, keyboard controls, reduced motion, real-motion pause/resume in Expand, overflow, and print hiding across Chromium/WebKit, desktop/phone, and light/dark. The older fixture clips ordinary phone playback after repeated screenshots, so production ordinary visible playback and complete static fallback remain separate integration checks. PNGs default to `/tmp/original-activation-checkpointing-review`.

This is agent review and browser verification, not real student testing or a learning-effectiveness result.
