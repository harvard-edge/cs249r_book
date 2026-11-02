# Why does a lost reply need a new attempt, but the same operation ID?

Follow one operation through a committed effect, lost response, safe retry, and rejected payload conflict while preserving identity.

Illustrative IDs op_A, a1/a2/a3, payload P, and retained result R on the original runtime/endpoint diagram. The endpoint authorizes each attempt within caller C/resource R and an active retention window. This example supports one atomic effect-and-result transaction.

Deduplication requires the endpoint contract and retention scope. It does not make arbitrary external effects transactional. A timeout leaves the client uncertain; a new attempt is not a new intent. Pending/unresolved branches remain reference alternatives, not extra executed effects.

The canonical SVG is cloned from verified artwork; original panel geometry and semantic colors remain. State values change inside its original objects. Local panning and shared expanded inspection preserve the active state. The shared player owns global timing, keyboard operation, reduced motion, and playback.

Run `build-preview.py` from the worktree root, then `check.py --engine chromium` or `--engine webkit`. The builder copies an existing complete chapter into ignored local review output and does not enroll the manuscript. It reads the canonical-art index at `/tmp/figures-animation-art-current/index.json`. Browser print is not a Quarto static-target build. Agent review is not a student study.
