# Why does stopping a child release no root funds?

Canonical dollar balances from the original figure. JavaScript uses integer mills to preserve the displayed budget; logical animation does not model billing time.

Guard: No reservation is released on local stop. Canonical values use integer mills; E+R+F always equals the root budget. Idempotent final billing cannot create extra free funds.

Source chapter: `books/vol3/19_agent_economics/19_agent_economics.qmd`. Original artwork is cloned only after loader verification; IDs are already isolated by the loader. Static figure, caption, and prose are untouched.

Build the full chapter QA fixture:

```sh
python3 binder/tools/scripts/build_original_figure_fixture.py --entry books/_prototypes/vol1-web-animations/original-v3-hierarchical-budget/entry.json
python3 books/_prototypes/vol1-web-animations/original-v3-hierarchical-budget/check.py
```

The fixture uses the shared chapter server at port 8882, original-player, native Diagram/Explore tabs, namespaced artwork, and global YAML playback timing. The check covers menu endpoints and model invariants, keyboard controls, every nonreduced transition paused in fresh contexts, reduced motion, expansion, Diagram/Explore preservation, print, and no-JavaScript fallback at desktop/phone widths and both actual themes. Motion capture uses viewport screenshots to avoid tall-element screenshot observer artifacts.

QA screenshots are written under `/tmp/original-v3-hierarchical-budget-review/`. Prototype only; production enrollment belongs to the integration owner. No real student evaluation is claimed.
