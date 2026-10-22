# Which values reach a query across a packed-example boundary?

Original two examples with five and three tokens. A declared toy row uses equal finite scores of zero; softmax is computed in JavaScript. No kernel speedup is implied.

Guard: Cross-example entries are removed before softmax, never after. Toy equal scores are explicitly scoped; no attention-kernel traffic or timing claim.

Source chapter: `books/vol3/16_fine_tuning/16_fine_tuning.qmd`. Original artwork is cloned only after loader verification; IDs are already isolated by the loader. Static figure, caption, and prose are untouched.

Build the full chapter QA fixture:

```sh
python3 binder/tools/scripts/build_original_figure_fixture.py --entry books/_prototypes/vol1-web-animations/original-v3-block-diagonal-mask/entry.json
python3 books/_prototypes/vol1-web-animations/original-v3-block-diagonal-mask/check.py
```

The fixture uses the shared chapter server at port 8882, original-player, native Diagram/Explore tabs, namespaced artwork, and global YAML playback timing. The check covers menu endpoints and model invariants, keyboard controls, every nonreduced transition paused in fresh contexts, reduced motion, expansion, Diagram/Explore preservation, print, and no-JavaScript fallback at desktop/phone widths and both actual themes. Motion capture uses viewport screenshots to avoid tall-element screenshot observer artifacts.

QA screenshots are written under `/tmp/original-v3-block-diagonal-mask-review/`. Prototype only; production enrollment belongs to the integration owner. No real student evaluation is claimed.
