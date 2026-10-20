# What does a durable intent establish after a crash?

Original event order and recovery bands. A marker follows one operation; it does not measure elapsed time.

Guard: Intent durability is not dispatch, outcome evidence, or exactly-once external execution. The original conservative in-doubt boundary is retained.

Source chapter: `books/vol3/11_durable_execution/11_durable_execution.qmd`. Original artwork is cloned only after loader verification; IDs are already isolated by the loader. Static figure, caption, and prose are untouched.

Build the full chapter QA fixture:

```sh
python3 binder/tools/scripts/build_original_figure_fixture.py --entry books/_prototypes/vol1-web-animations/original-v3-intent-before-dispatch/entry.json
python3 books/_prototypes/vol1-web-animations/original-v3-intent-before-dispatch/check.py
```

The fixture uses the shared chapter server at port 8882, original-player, native Diagram/Explore tabs, namespaced artwork, and global YAML playback timing. The check covers menu endpoints and model invariants, keyboard controls, every nonreduced transition paused in fresh contexts, reduced motion, expansion, Diagram/Explore preservation, print, and no-JavaScript fallback at desktop/phone widths and both actual themes. Motion capture uses viewport screenshots to avoid tall-element screenshot observer artifacts.

QA screenshots are written under `/tmp/original-v3-intent-before-dispatch-review/`. Prototype only; production enrollment belongs to the integration owner. No real student evaluation is claimed.
