# How does a private edit change only one task’s view?

The original baseline panel becomes a live overlay: Task A writes code, then deletes data. File-level OverlayFS semantics; no block-copy or storage-volume claim.

Guard: Copy-up is private; whiteout hides rather than mutates shared lower data. Copy-on-write is a storage mechanism, not by itself an execution isolation boundary.

Source chapter: `books/vol3/08_sandboxes/08_sandboxes.qmd`. Original artwork is cloned only after loader verification; IDs are already isolated by the loader. Static figure, caption, and prose are untouched.

Build the full chapter QA fixture:

```sh
python3 binder/tools/scripts/build_original_figure_fixture.py --entry books/_prototypes/vol1-web-animations/original-v3-workspace-overlay-visibility/entry.json
python3 books/_prototypes/vol1-web-animations/original-v3-workspace-overlay-visibility/check.py
```

The fixture uses the shared chapter server at port 8882, original-player, native Diagram/Explore tabs, namespaced artwork, and global YAML playback timing. The check covers menu endpoints and model invariants, keyboard controls, every nonreduced transition paused in fresh contexts, reduced motion, expansion, Diagram/Explore preservation, print, and no-JavaScript fallback at desktop/phone widths and both actual themes. Motion capture uses viewport screenshots to avoid tall-element screenshot observer artifacts.

QA screenshots are written under `/tmp/original-v3-workspace-overlay-visibility-review/`. Prototype only; production enrollment belongs to the integration owner. No real student evaluation is claimed.
