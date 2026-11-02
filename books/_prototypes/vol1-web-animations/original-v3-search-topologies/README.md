# Where does each search structure discard work?

Original illustrative topologies: four best-of-N candidates, beam width two, and one selected tree path. Counts describe this drawing; no quality or runtime advantage is implied.

Guard: Best-of-N selects only after completion. Beam compares partial candidates at step boundaries. Tree leaf values update path visits/mean and need not denote completed answers. No success, traffic, or latency claim.

Source chapter: `books/vol3/03_test_time_compute/03_test_time_compute.qmd`. Original artwork is cloned only after loader verification; IDs are already isolated by the loader. Static figure, caption, and prose are untouched.

Build the full chapter QA fixture:

```sh
python3 binder/tools/scripts/build_original_figure_fixture.py --entry books/_prototypes/vol1-web-animations/original-v3-search-topologies/entry.json
python3 books/_prototypes/vol1-web-animations/original-v3-search-topologies/check.py
```

The fixture uses the shared chapter server at port 8882, original-player, native Diagram/Explore tabs, namespaced artwork, and global YAML playback timing. The check covers menu endpoints and model invariants, keyboard controls, every nonreduced transition paused in fresh contexts, reduced motion, expansion, Diagram/Explore preservation, print, and no-JavaScript fallback at desktop/phone widths and both actual themes. Motion capture uses viewport screenshots to avoid tall-element screenshot observer artifacts.

QA screenshots are written under `/tmp/original-v3-search-topologies-review/`. Prototype only; production enrollment belongs to the integration owner. No real student evaluation is claimed.
