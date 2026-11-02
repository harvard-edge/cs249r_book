# When can a collection credit return?

One tracked run through the original four responsibilities; other in-flight credits remain unchanged. Motion is logical, not a rate model.

Guard: No credit release at rollout/capture/staging completion. Return follows a durable terminal record or acknowledged transfer to another bounded owner.

Source chapter: `books/vol3/15_trajectory_curation/15_trajectory_curation.qmd`. Original artwork is cloned only after loader verification; IDs are already isolated by the loader. Static figure, caption, and prose are untouched.

Build the full chapter QA fixture:

```sh
python3 binder/tools/scripts/build_original_figure_fixture.py --entry books/_prototypes/vol1-web-animations/original-v3-collection-pipeline/entry.json
python3 books/_prototypes/vol1-web-animations/original-v3-collection-pipeline/check.py
```

The fixture uses the shared chapter server at port 8882, original-player, native Diagram/Explore tabs, namespaced artwork, and global YAML playback timing. The check covers menu endpoints and model invariants, keyboard controls, every nonreduced transition paused in fresh contexts, reduced motion, expansion, Diagram/Explore preservation, print, and no-JavaScript fallback at desktop/phone widths and both actual themes. Motion capture uses viewport screenshots to avoid tall-element screenshot observer artifacts.

QA screenshots are written under `/tmp/original-v3-collection-pipeline-review/`. Prototype only; production enrollment belongs to the integration owner. No real student evaluation is claimed.
