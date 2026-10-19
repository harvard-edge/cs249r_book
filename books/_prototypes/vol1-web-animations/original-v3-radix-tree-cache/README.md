# Why must eviction proceed from dormant leaves?

Original radix prefix tree and active request path. Four drawn block symbols per node are illustrative ownership objects, not a byte-capacity model.

Guard: Only dormant leaves are evicted. Dormant internal ancestors remain while descendants are retained; active request paths stay intact. LRU chooses among eligible leaves only.

Source chapter: `books/vol3/05_kv_cache/05_kv_cache.qmd`. Original artwork is cloned only after loader verification; IDs are already isolated by the loader. Static figure, caption, and prose are untouched.

Build the full chapter QA fixture:

```sh
python3 binder/tools/scripts/build_original_figure_fixture.py --entry books/_prototypes/vol1-web-animations/original-v3-radix-tree-cache/entry.json
python3 books/_prototypes/vol1-web-animations/original-v3-radix-tree-cache/check.py
```

The fixture uses the shared chapter server at port 8882, original-player, native Diagram/Explore tabs, namespaced artwork, and global YAML playback timing. The check covers menu endpoints and model invariants, keyboard controls, every nonreduced transition paused in fresh contexts, reduced motion, expansion, Diagram/Explore preservation, print, and no-JavaScript fallback at desktop/phone widths and both actual themes. Motion capture uses viewport screenshots to avoid tall-element screenshot observer artifacts.

QA screenshots are written under `/tmp/original-v3-radix-tree-cache-review/`. Prototype only; production enrollment belongs to the integration owner. No real student evaluation is claimed.
