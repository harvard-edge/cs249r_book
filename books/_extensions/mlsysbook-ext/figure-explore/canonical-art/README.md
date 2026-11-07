# Reviewed HTML artwork

These five SVGs are recovered reviewed artifacts. Their bytes match the existing
`figure-registry.json` artwork identities and native lesson geometry. Their
historical generation environment is not asserted. They are ordinary versioned
book assets, not a cache to regenerate during publication.

The HTML-only `reviewed_figure_assets.lua` filter runs before the diagram filter.
For the owning chapter it checks the actual input path, exact source bytes and
registry source digest, parsed TikZ body and block options, effective inherited diagram metadata,
renderer/template/default-seed bytes, declared input bytes, and SVG bytes and
viewBox. It inserts the verified SVG into Pandoc's media bag. Other figures and
PDF/EPUB retain their existing rendering path. Browser artwork hashes, lesson
geometry guards, and the registry remain unchanged.

A known binding that fails stops the render. Do not remove a binding or change a
hash merely to make a build pass. If source, configuration, dependencies, or
artwork changes, review the rendered artwork and lesson mappings together before
updating the manifest. Add any new external figure inputs to the declared input
closure. Current figure bodies have no external data/image inputs; the manifest
also records the shared diagram defaults and renderer contract.

`binder/tests/test_reviewed_figure_assets.py` exercises actual Quarto rendering
and source, metadata, option, dependency, ownership, structure, and asset drift
controls. Full book builds and browser checks remain release requirements.
