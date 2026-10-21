# Concept Explorer

The main Quarto page is `index.qmd`, served at `/explore/`. It shares the site's
navigation, announcement, typography, color-mode switch, and footer. Its neutral
palette spans all four volumes. Taxonomy provides card and tree layouts; Concept
map displays every authored field relationship, with selection highlighting,
pan/zoom, and an optional neighborhood view.

`_workspace.qmd` is the shared interface body. The former `taxonomy-study.qmd`
URL includes the same body to preserve existing stateful links. The older
`field.qmd` browser remains available for compatibility.

The chapter-owned volume maps are preserved in `volume-maps.qmd`. Binder reads
that page's raw HTML as the template for portable chapter viewers and `atlas.html`;
its fallback to `index.qmd` supports older installations and fixtures. A guard
rejects accidentally using the field explorer as a chapter viewer template.

See [the authoring and pipeline guide](../../docs/CONCEPT_MAPS.md).

```bash
./binder/binder concepts refresh --vol1
./binder/binder concepts check --vol1
./binder/binder concepts build
```

Then run `quarto preview explore/index.qmd` from `site/`, or render the Model
Serving chapter through Binder. Normal HTML builds regenerate the maps.

Edit chapter YAML, not generated catalog JSON, map JSON, DOT, SVG, or chapter
viewers. The old manually maintained route file has been superseded.

Graphviz (`dot`) and PyYAML are build dependencies. The chapter/volume browser uses the vendored
Cytoscape.js 3.33.1; its MIT license is in `vendor/LICENSE.cytoscape`.
The browser does not require a remote model, CDN, or graph service.

The field explorer reads the independent field registry in
`shared/concepts/ml-systems-field.yml` and explicit book links in
`shared/concepts/mlsysbook-alignment.yml`. It keeps field identities separate
from chapter ownership. Generate it with the same commands above; open
`field-preview.html` for a portable local preview. Field and alignment records
remain editorial drafts, and unmapped does not mean uncovered.
