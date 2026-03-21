# Book contents layout

**Do not assume content lives only under `vol1/` and `vol2/`.** This directory has shared and front matter that both volumes depend on. Deleting or moving them will break builds.

## Directory structure

<table>
  <thead>
    <tr>
      <th width="25%">Path</th>
      <th width="75%">Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><code>shared/</code></b></td>
      <td>Content used by <b>both</b> Volume I and Volume II (e.g. notation, conventions).</td>
    </tr>
    <tr>
      <td><b><code>frontmatter/</code></b></td>
      <td>Site-level front matter (about, acknowledgements, Socratiq, etc.).</td>
    </tr>
    <tr>
      <td><b><code>index.qmd</code></b></td>
      <td>Root book index (under <code>frontmatter/</code> in some configs).</td>
    </tr>
    <tr>
      <td><b><code>vol1/</code></b></td>
      <td>Volume I: Introduction to Machine Learning Systems.</td>
    </tr>
    <tr>
      <td><b><code>vol2/</code></b></td>
      <td>Volume II: Machine Learning Systems at Scale.</td>
    </tr>
  </tbody>
</table>

## Required non-volume paths

- **`shared/`** — Must exist. Contains `notation.qmd` and any other shared reference material.
- **`frontmatter/`** — Must exist. Contains about, acknowledgements, and other front matter.

Scripts and tooling that iterate over “all content” should include `shared/` and `frontmatter/` (and the root `index.qmd` where applicable), not only `vol1/` and `vol2/`.
