# SocratiQ bundle

`bundle.js` is the production build of the SocratiQ reading widget. Its source
lives in [`socratiq/`](../../../../socratiq/); `npm run build:vite` there
regenerates this file.

The book site loads it from `/tools/scripts/socratiQ/bundle.js`, through the
`<script>` tag in [`books/config/_quarto-html-vol1.yml`](../../../../books/config/_quarto-html-vol1.yml)
and [`_quarto-html-vol2.yml`](../../../../books/config/_quarto-html-vol2.yml).

The bundle is committed. The `socratiq-bundle-drift` workflow rebuilds it on any
pull request that changes the SocratiQ sources, and fails if the committed file
no longer matches.

The widget itself is documented in [`bindery/socratiQ/README.md`](../../../socratiQ/README.md).
