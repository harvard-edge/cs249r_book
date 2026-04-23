# SocratiQ bundle (mirror)

This directory contains a single file, `bundle.js`, which is a **symlink** to
the canonical bundle that is actually served from the rendered book site:

    bundle.js -> ../../../quarto/tools/scripts/socratiQ/bundle.js

The served path on the deployed book is `/tools/scripts/socratiQ/bundle.js`,
which is included from [`book/quarto/config/_quarto-html-vol1.yml`](../../../quarto/config/_quarto-html-vol1.yml)
and [`_quarto-html-vol2.yml`](../../../quarto/config/_quarto-html-vol2.yml).

When a new SocratiQ widget bundle is published, replace
`book/quarto/tools/scripts/socratiQ/bundle.js`. This mirror updates
automatically via the symlink.

The SocratiQ widget itself is documented in [`book/socratiQ/README.md`](../../../socratiQ/README.md).
