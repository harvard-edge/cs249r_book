# Binder / Colab environment

**The live configuration is not in this directory.** repo2docker reads a
repository's Binder config from the repo root, from `binder/`, or from
`.binder/` — and this is a monorepo, so `tinytorch/binder/` is never consulted.

The files that actually build the Binder image are:

| File | Role |
|---|---|
| `../../binder/requirements.txt` | Python dependencies for the image |
| `../../binder/postBuild` | `cd tinytorch`, install the package, generate student notebooks |

This directory used to hold a second `requirements.txt` and `postBuild`, each
labelled "keep synchronized" with the live pair. They were not synchronized —
the pins had drifted apart (numpy `>=1.24` vs `>=2.2.6`, rich `>=13` vs
`>=15`, ipykernel `>=6.31` vs `>=7.2`) — and because repo2docker never read
them, the drift was invisible. They are gone; edit the live pair instead.

## Launch URLs

```
Binder: https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main
Colab:  https://colab.research.google.com/github/harvard-edge/cs249r_book/blob/main/tinytorch/<path>.ipynb
```

The module pages link to Binder with a `labpath` pointing at that module's
notebook under `tinytorch/modules/`.

## Notebook generation

`modules/` is gitignored; the notebooks are generated at image build time.
`postBuild` generates them through `convert_py_to_notebook(..., student=True)`,
the same path `tito module start` uses, so a Binder student gets the same
stubbed notebook a local student gets.

This matters: `postBuild` previously ran `jupytext --to notebook` on `src/`
directly, which is the `student=False` path, and shipped the full reference
solution in every notebook. `postBuild` now asserts after generation that no
notebook contains an unstubbed solution region, and fails the build if one
does.

## Verifying a build

Once Binder launches:

```python
import tinytorch
print(tinytorch.__version__)
```

Then open any notebook under `modules/` and confirm the exercises are stubs
(`raise NotImplementedError`), not completed implementations.
