# Scripts directory

Python and shell automation used by the **Machine Learning Systems** textbook tooling. Most behavior is exposed through the **Book Binder** CLI; this tree is the implementation layer.

## Use Binder first

From the **repository root**:

```bash
./book/binder validate all
./book/binder maintain repo-health
./book/binder help
```

If your shell is already in **`book/`**, use `./binder` instead of `./book/binder`.

Full command list, validation options, and setup: **[`book/docs/BINDER.md`](../../docs/BINDER.md)**.

Direct `python3 book/tools/scripts/...` use is for maintenance or cases not wired into Binder yet. Prefer `./book/binder` when a subcommand exists.

## Subfolder docs

| Area | README |
|------|--------|
| Content tools | [`content/README.md`](content/README.md) |
| Images | [`images/README.md`](images/README.md) |
| Glossary | [`glossary/README.md`](glossary/README.md) |
| Infra / CI helpers | [`infrastructure/README.md`](infrastructure/README.md) |
| Utilities | [`utilities/README.md`](utilities/README.md) |
| Extra script notes | [`docs/README.md`](docs/README.md) |

Other directories (`publish/`, `maintenance/`, `testing/`, `socratiQ/`) are documented here only where needed; see source and `--help` on individual scripts.

## Shell entrypoints (often run outside Binder)

These are common when you need the exact script interface:

```bash
./book/tools/scripts/publish/mit-press-release.sh --vol1
./book/tools/scripts/publish/publish.sh
./book/tools/scripts/maintenance/run_maintenance.sh
```

## Python scripts

Use Python 3. Most modules support `--help`:

```bash
python3 book/tools/scripts/publish/extract_figures.py --help
```
