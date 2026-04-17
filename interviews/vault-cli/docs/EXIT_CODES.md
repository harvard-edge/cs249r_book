# `vault` Exit-Code Taxonomy

> **Source of truth**: [`src/vault_cli/exit_codes.py`](../src/vault_cli/exit_codes.py).
> **Referenced from**: ARCHITECTURE.md §4.6.
> **Contract**: Codes are STABLE across releases. Never renumber. Scripts pin to these.

---

## Table

| Code | Symbol | Meaning | Typical cause |
|------|--------|---------|---------------|
| 0 | `SUCCESS` | Command completed successfully. | Happy path. |
| 1 | `VALIDATION_FAILURE` | A data invariant, schema rule, or integrity check failed. | Bad YAML, content-hash mismatch, registry inconsistency, rollback symmetry broken. |
| 2 | `USAGE_ERROR` | The command invocation itself is malformed. | Missing required arg, unknown flag, conflicting flags. Surfaced by Typer/Click. |
| 3 | `IO_ERROR` | A filesystem or local I/O operation failed. | Permission denied, disk full, missing expected file, symlink swap failed. |
| 4 | `NETWORK_ERROR` | A network call to D1, Cloudflare, an LLM API, or another external service failed. | D1 unreachable, timeout, 5xx from upstream, DNS failure. |
| 5 | `USER_ABORTED` | An interactive confirmation was declined or the command was Ctrl-C'd mid-confirmation. | User typed `n` or mismatched title on `vault rm --hard`. |
| 64–78 | — | Reserved for `sysexits.h` standard codes when a fit exists. | Use only when one of the above doesn't apply. |

## Why these specific categories

- **0 vs 1 distinction matters to CI**: CI pipelines need to tell "clean build, go ahead" from "the corpus has a problem". Using 1 for validation keeps `if vault check; then deploy; fi` idiomatic.
- **1 vs 2 distinction matters to operators**: Exit 1 = "your data is broken, fix it in git"; exit 2 = "you typed the command wrong, re-read --help". Different next actions.
- **3 vs 4 distinction matters to observability**: IO errors on local machines are almost always reproducible; network errors deserve retries and are the right thing to flag in Cloudflare logs.
- **5 is separate so scripts don't misinterpret user-cancel as a bug**. If CI runs a command that hangs waiting for confirm, then times out, you want that distinguishable from a genuine failure.

## Usage in code

```python
from vault_cli.exit_codes import ExitCode

raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
```

Never use raw integers — always the enum. Mypy catches typos.

## Usage in tests

```python
def test_rm_hard_without_confirm_aborts() -> None:
    result = runner.invoke(app, ["rm", "<id>", "--hard"], input="\n")
    assert result.exit_code == ExitCode.USER_ABORTED
```

## JSON-output integration

When `--json` is set on a failing command, stderr still exits with the correct
code AND stdout emits:

```json
{
  "ok": false,
  "exit_code": 1,
  "exit_symbol": "VALIDATION_FAILURE",
  "errors": [ ... ]
}
```

See [`JSON_OUTPUT.md`](JSON_OUTPUT.md).

## Evolution

Adding a new code to the enum:

1. Pick the next unused value in `[6..63]` or `[79..127]`.
2. Document here with a symbol and meaning.
3. Update the regression test `test_exit_code_taxonomy_is_stable`.
4. Never renumber existing codes.

---

**End of EXIT_CODES.md.**
