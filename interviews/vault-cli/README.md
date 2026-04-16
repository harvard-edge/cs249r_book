# `vault-cli` — StaffML Question Vault CLI

Authoring, building, and releasing the StaffML question vault.

> **Status**: Phase 0 scaffolding. Subcommands land in Phase 1+ per
> [`ARCHITECTURE.md`](../vault/ARCHITECTURE.md) §14.

## Install (local editable)

```bash
# from the monorepo root
pip install -e interviews/vault-cli/
vault --version
```

Python ≥3.12 required. CI pins 3.12 exactly for hash stability (see
[`docs/EXIT_CODES.md`](docs/EXIT_CODES.md) and ARCHITECTURE.md §3.5).

## Quickstart (available commands grow by phase)

**Phase 0** (current):

```bash
vault --version         # print version
vault --help            # show help
```

**Phase 1** (scaffolded next): `new`, `edit`, `move`, `rm`, `restore`, `build`,
`check`, `serve`, `api`. See ARCHITECTURE.md §4 for full surface.

## Run tests

```bash
pip install -e interviews/vault-cli/[dev]
pytest interviews/vault-cli/tests/
```

## Layout

```
vault-cli/
├── pyproject.toml
├── README.md              # this file
├── docs/
│   ├── EXIT_CODES.md      # stable exit-code taxonomy
│   ├── JSON_OUTPUT.md     # per-command --json schemas
│   └── CUTOVER_QA.md      # manual cutover QA checklist
├── src/vault_cli/
│   ├── __init__.py
│   ├── _version.py
│   ├── exit_codes.py
│   └── main.py            # Typer app entry
└── tests/
    └── test_smoke.py      # Phase 0 smoke tests
```

## Architecture

See the sibling [`vault/`](../vault/) directory:

- [`ARCHITECTURE.md`](../vault/ARCHITECTURE.md) — full design doc.
- [`REVIEWS.md`](../vault/REVIEWS.md) — adversarial review ledger.
- [`TESTING.md`](../vault/TESTING.md) — test plan.
- [`schema/EVOLUTION.md`](../vault/schema/EVOLUTION.md) — schema-version rules.

## Contributing

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## License

MIT — see project root.
