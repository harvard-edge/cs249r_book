# `vault` `--json` Output Schemas

> **Purpose**: Stable per-command JSON schemas for scripting, editor integration,
> and CI. Every subcommand that supports `--json` documents its schema here.
> **Referenced from**: ARCHITECTURE.md §4.5.
> **Contract**: Schemas are versioned with the CLI. Breaking changes bump the
> CLI minor version. Additive changes (new optional fields) don't.

---

## Envelope

All `--json` responses share an outer envelope:

```json
{
  "ok": true | false,
  "exit_code": 0..,
  "exit_symbol": "SUCCESS" | "VALIDATION_FAILURE" | ...,
  "command": "vault <subcommand>",
  "cli_version": "0.1.0",
  "data": <command-specific>,
  "errors": [<error>, ...],
  "warnings": [<warning>, ...]
}
```

On success: `ok=true`, `errors=[]`, `data` populated.
On failure: `ok=false`, `errors` populated, `data` may be partial.

Every command's `--json-schema` subcommand prints the full schema for that
command, e.g.:

```bash
vault check --json-schema | jq .
```

---

## Stable schemas

### `vault check --json`

LSP-diagnostic-shaped errors so editors render inline squiggles.

```json
{
  "ok": false,
  "exit_code": 1,
  "exit_symbol": "VALIDATION_FAILURE",
  "command": "vault check",
  "data": {
    "checks_run": 26,
    "checks_passed": 24,
    "checks_failed": 2,
    "tier": "structural"
  },
  "errors": [
    {
      "uri": "file:///.../questions/cloud/l4/diagnosis/foo-7f3a9c-0001.yaml",
      "range": { "start": { "line": 6, "character": 0 }, "end": { "line": 6, "character": 24 } },
      "severity": 1,
      "code": "topic-not-in-taxonomy",
      "source": "vault-check",
      "message": "topic 'kv-cachee' not found in taxonomy.yaml; did you mean 'kv-cache-management'?"
    }
  ]
}
```

Severity: 1=Error, 2=Warning, 3=Info, 4=Hint (LSP spec).

### `vault stats --json`

```json
{
  "ok": true,
  "data": {
    "release_id": "1.0.0",
    "release_hash": "<64 hex>",
    "counts": {
      "questions": { "total": 9199, "published": 9199, "draft": 0, "deprecated": 0 },
      "topics": 79,
      "chains": 127,
      "zones": 11
    },
    "provenance": { "human": 6420, "llm-draft": 0, "llm-then-human-edited": 2779, "imported": 0 },
    "by_track": { "cloud": 2140, "edge": 1980, "mobile": 1842, "tinyml": 1730, "global": 1507 },
    "by_level": { "l1": 1500, "l2": 1650, "l3": 1780, "l4": 1890, "l5": 1900, "l6": 479 }
  }
}
```

### `vault verify --json`

```json
{
  "ok": true,
  "data": {
    "release_id": "1.0.0",
    "expected_hash": "<64 hex>",
    "computed_hash": "<64 hex>",
    "leaves_verified": 9205,
    "match": true
  }
}
```

On mismatch (`match: false`, exit 1), `errors` enumerates the first 10 differing leaves.

### `vault doctor --json`

```json
{
  "ok": true,
  "data": {
    "checks": [
      { "check": "git-state", "status": "pass", "detail": "clean" },
      { "check": "schema-version", "status": "pass", "detail": "v1 (up to date)" },
      { "check": "registry-integrity", "status": "pass", "detail": "9199 entries, all files exist" },
      { "check": "release-integrity", "status": "pass", "detail": "release 1.0.0 manifest verified" },
      { "check": "d1-connectivity", "status": "skip", "detail": "no D1 credentials configured" },
      { "check": "content-hash-sample", "status": "pass", "detail": "20/20 sampled hashes match" },
      { "check": "llm-spend-ledger", "status": "pass", "detail": "$4.22 used today; ceiling $50.00" }
    ]
  }
}
```

### `vault diff --json`

```json
{
  "ok": true,
  "data": {
    "from": "0.9.0",
    "to": "1.0.0",
    "added": [{ "id": "...", "title": "..." }],
    "removed": [{ "id": "...", "title": "..." }],
    "modified": [{ "id": "...", "classification": "cosmetic" | "semantic" | "structural" }]
  }
}
```

---

## Schemas landing later (Phase 1+)

`new`, `edit`, `rm`, `restore`, `move`, `build`, `publish`, `snapshot`,
`migrations emit`, `export paper`, `tag`, `deploy`, `rollback`, `ship`,
`generate`, `promote`, `mark-exemplar`, `renumber`, `serve`, `api`.

Each lands with its schema appended here and a `test_<cmd>_json_shape` test.

---

## Versioning

This document is versioned with the CLI. Breaking the envelope (renaming
`ok`/`exit_code`/`data`) is a CLI major version bump. Adding fields inside
`data` per command is a minor bump.

---

**End of JSON_OUTPUT.md.**
