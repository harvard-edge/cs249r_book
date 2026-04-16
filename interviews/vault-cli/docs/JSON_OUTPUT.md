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

---

## Additional schemas (B.7 — remaining subcommands)

### `vault build --json`

```json
{
  "ok": true,
  "data": {
    "output": "interviews/vault/vault.db",
    "release_id": "0.9.0",
    "release_hash": "<64 hex>",
    "published_count": 9199,
    "policy_version": 1
  }
}
```

### `vault publish --json`

```json
{
  "ok": true,
  "data": {
    "version": "0.9.0",
    "staged_dir": "releases/.pending-0.9.0",
    "final_dir": "releases/0.9.0",
    "release_hash": "<64 hex>",
    "migration_stats": { "added": 12, "removed": 0, "modified": 3 }
  }
}
```

### `vault ship --json`

```json
{
  "ok": true,
  "data": {
    "version": "1.0.0",
    "env": "production",
    "journal_path": "releases/1.0.0/.ship-journal.json",
    "outcome": "success",
    "legs": [
      { "name": "d1",     "state": "deployed" },
      { "name": "nextjs", "state": "deployed" },
      { "name": "paper",  "state": "deployed" }
    ],
    "point_of_no_return": true
  }
}
```

On failure:

```json
{
  "ok": false,
  "exit_code": 4,
  "exit_symbol": "NETWORK_ERROR",
  "data": {
    "outcome": "failed_auto_rolled_back" | "failed_needs_manual",
    "legs": [ ... ]
  }
}
```

### `vault new --json`

```json
{
  "ok": true,
  "data": {
    "id": "cloud-l4-diagnosis-kv-cache-7f3a9c-0001",
    "path": "interviews/vault/questions/cloud/l4/diagnosis/kv-cache-7f3a9c-0001.yaml",
    "registry_appended": true
  }
}
```

### `vault rm --json`

```json
{
  "ok": true,
  "data": {
    "id": "global-0000",
    "action": "deprecated" | "hard-deleted",
    "chain_warning": "question was in chain 'kv-cache-depth' at position 2"
  }
}
```

### `vault move --json`

```json
{
  "ok": true,
  "data": {
    "id": "global-0000",
    "from": "global/l1/recall",
    "to": "cloud/l4/diagnosis",
    "renamed_path": "..."
  }
}
```

### `vault renumber --json`

```json
{
  "ok": true,
  "data": {
    "old_id": "global-l1-recall-foo-7f3a9c-0001",
    "new_id": "global-l1-recall-foo-7f3a9c-0002",
    "old_path": "...",
    "new_path": "..."
  }
}
```

### `vault restore --json`

```json
{ "ok": true, "data": { "id": "global-0000", "new_status": "published" } }
```

### `vault promote --json`

```json
{
  "ok": true,
  "data": {
    "promoted": [
      { "id": "...", "from": "vault/drafts/...", "to": "vault/questions/...",
        "new_provenance": "llm-then-human-edited", "reviewed_by": "user@example.com" }
    ],
    "count": 1
  }
}
```

### `vault mark-exemplar --json`

```json
{
  "ok": true,
  "data": {
    "id": "global-0000",
    "moved_from": "vault/questions/global/l1/recall/...",
    "moved_to": "vault/exemplars/global/l1/recall/..."
  }
}
```

### `vault snapshot --json`

```json
{
  "ok": true,
  "data": {
    "directory": "releases/.pending-1.0.0",
    "vault_db": "releases/.pending-1.0.0/vault.db",
    "release_json": "releases/.pending-1.0.0/release.json"
  }
}
```

### `vault migrations-emit --json`

```json
{
  "ok": true,
  "data": { "added": 12, "removed": 0, "modified": 3 }
}
```

### `vault export-paper --json`

```json
{
  "ok": true,
  "data": {
    "release_id": "0.9.0",
    "release_hash": "<64 hex>",
    "total_questions": 9199,
    "topics": 87,
    "chains": { "total": 964, "full": 77, "questions_in_chains": 2934, "chain_coverage_pct": 31.9 },
    "by_track": { "cloud": 4122, "edge": 1968, "mobile": 1641, "tinyml": 1163, "global": 305 },
    "by_level": { "l1": 1408, ... }
  }
}
```

### `vault tag --json`

```json
{ "ok": true, "data": { "tag": "v0.9.0", "pushed": false } }
```

### `vault deploy --json`

```json
{
  "ok": true,
  "data": {
    "env": "staging" | "production",
    "release_id": "1.0.0",
    "snapshot": "r2://staffml-vault-backups/pre-deploy-1.0.0.sqlite",
    "migration_applied": true,
    "pop_propagation": { "sampled": 8, "stale": 0 }
  }
}
```

### `vault rollback --json`

```json
{
  "ok": true,
  "data": {
    "method": "snapshot" | "sql",
    "env": "production",
    "restored_to": "0.9.0",
    "duration_seconds": 87
  }
}
```

### `vault generate --json`  _(Phase 7, deferred)_

Schema TBD. Placeholder shape:

```json
{
  "ok": true,
  "data": {
    "drafts_written": 3,
    "model": "claude-opus-4-6",
    "exemplar_ids": ["...", "..."],
    "prompt_hash": "<hex>",
    "cost_usd": 0.12
  }
}
```

### `vault serve --json`

Not applicable — `vault serve` launches Datasette, not a JSON-emitting command.

### `vault api --json`

Not applicable — `vault api` is a long-running HTTP server, not a
JSON-emitting command.

---

## Versioning

This document is versioned with the CLI. Breaking the envelope (renaming
`ok`/`exit_code`/`data`) is a CLI major version bump. Adding fields inside
`data` per command is a minor bump.

---

**End of JSON_OUTPUT.md.**
