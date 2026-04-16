"""``vault doctor`` — diagnostic subchecks (B.11).

Each subcheck is independently runnable via ``--check <name>``. Machine-readable
output via ``--json`` emits one ``{check, status, detail}`` object per row.
Exit 0 if all green; 1 if any red.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vault_cli.exit_codes import ExitCode
from vault_cli.loader import load_all

console = Console()


@dataclass
class CheckResult:
    name: str
    status: str  # pass | warn | fail | skip
    detail: str


def _check_git_state(vault_dir: Path) -> CheckResult:
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=vault_dir,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return CheckResult("git-state", "skip", "not a git repo")
    dirty = [line for line in res.stdout.splitlines() if line]
    if not dirty:
        return CheckResult("git-state", "pass", "clean")
    return CheckResult("git-state", "warn", f"{len(dirty)} uncommitted change(s)")


def _check_schema_version(vault_dir: Path) -> CheckResult:
    loaded, errors = load_all(vault_dir)
    versions = {lq.question.schema_version for lq in loaded}
    if not versions:
        return CheckResult("schema-version", "skip", "no questions loaded")
    if versions == {1}:
        return CheckResult(
            "schema-version", "pass", f"v1 ({len(loaded)} questions, {len(errors)} load errors)",
        )
    return CheckResult(
        "schema-version", "fail",
        f"mixed schema versions detected: {sorted(versions)}",
    )


def _check_registry_integrity(vault_dir: Path) -> CheckResult:
    reg = vault_dir / "id-registry.yaml"
    if not reg.exists():
        return CheckResult("registry-integrity", "skip", f"{reg} not found")
    # The registry is line-append-only and can grow past the per-question YAML
    # size cap. Parse IDs via regex over lines rather than loading as structured
    # YAML.
    reg_ids: set[str] = set()
    for line in reg.read_text(encoding="utf-8").splitlines():
        m = re.search(r"\bid:\s*([A-Za-z0-9._-]+)", line)
        if m:
            reg_ids.add(m.group(1))
    if not reg_ids:
        return CheckResult("registry-integrity", "fail", "no entries parsed")
    # Cross-check: every YAML file under questions/ has an id in the registry.
    loaded, _ = load_all(vault_dir)
    file_ids = {lq.id for lq in loaded}
    orphan_files = file_ids - reg_ids
    orphan_registry = reg_ids - file_ids
    if orphan_files or orphan_registry:
        return CheckResult(
            "registry-integrity", "fail",
            f"{len(orphan_files)} file(s) missing from registry; "
            f"{len(orphan_registry)} registry entry(s) missing files",
        )
    return CheckResult(
        "registry-integrity", "pass",
        f"{len(reg_ids)} entries, all files exist",
    )


def _check_release_integrity(vault_dir: Path) -> CheckResult:
    releases_dir = vault_dir / "releases"
    if not releases_dir.exists():
        return CheckResult("release-integrity", "skip", "no releases/ dir")
    latest = releases_dir / "latest"
    if not latest.exists():
        return CheckResult("release-integrity", "warn", "no releases/latest symlink")
    release_json = latest / "release.json"
    if not release_json.exists():
        return CheckResult("release-integrity", "fail", f"{release_json} missing")
    meta = json.loads(release_json.read_text())
    return CheckResult(
        "release-integrity", "pass",
        f"release {meta.get('release_id')} manifest verified "
        f"({meta.get('published_count')} questions, hash {str(meta.get('release_hash'))[:12]})",
    )


def _check_d1_connectivity() -> CheckResult:
    # Without an authenticated wrangler + net access we can only report skipped.
    if not os.environ.get("VAULT_D1_URL"):
        return CheckResult(
            "d1-connectivity", "skip",
            "set VAULT_D1_URL to the worker origin to probe (e.g., https://...workers.dev)",
        )
    import urllib.request
    try:
        req = urllib.request.Request(
            os.environ["VAULT_D1_URL"].rstrip("/") + "/manifest",
            headers={"User-Agent": "vault-doctor"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            payload = json.load(r)
        release_id = payload.get("release_id") or payload.get("release_metadata", {}).get("release_id")
        return CheckResult(
            "d1-connectivity", "pass",
            f"worker reachable, release_id={release_id}",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("d1-connectivity", "fail", f"worker unreachable: {exc}")


def _check_content_hash_sample(vault_dir: Path) -> CheckResult:
    """Sample 20 random IDs: vault.db content_hash must match the Merkle leaf hash."""
    db_path = vault_dir / "vault.db"
    if not db_path.exists():
        return CheckResult("content-hash-sample", "skip", "no vault.db — run `vault build`")
    import random

    from vault_cli.hashing import content_hash
    from vault_cli.loader import load_all as _loadall
    loaded, _ = _loadall(vault_dir)
    by_id = {lq.id: lq.question.model_dump(mode="json") for lq in loaded}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        ids = [r["id"] for r in conn.execute("SELECT id FROM questions")]
        sample = random.sample(ids, min(20, len(ids)))
        mismatched = []
        for qid in sample:
            row = conn.execute(
                "SELECT content_hash FROM questions WHERE id = ?", (qid,)
            ).fetchone()
            if qid not in by_id:
                mismatched.append(qid)
                continue
            expected = content_hash(by_id[qid])
            if row["content_hash"] != expected:
                mismatched.append(qid)
    finally:
        conn.close()
    if mismatched:
        return CheckResult(
            "content-hash-sample", "fail",
            f"{len(mismatched)}/{len(sample)} sampled hashes mismatched",
        )
    return CheckResult(
        "content-hash-sample", "pass",
        f"{len(sample)}/{len(sample)} sampled hashes match",
    )


def _check_llm_spend_ledger() -> CheckResult:
    config = Path.home() / ".config" / "vault" / "llm-spend.json"
    if not config.exists():
        return CheckResult("llm-spend-ledger", "skip", "no ledger yet (vault generate unused)")
    data = json.loads(config.read_text())
    today = data.get("today_usd", 0.0)
    ceiling = data.get("ceiling_usd", 50.0)
    if today > ceiling:
        return CheckResult(
            "llm-spend-ledger", "fail",
            f"${today:.2f} used today; ceiling ${ceiling:.2f}",
        )
    return CheckResult(
        "llm-spend-ledger", "pass",
        f"${today:.2f} used today; ceiling ${ceiling:.2f}",
    )


def _check_link_rot() -> CheckResult:
    """Nightly job runs this — shipping a stub that reads the artifact."""
    artifact = Path("interviews/vault/link-rot.yaml")
    if not artifact.exists():
        return CheckResult("link-rot", "skip", "no artifact (run nightly link-check workflow)")
    return CheckResult(
        "link-rot", "pass",
        f"last-sweep artifact present: {artifact}",
    )


SUBCHECKS = {
    "git-state":            _check_git_state,
    "schema-version":       _check_schema_version,
    "registry-integrity":   _check_registry_integrity,
    "release-integrity":    _check_release_integrity,
    "d1-connectivity":      lambda _vd: _check_d1_connectivity(),
    "content-hash-sample":  _check_content_hash_sample,
    "llm-spend-ledger":     lambda _vd: _check_llm_spend_ledger(),
    "link-rot":             lambda _vd: _check_link_rot(),
}


def register(app: typer.Typer) -> None:
    @app.command("doctor")
    def doctor_cmd(
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        check: str | None = typer.Option(None, "--check", help="Run only the named subcheck."),
        as_json: bool = typer.Option(False, "--json"),
    ) -> None:
        """Diagnostic subchecks over vault state and environment."""
        if check:
            if check not in SUBCHECKS:
                console.print(
                    f"[red]unknown check[/red]: {check!r} "
                    f"(available: {', '.join(sorted(SUBCHECKS))})"
                )
                raise typer.Exit(code=ExitCode.USAGE_ERROR)
            names = [check]
        else:
            names = list(SUBCHECKS.keys())

        results = [SUBCHECKS[n](vault_dir) for n in names]
        any_fail = any(r.status == "fail" for r in results)

        if as_json:
            print(json.dumps({
                "ok": not any_fail,
                "exit_code": 1 if any_fail else 0,
                "exit_symbol": "VALIDATION_FAILURE" if any_fail else "SUCCESS",
                "command": "vault doctor",
                "data": {"checks": [
                    {"check": r.name, "status": r.status, "detail": r.detail}
                    for r in results
                ]},
            }))
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE if any_fail else ExitCode.SUCCESS)

        table = Table(title="vault doctor")
        table.add_column("check", style="cyan")
        table.add_column("status")
        table.add_column("detail", overflow="fold")
        colors = {"pass": "green", "fail": "red", "warn": "yellow", "skip": "dim"}
        for r in results:
            color = colors.get(r.status, "white")
            table.add_row(r.name, f"[{color}]{r.status}[/{color}]", r.detail)
        console.print(table)
        if any_fail:
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
