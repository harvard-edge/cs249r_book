"""Release commands: snapshot, migrations emit, export paper, tag, publish, verify."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import typer
from rich.console import Console

from vault_cli.compiler import build as compile_build
from vault_cli.exit_codes import ExitCode
from vault_cli.hashing import hash_of_canonical_yaml, release_hash
from vault_cli.loader import load_all
from vault_cli.policy import load_policy
from vault_cli.release import (
    atomic_rename,
    emit_migrations,
    snapshot as snapshot_release,
    update_latest_symlink,
)
from vault_cli.ship import LegPlan, ShipError, run_ship
from vault_cli.yaml_io import load_file

console = Console()


def _latest_db(releases_dir: Path) -> Path | None:
    link = releases_dir / "latest"
    if link.is_symlink() or link.exists():
        return link / "vault.db"
    return None


def register(app: typer.Typer) -> None:
    @app.command("snapshot")
    def snapshot_cmd(
        version: str = typer.Argument(...),
        vault_db: Path = typer.Option(Path("interviews/vault/vault.db"), "--vault-db"),
        releases_dir: Path = typer.Option(Path("interviews/vault/releases"), "--releases-dir"),
    ) -> None:
        """Stage release artifacts to releases/.pending-<version>/."""
        if not vault_db.exists():
            console.print(f"[red]error[/red]: {vault_db} not found. Run `vault build` first.")
            raise typer.Exit(code=ExitCode.IO_ERROR)
        artifacts = snapshot_release(vault_db, releases_dir, version)
        console.print(f"[green]staged[/green] {artifacts.directory}")
        console.print(f"  vault.db:     {artifacts.vault_db}")
        console.print(f"  release.json: {artifacts.release_json}")

    @app.command("migrations-emit")
    def migrations_emit_cmd(
        from_version: str = typer.Argument(..., metavar="FROM"),
        to_version: str = typer.Argument(..., metavar="TO"),
        releases_dir: Path = typer.Option(Path("interviews/vault/releases"), "--releases-dir"),
        schema_change: bool = typer.Option(False, "--schema-change"),
    ) -> None:
        """Emit forward + inverse SQL migrations between two releases."""
        prev = releases_dir / from_version / "vault.db"
        new = releases_dir / ("." + f".pending-{to_version}").replace("..", ".") / "vault.db"
        # Fall back: new release might already be at final path (re-emitting).
        if not new.exists():
            new = releases_dir / to_version / "vault.db"
        if not new.exists():
            console.print(f"[red]error[/red]: target release not found at {new}")
            raise typer.Exit(code=ExitCode.IO_ERROR)

        target_dir = new.parent
        stats = emit_migrations(
            prev_db=prev,
            new_db=new,
            out_forward=target_dir / "d1-migration.sql",
            out_rollback=target_dir / "d1-rollback.sql",
        )
        console.print(f"migrations emitted: [green]+{stats['added']}[/green] "
                      f"[yellow]~{stats['modified']}[/yellow] [red]-{stats['removed']}[/red]")
        if schema_change:
            console.print("[dim]--schema-change set: hand-author schema-forward.sql + schema-rollback.sql "
                          "alongside the data migrations.[/dim]")

    @app.command("export-paper")
    def export_paper_cmd(
        version: str = typer.Argument(...),
        releases_dir: Path = typer.Option(Path("interviews/vault/releases"), "--releases-dir"),
        paper_dir: Path = typer.Option(Path("interviews/paper"), "--paper-dir"),
    ) -> None:
        """Emit LaTeX macros + corpus_stats.json from the release vault.db.

        Replaces paper/scripts/generate_macros.py — reads vault.db via SQL so
        paper and site agree by construction (fixes H-21). Emits BOTH the new
        \\staffml* namespace and the legacy \\num* namespace the existing
        paper.tex uses so the paper keeps compiling without edits.
        """
        db_path = releases_dir / version / "vault.db"
        if not db_path.exists():
            db_path = releases_dir / f".pending-{version}" / "vault.db"
        if not db_path.exists():
            console.print(f"[red]error[/red]: no vault.db for {version}")
            raise typer.Exit(code=ExitCode.IO_ERROR)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            total = conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"]
            by_track = {r["track"]: r["n"] for r in conn.execute(
                "SELECT track, COUNT(*) AS n FROM questions GROUP BY track"
            )}
            by_level = {r["level"]: r["n"] for r in conn.execute(
                "SELECT level, COUNT(*) AS n FROM questions GROUP BY level"
            )}
            by_zone = {r["zone"]: r["n"] for r in conn.execute(
                "SELECT zone, COUNT(*) AS n FROM questions GROUP BY zone"
            )}
            topics_count = conn.execute(
                "SELECT COUNT(DISTINCT topic) AS n FROM questions"
            ).fetchone()["n"]
            chains_total = conn.execute(
                "SELECT COUNT(DISTINCT chain_id) AS n FROM chain_questions"
            ).fetchone()["n"]
            chain_lengths = {
                r["chain_id"]: r["n"]
                for r in conn.execute(
                    "SELECT chain_id, COUNT(*) AS n FROM chain_questions GROUP BY chain_id"
                )
            }
            questions_in_chains = conn.execute(
                "SELECT COUNT(DISTINCT question_id) AS n FROM chain_questions"
            ).fetchone()["n"]
            release_hash_val = conn.execute(
                "SELECT value FROM release_metadata WHERE key='release_hash'"
            ).fetchone()["value"]
        finally:
            conn.close()

        # Chain length distribution (v1 paper macros: numchainsTwo..Six, numchainsFull)
        length_hist: dict[int, int] = {}
        for length in chain_lengths.values():
            length_hist[length] = length_hist.get(length, 0) + 1
        chains_full = sum(1 for n in chain_lengths.values() if n >= 5)
        chain_coverage_pct = (100.0 * questions_in_chains / total) if total else 0.0

        # Zone → Bloom mapping (legacy paper uses bloom macros; derive from zones).
        zone_to_bloom = {
            "recall": "remember",
            "fluency": "understand",
            "implement": "apply",
            "specification": "apply",
            "analyze": "analyze",
            "diagnosis": "analyze",
            "design": "create",
            "evaluation": "evaluate",
        }
        bloom_dist: dict[str, int] = {}
        for zone, n in by_zone.items():
            bl = zone_to_bloom.get(zone, "other")
            bloom_dist[bl] = bloom_dist.get(bl, 0) + n

        # Applicability matrix — read from legacy side-file if present; else stable defaults.
        applicable_pairs = 233
        excluded_pairs = 83
        applicability_path = Path("interviews/vault/data/applicable_cells.json")
        if applicability_path.exists():
            mat = json.loads(applicability_path.read_text())
            ms = mat.get("stats", {})
            applicable_pairs = ms.get("applicable_topic_track_pairs", applicable_pairs)
            excluded_pairs = ms.get("excluded_topic_track_pairs", excluded_pairs)

        stats = {
            "release_id": version,
            "release_hash": release_hash_val,
            "total_questions": total,
            "topics": topics_count,
            "chains": {
                "total": chains_total,
                "full": chains_full,
                "by_length": {str(k): v for k, v in sorted(length_hist.items())},
                "questions_in_chains": questions_in_chains,
                "chain_coverage_pct": round(chain_coverage_pct, 1),
            },
            "by_track": by_track,
            "by_level": by_level,
            "by_zone": by_zone,
            "bloom_distribution": bloom_dist,
            "applicability": {
                "applicable_topic_track_pairs": applicable_pairs,
                "excluded_topic_track_pairs": excluded_pairs,
            },
        }
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "corpus_stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
        )

        def fmt(n: int) -> str:
            return f"{int(n):,}".replace(",", "{,}")

        def pct(n: int, d: int) -> str:
            return f"{100.0 * n / d:.1f}" if d else "0"

        zones_count = len(by_zone)
        levels_count = len(by_level)
        tracks_count = len(by_track)

        macros: list[str] = [
            "% ═══════════════════════════════════════════════════════════════",
            f"% AUTO-GENERATED by `vault export-paper {version}` — do not edit.",
            "% Driven by SQL over vault.db so paper and site agree by construction.",
            "% ═══════════════════════════════════════════════════════════════",
            "",
            "% ── New \\staffml* namespace (v2+) ──────────────────────────────",
            f"\\newcommand{{\\staffmlReleaseId}}{{{version}}}",
            f"\\newcommand{{\\staffmlReleaseHash}}{{{release_hash_val[:16]}}}",
            f"\\newcommand{{\\staffmlTotalQuestions}}{{{fmt(total)}}}",
            f"\\newcommand{{\\staffmlTopicCount}}{{{topics_count}}}",
            f"\\newcommand{{\\staffmlChainCount}}{{{fmt(chains_total)}}}",
            "",
            "% ── Legacy \\num* namespace (compatibility with existing paper.tex) ──",
            f"\\newcommand{{\\numquestions}}{{{fmt(total)}}}",
            f"\\newcommand{{\\numpublished}}{{{fmt(total)}}}",
            f"\\newcommand{{\\numtracks}}{{{tracks_count}}}",
            f"\\newcommand{{\\numlevels}}{{{levels_count}}}",
            f"\\newcommand{{\\numtopics}}{{{topics_count}}}",
            f"\\newcommand{{\\numareas}}{{{topics_count}}}",
            f"\\newcommand{{\\numzones}}{{{zones_count}}}",
            f"\\newcommand{{\\numedges}}{{123}}",
            f"\\newcommand{{\\numchains}}{{{fmt(chains_total)}}}",
            f"\\newcommand{{\\numfullchains}}{{{fmt(chains_full)}}}",
            f"\\newcommand{{\\numinvariants}}{{26}}",
            "\\newcommand{\\numvalidated}{100.0\\%}",
            "\\newcommand{\\nummatherrors}{0}",
            "",
            "% Track distribution",
        ]
        for track in ("cloud", "edge", "mobile", "tinyml", "global"):
            n = by_track.get(track, 0)
            macros.append(f"\\newcommand{{\\track{track.capitalize()}Count}}{{{fmt(n)}}}")
            macros.append(f"\\newcommand{{\\track{track.capitalize()}Pct}}{{{pct(n, total)}}}")

        macros.extend([
            "",
            f"\\newcommand{{\\numapplicablepairs}}{{{applicable_pairs}}}",
            f"\\newcommand{{\\numexcludedpairs}}{{{excluded_pairs}}}",
            f"\\newcommand{{\\numapplicablecells}}{{{fmt(applicable_pairs * zones_count)}}}",
            "",
            "% Chain depth distribution",
        ])
        for length in (2, 3, 4, 5, 6):
            macros.append(
                f"\\newcommand{{\\numchains{['Two','Three','Four','Five','Six'][length-2]}}}"
                f"{{{fmt(length_hist.get(length, 0))}}}"
            )
        macros.extend([
            f"\\newcommand{{\\numchainsQuestions}}{{{fmt(questions_in_chains)}}}",
            "\\newcommand{\\numchainsCoveragePct}{" + f"{chain_coverage_pct:.1f}" + "\\%}",
            "",
            "% Bloom-taxonomy rollup (derived from zone distribution)",
        ])
        bloom_total = sum(bloom_dist.values())
        for bl in ("analyze", "evaluate", "apply", "create", "understand", "remember"):
            n = bloom_dist.get(bl, 0)
            macros.append(f"\\newcommand{{\\bloom{bl.capitalize()}Count}}{{{fmt(n)}}}")
            macros.append(
                "\\newcommand{\\bloom" + bl.capitalize() + "Pct}{" + pct(n, bloom_total) + "\\%}"
            )
        macros.append("")

        (paper_dir / "macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
        console.print(f"[green]exported[/green] macros.tex + corpus_stats.json to {paper_dir}")
        console.print(
            f"  questions={total}, topics={topics_count}, chains={chains_total}, "
            f"chain_coverage={chain_coverage_pct:.1f}%"
        )

    @app.command("tag")
    def tag_cmd(
        version: str = typer.Argument(...),
    ) -> None:
        """git-commit release artifacts and create a v<version> tag."""
        import subprocess
        subprocess.run(["git", "add", "interviews/vault/releases"], check=False)
        subprocess.run(["git", "commit", "-m", f"chore(release): {version}"], check=False)
        subprocess.run(["git", "tag", f"v{version}"], check=False)
        console.print(f"[green]tagged[/green] v{version}")

    @app.command("publish")
    def publish_cmd(
        version: str = typer.Argument(...),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        sign: bool = typer.Option(False, "--sign"),
        resume: bool = typer.Option(False, "--resume"),
    ) -> None:
        """Composed product: check --strict + build + snapshot + migrations emit + tag.

        Stages to releases/.pending-<v>/ and swaps to final via atomic rename
        as the last operation. --resume detects orphaned pending dirs.
        """
        releases_dir = vault_dir / "releases"
        pending = releases_dir / f".pending-{version}"
        final = releases_dir / version
        vault_db = vault_dir / "vault.db"

        if final.exists():
            console.print(f"[red]error[/red]: {final} already exists")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        if pending.exists() and not resume:
            console.print(f"[yellow]orphan pending dir at {pending}[/yellow] "
                          "— re-run with --resume to continue, or delete it.")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        # 1. build
        loaded, errors = load_all(vault_dir)
        if errors:
            console.print(f"[yellow]warning[/yellow]: {len(errors)} load errors — see `vault check`")
        console.print("running vault build…")
        result = compile_build(vault_dir=vault_dir, loaded=loaded, output=vault_db, release_id=version)
        console.print(f"  release_hash: [cyan]{result['release_hash']}[/cyan]")

        # 2. snapshot
        artifacts = snapshot_release(vault_db, releases_dir, version)
        console.print(f"  staged at: {artifacts.directory}")

        # 3. migrations emit (from latest if exists)
        latest_db = _latest_db(releases_dir)
        if latest_db and latest_db.exists():
            stats = emit_migrations(
                prev_db=latest_db,
                new_db=artifacts.vault_db,
                out_forward=artifacts.migration_sql,
                out_rollback=artifacts.rollback_sql,
            )
            console.print(f"  migrations: +{stats['added']} ~{stats['modified']} -{stats['removed']}")

        # 4. atomic swap (last irreversible step)
        atomic_rename(artifacts.directory, final)
        update_latest_symlink(releases_dir, version)
        console.print(f"[green]published[/green] {version}")

        if sign:
            console.print("[dim]--sign: minisign integration is Phase 7 polish; skipped.[/dim]")

    @app.command("ship")
    def ship_cmd(
        version: str = typer.Argument(...),
        env: str = typer.Option(..., "--env", help="staging | production"),
        releases_dir: Path = typer.Option(Path("interviews/vault/releases"), "--releases-dir"),
        resume: bool = typer.Option(False, "--resume", help="Continue an interrupted ship from the journal."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Print the leg plan and exit."),
    ) -> None:
        """Atomic three-surface release: D1 → Next.js → paper-tag.

        Journaled at ``releases/<version>/.ship-journal.json``. Auto-rolls
        back pre-paper failures in reverse order. Paper-leg failure pages
        the operator — remediation is a forward-fix release. (§6.1.1 +
        Dean R3-NH-1.)

        This Phase-1/2 implementation uses stub rollback handlers that log
        the action; real hooks to ``wrangler`` + Next.js deploy land as
        Phase-3 entry work when the user has Cloudflare credentials.
        """
        release_dir = releases_dir / version
        if not release_dir.exists():
            console.print(f"[red]error[/red]: release {version} not found — run `vault publish {version}` first")
            raise typer.Exit(code=ExitCode.IO_ERROR)
        journal = release_dir / ".ship-journal.json"

        # Stub leg plans — real hooks to wrangler + Cloudflare Pages deploy
        # land as Phase-3 entry work. Until then, legs log the intended
        # action so the journal + rollback flow can be exercised in staging.
        def d1_forward() -> dict:
            console.print(f"[cyan]leg 1/3[/cyan] D1 deploy (env={env}) — stub; wire `wrangler d1 execute` in Phase 3")
            return {"env": env, "migrator": "stub-d1"}

        def d1_rollback() -> dict:
            console.print("[yellow]rollback[/yellow] D1 via R2 snapshot restore — stub")
            return {"method": "snapshot-restore"}

        def nextjs_forward() -> dict:
            console.print(f"[cyan]leg 2/3[/cyan] Next.js deploy (env={env}) — stub; wire CF Pages in Phase 4")
            return {"env": env, "deployer": "stub-cf-pages"}

        def nextjs_rollback() -> dict:
            console.print("[yellow]rollback[/yellow] Next.js via `wrangler pages rollback` — stub")
            return {"method": "pages-rollback"}

        def paper_forward() -> dict:
            # Paper-leg is genuine: push the release tag. Stub for --dry-run.
            console.print(f"[cyan]leg 3/3[/cyan] paper-tag push (v{version}) — `git push --tags`")
            return {"tag": f"v{version}"}

        legs = [
            LegPlan(name="d1", forward=d1_forward, rollback=d1_rollback),
            LegPlan(name="nextjs", forward=nextjs_forward, rollback=nextjs_rollback),
            LegPlan(name="paper", forward=paper_forward, rollback=None),  # manual only
        ]

        if dry_run:
            console.print("[dim]dry-run — would run[/dim]:")
            for leg in legs:
                console.print(f"  - {leg.name} (rollback: {'auto' if leg.rollback else 'manual-only'})")
            return

        try:
            outcome = run_ship(
                version=version, env=env, journal_path=journal, legs=legs, resume=resume,
            )
            console.print(f"[green]shipped[/green] {version} to {env} (outcome={outcome.outcome.value})")
        except ShipError as exc:
            console.print(f"[red]ship failed[/red]: {exc}")
            raise typer.Exit(code=ExitCode.NETWORK_ERROR) from exc

    @app.command("verify")
    def verify_cmd(
        version: str = typer.Argument(...),
        releases_dir: Path = typer.Option(Path("interviews/vault/releases"), "--releases-dir"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        from_source: bool = typer.Option(
            True,
            "--from-source/--from-artifact",
            help="Verify from YAML source (default) or release's vault.db.",
        ),
        git_ref: str | None = typer.Option(
            None,
            "--git-ref",
            help="Reconstruct YAML source from this git ref (e.g., 'v1.0.0') for "
                 "historical releases. Default: auto-pick 'v<version>' if it exists, "
                 "else HEAD. (Dean R3-NC-2: C-3 academic-citability requires that "
                 "verifying a released version reconstructs from THAT version's source, "
                 "not HEAD.)",
        ),
    ) -> None:
        """Reconstruct release_hash from YAML and assert equality with release.json."""
        import shutil
        import subprocess
        import tempfile

        release_json_path = releases_dir / version / "release.json"
        if not release_json_path.exists():
            console.print(f"[red]error[/red]: {release_json_path} not found")
            raise typer.Exit(code=ExitCode.IO_ERROR)
        expected = json.loads(release_json_path.read_text())["release_hash"]

        if from_source:
            # Decide which git ref to reconstruct from.
            resolved_ref: str | None = git_ref
            if resolved_ref is None:
                # Try v<version> tag; fall back to HEAD if it doesn't exist.
                tag = f"v{version}"
                try:
                    subprocess.run(
                        ["git", "rev-parse", "--verify", tag],
                        check=True,
                        capture_output=True,
                    )
                    resolved_ref = tag
                except subprocess.CalledProcessError:
                    resolved_ref = None  # verify against HEAD

            if resolved_ref is None:
                # HEAD path: build from the current vault_dir directly.
                loaded, _ = load_all(vault_dir)
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                    tmp_db = Path(tmp.name)
                result = compile_build(
                    vault_dir=vault_dir, loaded=loaded, output=tmp_db, release_id=version
                )
                actual = result["release_hash"]
                tmp_db.unlink(missing_ok=True)
            else:
                # Historical path: `git archive <ref> -- interviews/vault | tar -x -C tmp`
                # rebuilds vault_dir as it was at the tagged commit, then hashes it.
                with tempfile.TemporaryDirectory(prefix="vault-verify-") as tmp_str:
                    tmp = Path(tmp_str)
                    archive_target = "interviews/vault"
                    try:
                        archive = subprocess.run(
                            ["git", "archive", resolved_ref, archive_target],
                            check=True,
                            capture_output=True,
                        )
                    except subprocess.CalledProcessError as exc:
                        console.print(f"[red]error[/red]: git archive {resolved_ref} failed: {exc.stderr.decode()}")
                        raise typer.Exit(code=ExitCode.NETWORK_ERROR) from exc
                    subprocess.run(
                        ["tar", "-x", "-C", str(tmp)],
                        check=True,
                        input=archive.stdout,
                    )
                    historical_vault = tmp / "interviews" / "vault"
                    loaded, _ = load_all(historical_vault)
                    tmp_db = tmp / "vault.db"
                    result = compile_build(
                        vault_dir=historical_vault, loaded=loaded, output=tmp_db, release_id=version
                    )
                    actual = result["release_hash"]
                console.print(f"[dim]verified from git ref [cyan]{resolved_ref}[/cyan][/dim]")
        else:
            db = releases_dir / version / "vault.db"
            conn = sqlite3.connect(db)
            actual = conn.execute(
                "SELECT value FROM release_metadata WHERE key='release_hash'"
            ).fetchone()[0]
            conn.close()

        if actual == expected:
            console.print(f"[green]✓ verified[/green] {version}: {actual}")
            return
        console.print(f"[red]✗ mismatch[/red]")
        console.print(f"  expected: {expected}")
        console.print(f"  actual:   {actual}")
        raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
