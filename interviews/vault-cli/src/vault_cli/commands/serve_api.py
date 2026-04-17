"""``vault api`` and ``vault serve`` — local dev surfaces.

``vault api`` mirrors the production Worker endpoint surface from a local
vault.db so contributors can run the site without a Cloudflare account
(REVIEWS.md H-17 resolution).

``vault serve`` launches Datasette for ad-hoc exploration of vault.db.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import typer
from rich.console import Console

from vault_cli.exit_codes import ExitCode

console = Console()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    # sqlite3.Row iterates as VALUES, not keys; coerce explicitly.
    return dict(row)


class _VaultAPIHandler(BaseHTTPRequestHandler):
    db_path: Path = Path("vault.db")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence default stderr spam; operators can re-enable if needed.
        return

    def _json(self, code: int, payload: Any, extra_headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Vault-Api-Shim", "true")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — stdlib interface
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if path == "/manifest":
                meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM release_metadata")}
                self._json(200, {"release_metadata": meta})
            elif path == "/taxonomy":
                self._json(200, {"note": "local shim: taxonomy served from vault dir, not D1"})
            elif path.startswith("/questions/"):
                qid = path.split("/", 2)[2]
                cur = conn.execute("SELECT * FROM questions WHERE id = ?", (qid,))
                row = cur.fetchone()
                if row is None:
                    self._json(404, {"error": "not-found", "id": qid})
                else:
                    self._json(200, _row_to_dict(row))
            elif path == "/questions":
                q = "SELECT id, title, topic, track, level, zone FROM questions"
                where = []
                args: list[Any] = []
                for key in ("track", "level", "zone", "topic"):
                    if key in params:
                        where.append(f"{key} = ?")
                        args.append(params[key])
                if where:
                    q += " WHERE " + " AND ".join(where)
                q += " ORDER BY id LIMIT ?"
                args.append(int(params.get("limit", "50")))
                rows = conn.execute(q, args).fetchall()
                self._json(200, {"questions": [_row_to_dict(r) for r in rows]})
            elif path == "/stats":
                total = conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"]
                self._json(200, {"count": total})
            else:
                self._json(404, {"error": "unknown-endpoint", "path": path})
        finally:
            conn.close()


def register(app: typer.Typer) -> None:
    @app.command("api")
    def api_cmd(
        db: Path = typer.Option(Path("interviews/vault/vault.db"), "--db", help="vault.db to serve."),
        port: int = typer.Option(8002, "--port"),
    ) -> None:
        """Serve the Worker endpoint surface from a local vault.db."""
        if not db.exists():
            console.print(f"[red]error[/red]: {db} not found. Run `vault build` first.")
            raise typer.Exit(code=ExitCode.IO_ERROR)
        _VaultAPIHandler.db_path = db
        server = ThreadingHTTPServer(("127.0.0.1", port), _VaultAPIHandler)
        console.print(f"[green]vault api[/green] serving {db} on http://127.0.0.1:{port}")
        console.print("[dim]shim mirrors the Worker endpoint surface; deliberate divergences in CORS, "
                      "rate-limit, and edge cache — see CONTRIBUTING.md[/dim]")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            console.print("[yellow]shutting down[/yellow]")
            server.shutdown()

    @app.command("serve")
    def serve_cmd(
        db: Path = typer.Option(Path("interviews/vault/vault.db"), "--db"),
        port: int = typer.Option(8001, "--port"),
    ) -> None:
        """Launch Datasette on vault.db for ad-hoc exploration (127.0.0.1 only)."""
        if not db.exists():
            console.print(f"[red]error[/red]: {db} not found. Run `vault build` first.")
            raise typer.Exit(code=ExitCode.IO_ERROR)
        try:
            subprocess.run(["datasette", "serve", str(db), "--host", "127.0.0.1", "--port", str(port)], check=True)
        except FileNotFoundError:
            console.print("[red]datasette not installed[/red]: pip install datasette")
            raise typer.Exit(code=ExitCode.IO_ERROR) from None
