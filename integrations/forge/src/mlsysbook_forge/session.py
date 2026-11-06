"""Explicit sessions with a local journal and lazy cloud delivery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import platform
import re
import sqlite3
from typing import Any, Callable
import uuid
import warnings

from .evidence import canonical, digest, numeric_metrics, sanitize


class ResultKind(str, Enum):
    ANALYTICAL = "analytical_simulation"
    SUPPLIED = "supplied_case"
    SMOKE = "functional_smoke"
    MEASURED = "measured_experiment"
    HOSTED = "hosted_model_execution"


@dataclass(frozen=True)
class RunContext:
    lab_id: str
    source_commit: str
    result_kind: ResultKind
    config: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    units: dict[str, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.lab_id or not re.fullmatch(r"[0-9a-f]{40}", self.source_commit):
            raise ValueError("A lab ID and full source commit SHA are required")
        ResultKind(self.result_kind)


class ForgeSession:
    """Publish only when callers explicitly log; default mode does nothing.

    Supply the same session_id and action_id to deduplicate notebook reruns.
    A failed remote attempt is ambiguous and is never retried automatically.
    Local evidence remains available even when the SDK or service is unavailable.
    """

    def __init__(
        self, context: RunContext, *, mode: str = "disabled",
        directory: str | Path = ".forge/runs", session_id: str | None = None,
        entity: str | None = None, project: str | None = None,
        backend_factory: Callable[..., Any] | None = None,
    ):
        if mode not in {"disabled", "offline", "online"}:
            raise ValueError("mode must be disabled, offline, or online")
        if mode == "online" and (not entity or not project):
            raise ValueError("Online publishing requires explicit entity and project")
        self.context = context
        self.mode = mode
        self.entity, self.project = entity, project
        self.session_id = session_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.session_id):
            raise ValueError("session_id must contain 1–64 letters, numbers, '_' or '-'")
        self.directory = Path(directory) / self.session_id
        self.backend_factory = backend_factory
        self._backend = None
        self._db = None
        self.closed = False

    def _journal(self):
        if self._db is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.directory / "journal.sqlite3", timeout=30)
            db.execute("CREATE TABLE IF NOT EXISTS context (id INTEGER PRIMARY KEY, payload TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, payload TEXT, status TEXT, error TEXT)")
            context = canonical(sanitize(self.context))
            with db:
                db.execute("INSERT OR IGNORE INTO context VALUES (1, ?)", (context,))
            if db.execute("SELECT payload FROM context WHERE id=1").fetchone()[0] != context:
                db.close()
                raise ValueError("This session_id already belongs to a different experiment")
            self._db = db
        return self._db

    def _deliver(self, event):
        if self._backend is None:
            factory = self.backend_factory
            if factory is None:
                from .wandb_backend import WandbBackend
                factory = WandbBackend
            self._backend = factory(
                context=sanitize(self.context), session_id=self.session_id,
                entity=self.entity, project=self.project,
            )
        self._backend.publish(event, self.directory / f"{event['id']}.json")

    def _record(self, action_id: str, category: str, payload: Any):
        if self.closed:
            raise RuntimeError("The session is closed")
        if self.mode == "disabled":
            return {"status": "disabled"}
        if not action_id:
            raise ValueError("An explicit action_id is required")
        clean = sanitize(payload)
        event_id = digest({"session": self.session_id, "action": action_id})
        event = {
            "schema_version": 1, "id": event_id, "category": category,
            "context": sanitize(self.context), "payload": clean,
            "environment": {"python": platform.python_version(), "platform": platform.system()},
        }
        encoded = canonical(event)
        db = self._journal()
        # Claim the action before any network I/O. Separate cells/processes cannot
        # both send the same action. A crash leaves a visible unresolved receipt.
        with db:
            inserted = db.execute(
                "INSERT OR IGNORE INTO events VALUES (?, ?, ?, NULL)",
                (event_id, encoded, "sending" if self.mode == "online" else "saved"),
            ).rowcount
        prior = db.execute("SELECT payload, status, error FROM events WHERE id=?", (event_id,)).fetchone()
        if not inserted:
            if prior[0] != encoded:
                raise ValueError("action_id was reused with different evidence; use a new action ID")
            return {"id": event_id, "status": prior[1], "duplicate": True, "error": prior[2]}
        try:
            (self.directory / f"{event_id}.json").write_text(encoded + "\n", encoding="utf-8")
            if self.mode == "online":
                self._deliver(event)
            status, error = ("sent" if self.mode == "online" else "saved"), None
        except Exception as exc:
            # Exception messages can contain credential-bearing URLs. The local
            # sanitized payload is the recovery artifact, not the raw exception.
            status, error = "unconfirmed", type(exc).__name__
            warnings.warn(f"Forge delivery unconfirmed ({error}); evidence retained in the local journal", RuntimeWarning, stacklevel=2)
        with db:
            db.execute("UPDATE events SET status=?, error=? WHERE id=?", (status, error, event_id))
        return {"id": event_id, "status": status, "error": error}

    def capture(self, evidence: Any, *, action_id: str):
        return self._record(action_id, "evidence", evidence)

    def log_metrics(self, metrics: dict, *, step: int, action_id: str):
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        clean = sanitize(metrics)
        if len(numeric_metrics(clean)) != len(clean) or any(isinstance(v, dict) for v in clean.values()):
            raise ValueError("metrics must be a flat mapping of finite numbers")
        return self._record(action_id, "metrics", {"step": step, "metrics": clean})

    def log_artifact(self, path: str | Path, *, action_id: str):
        """Publish a sanitized JSON artifact; arbitrary files are never uploaded."""
        if self.mode == "disabled":
            return {"status": "disabled"}
        path = Path(path)
        if path.suffix.lower() != ".json":
            raise ValueError("Only structured JSON evidence is supported; export other files to JSON first")
        return self._record(action_id, "artifact", {"name": path.name, "data": json.loads(path.read_text(encoding="utf-8"))})

    def receipts(self):
        if self._db is None:
            return []
        return [dict(zip(("id", "status", "error"), row)) for row in self._db.execute("SELECT id, status, error FROM events ORDER BY rowid")]

    def finish(self):
        if self.closed:
            return
        try:
            if self._backend is not None:
                self._backend.finish()
        except Exception as exc:
            warnings.warn(f"Forge finish unconfirmed ({type(exc).__name__}); inspect remote run status", RuntimeWarning, stacklevel=2)
        finally:
            if self._db is not None:
                self._db.close()
                self._db = None
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finish()
