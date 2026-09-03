# mlsysim/labs/state.py
# Persistent state management for the MLSys Design Ledger.
# Handles CLI (Local File) and Web (Browser IndexedDB) persistence.

import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class LedgerState:
    """The schema for the persistent student state."""

    track: Optional[str] = None
    current_step: int = 0
    history: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    last_updated: str = ""


class DesignLedger:
    """
    The 'Save Game' manager for the MLSys curriculum.

    Ensures that decisions made in Lab 00 persist through Lab 32.
    """

    def __init__(self):
        self.config_dir = Path.home() / ".mlsys"
        self.file_path = self.config_dir / "ledger.json"

        self._state = LedgerState()
        self._last_load_error: Optional[str] = None

        # WASM save tasks remain tracked until flush() observes them. Keeping
        # completed tasks lets a later flush() re-raise persistence failures.
        self._pending_save_tasks = set()

        # Error from the most recent failed background save.
        self._last_save_error: Optional[str] = None

        self.load()

    @property
    def last_save_error(self) -> Optional[str]:
        """
        Error message from the most recent failed background save,
        if any.
        """
        return self._last_save_error

    @property
    def last_load_error(self) -> Optional[str]:
        """Error message from the most recent failed load."""
        return self._last_load_error

    @property
    def save_pending(self) -> bool:
        """True while at least one WASM background save is still running."""
        return any(not task.done() for task in self._pending_save_tasks)

    @property
    def is_wasm(self) -> bool:
        """Detect if we are running in a browser environment (Pyodide)."""
        return sys.platform == "emscripten"

    _LOCALSTORAGE_KEY = "mlsys_design_ledger"

    def _parse_history(self, data: dict) -> dict:
        """Normalize history from either legacy list or dict format."""
        history_data = data.get("history", {})

        if isinstance(history_data, list):
            return {int(entry.get("step", 0)): entry.get("design", {}) for entry in history_data}

        elif isinstance(history_data, dict):
            return {int(k) if str(k).isdigit() else k: v for k, v in history_data.items()}

        return {}

    def load(self) -> LedgerState:
        """Loads the ledger from the best available persistent storage."""
        self._last_load_error = None

        # WASM loading is asynchronous, so synchronous load()
        # simply returns the current in-memory state.
        if self.is_wasm:
            return self._state

        # Native/local filesystem persistence.
        if self.file_path.exists():
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)

                data["history"] = self._parse_history(data)
                self._state = LedgerState(**data)

            except Exception as e:
                self._last_load_error = f"{type(e).__name__}: {e}"
                self._state = LedgerState()

        return self._state

    async def load_async(self) -> LedgerState:
        """
        Async load for WASM environments using IndexedDB.
        """
        self._last_load_error = None

        if not self.is_wasm:
            return self.load()

        try:
            from pyodide.code import run_js

            js_code = """
            (async () => new Promise((resolve, reject) => {
                const request = indexedDB.open("mlsys_ledger_db", 1);

                request.onupgradeneeded = (e) => {
                    const db = e.target.result;

                    if (!db.objectStoreNames.contains("ledger")) {
                        db.createObjectStore("ledger");
                    }
                };

                request.onsuccess = (e) => {
                    const db = e.target.result;

                    if (!db.objectStoreNames.contains("ledger")) {
                        db.close();
                        resolve(null);
                        return;
                    }

                    try {
                        const tx = db.transaction(
                            "ledger",
                            "readonly"
                        );

                        const store = tx.objectStore("ledger");

                        const getReq = store.get(
                            "mlsys_design_ledger"
                        );

                        getReq.onsuccess = () => {
                            const result = getReq.result;

                            db.close();

                            resolve(result);
                        };

                        getReq.onerror = () => {
                            const error = getReq.error
                                ? getReq.error.message
                                : "unknown error";

                            db.close();

                            reject(new Error(
                                "IndexedDB read failed: " + error
                            ));
                        };

                    } catch (err) {
                        db.close();

                        reject(new Error(
                            "IndexedDB transaction failed: "
                            + err.message
                        ));
                    }
                };

                request.onerror = () => {
                    reject(new Error(
                        "indexedDB.open failed: "
                        + (
                            request.error
                                ? request.error.message
                                : "unknown error"
                        )
                    ));
                };
            }))()
            """

            raw = await run_js(js_code)

            if raw:
                data = json.loads(raw)
                data["history"] = self._parse_history(data)
                self._state = LedgerState(**data)

        except Exception as e:
            self._last_load_error = f"{type(e).__name__}: {e}"
            print(f"Failed to load from IndexedDB: {e}")
            self._state = LedgerState()

        return self._state

    async def save_async(self):
        """
        Async save for WASM environments using IndexedDB.

        The Promise resolves only after the IndexedDB transaction
        has successfully completed, rather than immediately after
        the put request succeeds.
        """

        if not self.is_wasm:
            return True

        import json

        from pyodide.code import run_js
        from js import globalThis

        state_json = json.dumps(asdict(self._state))

        # A double-underscore attribute is name-mangled inside this class.
        # Keep this name in sync with the JavaScript reference below.
        globalThis._mlsys_temp_state = state_json

        js_code = """
        (async () => new Promise((resolve, reject) => {
            const request = indexedDB.open("mlsys_ledger_db", 1);

            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains("ledger")) {
                    db.createObjectStore("ledger");
                }
            };

            request.onsuccess = (e) => {
                const db = e.target.result;

                try {
                    const tx = db.transaction("ledger", "readwrite");

                    tx.oncomplete = () => {
                        db.close();
                        resolve(true);
                    };

                    tx.onerror = () => {
                        const error = tx.error
                            ? tx.error.message
                            : "unknown transaction error";
                        db.close();
                        reject(new Error(
                            "IndexedDB transaction failed: "
                            + error
                        ));
                    };

                    tx.onabort = () => {
                        const error = tx.error
                            ? tx.error.message
                            : "transaction aborted";
                        db.close();
                        reject(new Error(
                            "IndexedDB transaction aborted: "
                            + error
                        ));
                    };

                    tx.objectStore("ledger").put(
                        globalThis._mlsys_temp_state,
                        "mlsys_design_ledger"
                    );
                } catch (err) {
                    db.close();
                    reject(new Error(
                        "IndexedDB transaction failed: "
                        + err.message
                    ));
                }
            };

            request.onerror = () => {
                const error = request.error
                    ? request.error.message
                    : "unknown error";
                reject(new Error(
                    "indexedDB.open failed: "
                    + error
                    + " (storage may be disabled, full, "
                    + "or unavailable in private browsing)"
                ));
            };
        }))()
        """

        await run_js(js_code)
        return True

    def _apply_pending_state(self, track, step, design, chapter):
        """
        Shared bookkeeping used by both save() and asave().
        """

        if track:
            self._state.track = track

        step_id = step if step is not None else chapter

        if step_id is None:
            step_id = self._state.current_step

        else:
            self._state.current_step = step_id

        if design is not None:
            self._state.history[step_id] = design

    def _on_save_task_done(self, task):
        """
        Done-callback for the background WASM save task.

        Captures exceptions that would otherwise be lost from
        the background asyncio task.
        """

        try:
            task.result()
        except Exception as e:
            self._last_save_error = str(e)
            message = f"[DesignLedger] SAVE FAILED - progress was NOT persisted: {e}"

            try:
                from js import console

                console.error(message)

            except Exception:
                print(message, file=sys.stderr)

    def save(self, track: str = None, step: int = None, design: dict = None, chapter: int = None):
        """
        Persists the design decisions to storage.

        In WASM, the save runs asynchronously in the background.
        Call flush() when the caller needs to wait for completion.
        """

        self._apply_pending_state(track, step, design, chapter)

        if self.is_wasm:
            import asyncio

            task = asyncio.ensure_future(self.save_async())
            self._pending_save_tasks.add(task)
            task.add_done_callback(self._on_save_task_done)
            return task

        else:
            # Native/local filesystem persistence.
            self.config_dir.mkdir(exist_ok=True)

            with open(self.file_path, "w") as f:
                json.dump(asdict(self._state), f, indent=2)

    async def asave(self, track: str = None, step: int = None, design: dict = None, chapter: int = None):
        """
        Async variant of save() that awaits persistence.

        In WASM, this guarantees that save_async() completes
        before returning, or raises an exception if persistence
        fails.
        """

        self._apply_pending_state(track, step, design, chapter)

        if self.is_wasm:
            try:
                await self.save_async()
            except Exception as e:
                self._last_save_error = str(e)
                raise
            else:
                self._last_save_error = None

        else:
            # Native/local filesystem persistence.
            self.config_dir.mkdir(exist_ok=True)

            with open(self.file_path, "w") as f:
                json.dump(asdict(self._state), f, indent=2)

    async def flush(self):
        """Await all background saves scheduled since the previous flush."""
        import asyncio

        tasks = tuple(self._pending_save_tasks)
        if not tasks:
            return

        try:
            await asyncio.gather(*tasks)
        except Exception:
            raise
        else:
            self._last_save_error = None
        finally:
            self._pending_save_tasks.difference_update(task for task in tasks if task.done())

    def get_design(self, step_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves the design dictionary for a specific step."""
        return self._state.history.get(step_id)

    def get_track(self) -> str:
        """Returns the current track or NONE."""
        return self._state.track or "NONE"

    def get_baseline(self, step_id: int) -> dict:
        """
        Provides the 'Gold Standard' baseline if the student
        hasn't completed previous labs.
        """
        return {}

    def __repr__(self):
        return f"DesignLedger(track={self._state.track}, step={self._state.current_step})"
