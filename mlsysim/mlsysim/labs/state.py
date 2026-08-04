# mlsysim/labs/state.py
# Persistent state management for the MLSys Design Ledger.
# Handles CLI (Local File) and Web (Browser IndexedDB) persistence.

import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
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

        # Background WASM save task, if one is currently running.
        self._pending_save_task = None

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
    def save_pending(self) -> bool:
        """True while a WASM background save has not finished yet."""
        return (
            self._pending_save_task is not None
            and not self._pending_save_task.done()
        )

    @property
    def is_wasm(self) -> bool:
        """Detect if we are running in a browser environment (Pyodide)."""
        return sys.platform == "emscripten"

    _LOCALSTORAGE_KEY = "mlsys_design_ledger"

    def _parse_history(self, data: dict) -> dict:
        """Normalize history from either legacy list or dict format."""
        history_data = data.get("history", {})

        if isinstance(history_data, list):
            return {
                int(entry.get("step", 0)): entry.get("design", {})
                for entry in history_data
            }

        elif isinstance(history_data, dict):
            return {
                int(k) if str(k).isdigit() else k: v
                for k, v in history_data.items()
            }

        return {}

    def load(self) -> LedgerState:
        """Loads the ledger from the best available persistent storage."""

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

            except Exception:
                self._state = LedgerState()

        return self._state

    async def load_async(self) -> LedgerState:
        """
        Async load for WASM environments using IndexedDB.
        """

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
            print(
                f"Failed to load from IndexedDB: {e}"
            )

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

        # Serialize the current state to JSON.
        state_json = json.dumps(
            asdict(self._state)
        )

        # Store the serialized state on the JS global object so
        # IndexedDB can access it.
        # NOTE: deliberately NOT double-underscore-prefixed. Any "__name"
        # written inside a class body gets silently rewritten by Python's
        # name mangling (e.g. to "_DesignLedger__mlsys_temp_state"), which
        # previously desynced this Python-side assignment from the plain
        # "__mlsys_temp_state" the JS string below reads -- causing a
        # bogus, unmangled `undefined` to be persisted while save_async()
        # still reported success. See PR #1988 for the full investigation.
        globalThis._mlsys_temp_state = state_json

        js_code = """
        (async () => new Promise((resolve, reject) => {
            const request = indexedDB.open(
                "mlsys_ledger_db",
                1
            );

            request.onupgradeneeded = (e) => {
                const db = e.target.result;

                if (!db.objectStoreNames.contains("ledger")) {
                    db.createObjectStore("ledger");
                }
            };

            request.onsuccess = (e) => {
                const db = e.target.result;

                try {
                    const tx = db.transaction(
                        "ledger",
                        "readwrite"
                    );

                    const store = tx.objectStore(
                        "ledger"
                    );

                    // Perform the write.
                   store.put(
    globalThis._mlsys_temp_state,
    "mlsys_design_ledger"
);

tx.oncomplete = () => {
    db.close();
    resolve(true);
};

                    // Handle transaction errors.
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

                    // Handle transaction abortion.
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

                } catch (err) {
                    db.close();

                    reject(new Error(
                        "IndexedDB transaction failed: "
                        + err.message
                    ));
                }
            };

            // Handle failure to open IndexedDB.
            request.onerror = () => {
                reject(new Error(
                    "indexedDB.open failed: "
                    + (
                        request.error
                            ? request.error.message
                            : "unknown error"
                    )
                    + " (storage may be disabled, full, "
                    + "or unavailable in private browsing)"
                ));
            };
        }))()
        """

        # Wait for the IndexedDB transaction to complete.
        # If the transaction fails, this raises the exception.
        await run_js(js_code)

        return True

    def _apply_pending_state(
        self,
        track,
        step,
        design,
        chapter
    ):
        """
        Shared bookkeeping used by both save() and asave().
        """

        if track:
            self._state.track = track

        step_id = (
            step
            if step is not None
            else chapter
        )

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
            # task.result() re-raises any exception that occurred
            # inside save_async().
            task.result()

            # Save completed successfully.
            self._last_save_error = None

        except Exception as e:
            # Save failed.
            self._last_save_error = str(e)

            message = (
                "[DesignLedger] SAVE FAILED - "
                "progress was NOT persisted: "
                f"{e}"
            )

            try:
                from js import console

                console.error(message)

            except Exception:
                print(
                    message,
                    file=sys.stderr
                )

        finally:
            # Clear the task only if this is still the currently
            # tracked save task.
            if self._pending_save_task is task:
                self._pending_save_task = None

    def save(
        self,
        track: str = None,
        step: int = None,
        design: dict = None,
        chapter: int = None
    ):
        """
        Persists the design decisions to storage.

        In WASM, the save runs asynchronously in the background.
        Call flush() when the caller needs to wait for completion.
        """

        self._apply_pending_state(
            track,
            step,
            design,
            chapter
        )

        if self.is_wasm:
            import asyncio

            task = asyncio.ensure_future(
                self.save_async()
            )

            task.add_done_callback(
                self._on_save_task_done
            )

            self._pending_save_task = task

        else:
            # Native/local filesystem persistence.
            self.config_dir.mkdir(
                exist_ok=True
            )

            with open(
                self.file_path,
                "w"
            ) as f:
                json.dump(
                    asdict(self._state),
                    f,
                    indent=2
                )

    async def asave(
        self,
        track: str = None,
        step: int = None,
        design: dict = None,
        chapter: int = None
    ):
        """
        Async variant of save() that awaits persistence.

        In WASM, this guarantees that save_async() completes
        before returning, or raises an exception if persistence
        fails.
        """

        self._apply_pending_state(
            track,
            step,
            design,
            chapter
        )

        if self.is_wasm:
            await self.save_async()

            self._last_save_error = None

        else:
            # Native/local filesystem persistence.
            self.config_dir.mkdir(
                exist_ok=True
            )

            with open(
                self.file_path,
                "w"
            ) as f:
                json.dump(
                    asdict(self._state),
                    f,
                    indent=2
                )

    async def flush(self):
        """
        Await any in-flight background save scheduled by save().
        """

        task = self._pending_save_task

        if task is not None:
            await task

    def get_design(
        self,
        step_id: int
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the design dictionary for a specific step."""
        return self._state.history.get(
            step_id
        )

    def get_track(self) -> str:
        """Returns the current track or NONE."""
        return self._state.track or "NONE"

    def get_baseline(
        self,
        step_id: int
    ) -> dict:
        """
        Provides the 'Gold Standard' baseline if the student
        hasn't completed previous labs.
        """
        return {}

    def __repr__(self):
        return (
            f"DesignLedger("
            f"track={self._state.track}, "
            f"step={self._state.current_step}"
            f")"
        )
