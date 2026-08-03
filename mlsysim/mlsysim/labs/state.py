# mlsysim/labs/state.py
# Persistent state management for the MLSys Design Ledger.
# Handles CLI (Local File) and Web (Browser LocalStorage) persistence.

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
        self._pending_save_task = None
        self._last_save_error: Optional[str] = None
        self.load()

    @property
    def last_save_error(self) -> Optional[str]:
        """Error message from the most recent failed background save, if any.

        Callers in WASM environments should check this after calling
        ``save()`` (e.g. on the next reactive cell run) since ``save()``
        cannot block on the result inside a synchronous notebook cell.
        """
        return self._last_save_error

    @property
    def save_pending(self) -> bool:
        """True while a WASM background save has not finished yet."""
        return self._pending_save_task is not None and not self._pending_save_task.done()

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
        if self.is_wasm:
            # Synchronous load is not possible in WASM with IndexedDB.
            # Labs must call `await ledger.load_async()` during setup.
            return self._state

        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    data["history"] = self._parse_history(data)
                    self._state = LedgerState(**data)
            except Exception:
                self._state = LedgerState()
        return self._state

    async def load_async(self) -> LedgerState:
        """Async load for WASM environments using IndexedDB."""
        if not self.is_wasm:
            return self.load()
        
        try:
            from pyodide.code import run_js
            
            js_code = """
            (async () => {
                return new Promise((resolve, reject) => {
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
                            resolve(null);
                            return;
                        }
                        try {
                            const tx = db.transaction("ledger", "readonly");
                            const store = tx.objectStore("ledger");
                            const getReq = store.get("mlsys_design_ledger");
                            getReq.onsuccess = () => resolve(getReq.result);
                            getReq.onerror = () => resolve(null);
                        } catch (err) {
                            resolve(null);
                        }
                    };
                    request.onerror = () => resolve(null);
                });
            })()
            """
            raw = await run_js(js_code)
            if raw:
                data = json.loads(raw)
                data["history"] = self._parse_history(data)
                self._state = LedgerState(**data)
        except Exception as e:
            print(f"Failed to load from IndexedDB: {e}")
            self._state = LedgerState()
        return self._state

    async def save_async(self):
        """Async save for WASM environments using IndexedDB.

        Unlike the previous implementation, this raises on failure instead
        of printing to the console and returning ``None``. Callers that can
        await (e.g. :meth:`asave`, tests) will see the real exception.
        Callers that cannot await (the synchronous :meth:`save`) capture the
        exception via a done-callback on the background task instead.
        """
        if not self.is_wasm:
            return True

        import json
        from pyodide.code import run_js
        from js import globalThis

        state_json = json.dumps(asdict(self._state))
        globalThis.__mlsys_temp_state = state_json

        js_code = """
        (async () => {
            return new Promise((resolve, reject) => {
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
                        const store = tx.objectStore("ledger");
                        const putReq = store.put(globalThis.__mlsys_temp_state, "mlsys_design_ledger");
                        putReq.onsuccess = () => resolve(true);
                        putReq.onerror = () => reject(new Error(
                            "IndexedDB put failed: " + (putReq.error ? putReq.error.message : "unknown error")
                        ));
                    } catch (err) {
                        reject(new Error("IndexedDB transaction failed: " + err.message));
                    }
                };
                request.onerror = () => reject(new Error(
                    "indexedDB.open failed: " + (request.error ? request.error.message : "unknown error")
                    + " (storage may be disabled, full, or unavailable in private browsing)"
                ));
            });
        })()
        """
        # If the JS promise rejects, Pyodide raises the corresponding
        # exception here rather than us having to poll a boolean result.
        await run_js(js_code)
        return True

    def _apply_pending_state(self, track, step, design, chapter):
        """Shared bookkeeping used by both save() and asave()."""
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
        """Done-callback for the background WASM save task.

        This is what actually fixes the silent-failure bug: instead of the
        exception from `save_async()` disappearing into an unobserved task,
        we record it and log it loudly so it's impossible to miss in the
        browser console, and expose it via `last_save_error` /
        `save_pending` so the notebook UI can surface it to the student.
        """
        try:
            task.result()
            self._last_save_error = None
        except Exception as e:
            self._last_save_error = str(e)
            message = f"[DesignLedger] SAVE FAILED - progress was NOT persisted: {e}"
            try:
                from js import console
                console.error(message)
            except Exception:
                print(message, file=sys.stderr)

    def save(self, track: str = None, step: int = None, design: dict = None, chapter: int = None):
        """Persists the design decisions to storage.

        ``chapter`` is kept as a compatibility alias for existing Co-Labs,
        while ``step`` is the newer generic ledger key.

        In WASM this schedules a background save (marimo cells are
        synchronous, so we can't await here), but unlike the previous
        implementation the resulting task's outcome is observed: failures
        are logged loudly and recorded on ``self.last_save_error`` instead
        of vanishing silently. Callers that need a hard persistence
        guarantee (tests, async code) should use :meth:`asave` instead.
        """
        self._apply_pending_state(track, step, design, chapter)

        if self.is_wasm:
            import asyncio
            task = asyncio.ensure_future(self.save_async())
            task.add_done_callback(self._on_save_task_done)
            self._pending_save_task = task
        else:
            self.config_dir.mkdir(exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(asdict(self._state), f, indent=2)

    async def asave(self, track: str = None, step: int = None, design: dict = None, chapter: int = None):
        """Async variant of :meth:`save` that awaits persistence.

        Guarantees the data is written (or raises) before returning. Use
        this from async contexts (tests, async marimo cells) whenever you
        need certainty that a save actually succeeded.
        """
        self._apply_pending_state(track, step, design, chapter)

        if self.is_wasm:
            await self.save_async()
            self._last_save_error = None
        else:
            self.config_dir.mkdir(exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(asdict(self._state), f, indent=2)

    async def flush(self):
        """Await any in-flight background save scheduled by `save()`.

        Raises the underlying exception if that save failed.
        """
        if self._pending_save_task is not None:
            await self._pending_save_task

    def get_design(self, step_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves the design dictionary for a specific step."""
        return self._state.history.get(step_id)

    def get_track(self) -> str:
        return self._state.track or "NONE"

    def get_baseline(self, step_id: int) -> dict:
        """
        Provides the 'Gold Standard' baseline if the student 
        hasn't completed previous labs.
        """
        # Logic to return pre-computed design for a specific track/step.
        return {}

    def __repr__(self):
        return f"DesignLedger(track={self._state.track}, step={self._state.current_step})"
