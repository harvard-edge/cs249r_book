# mlsysim/labs/state.py
# Persistent state management for the MLSys Design Ledger.
# Handles CLI (Local File) and Web (Browser LocalStorage) persistence.

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List

@dataclass
class LedgerState:
    """The schema for the persistent student state."""
    track: Optional[str] = None
    current_chapter: int = 0
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
        self.load()

    @property
    def is_wasm(self) -> bool:
        """Detect if we are running in a browser environment (Pyodide)."""
        return sys.platform == "emscripten"

    _LOCALSTORAGE_KEY = "mlsys_design_ledger"

    def _parse_history(self, data: dict) -> dict:
        """Normalize history from either legacy list or dict format."""
        history_data = data.get("history", {})
        if isinstance(history_data, list):
            return {int(entry.get("chapter", 0)): entry.get("design", {}) for entry in history_data}
        elif isinstance(history_data, dict):
            return {int(k) if str(k).isdigit() else k: v for k, v in history_data.items()}
        return {}

    def load(self) -> LedgerState:
        """Loads the ledger from the best available persistent storage."""
        if self.is_wasm:
            try:
                from js import localStorage  # Pyodide bridge to browser API
                raw = localStorage.getItem(self._LOCALSTORAGE_KEY)
                if raw:
                    data = json.loads(raw)
                    data["history"] = self._parse_history(data)
                    self._state = LedgerState(**data)
            except Exception:
                self._state = LedgerState()
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

    def save(self, track: str = None, chapter: int = None, design: dict = None):
        """Persists the design decisions to storage."""
        if track:
            self._state.track = track

        ch_id = chapter if chapter is not None else self._state.current_chapter
        if chapter is not None:
            self._state.current_chapter = chapter

        if design:
            self._state.history[ch_id] = design

        if self.is_wasm:
            try:
                from js import localStorage
                localStorage.setItem(
                    self._LOCALSTORAGE_KEY,
                    json.dumps(asdict(self._state)),
                )
            except Exception:
                pass  # Degrade gracefully if localStorage unavailable
        else:
            self.config_dir.mkdir(exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(asdict(self._state), f, indent=2)

    def get_design(self, chapter_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves the design dictionary for a specific chapter."""
        return self._state.history.get(chapter_id)

    def get_track(self) -> str:
        return self._state.track or "NONE"

    def get_baseline(self, chapter_id: int) -> dict:
        """
        Provides the 'Gold Standard' baseline if the student 
        hasn't completed previous labs.
        """
        # Logic to return pre-computed design for specific track/chapter
        return {}

    def __repr__(self):
        return f"DesignLedger(track={self._state.track}, ch={self._state.current_chapter})"
